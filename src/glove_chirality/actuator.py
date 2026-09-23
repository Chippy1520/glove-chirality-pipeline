"""Optional non-blocking serial reject actuator.

The caller decides when a REJECT is warranted. This module only formats that
command and, when a port is connected, writes it from a worker thread. ACK
lines are recorded and never generate another REJECT. pyserial is imported
only when a port is listed or opened.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable

_INSTALL = 'python -m pip install -e ".[factory]"'
_STOP = object()


def reject_command(event_id: str, delay_ms: int) -> str:
    """Return the exact REJECT line, including the trailing newline."""
    if not isinstance(event_id, str) or "|" in event_id or "\n" in event_id or "\r" in event_id:
        raise ValueError("event_id must not contain '|' or newlines")
    if delay_ms < 0:
        raise ValueError("delay_ms must be non-negative")
    return f"REJECT|{event_id}|{int(delay_ms)}\n"


def command_for_event(
    *,
    mode: str,
    status: str,
    prediction: str | None,
    reject_class: str,
    event_id: str,
    delay_ms: int,
    commanded: set[str],
) -> str | None:
    """Return a REJECT command only for a new armed acceptance of reject_class.

    Does not mutate ``commanded``. preview, shadow, a non-reject prediction,
    partial / multiple_candidates, and an id already in ``commanded`` return None.
    """
    if mode != "armed" or status != "accepted" or prediction != reject_class:
        return None
    if event_id in commanded:
        return None
    return reject_command(event_id, delay_ms)


def list_serial_ports() -> list[dict[str, str]]:
    """List serial ports. Raises RuntimeError with install instructions if pyserial is missing."""
    try:
        from serial.tools import list_ports
    except ImportError as exc:
        raise RuntimeError(f"pyserial is not installed. Run: {_INSTALL}") from exc
    ports: list[dict[str, str]] = []
    for info in list_ports.comports():
        device = str(getattr(info, "device", "") or "")
        description = str(getattr(info, "description", "") or "")
        ports.append(
            {
                "device": device,
                "description": description,
                "label": f"{device} | {description}",
            }
        )
    return ports


def _pyserial_available() -> bool:
    try:
        import serial as _serial
    except ImportError:
        return False
    return _serial is not None


def _default_port_factory(port: str, baud: int):
    import serial

    return serial.Serial(port, baud, timeout=0.05)


class SerialActuator:
    """Serial writer whose read/write path lives entirely on a worker thread."""

    def __init__(self) -> None:
        self._port = None
        self._port_name: str | None = None
        self._baud: int | None = None
        self._connected = False
        self._last_command: str | None = None
        self._last_ack: str | None = None
        self._fault: str | None = None
        self._fault_reported = False
        self._on_fault: Callable[[str], None] | None = None
        self._on_ack: Callable[[str], None] | None = None
        self._on_command: Callable[[str], None] | None = None
        self._queue: queue.Queue = queue.Queue()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._rx = bytearray()

    def available(self) -> bool:
        return _pyserial_available()

    def list_ports(self) -> list[dict[str, str]]:
        return list_serial_ports()

    def connect(
        self,
        port: str,
        baud: int = 115200,
        *,
        port_factory=None,
        on_fault: Callable[[str], None] | None = None,
        on_ack: Callable[[str], None] | None = None,
        on_command: Callable[[str], None] | None = None,
    ) -> dict:
        if self._thread is not None and self._thread.is_alive():
            self.disconnect()
        factory = port_factory or _default_port_factory
        handle = factory(port, baud)
        self._drain()
        self._rx.clear()
        self._stop.clear()
        self._on_fault = on_fault
        self._on_ack = on_ack
        self._on_command = on_command
        with self._lock:
            self._port = handle
            self._port_name = port
            self._baud = int(baud)
            self._connected = True
            self._last_command = None
            self._last_ack = None
            self._fault = None
            self._fault_reported = False
        self._thread = threading.Thread(
            target=self._worker,
            name="glove-serial-actuator",
            daemon=True,
        )
        self._thread.start()
        return self.snapshot()

    def submit(self, command: str) -> None:
        """Queue a command without waiting for serial read or write."""
        if not isinstance(command, str):
            raise TypeError("command must be a string")
        with self._lock:
            connected = self._connected
        if not connected:
            raise RuntimeError("actuator is not connected")
        self._queue.put_nowait(command)

    def disconnect(self) -> None:
        """Stop the worker and close the port. Safe to call from on_fault."""
        self._stop.set()
        self._wake()
        thread = self._thread
        if (
            thread is not None
            and thread.is_alive()
            and thread is not threading.current_thread()
        ):
            thread.join(timeout=2.0)
        self._close_port()
        with self._lock:
            self._connected = False

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "connected": self._connected,
                "port": self._port_name,
                "baud": self._baud,
                "last_command": self._last_command,
                "last_ack": self._last_ack,
                "fault": self._fault,
            }

    def _worker(self) -> None:
        try:
            while not self._stop.is_set():
                command = self._poll_command()
                if self._stop.is_set():
                    break
                if command is not None:
                    self._write_command(command)
                if self._stop.is_set():
                    break
                self._read_acks()
        except (OSError, AttributeError) as exc:
            self._fail(exc)
        finally:
            self._close_port()
            with self._lock:
                self._connected = False

    def _poll_command(self) -> str | None:
        try:
            item = self._queue.get(timeout=0.05)
        except queue.Empty:
            return None
        if item is _STOP:
            self._stop.set()
            return None
        return item

    def _write_command(self, command: str) -> None:
        port = self._require_port()
        payload = command.encode("utf-8")
        written = port.write(payload)
        if isinstance(written, int) and written < len(payload):
            raise OSError(f"short serial write: {written} of {len(payload)} bytes")
        stripped = command.strip()
        with self._lock:
            self._last_command = stripped
        self._emit(self._on_command, stripped)

    def _read_acks(self) -> None:
        port = self._require_port()
        data = self._read_available(port)
        if not data:
            return
        for line in self._consume_lines(data):
            if not line.startswith("ACK|"):
                continue
            with self._lock:
                self._last_ack = line
            self._emit(self._on_ack, line)

    def _read_available(self, port) -> bytes:
        if hasattr(port, "in_waiting"):
            waiting = int(port.in_waiting or 0)
            if waiting <= 0:
                return b""
            chunk = port.read(waiting)
            return chunk or b""
        if hasattr(port, "readline"):
            chunk = port.readline()
            return chunk or b""
        return b""

    def _consume_lines(self, data: bytes) -> list[str]:
        self._rx.extend(data)
        lines: list[str] = []
        while True:
            newline = self._rx.find(b"\n")
            if newline < 0:
                break
            raw = bytes(self._rx[:newline])
            del self._rx[: newline + 1]
            line = raw.decode("utf-8", errors="replace").strip()
            if line:
                lines.append(line)
        return lines

    def _require_port(self):
        with self._lock:
            port = self._port
        if port is None:
            raise OSError("serial port is disconnected")
        is_open = getattr(port, "is_open", True)
        if not is_open:
            raise OSError("serial port is closed")
        return port

    def _fail(self, exc: BaseException) -> None:
        message = str(exc) or exc.__class__.__name__
        with self._lock:
            first = not self._fault_reported
            if first:
                self._fault = message
                self._connected = False
                self._fault_reported = True
            callback = self._on_fault
        self._stop.set()
        if first:
            self._emit(callback, message)

    def _emit(self, callback: Callable[[str], None] | None, message: str) -> None:
        if callback is None:
            return
        try:
            callback(message)
        except Exception:  # noqa: BLE001 — a listener must not retry or fault the link
            return

    def _close_port(self) -> None:
        with self._lock:
            port = self._port
            self._port = None
        if port is None:
            return
        try:
            closer = getattr(port, "close", None)
            if closer is not None:
                closer()
        except (OSError, AttributeError) as exc:
            self._fail(exc)

    def _wake(self) -> None:
        try:
            self._queue.put_nowait(_STOP)
        except queue.Full:
            return

    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                return
