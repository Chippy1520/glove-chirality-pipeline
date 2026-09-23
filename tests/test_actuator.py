import sys
import threading
import time

import pytest

from glove_chirality.actuator import (
    SerialActuator,
    command_for_event,
    list_serial_ports,
    reject_command,
)


def test_reject_command_protocol_is_exact():
    assert reject_command("evt-1", 850) == "REJECT|evt-1|850\n"
    assert reject_command("evt-1", 850.9) == "REJECT|evt-1|850\n"
    assert reject_command("evt-1", 0) == "REJECT|evt-1|0\n"
    with pytest.raises(ValueError):
        reject_command("bad|id", 10)
    with pytest.raises(ValueError):
        reject_command("bad\nid", 10)
    with pytest.raises(ValueError):
        reject_command("evt-1", -1)


def test_command_for_event_gates_and_does_not_mutate_commanded():
    commanded: set[str] = set()
    common = {
        "reject_class": "right",
        "event_id": "evt-1",
        "delay_ms": 850,
        "commanded": commanded,
    }
    assert command_for_event(mode="shadow", status="accepted", prediction="right", **common) is None
    assert command_for_event(mode="preview", status="accepted", prediction="right", **common) is None
    assert command_for_event(mode="armed", status="accepted", prediction="left", **common) is None
    assert (
        command_for_event(
            mode="armed",
            status="multiple_candidates",
            prediction="right",
            **common,
        )
        is None
    )
    assert command_for_event(mode="armed", status="partial", prediction="right", **common) is None

    command = command_for_event(mode="armed", status="accepted", prediction="right", **common)
    assert command == "REJECT|evt-1|850\n"
    assert commanded == set()

    commanded.add("evt-1")
    assert command_for_event(mode="armed", status="accepted", prediction="right", **common) is None
    assert commanded == {"evt-1"}


class FakePort:
    def __init__(self, *, fail_write: bool = False, write_delay_s: float = 0.0):
        self.fail_write = fail_write
        self.write_delay_s = write_delay_s
        self.writes: list[bytes] = []
        self.write_count = 0
        self.is_open = True
        self.closed = False
        self.write_entered = threading.Event()
        self._rx = bytearray()
        self._lock = threading.Lock()

    def write(self, data: bytes) -> int:
        self.write_entered.set()
        if self.write_delay_s:
            time.sleep(self.write_delay_s)
        with self._lock:
            self.write_count += 1
            if self.fail_write:
                raise OSError("broken write")
            self.writes.append(bytes(data))
            return len(data)

    @property
    def in_waiting(self) -> int:
        with self._lock:
            return len(self._rx)

    def read(self, count: int) -> bytes:
        with self._lock:
            chunk = bytes(self._rx[:count])
            del self._rx[:count]
            return chunk

    def push(self, data: bytes) -> None:
        with self._lock:
            self._rx.extend(data)

    def close(self) -> None:
        self.is_open = False
        self.closed = True


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_submit_returns_immediately_and_ack_does_not_write_again():
    port = FakePort(write_delay_s=0.3)
    acks: list[str] = []
    commands: list[str] = []
    actuator = SerialActuator()
    try:
        actuator.connect(
            "COM_TEST",
            115200,
            port_factory=lambda port_name, baud: port,
            on_ack=acks.append,
            on_command=commands.append,
        )
        started = time.perf_counter()
        actuator.submit("REJECT|evt-1|850\n")
        elapsed = time.perf_counter() - started
        assert elapsed < 0.1
        assert port.write_entered.wait(2.0)
        assert _wait_until(lambda: port.write_count == 1)
        assert port.writes == [b"REJECT|evt-1|850\n"]
        assert commands == ["REJECT|evt-1|850"]
        port.push(b"ACK|evt-1|ok\n")
        assert _wait_until(lambda: actuator.snapshot()["last_ack"] == "ACK|evt-1|ok")
        assert acks == ["ACK|evt-1|ok"]
        time.sleep(0.15)
        assert port.write_count == 1
        snapshot = actuator.snapshot()
        assert snapshot["connected"] is True
        assert snapshot["port"] == "COM_TEST"
        assert snapshot["baud"] == 115200
        assert snapshot["last_command"] == "REJECT|evt-1|850"
        assert snapshot["fault"] is None
    finally:
        actuator.disconnect()
        assert port.closed is True


def test_broken_write_faults_once_and_does_not_retry():
    port = FakePort(fail_write=True)
    faults: list[str] = []
    finished = threading.Event()
    actuator = SerialActuator()

    def on_fault(message: str) -> None:
        faults.append(message)
        actuator.disconnect()
        finished.set()

    actuator.connect(
        "COM_BROKEN",
        9600,
        port_factory=lambda port_name, baud: port,
        on_fault=on_fault,
    )
    actuator.submit("REJECT|evt-9|10\n")
    assert finished.wait(2.0)
    assert faults == ["broken write"]
    assert port.write_count == 1
    try:
        actuator.submit("REJECT|evt-10|10\n")
    except RuntimeError:
        pass
    time.sleep(0.15)
    assert port.write_count == 1
    snapshot = actuator.snapshot()
    assert snapshot["connected"] is False
    assert snapshot["fault"] == "broken write"
    assert snapshot["last_ack"] is None


def test_list_ports_tells_user_how_to_install_pyserial(monkeypatch):
    for name in list(sys.modules):
        if name == "serial" or name.startswith("serial."):
            monkeypatch.delitem(sys.modules, name, raising=False)
    monkeypatch.setitem(sys.modules, "serial", None)
    actuator = SerialActuator()
    assert actuator.available() is False
    with pytest.raises(RuntimeError, match=r'python -m pip install -e "\.\[factory\]"'):
        actuator.list_ports()
    with pytest.raises(RuntimeError, match=r'python -m pip install -e "\.\[factory\]"'):
        list_serial_ports()


def test_module_import_does_not_import_pyserial():
    import ast
    from pathlib import Path

    from glove_chirality import actuator

    tree = ast.parse(Path(actuator.__file__).read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module or ""]
        else:
            continue
        assert not any(name == "serial" or name.startswith("serial.") for name in names)
