from __future__ import annotations

import importlib.util
import os
import socket
import subprocess
import threading
import uuid
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from glove_chirality import gui_commands
from glove_chirality.comparison import discover_model_runs, sort_model_runs
from glove_chirality.gui_processes import ProcessSlots


@dataclass(frozen=True)
class LogEntry:
    sequence: int
    slot: str
    text: str


@dataclass
class JobState:
    job_id: str
    action: str
    status: str
    started_at: str
    finished_at: str | None = None
    exit_code: int | None = None


def _text(payload: dict[str, Any], key: str, default: str = "") -> str:
    return str(payload.get(key, default)).strip()


def _integer(payload: dict[str, Any], key: str, default: int) -> int:
    return int(payload.get(key, default))


def _number(payload: dict[str, Any], key: str, default: float) -> float:
    return float(payload.get(key, default))


def _boolean(payload: dict[str, Any], key: str, default: bool = False) -> bool:
    value = payload.get(key, default)
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def build_web_command(action: str, payload: dict[str, Any]) -> tuple[str, list[str]]:
    """Map a named web action to an existing, typed CLI command builder."""
    if action == "extract_dataset":
        command = gui_commands.extract_dataset(
            _text(payload, "left"),
            _text(payload, "right"),
            _text(payload, "output"),
            _text(payload, "config"),
        )
    elif action == "extract_single":
        command = gui_commands.extract_single(
            _text(payload, "input"),
            _text(payload, "output"),
            _text(payload, "label", "unknown"),
            _text(payload, "config"),
        )
    elif action == "preview":
        command = gui_commands.preview(
            _text(payload, "video"),
            _text(payload, "output"),
            _number(payload, "seconds", 0.0),
            _text(payload, "config"),
            _number(payload, "warmup_seconds", 2.0),
        )
    elif action == "train":
        command = gui_commands.train(
            manifest=_text(payload, "manifest"),
            output=_text(payload, "output"),
            model=_text(payload, "model", "resnet18"),
            epochs=_integer(payload, "epochs", 20),
            batch_size=_integer(payload, "batch_size", 32),
            image_size=_integer(payload, "image_size", 224),
            learning_rate=_number(payload, "learning_rate", 0.001),
            validation_fraction=_number(payload, "validation_fraction", 0.2),
            seed=_integer(payload, "seed", 42),
            device=_text(payload, "device", "auto"),
            workers=_integer(payload, "workers", 0),
            amp=_boolean(payload, "amp"),
            loss=_text(payload, "loss", "weighted_cross_entropy"),
            recall_target=_text(payload, "recall_target", "right"),
            recall_weight=_number(payload, "recall_weight", 1.0),
            selection_metric=_text(payload, "selection_metric", "macro_recall"),
            augmentation=_text(payload, "augmentation", "standard"),
            tensorboard_logdir=_text(payload, "tensorboard_logdir"),
        )
    elif action == "infer_video":
        command = gui_commands.infer_video(
            _text(payload, "video"),
            _text(payload, "checkpoint"),
            _text(payload, "output"),
            _text(payload, "config"),
            _text(payload, "device", "auto"),
            _text(payload, "decision_class", "argmax"),
            _number(payload, "decision_threshold", 0.5),
        )
    elif action == "infer_images":
        command = gui_commands.infer_images(
            _text(payload, "input"),
            _text(payload, "checkpoint"),
            _text(payload, "output"),
            _text(payload, "device", "auto"),
            _text(payload, "decision_class", "argmax"),
            _number(payload, "decision_threshold", 0.5),
        )
    elif action == "infer_live":
        command = gui_commands.infer_live(
            _text(payload, "source", "0"),
            _text(payload, "checkpoint"),
            _text(payload, "output"),
            _text(payload, "config"),
            _text(payload, "device", "auto"),
            _boolean(payload, "amp"),
            _text(payload, "decision_class", "argmax"),
            _number(payload, "decision_threshold", 0.5),
        )
    elif action == "explain":
        command = gui_commands.explain(
            _text(payload, "image"),
            _text(payload, "checkpoint"),
            _text(payload, "output"),
            _text(payload, "device", "auto"),
            _text(payload, "method", "smoothgrad"),
            _text(payload, "target_class", "predicted"),
        )
    elif action == "tensorboard":
        command = gui_commands.tensorboard(
            _text(payload, "logdir"),
            _integer(payload, "port", 6006),
        )
        return "tensorboard", command
    else:
        raise ValueError(f"Unknown web action: {action}")
    return "pipeline", command


def _command_option(command: list[str], option: str) -> str:
    try:
        return command[command.index(option) + 1]
    except (ValueError, IndexError) as error:
        raise ValueError(f"Missing required TensorBoard option: {option}") from error


def _validate_tensorboard(command: list[str], workdir: Path) -> None:
    if importlib.util.find_spec("tensorboard") is None:
        raise ValueError("TensorBoard is not installed in this Python environment")
    logdir = Path(_command_option(command, "--logdir")).expanduser()
    if not logdir.is_absolute():
        logdir = workdir / logdir
    if not logdir.resolve().is_dir():
        raise ValueError(f"TensorBoard log directory not found: {logdir.resolve()}")
    port = int(_command_option(command, "--port"))
    if not 1 <= port <= 65535:
        raise ValueError("TensorBoard port must be in [1, 65535]")
    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.bind(("127.0.0.1", port))
    except OSError as error:
        raise ValueError(f"TensorBoard port {port} is already in use") from error
    finally:
        probe.close()


def _terminate_process(process: subprocess.Popen[str]) -> None:
    if os.name == "nt":
        result = subprocess.run(
            ["taskkill.exe", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if result.returncode != 0 and process.poll() is None:
            process.terminate()
    else:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


class CommandService:
    """Thread-safe subprocess state shared by Flask request threads."""

    def __init__(self, workdir: str | Path, max_log_entries: int = 2000):
        self.workdir = Path(workdir).resolve()
        self.processes = ProcessSlots()
        self._lock = threading.RLock()
        self._logs: deque[LogEntry] = deque(maxlen=max_log_entries)
        self._sequence = 0
        self._jobs: dict[str, JobState | None] = {"pipeline": None, "tensorboard": None}
        self.comparison_root = self.workdir / "outputs"

    def _append(self, slot: str, text: str) -> None:
        with self._lock:
            self._sequence += 1
            self._logs.append(LogEntry(self._sequence, slot, text.rstrip("\n")))

    def start(self, slot: str, command: list[str], *, action: str | None = None) -> str:
        with self._lock:
            self.processes.ensure_available(slot)
            if slot == "tensorboard":
                _validate_tensorboard(command, self.workdir)
            flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            environment = os.environ.copy()
            environment["PYTHONUNBUFFERED"] = "1"
            process = subprocess.Popen(
                command,
                cwd=self.workdir,
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=flags,
            )
            self.processes.claim(slot, process)
            job = JobState(
                job_id=uuid.uuid4().hex,
                action=action or slot,
                status="running",
                started_at=datetime.now(timezone.utc).isoformat(),
            )
            self._jobs[slot] = job
            self._append(slot, "$ " + subprocess.list2cmdline(command))

        def collect() -> None:
            assert process.stdout is not None
            for line in process.stdout:
                self._append(slot, line)
            code = process.wait()
            with self._lock:
                self.processes.release(slot, process)
                if self._jobs.get(slot) is job:
                    if job.status == "stopping":
                        job.status = "cancelled"
                    else:
                        job.status = "succeeded" if code == 0 else "failed"
                    job.finished_at = datetime.now(timezone.utc).isoformat()
                    job.exit_code = code
            self._append(slot, f"Process finished with exit code {code}.")

        threading.Thread(target=collect, daemon=True).start()
        return job.job_id

    def stop(self, slot: str) -> bool:
        with self._lock:
            process = self.processes.get(slot)
            if process is None or process.poll() is not None:
                return False
            job = self._jobs.get(slot)
            if job is not None:
                job.status = "stopping"
            _terminate_process(process)
            self._append(slot, "Stop requested.")
            return True

    def snapshot(self, include_logs: bool) -> dict[str, Any]:
        with self._lock:
            running = {
                slot: (process := self.processes.get(slot)) is not None
                and process.poll() is None
                for slot in ("pipeline", "tensorboard")
            }
            return {
                "running": running,
                "jobs": {
                    slot: asdict(job) if job is not None else None
                    for slot, job in self._jobs.items()
                },
                "logs": [asdict(entry) for entry in self._logs] if include_logs else [],
                "last_sequence": self._sequence,
            }

    def comparison(self, metric: str, reveal_paths: bool) -> list[dict[str, Any]]:
        runs = sort_model_runs(
            discover_model_runs([self.comparison_root]),
            metric,
        )
        rows = []
        for run in runs:
            row = asdict(run)
            if not reveal_paths:
                row["source"] = Path(run.source).name
            rows.append(row)
        return rows

    def set_comparison_root(self, path: str | Path) -> Path:
        candidate = self.resolve_path(path)
        if not candidate.is_dir():
            raise ValueError(f"Comparison directory not found: {candidate}")
        self.comparison_root = candidate
        return candidate

    def resolve_path(self, raw_path: str | Path) -> Path:
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = self.workdir / path
        return path.resolve()

    def browse(self, raw_path: str | Path | None = None) -> dict[str, Any]:
        path = self.resolve_path(raw_path or self.workdir)
        if path.is_file():
            path = path.parent
        if not path.is_dir():
            raise ValueError(f"Directory not found: {path}")
        entries = []
        for child in sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower())):
            if child.name.startswith("."):
                continue
            entries.append({
                "name": child.name,
                "path": str(child),
                "is_directory": child.is_dir(),
            })
        return {
            "path": str(path),
            "parent": str(path.parent) if path.parent != path else None,
            "entries": entries,
        }

    def shutdown(self) -> None:
        for slot in ("pipeline", "tensorboard"):
            self.stop(slot)
