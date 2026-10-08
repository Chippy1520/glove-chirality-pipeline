from __future__ import annotations

import importlib.metadata
import importlib.util
import json
import math
import os
import socket
import subprocess
import threading
import time
import uuid
from collections import deque
from dataclasses import asdict, dataclass, field
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
    timestamp: str = ""
    level: str = "info"
    stream: str = "service"
    job_id: str | None = None


@dataclass
class JobState:
    job_id: str
    action: str
    status: str
    started_at: str
    finished_at: str | None = None
    exit_code: int | None = None
    workflow: str = "pipeline"
    stage: str = "starting"
    progress: float | dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    result: dict[str, Any] = field(default_factory=dict)
    artifacts: list[dict[str, Any]] = field(default_factory=list)
    output: str = ""
    command: list[str] = field(default_factory=list)
    preflight: dict[str, Any] = field(default_factory=dict)


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
    if action == "audit_dataset":
        gui_commands._required(manifest=_text(payload, "manifest"), output=_text(payload, "output"))
        command = gui_commands._base() + ["audit-dataset",
                   "--manifest", _text(payload, "manifest"), "--output", _text(payload, "output")]
    elif action == "extract_dataset":
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
            epochs=int(str(payload.get("epochs", 20))),
            batch_size=_integer(payload, "batch_size", 32),
            image_size=_integer(payload, "image_size", 224),
            learning_rate=_number(payload, "learning_rate", 0.001),
            head_only_epochs=int(str(payload.get("head_only_epochs", 0))),
            backbone_learning_rate=(
                _number(payload, "backbone_learning_rate", 0.0001)
                if payload.get("backbone_learning_rate") is not None
                and _text(payload, "backbone_learning_rate") else None
            ),
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
        self._history: dict[str, JobState] = {}
        self._cancel: dict[str, threading.Event] = {}
        self._started: dict[str, float] = {}
        self._elapsed: dict[str, float] = {}
        self._payloads: dict[str, dict] = {}
        self._baseline: dict[str, dict[str, tuple]] = {}

    def _job_dir(self, job_id: str) -> Path:
        return self.workdir / "outputs" / "jobs" / job_id

    def _persist(self, job: JobState) -> None:
        directory = self._job_dir(job.job_id)
        directory.mkdir(parents=True, exist_ok=True)
        data = self.get_job(job.job_id)
        temporary = directory / "job.json.tmp"
        temporary.write_text(json.dumps(data, indent=2, allow_nan=False), encoding="utf-8")
        temporary.replace(directory / "job.json")

    def _available(self, slot: str) -> None:
        self.processes.ensure_available(slot)
        active = self._jobs.get(slot)
        if active and active.status in {"preflight", "starting", "running", "stopping"}:
            raise RuntimeError(f"Process slot {slot!r} is already running")

    def _reserve(self, slot: str, action: str, status: str, payload: dict) -> JobState:
        self._available(slot)
        job = JobState(uuid.uuid4().hex, action, status,
                       datetime.now(timezone.utc).isoformat(), workflow=slot, stage=status,
                       output=str(payload.get("output", "")))
        self._jobs[slot] = job
        self._history[job.job_id] = job
        self._cancel[job.job_id] = threading.Event()
        self._started[job.job_id] = time.monotonic()
        self._payloads[job.job_id] = payload.copy()
        self._persist(job)
        return job

    def start_workflow(self, action: str, payload: dict[str, Any]) -> str:
        from glove_chirality.preflight import _REQUIRED

        if action not in _REQUIRED or action == "factory":
            raise ValueError(f"Unknown web action: {action}")
        # Reserve before starting the checker: concurrent preflights can open the
        # same camera/model and must not race just because no subprocess exists yet.
        slot = "tensorboard" if action == "tensorboard" else "pipeline"
        with self._lock:
            job = self._reserve(slot, action, "preflight", payload)

        def prepare():
            try:
                from glove_chirality.preflight import check_workflow

                report = check_workflow(action, payload, self.workdir)
                with self._lock:
                    job.preflight = report
                    if self._cancel[job.job_id].is_set():
                        self._finish(job, "cancelled")
                    elif not report["ok"]:
                        job.error = {"stage": "preflight", "message": report["summary"],
                                     "errors": report["errors"]}
                        self._finish(job, "failed")
                    else:
                        _, command = build_web_command(action, payload)
                        self._spawn(slot, command, job)
            except Exception as error:  # noqa: BLE001 - never lose an asynchronous job failure
                with self._lock:
                    job.error = {"stage": job.stage, "message": str(error)}
                    self._append(slot, str(error), job_id=job.job_id, level="error")
                    self._finish(job, "cancelled" if self._cancel[job.job_id].is_set() else "failed")
        threading.Thread(target=prepare, daemon=True).start()
        return job.job_id

    def _finish(self, job: JobState, status: str, code: int | None = None) -> None:
        job.status = status
        job.stage = status
        job.finished_at = datetime.now(timezone.utc).isoformat()
        job.exit_code = code
        self._elapsed[job.job_id] = time.monotonic() - self._started[job.job_id]
        if status == "succeeded" and not isinstance(job.progress, dict):
            job.progress = 1.0
        self._persist(job)

    def get_job(self, job_id: str, reveal_paths: bool = True) -> dict[str, Any]:
        with self._lock:
            job = self._history[job_id]
            data = asdict(job)
            data.update(id=job_id, state=job.status, start=job.started_at, end=job.finished_at,
                        elapsed=self._elapsed.get(job_id, time.monotonic() - self._started[job_id]))
            if not reveal_paths:
                # Allowlist the complete object, rather than trying to scrub nested
                # CLI results/preflight errors, which may contain arbitrary secrets.
                public = {key: data[key] for key in (
                    "id", "job_id", "workflow", "action", "state", "status", "start", "end",
                    "started_at", "finished_at", "elapsed", "exit_code", "progress")}
                public["stage"] = job.status
                if isinstance(job.progress, dict):
                    public["progress"] = {key: value for key, value in job.progress.items()
                                          if key in {"epoch", "epochs", "frame", "frames", "total_frames",
                                                     "fps", "train_loss", "validation_loss", "elapsed_time"}
                                          and isinstance(value, (int, float)) and not isinstance(value, bool)}
                public["error"] = {"message": "Job failed; details are host-only"} if job.error else None
                public["result"] = {}
                public["artifacts"] = []
                public["preflight"] = {"ok": job.preflight.get("ok")} if job.preflight else {}
                return public
            return data

    def job_logs(self, job_id: str) -> list[dict[str, Any]]:
        with self._lock:
            if job_id not in self._history:
                raise KeyError(job_id)
            file = self._job_dir(job_id) / "logs.jsonl"
            return [json.loads(line) for line in file.read_text(encoding="utf-8").splitlines()] \
                if file.is_file() else []

    def job_artifacts(self, job_id: str) -> list[dict[str, Any]]:
        job = self.get_job(job_id)
        return job["artifacts"] if job["status"] in {"succeeded", "failed"} else []

    def stop_job(self, job_id: str) -> bool:
        with self._lock:
            job = self._history[job_id]
            if job.status not in {"preflight", "starting", "running", "stopping"}:
                return False
            self._cancel[job_id].set()
            job.status = "stopping"
            process = self.processes.get(job.workflow)
            self._persist(job)
        if process is not None and process.poll() is None:
            _terminate_process(process)
        self._append(job.workflow, "Stop requested.", job_id=job_id)
        return True

    def _artifact_candidates(self, job: JobState) -> list[Path]:
        if not job.output:
            return []
        output = self.resolve_path(job.output)
        if job.action in {"extract_dataset", "extract_single", "infer_video", "infer_live"}:
            return [output / "manifest.csv", output / "event_report.csv",
                    output / "predictions.csv", output / "events.jsonl"]
        if job.action == "train":
            return [output, output.with_suffix(output.suffix + ".metrics.json")]
        return [output]

    @staticmethod
    def _fingerprint(path: Path) -> tuple | None:
        try:
            stat = path.stat()
            return stat.st_mtime_ns, stat.st_size
        except OSError:
            return None

    def _collect_artifacts(self, job: JobState) -> None:
        from glove_chirality.dataset_audit import audit_dataset

        candidates = self._artifact_candidates(job)
        baseline = self._baseline.get(job.job_id, {})
        for path in candidates:
            fingerprint = self._fingerprint(path)
            if not path.is_file() or not fingerprint or fingerprint == baseline.get(str(path)):
                continue
            if path.stat().st_size == 0:
                continue
            if path.suffix == ".json":
                data = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    job.result.update(data)
            job.artifacts.append({"name": path.name, "path": str(path), "size": path.stat().st_size})
        if job.action in {"extract_dataset", "extract_single"}:
            manifest = self.resolve_path(job.output) / "manifest.csv"
            if not any(item["path"] == str(manifest) for item in job.artifacts):
                raise ValueError("Extraction did not produce a new manifest")
            report = audit_dataset(manifest, self._job_dir(job.job_id) / "dataset_audit.json")
            job.result.update(dataset_audit=report, dataset_ready=report["dataset_ready"])
            if not report["validity"]["ok"]:
                raise ValueError("Extracted dataset failed audit: " + "; ".join(report["errors"]))
        if job.action == "train":
            job.result["checkpoint_valid"] = bool(job.artifacts and
                any(item["path"] == str(self.resolve_path(job.output)) for item in job.artifacts))
            if not job.result["checkpoint_valid"]:
                raise ValueError("Training did not produce a new checkpoint")
            from glove_chirality.preflight import probe_classifier

            probe_classifier(self.resolve_path(job.output), "cpu", {})
            metrics_path = self.resolve_path(job.output).with_suffix(
                self.resolve_path(job.output).suffix + ".metrics.json")
            if not any(item["path"] == str(metrics_path) for item in job.artifacts):
                raise ValueError("Training did not produce final metrics")

    def _progress(self, job: JobState, text: str) -> None:
        if not text.startswith("GRIP_PROGRESS ") or len(text) > 65536:
            return
        try:
            data = json.loads(text[len("GRIP_PROGRESS "):])
            if not isinstance(data, dict):
                return
            progress = data.get("progress")
            if isinstance(progress, dict):
                if len(progress) > 100:
                    return
                json.dumps(progress, allow_nan=False)
                for key in ("epoch", "epochs", "frame", "frames", "total_frames", "elapsed_time", "fps"):
                    if key in progress and (isinstance(progress[key], bool)
                                            or not isinstance(progress[key], (float, int))
                                            or progress[key] < 0):
                        return
                if ("epoch" in progress and "epochs" in progress
                        and progress["epoch"] > progress["epochs"]):
                    return
            elif (isinstance(progress, bool) or not isinstance(progress, (float, int))
                  or not math.isfinite(progress) or not 0 <= progress <= 1):
                return
            stage = data.get("stage")
            if not isinstance(stage, str) or not stage or len(stage) > 128:
                return
            if isinstance(data.get("result"), dict):
                # Reject non-JSON finite payloads before mutating retained state.
                json.dumps(data["result"], allow_nan=False)
                job.result.update(data["result"])
            job.stage = stage
            job.progress = progress if isinstance(progress, dict) else float(progress)
            self._persist(job)
        except (ValueError, TypeError, OverflowError):
            pass

    def _spawn(self, slot: str, command: list[str], job: JobState) -> None:
        job.status = job.stage = "starting"
        job.command = command.copy()
        self._baseline[job.job_id] = {str(path): self._fingerprint(path)
                                      for path in self._artifact_candidates(job)}
        versions = {}
        for package in ("glove-chirality", "torch", "torchvision", "opencv-python", "timm"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                pass
        configuration = {"action": job.action, "command": command,
                         "payload": self._payloads[job.job_id], "versions": versions}
        manifest = self._payloads[job.job_id].get("manifest")
        if manifest:
            from glove_chirality.dataset_audit import file_sha256

            configuration["dataset_sha256"] = file_sha256(self.resolve_path(manifest))
        (self._job_dir(job.job_id) / "config.json").write_text(
            json.dumps(configuration, indent=2, allow_nan=False), encoding="utf-8")
        flags = subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0
        environment = os.environ.copy()
        environment["PYTHONUNBUFFERED"] = "1"
        process = subprocess.Popen(command, cwd=self.workdir, env=environment,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                                   encoding="utf-8", errors="replace", bufsize=1, creationflags=flags)
        self.processes.claim(slot, process)
        job.status = job.stage = "running"
        self._persist(job)
        self._append(slot, "$ " + subprocess.list2cmdline(command), job_id=job.job_id)

        def reader(pipe, stream):
            try:
                for line in pipe:
                    self._append(slot, line, stream=stream, job_id=job.job_id,
                                 level="error" if stream == "stderr" else "info")
                    with self._lock:
                        self._progress(job, line.rstrip("\n"))
            finally:
                pipe.close()

        def collect():
            readers = [threading.Thread(target=reader, args=(pipe, stream), daemon=True)
                       for pipe, stream in ((process.stdout, "stdout"), (process.stderr, "stderr"))]
            for thread in readers:
                thread.start()
            code = process.wait()
            for thread in readers:
                thread.join()
            with self._lock:
                status = "cancelled" if self._cancel[job.job_id].is_set() else (
                    "succeeded" if code == 0 else "failed")
                if status == "succeeded" and job.output:
                    job.stage = "artifacts"
            # Audits/model checks may be slow: do not block state polling or Stop.
            if status == "succeeded" and job.output:
                try:
                    self._collect_artifacts(job)
                except Exception as error:  # noqa: BLE001 - retain post-run failures
                    status = "failed"
                    job.error = {"stage": "artifacts", "message": str(error)}
                    if job.action == "train":
                        job.result["checkpoint_valid"] = False
                        job.artifacts = []
            elif status == "failed":
                job.error = {"stage": job.stage, "message": f"Process exited with code {code}"}
            with self._lock:
                if self._cancel[job.job_id].is_set():
                    status = "cancelled"
                self._append(slot, f"Process finished with exit code {code}.", job_id=job.job_id)
                self._finish(job, status, code)
                self.processes.release(slot, process)
        threading.Thread(target=collect, daemon=True).start()

    def _append(self, slot: str, text: str, *, stream: str = "service",
                job_id: str | None = None, level: str = "info") -> None:
        with self._lock:
            self._sequence += 1
            entry = LogEntry(self._sequence, slot, text.rstrip("\n"),
                             datetime.now(timezone.utc).isoformat(), level, stream, job_id)
            self._logs.append(entry)
            if job_id is not None:
                with (self._job_dir(job_id) / "logs.jsonl").open("a", encoding="utf-8") as file:
                    file.write(json.dumps(asdict(entry)) + "\n")

    def start(self, slot: str, command: list[str], *, action: str | None = None) -> str:
        """Compatibility entry point for already-built commands (no workflow preflight)."""
        with self._lock:
            self._available(slot)
            if slot == "tensorboard":
                _validate_tensorboard(command, self.workdir)
            job = self._reserve(slot, action or slot, "starting", {})
            try:
                self._spawn(slot, command, job)
            except Exception as error:
                job.error = {"stage": "starting", "message": str(error)}
                self._finish(job, "failed")
                raise
            return job.job_id

    def stop(self, slot: str) -> bool:
        with self._lock:
            job = self._jobs.get(slot)
            return self.stop_job(job.job_id) if job is not None else False

    def snapshot(self, include_logs: bool = True) -> dict[str, Any]:
        with self._lock:
            running = {slot: job is not None and job.status in {
                "preflight", "starting", "running", "stopping"}
                for slot, job in self._jobs.items()}
            return {
                "running": running,
                "jobs": {slot: self.get_job(job.job_id, reveal_paths=include_logs)
                         if job is not None else None for slot, job in self._jobs.items()},
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
