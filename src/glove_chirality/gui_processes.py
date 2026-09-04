from __future__ import annotations

from typing import Protocol


class ManagedProcess(Protocol):
    def poll(self) -> int | None: ...

    def terminate(self) -> None: ...


class ProcessSlots:
    """Track independent long-running GUI processes without cross-slot blocking."""

    def __init__(self, names=("pipeline", "tensorboard")):
        self._processes: dict[str, ManagedProcess | None] = {name: None for name in names}

    def ensure_available(self, name: str) -> None:
        if name not in self._processes:
            raise ValueError(f"Unknown process slot: {name}")
        process = self._processes[name]
        if process is not None and process.poll() is None:
            raise RuntimeError(f"Process slot {name!r} is already running")

    def claim(self, name: str, process: ManagedProcess) -> None:
        self.ensure_available(name)
        self._processes[name] = process

    def get(self, name: str) -> ManagedProcess | None:
        return self._processes.get(name)

    def release(self, name: str, process: ManagedProcess) -> bool:
        if self._processes.get(name) is not process:
            return False
        self._processes[name] = None
        return True

    def terminate_all(self) -> None:
        for process in self._processes.values():
            if process is not None and process.poll() is None:
                process.terminate()
