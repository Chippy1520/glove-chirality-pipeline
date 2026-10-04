"""Optional system GPU telemetry; never confuse parent allocator with child VRAM."""
from __future__ import annotations

import csv
import io
import math
import shutil
import subprocess
import threading
import time

_lock = threading.Lock()
_cached: dict = {"status": "unavailable", "gpus": []}
_checked = float("-inf")


def gpu_telemetry() -> dict:
    """Read actual NVIDIA counters, bounded/cached; missing tooling is not a failure."""
    global _cached, _checked
    with _lock:
        if time.monotonic() - _checked < 5:
            return dict(_cached)
        _checked = time.monotonic()
        executable = shutil.which("nvidia-smi")
        if executable is None:
            _cached = {"status": "unavailable", "gpus": []}
            return dict(_cached)
        try:
            result = subprocess.run(
                [executable, "--query-gpu=index,name,utilization.gpu,memory.used,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=1.5, check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
            devices = []
            for row in csv.reader(io.StringIO(result.stdout)):
                if len(row) != 5:
                    continue
                index, name, utilization, used, total = [part.strip() for part in row]
                counters = [float(value) for value in (utilization, used, total)]
                if not all(math.isfinite(value) and value >= 0 for value in counters):
                    continue
                devices.append({"index": int(index), "name": name,
                                "utilization_percent": counters[0], "memory_used_mb": counters[1],
                                "memory_total_mb": counters[2]})
            _cached = {"status": "available" if devices else "unavailable", "gpus": devices}
        except (OSError, subprocess.SubprocessError, ValueError):
            _cached = {"status": "unavailable", "gpus": []}
        return dict(_cached)
