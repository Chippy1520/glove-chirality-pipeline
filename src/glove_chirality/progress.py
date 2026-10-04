"""Optional CLI telemetry: orchestration consumes data, never synthesizes progress."""
from __future__ import annotations

import json


def emit_progress(payload: dict[str, object]) -> None:
    """Emit one unbuffered machine-readable record alongside normal CLI output."""
    print("GRIP_PROGRESS " + json.dumps(payload, allow_nan=False), flush=True)
