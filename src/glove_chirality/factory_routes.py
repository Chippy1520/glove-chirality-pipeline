from __future__ import annotations

from functools import wraps
from pathlib import Path

from flask import Response, jsonify, request

from glove_chirality.factory_live import (
    FactoryLiveSession,
    device_status,
    discover_checkpoints,
    discover_configs,
    load_settings,
    save_settings,
    scan_cameras,
    summarize_checkpoint,
)


def _payload() -> dict:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        raise ValueError("Request body must be a JSON object")  # noqa: TRY004 - mapped to HTTP 400
    return payload


def _flag(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def register_factory_routes(app, factory: FactoryLiveSession, host_only, *,
                            prefix: str = "/api/factory", hardware_enabled: bool | None = None) -> None:
    hardware = factory.hardware_allowed if hardware_enabled is None else (hardware_enabled and factory.hardware_allowed)
    namespace = prefix.strip("/").replace("/", "_")

    def route(path, method):
        def decorate(function):
            @wraps(function)
            def guarded(*args, **kwargs):
                try:
                    return function(*args, **kwargs)
                except (ValueError, TypeError, RuntimeError, OSError, OverflowError) as exc:
                    return jsonify(error=factory._safe_error(exc), checks=[{
                        "field": "request", "label": "Request", "status": "failed",
                        "message": factory._safe_error(exc),
                    }]), 400
            suffix = path.removeprefix("/api/factory")
            dangerous = suffix.startswith("/serial/") or suffix in {"/ports", "/mode", "/manual-trigger"}
            if hardware or not dangerous:
                app.add_url_rule(prefix + suffix, endpoint=namespace + "_" + function.__name__,
                                 view_func=guarded, methods=[method])
            return guarded
        return decorate

    def get(path):
        return route(path, "GET")

    def post(path):
        return route(path, "POST")

    @get("/api/factory/decision")
    def factory_decision():
        return jsonify(factory.decision())

    @get("/api/factory/status")
    @host_only
    def factory_status():
        return jsonify(factory.snapshot(reveal_paths=True))

    @get("/api/factory/cameras")
    @host_only
    def factory_cameras():
        if factory.running:
            raise ValueError("Stop the live session before scanning cameras")
        return jsonify(cameras=scan_cameras())

    @get("/api/factory/checkpoints")
    @host_only
    def factory_checkpoints():
        root = request.args.get("root") or load_settings(factory.workdir).get("models_root")
        return jsonify(root=str(root), checkpoints=discover_checkpoints(root))

    @post("/api/factory/models-root")
    @host_only
    def factory_models_root():
        path = str(_payload().get("path", "")).strip()
        if not path:
            raise ValueError("Models root is required")
        resolved = Path(path)
        if not resolved.exists():
            raise ValueError(f"Models root not found: {resolved}")
        saved = save_settings(factory.workdir, {"models_root": str(resolved)})
        return jsonify(settings=saved, checkpoints=discover_checkpoints(resolved))

    @get("/api/factory/checkpoint")
    @host_only
    def factory_checkpoint():
        path = request.args.get("path", "")
        return jsonify(summarize_checkpoint(path, compute_hash=_flag(request.args.get("sha256"))))

    @get("/api/factory/configs")
    @host_only
    def factory_configs():
        return jsonify(configs=discover_configs(factory.workdir))

    @get("/api/factory/devices")
    @host_only
    def factory_devices():
        return jsonify(device_status())

    @get("/api/factory/ports")
    @host_only
    def factory_ports():
        return jsonify(ports=factory.serial_ports())

    @post("/api/factory/serial/connect")
    @host_only
    def factory_serial_connect():
        payload = _payload()
        baud = int(payload.get("baud", 115200))
        return jsonify(factory.connect_serial(str(payload.get("port", "")), baud))

    @post("/api/factory/serial/disconnect")
    @host_only
    def factory_serial_disconnect():
        return jsonify(factory.disconnect_serial())

    @post("/api/factory/display")
    @host_only
    def factory_display():
        payload = _payload()
        factory.set_display(show_size_rejected=_flag(payload.get("show_size_rejected")))
        return jsonify(show_size_rejected=factory._show_rejected)

    @post("/api/factory/start")
    @host_only
    def factory_start():
        payload = _payload()
        for key in ("amp", "trigger_line_enabled", "continue_counters", "confirm_armed"):
            if key in payload:
                payload[key] = _flag(payload[key])
        return jsonify(factory.start(payload)), 202

    @post("/api/factory/stop")
    @host_only
    def factory_stop():
        factory.stop()
        return jsonify(factory.snapshot(reveal_paths=True))

    @post("/api/factory/mode")
    @host_only
    def factory_mode():
        payload = _payload()
        mode = factory.set_mode(str(payload.get("mode", "")), confirm=_flag(payload.get("confirm")))
        return jsonify(mode=mode, fault=factory.fault)

    @post("/api/factory/counters/reset")
    @host_only
    def factory_reset_counters():
        confirm = _flag(_payload().get("confirm"))
        return jsonify(counters=factory.reset_counters(confirm=confirm))

    @get("/api/factory/frame.jpg")
    @host_only
    def factory_frame():
        encoded = factory.frame_jpeg()
        if not encoded:
            return Response(status=204)
        return Response(encoded, mimetype="image/jpeg")

    @get("/api/factory/crop.jpg")
    @host_only
    def factory_crop():
        encoded = factory.crop_jpeg()
        if not encoded:
            return Response(status=204)
        return Response(encoded, mimetype="image/jpeg")

    @get("/api/factory/export")
    @host_only
    def factory_export():
        if factory.session_dir is None:
            raise ValueError("No Factory Live session log is available")
        kind = request.args.get("format", "jsonl")
        if kind == "csv":
            return Response(
                factory.export_csv(),
                mimetype="text/csv",
                headers={"Content-Disposition": "attachment; filename=events.csv"},
            )
        if kind != "jsonl":
            raise ValueError("format must be jsonl or csv")
        path = factory.session_dir / "events.jsonl"
        if not path.is_file():
            raise ValueError("No Factory Live session log is available")
        return Response(
            path.read_text(encoding="utf-8"),
            mimetype="application/x-ndjson",
            headers={"Content-Disposition": "attachment; filename=events.jsonl"},
        )

    @post("/api/factory/preflight")
    @host_only
    def factory_preflight():
        return jsonify(factory.preflight(_payload()))

    @post("/api/factory/controls")
    @host_only
    def factory_controls():
        return jsonify(factory.update_controls(_payload()))

    @post("/api/factory/playback")
    @host_only
    def factory_playback():
        payload = _payload()
        return jsonify(factory.playback(pause=payload.get("pause", payload.get("paused")), speed=payload.get("speed")))

    @post("/api/factory/manual-trigger")
    @host_only
    def factory_manual_trigger():
        return jsonify(factory.manual_trigger(confirm=_payload().get("confirm") is True))

    if not hardware:
        @post("/api/factory/simulate-trigger")
        @host_only
        def inference_simulate_trigger():
            return jsonify(factory.simulate_trigger())



def register_factory_viewer(app, factory: FactoryLiveSession, *, prefix: str = "/api/factory",
                            hardware_enabled: bool | None = None) -> None:
    """Read-only status. No frames, serial, start, or configuration routes."""

    namespace = prefix.strip("/").replace("/", "_")

    @app.get(prefix + "/decision", endpoint=namespace + "_viewer_decision")
    def factory_viewer_decision():
        return jsonify(factory.decision())

    @app.get(prefix + "/status", endpoint=namespace + "_viewer_status")
    def factory_viewer_status():
        return jsonify(factory.snapshot(reveal_paths=False))
