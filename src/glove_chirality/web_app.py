from __future__ import annotations

import argparse
import hmac
import ipaddress
import secrets
import socket
import tempfile
import threading
import webbrowser
from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Any

import yaml
from flask import Flask, abort, jsonify, render_template, request

from glove_chirality.comparison import COMPARISON_METRICS
from glove_chirality.config import ExtractionConfig
from glove_chirality.models import CLASSIFIER_CHOICES
from glove_chirality.ui_presets import (
    custom_yolo_segmentation_preset,
    tight_detection_crop_preset,
)
from glove_chirality.web_service import CommandService, build_web_command


def _is_loopback(address: str | None) -> bool:
    if not address:
        return False
    try:
        return ipaddress.ip_address(address.split("%", 1)[0]).is_loopback
    except ValueError:
        return address == "localhost"


def _lan_address() -> str:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("8.8.8.8", 80))
        return str(sock.getsockname()[0])
    except OSError:
        return socket.gethostbyname(socket.gethostname())
    finally:
        sock.close()


def _json_payload() -> dict[str, Any]:
    payload = request.get_json(silent=True)
    if not isinstance(payload, dict):
        abort(400, description="Request body must be a JSON object")
    return payload


def create_app(
    service: CommandService,
    *,
    allow_lan: bool = False,
    lan_token: str = "",
) -> Flask:
    if allow_lan and not lan_token:
        raise ValueError("LAN mode requires a non-empty access token")
    app = Flask(
        __name__,
        template_folder="web/templates",
        static_folder="web/static",
    )
    app.config.update(
        COMMAND_SERVICE=service,
        ALLOW_LAN=allow_lan,
        LAN_TOKEN=lan_token,
    )

    def local_request() -> bool:
        return _is_loopback(request.remote_addr)

    @app.after_request
    def security_headers(response):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data:; style-src 'self'; "
            "script-src 'self'; connect-src 'self'; frame-ancestors 'none'"
        )
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["Referrer-Policy"] = "no-referrer"
        if request.path.startswith("/api/"):
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.before_request
    def authenticate_lan():
        if local_request() or request.endpoint in {"static", "health"}:
            return
        if not app.config["ALLOW_LAN"]:
            abort(403, description="LAN access is disabled")
        if request.endpoint == "index":
            return
        supplied = request.headers.get("X-GRIP-Token", "")
        expected = app.config["LAN_TOKEN"]
        if not supplied or not hmac.compare_digest(supplied, expected):
            abort(401, description="A valid LAN access token is required")
        return

    def host_only(view: Callable):
        @wraps(view)
        def wrapped(*args, **kwargs):
            if not local_request():
                abort(403, description="This action is available only on the host machine")
            return view(*args, **kwargs)

        return wrapped

    @app.errorhandler(400)
    @app.errorhandler(401)
    @app.errorhandler(403)
    @app.errorhandler(404)
    def http_error(error):
        return jsonify(error=str(error.description)), error.code

    @app.errorhandler(OSError)
    @app.errorhandler(ValueError)
    def input_error(error):
        return jsonify(error=str(error)), 400

    @app.get("/")
    def index():
        return render_template(
            "index.html",
            classifier_choices=CLASSIFIER_CHOICES,
            comparison_metrics=COMPARISON_METRICS,
            initial_can_edit=local_request(),
        )

    @app.get("/api/health")
    def health():
        return jsonify(status="ok")

    @app.get("/api/state")
    def state():
        can_edit = local_request()
        snapshot = service.snapshot(include_logs=can_edit)
        snapshot.update(
            can_edit=can_edit,
            lan_enabled=bool(app.config["ALLOW_LAN"]),
            comparison_root=str(service.comparison_root) if can_edit else None,
        )
        return jsonify(snapshot)

    @app.post("/api/run")
    @host_only
    def run_command():
        payload = _json_payload()
        action = str(payload.pop("action", "")).strip()
        slot, command = build_web_command(action, payload)
        service.start(slot, command)
        return jsonify(slot=slot, status="started"), 202

    @app.post("/api/stop/<slot>")
    @host_only
    def stop_command(slot: str):
        if slot not in {"pipeline", "tensorboard"}:
            raise ValueError("slot must be pipeline or tensorboard")
        return jsonify(slot=slot, stopped=service.stop(slot))

    @app.get("/api/paths")
    @host_only
    def paths():
        return jsonify(service.browse(request.args.get("path")))

    @app.get("/api/config")
    @host_only
    def load_config():
        path = service.resolve_path(request.args.get("path", ""))
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("Configuration path must be a YAML file")
        ExtractionConfig.from_yaml(path)
        return jsonify(path=str(path), text=path.read_text(encoding="utf-8"))

    @app.put("/api/config")
    @host_only
    def save_config():
        payload = _json_payload()
        path = service.resolve_path(str(payload.get("path", "")))
        if path.suffix.lower() not in {".yaml", ".yml"}:
            raise ValueError("Configuration path must be a YAML file")
        text = str(payload.get("text", ""))
        raw = yaml.safe_load(text)
        if raw is not None and not isinstance(raw, dict):
            raise ValueError("Configuration YAML must contain a mapping")
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".yaml",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
        ) as stream:
            stream.write(text)
            temporary = Path(stream.name)
        try:
            config = ExtractionConfig.from_yaml(temporary)
            config.to_yaml(temporary)
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
        return jsonify(path=str(path), status="saved")

    @app.post("/api/config/preset")
    @host_only
    def config_preset():
        payload = _json_payload()
        path = service.resolve_path(str(payload.get("path", "")))
        config = ExtractionConfig.from_yaml(path)
        preset = str(payload.get("preset", ""))
        if preset == "custom_yolo":
            values = custom_yolo_segmentation_preset(str(payload.get("model", "")))
            for key, value in values.items():
                setattr(config.detector, key, value)
            config.detector.validate()
        elif preset == "tight_crop":
            for key, value in tight_detection_crop_preset().items():
                setattr(config.event, key, value)
            config.event.validate()
        else:
            raise ValueError("preset must be custom_yolo or tight_crop")
        with tempfile.NamedTemporaryFile(suffix=".yaml", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            config.to_yaml(temporary)
            text = temporary.read_text(encoding="utf-8")
        finally:
            temporary.unlink(missing_ok=True)
        return jsonify(path=str(path), text=text, status="ready")

    @app.get("/api/comparison")
    def comparison():
        metric = request.args.get("metric", "recall_right")
        return jsonify(
            metric=metric,
            runs=service.comparison(metric, reveal_paths=local_request()),
        )

    @app.put("/api/comparison/root")
    @host_only
    def comparison_root():
        payload = _json_payload()
        path = service.set_comparison_root(str(payload.get("path", "")))
        return jsonify(path=str(path))

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GRIP browser workstation")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument(
        "--lan",
        action="store_true",
        help="Allow authenticated read-only access from other devices on the local network",
    )
    parser.add_argument(
        "--token",
        help="LAN viewer token; generated when omitted (valid only with --lan)",
    )
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--workdir", default=".")
    parser.add_argument("--smoke-test", action="store_true", help=argparse.SUPPRESS)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise ValueError("port must be in [1, 65535]")
    if args.token and not args.lan:
        raise ValueError("--token requires --lan")

    token = args.token or (secrets.token_urlsafe(18) if args.lan else "")
    service = CommandService(args.workdir)
    app = create_app(service, allow_lan=args.lan, lan_token=token)
    if args.smoke_test:
        client = app.test_client()
        response = client.get("/api/health")
        if response.status_code != 200:
            raise RuntimeError("Web application health check failed")
        return

    local_url = f"http://127.0.0.1:{args.port}"
    print(f"GRIP host controls: {local_url}", flush=True)
    if args.lan:
        print(
            f"GRIP LAN viewer: http://{_lan_address()}:{args.port}/#token={token}",
            flush=True,
        )
        print("LAN clients are read-only; host paths, raw logs, and actions stay local.", flush=True)
    if not args.no_browser:
        threading.Timer(0.6, lambda: webbrowser.open(local_url)).start()

    from waitress import serve

    try:
        serve(app, host="0.0.0.0" if args.lan else "127.0.0.1", port=args.port, threads=6)
    finally:
        service.shutdown()


if __name__ == "__main__":
    main()
