"""Supervised service/CLI entrypoint for the native desktop executable.

The GUI is C++/Qt Quick. This process retains the existing Python ML contracts.
"""
from __future__ import annotations

import argparse
import hmac
import ipaddress
import json
import multiprocessing
import secrets
import sys
import threading
import time
from pathlib import Path


def create_desktop_app(service, factory, inference, token: str, shutdown=None):
    from flask import abort, jsonify, request

    from glove_chirality.desktop_schema import workstation_schema
    from glove_chirality.web_app import create_app

    if not token:
        raise ValueError("A desktop capability token is required")
    app = create_app(service, factory_session=factory, inference_session=inference)
    app.config["DESKTOP_LAST_SEEN"] = time.monotonic()

    @app.before_request
    def authorize_desktop():
        supplied = request.headers.get("X-GRIP-Desktop", "")
        if not hmac.compare_digest(supplied, token):
            abort(403, description="Native desktop capability required")
        app.config["DESKTOP_LAST_SEEN"] = time.monotonic()

    # Check capabilities before admission checks or resource probing.
    app.before_request_funcs[None].remove(authorize_desktop)
    app.before_request_funcs[None].insert(0, authorize_desktop)

    @app.get("/api/desktop/schema")
    def schema():
        return jsonify(workstation_schema())

    @app.get("/api/desktop/state")
    def native_state():
        # Full raw logs/GPU probes are explicit diagnostics requests, not hot polling.
        result = service.snapshot(include_logs=False)
        result.update(factory=factory.snapshot(reveal_paths=True),
                      inference=inference.snapshot(reveal_paths=True),
                      comparison_root=str(service.comparison_root))
        return jsonify(result)

    @app.post("/api/desktop/heartbeat")
    def heartbeat():
        return jsonify(status="ok")

    @app.post("/api/desktop/shutdown")
    def stop():
        if shutdown is not None:
            threading.Thread(target=shutdown, name="desktop-shutdown", daemon=True).start()
        return jsonify(status="stopping")

    return app


def serve(args) -> None:
    from waitress import create_server

    from glove_chirality.factory_live import FactoryLiveSession
    from glove_chirality.web_app import _lan_address, create_viewer_app
    from glove_chirality.web_service import CommandService

    workdir = Path(args.workdir).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    service = CommandService(workdir)
    factory = FactoryLiveSession(workdir)
    inference = FactoryLiveSession(workdir, hardware_allowed=False)
    token = secrets.token_urlsafe(32)
    stopped = threading.Event()
    servers = []
    close_lock = threading.Lock()

    def shutdown():
        with close_lock:
            if stopped.is_set():
                return
            stopped.set()
        # Backend guards disarm during stop. Already sent firmware delays cannot
        # be recalled; hardware acceptance must account for that limitation.
        factory.stop()
        inference.stop()
        service.shutdown()
        for server in servers:
            server.close()

    app = create_desktop_app(service, factory, inference, token, shutdown)
    server = create_server(app, host="127.0.0.1", port=0, threads=6)
    servers.append(server)
    viewer = {"url": None}

    @app.post("/api/desktop/sharing")
    def sharing():
        from flask import jsonify, request

        if viewer["url"]:
            return jsonify(url=viewer["url"])
        payload = request.get_json(silent=True) or {}
        if payload.get("confirm_private_network") is not True:
            return jsonify(error="Explicit trusted-private-network confirmation required"), 400
        address = _lan_address()
        parsed = ipaddress.ip_address(address)
        if not parsed.is_private or parsed.is_loopback:
            return jsonify(error="No trusted private network interface found"), 400
        viewer_token = secrets.token_urlsafe(24)
        viewer_app = create_viewer_app(service, lan_token=viewer_token,
                                       factory_session=factory, inference_session=inference)
        viewer_server = create_server(viewer_app, host=address, port=0, threads=4)
        servers.append(viewer_server)
        threading.Thread(target=viewer_server.run, name="desktop-readonly-viewer", daemon=True).start()
        viewer["url"] = f"http://{address}:{viewer_server.effective_port}/#token={viewer_token}"
        app.config.update(LAN_ENABLED=True, LAN_VIEWER_URL=viewer["url"])
        return jsonify(url=viewer["url"])

    def watchdog():
        while not stopped.wait(1):
            if time.monotonic() - app.config["DESKTOP_LAST_SEEN"] > args.owner_timeout:
                shutdown()
                return

    ready = Path(args.ready_file)
    ready.parent.mkdir(parents=True, exist_ok=True)
    payload = {"url": f"http://127.0.0.1:{server.effective_port}", "token": token}
    temporary = ready.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload), encoding="utf-8")
    temporary.replace(ready)
    threading.Thread(target=watchdog, name="desktop-owner-watchdog", daemon=True).start()
    try:
        server.run()
    finally:
        shutdown()
        ready.unlink(missing_ok=True)


def main(argv=None):
    multiprocessing.freeze_support()
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "cli":
        from glove_chirality.cli import main as cli_main
        return cli_main(argv[1:])
    if argv and argv[0] == "tensorboard":
        from tensorboard.main import run_main
        sys.argv = [sys.argv[0], *argv[1:]]
        return run_main()
    parser = argparse.ArgumentParser(description="Managed GRIP native worker")
    parser.add_argument("command", choices=["serve"])
    parser.add_argument("--workdir", required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--owner-timeout", type=float, default=15)
    args = parser.parse_args(argv)
    if args.owner_timeout < 5:
        parser.error("owner-timeout must be at least 5 seconds")
    return serve(args)


if __name__ == "__main__":
    main()
