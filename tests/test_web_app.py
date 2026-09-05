import json
from pathlib import Path

import pytest

from glove_chirality.web_app import _is_loopback, create_app, create_viewer_app
from glove_chirality.web_service import CommandService


@pytest.fixture
def service(tmp_path):
    (tmp_path / "outputs").mkdir()
    return CommandService(tmp_path)


@pytest.fixture
def app(service):
    application = create_app(service, lan_enabled=True)
    application.config.update(TESTING=True)
    return application


@pytest.fixture
def viewer_app(service):
    application = create_viewer_app(service, lan_token="viewer-secret")
    application.config.update(TESTING=True)
    return application


def remote(headers=None):
    return {
        "environ_base": {"REMOTE_ADDR": "192.168.1.50"},
        "headers": headers or {},
    }


def test_browser_shell_is_responsive_and_uses_tick_controls(app):
    response = app.test_client().get("/")
    text = response.get_data(as_text=True)

    assert response.status_code == 200
    assert 'name="viewport"' in text
    assert 'class="tick"' in text
    assert "Compare historical models" in text
    assert "Run log" in text
    assert response.headers["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in response.headers["Content-Security-Policy"]


def test_lan_mode_requires_nonempty_token(service):
    with pytest.raises(ValueError, match="non-empty access token"):
        create_viewer_app(service, lan_token="")


def test_loopback_detection_does_not_trust_lan_addresses():
    assert _is_loopback("127.0.0.1") is True
    assert _is_loopback("::1") is True
    assert _is_loopback("192.168.1.20") is False


def test_malformed_json_returns_clear_noncacheable_error(app):
    response = app.test_client().post(
        "/api/run",
        data="[]",
        content_type="application/json",
    )

    assert response.status_code == 400
    assert response.get_json()["error"] == "Request body must be a JSON object"
    assert response.headers["Cache-Control"] == "no-store"


def test_host_receives_controls_and_raw_logs(app, service):
    service._append("pipeline", "C:/private/path/video.mkv")

    payload = app.test_client().get("/api/state").get_json()

    assert payload["can_edit"] is True
    assert payload["logs"][0]["text"] == "C:/private/path/video.mkv"
    assert payload["comparison_root"] is not None


def test_lan_viewer_requires_token_and_has_no_mutation_routes(viewer_app):
    client = viewer_app.test_client()

    assert client.get("/api/state", **remote()).status_code == 401
    authorized = remote({"X-GRIP-Token": "viewer-secret"})
    state = client.get("/api/state", **authorized)
    mutation = client.post(
        "/api/stop/pipeline",
        **authorized,
    )
    paths = client.get("/api/paths", **authorized)

    assert state.status_code == 200
    assert state.get_json()["can_edit"] is False
    assert state.get_json()["logs"] == []
    assert state.get_json()["comparison_root"] is None
    assert mutation.status_code == 404
    assert paths.status_code == 404


def test_remote_index_requires_explicit_lan_mode(service):
    application = create_app(service)
    application.config.update(TESTING=True)

    response = application.test_client().get("/", **remote())

    assert response.status_code == 403


def test_host_controller_rejects_dns_rebinding_host(app):
    response = app.test_client().get("/api/state", headers={"Host": "attacker.example"})

    assert response.status_code == 403
    assert response.get_json()["error"] == "Untrusted Host header"


def test_host_controller_rejects_cross_origin_mutation(app):
    response = app.test_client().post(
        "/api/run",
        json={"action": "extract_single"},
        headers={"Origin": "https://attacker.example"},
    )

    assert response.status_code == 403
    assert response.get_json()["error"] == "Untrusted request origin"


def test_lan_shell_hides_host_controls_before_javascript(viewer_app):
    response = viewer_app.test_client().get("/", **remote())

    assert response.status_code == 200
    assert '<body data-local="false">' in response.get_data(as_text=True)


def test_host_run_route_uses_allowlisted_command_builder(app, service, monkeypatch):
    captured = {}

    def capture(slot, command, *, action=None):
        captured.update(slot=slot, command=command, action=action)
        return "test-job-id"

    monkeypatch.setattr(service, "start", capture)
    response = app.test_client().post(
        "/api/run",
        json={
            "action": "extract_dataset",
            "left": "left-videos",
            "right": "right-videos",
            "output": "dataset",
            "config": "config.yaml",
        },
    )

    assert response.status_code == 202
    assert response.get_json()["job_id"] == "test-job-id"
    assert captured["slot"] == "pipeline"
    assert captured["action"] == "extract_dataset"
    assert "extract-dataset" in captured["command"]
    assert "left-videos" in captured["command"]


def test_unknown_web_action_is_rejected_without_starting_process(app, service, monkeypatch):
    started = False

    def capture(_slot, _command, *, action=None):
        nonlocal started
        started = True

    monkeypatch.setattr(service, "start", capture)
    response = app.test_client().post(
        "/api/run",
        json={"action": "shell", "command": "rm -rf /"},
    )

    assert response.status_code == 400
    assert started is False


def test_host_can_browse_and_round_trip_validated_yaml(app, tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("detector:\n  backend: belt_foreground\n", encoding="utf-8")
    client = app.test_client()

    listing = client.get(f"/api/paths?path={tmp_path}")
    loaded = client.get(f"/api/config?path={config}")
    saved = client.put(
        "/api/config",
        json={"path": str(config), "text": "event:\n  make_square: false\n"},
    )

    assert listing.status_code == 200
    assert any(entry["name"] == "config.yaml" for entry in listing.get_json()["entries"])
    assert loaded.status_code == 200
    assert saved.status_code == 200
    assert "make_square: false" in config.read_text(encoding="utf-8")


def test_missing_host_file_returns_json_error(app, tmp_path):
    response = app.test_client().get(f"/api/config?path={tmp_path / 'missing.yaml'}")

    assert response.status_code == 400
    assert "missing.yaml" in response.get_json()["error"]


def test_host_can_prepare_layer_one_presets_without_implicit_save(app, tmp_path):
    config = tmp_path / "config.yaml"
    model = tmp_path / "best.pt"
    config.write_text("{}\n", encoding="utf-8")
    model.write_bytes(b"placeholder")
    client = app.test_client()

    response = client.post(
        "/api/config/preset",
        json={
            "path": str(config),
            "preset": "custom_yolo",
            "model": str(model),
        },
    )

    assert response.status_code == 200
    assert "backend: yolo" in response.get_json()["text"]
    assert "yolo_require_masks: true" in response.get_json()["text"]
    assert config.read_text(encoding="utf-8") == "{}\n"


def test_lan_comparison_redacts_host_path(viewer_app, service, tmp_path):
    metrics = tmp_path / "outputs" / "model.pt.metrics.json"
    metrics.write_text(
        json.dumps({
            "model": "resnet18",
            "best_validation": {
                "accuracy": 0.8,
                "macro_recall": 0.75,
                "macro_f1": 0.74,
                "recall_per_class": [0.7, 0.8],
            },
        }),
        encoding="utf-8",
    )
    authorized = remote({"X-GRIP-Token": "viewer-secret"})

    payload = viewer_app.test_client().get("/api/comparison", **authorized).get_json()

    assert payload["runs"][0]["source"] == Path(metrics).name
