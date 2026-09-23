import threading
from pathlib import Path

import numpy as np
import pytest

from glove_chirality.config import ExtractionConfig
from glove_chirality.events import FrameResult
from glove_chirality.factory_live import FactoryLiveSession
from glove_chirality.web_app import create_app, create_viewer_app
from glove_chirality.web_service import CommandService


def _config(path: Path) -> None:
    path.write_text(
        "detector:\n  backend: belt_foreground\n  roi: [0, 0, 1, 1]\n  trigger_zone: [0, 0, 1, 1]\n"
        "event:\n  output_size: 32\n  letterbox_fill: 114\n",
        encoding="utf-8",
    )


def _checkpoint(path: Path) -> None:
    path.write_bytes(b"not-a-real-weight")


def test_viewer_has_no_factory_mutation_or_frame_routes(tmp_path):
    service = CommandService(tmp_path)
    factory = FactoryLiveSession(tmp_path)
    viewer = create_viewer_app(service, lan_token="viewer-secret", factory_session=factory)
    viewer.config.update(TESTING=True)
    client = viewer.test_client()
    headers = {"X-GRIP-Token": "viewer-secret"}

    status = client.get("/api/factory/status", headers=headers)
    assert status.status_code == 200
    assert status.get_json()["mode"] == "shadow"
    assert "checkpoint" not in status.get_json()

    for path in (
        "/api/factory/start",
        "/api/factory/stop",
        "/api/factory/mode",
        "/api/factory/serial/connect",
        "/api/factory/serial/disconnect",
        "/api/factory/counters/reset",
        "/api/factory/models-root",
    ):
        assert client.post(path, json={}, headers=headers).status_code == 404
    for path in ("/api/factory/frame.jpg", "/api/factory/crop.jpg", "/api/factory/ports", "/api/factory/cameras"):
        assert client.get(path, headers=headers).status_code == 404


def test_factory_page_is_present_and_scrollable(tmp_path):
    service = CommandService(tmp_path)
    app = create_app(service, factory_session=FactoryLiveSession(tmp_path))
    app.config.update(TESTING=True)
    text = app.test_client().get("/").get_data(as_text=True)
    assert 'id="factory"' in text
    assert "ARMED mode can activate physical machinery" in text
    assert "Classifier not run" not in text or "factory-events" in text
    assert "overflow: hidden" not in text.split('id="factory"')[1][:400]
    assert "Anti-spurious augmentation intentionally varies" in text


def test_host_rejects_invalid_factory_start_without_launching(tmp_path):
    service = CommandService(tmp_path)
    app = create_app(service, factory_session=FactoryLiveSession(tmp_path))
    app.config.update(TESTING=True)
    response = app.test_client().post("/api/factory/start", json={"mode": "armed"})
    assert response.status_code == 400
    assert "confirmation" in response.get_json()["error"]
    assert app.config["FACTORY"].running is False


def test_factory_session_start_stop_frame_and_event(tmp_path):
    config = tmp_path / "factory.yaml"
    checkpoint = tmp_path / "classifier.pt"
    _config(config)
    _checkpoint(checkpoint)
    session = FactoryLiveSession(tmp_path)
    frame = np.zeros((16, 16, 3), dtype=np.uint8)
    started = threading.Event()

    def runner(*_args, **kwargs):
        kwargs["status_callback"]("RUNNING")
        kwargs["frame_callback"](frame, FrameResult((), (), 1.0, 0.2), 0.1)
        kwargs["event_callback"](
            {
                "event_id": "live__camera_0__e000001",
                "status": "accepted",
                "prediction": "right",
                "confidence": 0.99,
                "detector_confidence": 0.91,
                "timestamp_s": 1.0,
                "wall_time_iso": "2026-09-23T10:14:32.417+05:30",
                "crop": frame,
            }
        )
        started.set()
        kwargs["stop_event"].wait(2)

    class _Serial:
        def __init__(self):
            self.commands = []
            self.connected = True

        def submit(self, command):
            self.commands.append(command)

        def snapshot(self):
            return {"connected": True, "port": "COM5", "baud": 115200, "last_command": None, "last_ack": None, "fault": None}

    session._serial = _Serial()
    session.start(
        {
            "source": "0",
            "checkpoint": str(checkpoint),
            "config": str(config),
            "device": "cpu",
            "mode": "shadow",
            "reject_class": "right",
        },
        runner=runner,
        detector=object(),
        classifier=type("C", (), {"device": "cpu", "use_amp": False, "model_name": "resnet18"})(),
        capture=object(),
    )
    assert started.wait(2)
    status = session.snapshot(reveal_paths=True)
    assert status["status"] == "RUNNING"
    assert status["latest"]["prediction"] == "right"
    assert status["latest"]["result"] == "REJECT"
    assert status["latest"]["actuator_command"] is None
    assert status["counters"]["right_reject"] == 1
    assert status["counters"]["commands_sent"] == 0
    assert session.frame_jpeg()
    assert session.crop_jpeg()
    session.set_mode("armed", confirm=True)
    session._on_event(
        {
            "event_id": "live__camera_0__e000002",
            "status": "accepted",
            "prediction": "right",
            "confidence": 0.98,
        }
    )
    assert session._serial.commands == ["REJECT|live__camera_0__e000002|850\n"]
    session._on_event(
        {
            "event_id": "live__camera_0__e000002",
            "status": "accepted",
            "prediction": "right",
            "confidence": 0.98,
        }
    )
    assert len(session._serial.commands) == 1
    session._on_serial_fault("disconnected")
    assert session.mode == "shadow"
    assert session.fault.startswith("Serial fault")
    session.stop()
    assert session.running is False
    assert session.mode == "shadow"
    assert (session.session_dir / "events.jsonl").is_file()
    assert (session.session_dir / "session.json").is_file()


def test_pipeline_reject_does_not_command_or_invent_confidence(tmp_path):
    session = FactoryLiveSession(tmp_path)
    session.mode = "armed"
    session._serial = type(
        "S",
        (),
        {
            "commands": [],
            "submit": lambda self, command: self.commands.append(command),
            "snapshot": lambda self: {"connected": True},
        },
    )()
    session._on_event(
        {
            "event_id": "e1",
            "status": "multiple_candidates",
            "reject_reason": "multiple_candidates",
            "prediction": None,
            "confidence": None,
        }
    )
    session._on_event(
        {"event_id": "e2", "status": "partial", "reject_reason": "partial", "prediction": None}
    )
    session._on_event(
        {"event_id": "e3", "status": "accepted", "prediction": "left", "confidence": 0.8}
    )
    assert session._serial.commands == []
    assert session.snapshot(reveal_paths=False)["latest"]["confidence"] is None or session._events[-1]["prediction"] == "left"
    assert session._counters["commands_sent"] == 0
    assert session._counters["multiple_candidates"] == 1
    assert session._counters["partial"] == 1


def test_factory_config_template_is_valid_and_portable():
    config = ExtractionConfig.from_yaml(Path("configs/factory.yaml"))
    assert config.event.letterbox_fill == 114
    assert config.event.output_size == 256
    assert config.detector.yolo_require_masks is True
    assert not config.detector.yolo_model.startswith(("\\", "/"))
    assert config.detector.yolo_min_box_area_ratio == pytest.approx(0.03)
