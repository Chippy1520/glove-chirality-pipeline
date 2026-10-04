import threading

import numpy as np
import pytest
from flask import Flask

from glove_chirality.actuator import SerialActuator
from glove_chirality.factory_live import FactoryLiveSession
from glove_chirality.factory_routes import register_factory_routes, register_factory_viewer
from glove_chirality.live import CapturedFrame


class SerialMock:
    def __init__(self):
        self.commands = []
        self.connected = True
        self.cancelled = 0

    def snapshot(self):
        return {"connected": self.connected, "last_ack": None}

    def submit(self, command):
        self.commands.append(command)

    def cancel_pending(self):
        self.cancelled += 1

    def disconnect(self):
        self.connected = False


def options(tmp_path, **updates):
    config = tmp_path / "config.yaml"
    config.write_text("runtime:\n  stage_pipeline: true\n  report_interval_seconds: 60\n")
    checkpoint = tmp_path / "classifier.pt"
    checkpoint.write_bytes(b"injected-classifier")
    return {"source": "0", "config": str(config), "checkpoint": str(checkpoint),
            "device": "cpu", **updates}


def test_invalid_start_is_async_accumulates_checks_and_never_loads_models(tmp_path, monkeypatch):
    session = FactoryLiveSession(tmp_path)
    monkeypatch.setattr("glove_chirality.detection.build_detector", lambda _: pytest.fail("model operation"))
    monkeypatch.setattr("glove_chirality.factory_live.device_status", lambda: pytest.fail("device operation"))
    response = session.start({"mode": "armed", "device": "invalid", "delay_ms": -1,
                              "decision_threshold": float("nan"), "reject_class": "bad"})
    assert response["job_id"] and response["status"] == "preflight"
    session._thread.join(2)
    status = session.snapshot(reveal_paths=True)
    assert status["status"] == "fault" and not status["running"]
    failed = {c["field"] for c in status["checks"] if c["status"] == "failed"}
    assert {"source", "config", "checkpoint", "confirm_armed", "device",
            "actuator_delay_ms", "decision_threshold", "reject_class"} <= failed
    assert status["end_time"] and status["elapsed_s"] >= 0
    assert all(set(c) == {"field", "label", "status", "message"} for c in status["checks"])


def test_detector_runs_during_classifier_warmup_and_arm_waits_for_readiness(tmp_path):
    session = FactoryLiveSession(tmp_path)
    session._serial = SerialMock()
    detected = threading.Event()
    warmed = threading.Event()
    release = threading.Event()

    class Capture:
        captured_frames = 1
        dropped_frames = 0
        exhausted = False

        def __init__(self):
            self.first = True
            self.stopped = threading.Event()

        def start(self):
            return self

        def read(self, timeout=0.1):
            if self.first:
                self.first = False
                import time
                return CapturedFrame(0, time.monotonic(), np.zeros((40, 40, 3), np.uint8))
            self.stopped.wait(timeout)
            self.exhausted = self.stopped.is_set()
            return None

        def stop(self):
            self.stopped.set()

    class Detector:
        def warmup(self, _frame):
            pass

        def detect(self, _frame):
            detected.set()
            return []

    class Classifier:
        device = "cpu"

        def warmup(self):
            warmed.set()
            assert release.wait(3)

    session.start(options(tmp_path, mode="armed", confirm_armed=True), detector=Detector(),
                  classifier=Classifier(), capture=Capture())
    try:
        assert warmed.wait(2) and detected.wait(2)
        assert session.mode == "shadow" and session.status == "starting"
        with pytest.raises(ValueError, match="successfully running"):
            session.set_mode("armed", confirm=True)
        release.set()
        import time
        deadline = time.monotonic() + 2
        while session.status != "running" and time.monotonic() < deadline:
            time.sleep(0.005)
        assert session.status == "running"
        with pytest.raises(ValueError, match="confirmation"):
            session.set_mode("armed")
        session.set_mode("armed", confirm=True)
        session.manual_trigger(confirm=True)
        assert len(session._serial.commands) == 1
        assert session._serial.snapshot()["last_ack"] is None
        session.disconnect_serial()
        assert session.mode == "shadow" and session._serial.cancelled >= 1
    finally:
        release.set()
        session.stop()
    assert not session._thread.is_alive()


def test_generic_inference_never_actuates_even_with_spoofed_state(tmp_path):
    session = FactoryLiveSession(tmp_path, hardware_allowed=False)
    session._serial = SerialMock()
    session.mode = "armed"
    session.running = True
    session.status = "running"
    for operation in (lambda: session.connect_serial("COM5"),
                      lambda: session.set_mode("armed", confirm=True),
                      lambda: session.manual_trigger(confirm=True), session._serial_actor):
        with pytest.raises(ValueError, match="disabled"):
            operation()
    session._on_event({"event_id": "e1", "status": "accepted", "prediction": "right"})
    assert not session._serial.commands
    assert session.snapshot(reveal_paths=False)["actuator_state"] == "DISABLED"
    assert session.simulate_trigger()["actuated"] is False


@pytest.mark.parametrize("payload", [{"threshold": float("nan")}, {"threshold": float("inf")},
                                     {"threshold": -0.1}, {"delay_ms": -1}, {"delay_ms": 60001},
                                     {"delay_ms": 1.5}, {"decision_policy": "bad"},
                                     {"reject_class": "bad"}, {"decision_class": "bad"}])
def test_controls_invalid_update_is_atomic(tmp_path, payload):
    session = FactoryLiveSession(tmp_path)
    before = dict(session._controls)
    with pytest.raises((ValueError, TypeError)):
        session.update_controls(payload)
    assert session._controls == before


def test_controls_record_actual_classification_policy_and_disarm_pending(tmp_path):
    session = FactoryLiveSession(tmp_path)
    session._serial = SerialMock()
    session.update_controls({"decision_policy": "threshold", "decision_class": "right",
                             "threshold": 0.8, "delay_ms": 100, "reject_class": "right"})

    class Classifier:
        def predict_array(self, _crop):
            assert self.decision_class == "right" and self.decision_threshold == 0.8
            return "right", 0.9

    prediction, confidence, used = session._classify(np.zeros((8, 8, 3)), Classifier())
    session.running = True
    session.status = "running"
    session.set_mode("armed", confirm=True)
    session.update_controls({"threshold": 0.2, "delay_ms": 200, "reject_class": "left"})
    assert session.mode == "shadow"
    session.set_mode("armed", confirm=True)
    session._on_event({"event_id": "e1", "status": "accepted", "prediction": prediction,
                       "confidence": confidence, **used})
    latest = session.snapshot(reveal_paths=True)["latest"]
    assert latest["decision_threshold"] == 0.8 and latest["actuator_delay_ms"] == 100
    assert latest["reject_class"] == "right" and latest["result"] == "REJECT"
    assert not session._serial.commands  # A decision from before rearming is stale.


def test_video_cannot_spoof_camera_or_enable_hardware(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fixture")
    session = FactoryLiveSession(tmp_path)
    result = session.preflight(options(tmp_path, source=str(video), source_type="camera"))
    assert result["status"] == "failed"
    assert next(c for c in result["checks"] if c["field"] == "source")["status"] == "failed"
    session.source_type = "video"
    with pytest.raises(ValueError, match="disabled"):
        session.connect_serial("COM5")
    assert session.playback(pause=True, speed=0.5) == {"paused": True, "speed": 0.5}


def test_routes_are_namespaced_generic_has_no_hardware_and_viewer_is_readonly(tmp_path):
    app = Flask(__name__)
    factory = FactoryLiveSession(tmp_path)
    inference = FactoryLiveSession(tmp_path, hardware_allowed=False)
    register_factory_routes(app, factory, lambda f: f)
    register_factory_routes(app, inference, lambda f: f, prefix="/api/inference")
    client = app.test_client()
    assert client.get("/api/factory/status").status_code == 200
    assert client.get("/api/inference/status").get_json()["workflow"] == "inference"
    assert client.post("/api/inference/start", json={}).status_code == 202
    inference._thread.join(2)
    assert client.post("/api/inference/simulate-trigger", json={}).get_json()["actuated"] is False
    for suffix in ("/serial/connect", "/manual-trigger", "/mode"):
        assert client.post("/api/inference" + suffix, json={}).status_code == 404
    response = client.post("/api/factory/controls", json={"threshold": "invalid"})
    assert response.status_code == 400 and response.get_json()["checks"]
    viewer = Flask("viewer")
    register_factory_viewer(viewer, factory)
    register_factory_viewer(viewer, inference, prefix="/api/inference")
    factory.fault = "private/path rtsp://name:secret@example.org/video"
    status = viewer.test_client().get("/api/factory/status").get_json()
    assert "secret" not in str(status) and "private/path" not in str(status)
    assert viewer.test_client().post("/api/factory/controls", json={}).status_code == 404
    assert viewer.test_client().get("/api/factory/frame.jpg").status_code == 404


def test_serial_pending_cancellation_uses_generation_without_fabricating_ack():
    actor = SerialActuator()
    actor._connected = True  # No physical port: test only the queue boundary.
    actor.submit("REJECT|e1|100\n")
    generation, command = actor._poll_command()
    actor.cancel_pending()
    assert generation != actor._generation and command == "REJECT|e1|100\n"
    assert actor.snapshot()["last_ack"] is None


def test_video_requested_armed_runs_shadow_and_eof_is_not_a_fault(tmp_path):
    video = tmp_path / "input.mp4"
    video.write_bytes(b"injected-source")
    session = FactoryLiveSession(tmp_path)
    session._serial = SerialMock()

    class Capture:
        def playback(self, **_kwargs):
            pass

    def runner(*_args, **kwargs):
        assert session.mode == "shadow"
        for stage in ("classifier_warmup", "first_frame", "detector_inference"):
            kwargs["on_stage"](stage, "passed")
        assert session.status == "running"
        kwargs["event_callback"]({"event_id": "video1", "status": "accepted", "prediction": "right"})

    session.start(options(tmp_path, source=str(video), source_type="video", mode="armed"),
                  runner=runner, detector=object(),
                  classifier=type("Classifier", (), {"device": "cpu"})(), capture=Capture())
    session._thread.join(2)
    status = session.snapshot(reveal_paths=True)
    assert status["status"] == "stopped" and status["fault"] is None
    assert status["mode"] == "shadow" and status["actuator_state"] == "DISABLED"
    assert session._serial.commands == []


def test_bad_yaml_does_not_prevent_accumulating_other_plain_checks(tmp_path):
    payload = options(tmp_path, reject_class="invalid", decision_threshold=2)
    from pathlib import Path
    Path(payload["config"]).write_text("detector: [not valid yaml")
    result = FactoryLiveSession(tmp_path).preflight(payload)
    failed = {c["field"] for c in result["checks"] if c["status"] == "failed"}
    assert {"config", "reject_class", "decision_threshold"} <= failed


def test_playback_route_accepts_frontend_paused_alias(tmp_path):
    app = Flask("playback")
    session = FactoryLiveSession(tmp_path, hardware_allowed=False)
    session.source_type = "video"
    register_factory_routes(app, session, lambda f: f, prefix="/api/inference")
    response = app.test_client().post("/api/inference/playback", json={"paused": True, "speed": 0.25})
    assert response.status_code == 200
    assert response.get_json() == {"paused": True, "speed": 0.25}
