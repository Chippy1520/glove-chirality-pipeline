"""End-of-source safety must remain distinct from an operator-requested stop."""
import pytest

from glove_chirality.config import ExtractionConfig
from glove_chirality.factory_live import FactoryLiveSession
from glove_chirality.live import LiveMetrics


@pytest.mark.parametrize("source_type", ["camera", "video"])
@pytest.mark.parametrize("exhausted", [False, True])
def test_factory_retains_camera_eof_fault_after_safety_stop(tmp_path, source_type, exhausted):
    session = FactoryLiveSession(tmp_path)
    session._config = ExtractionConfig()
    session.source_type = source_type
    session._options = {"source": "0", "device": "cpu"}

    def runner(*_args, **kwargs):
        kwargs["stop_event"].set()
        return LiveMetrics(source_exhausted=exhausted)

    session._injected = {
        "runner": runner,
        "detector": object(),
        "classifier": type("Classifier", (), {"device": "cpu"})(),
        "capture": type("Capture", (), {"playback": lambda self, **kwargs: None})(),
    }
    session._run()
    if source_type == "camera" and exhausted:
        assert session.status == "fault" and "Camera stream ended" in session.fault
    else:
        assert session.status == "stopped" and not session.fault, session.fault
    assert session.mode == "shadow" and not session.running
