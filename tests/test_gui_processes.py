import pytest

from glove_chirality.gui_processes import ProcessSlots


class FakeProcess:
    def __init__(self, running=True):
        self.running = running
        self.terminated = False

    def poll(self):
        return None if self.running else 0

    def terminate(self):
        self.terminated = True
        self.running = False


def test_pipeline_and_tensorboard_can_run_concurrently():
    slots = ProcessSlots()
    pipeline = FakeProcess()
    tensorboard = FakeProcess()

    slots.claim("pipeline", pipeline)
    slots.claim("tensorboard", tensorboard)

    assert slots.get("pipeline") is pipeline
    assert slots.get("tensorboard") is tensorboard
    with pytest.raises(RuntimeError, match="already running"):
        slots.claim("pipeline", FakeProcess())


def test_slots_release_only_the_matching_process_and_stop_independently():
    slots = ProcessSlots()
    pipeline = FakeProcess()
    tensorboard = FakeProcess()
    slots.claim("pipeline", pipeline)
    slots.claim("tensorboard", tensorboard)

    assert slots.release("pipeline", FakeProcess()) is False
    assert slots.release("pipeline", pipeline) is True
    assert slots.get("pipeline") is None
    assert slots.get("tensorboard") is tensorboard

    slots.terminate_all()
    assert tensorboard.terminated is True
    assert pipeline.terminated is False


def test_unknown_slot_is_rejected():
    slots = ProcessSlots()
    with pytest.raises(ValueError, match="Unknown process slot"):
        slots.ensure_available("other")
