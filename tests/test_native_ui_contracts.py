"""Source-contract regressions; native execution/exports are verified separately."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / "desktop"


def compact(name):
    return "".join((ROOT / name).read_text(encoding="utf-8").split())


def test_native_arm_and_counter_confirmations_match_backend_routes():
    source = compact("Inspect.qml")
    assert '{mode:"armed",confirm:true}' in source
    assert 'onAccepted:client.request("reset",' in source and '{confirm:true}' in source
    assert 'if(shadowOnly||payload.mode==="armed")payload.mode="shadow"' in source
    assert 'onClosed:operatingMode.currentIndex=Qt.binding(' in source
    assert 'setupForm.applyErrors(session)' in source
    assert 'session.status==="fault"' in source


def test_tensorboard_form_stops_its_own_slot():
    assert '"/api/stop/"+form.stopSlot' in compact("WorkflowForm.qml")
    assert 'definition:tools.form("tensorboard");stopSlot:"tensorboard"' in compact("Tools.qml")


def test_native_network_and_preview_responses_are_not_silently_stale():
    source = compact("GripClient.cpp")
    assert 'reply->error()==QNetworkReply::NoError' in source
    assert 'generation!=previewGeneration_' in source
    assert 'previewSession_=session;clearPreview();' in source
    assert 'QDir(workdir_).absoluteFilePath(target)' in source
    assert 'newQSaveFile(target,reply)' in source
    assert 'finishExport(*file,*reply,complete)' in source
