"""Exercise a deployed frozen GRIP worker with real CPU jobs and synthetic fixtures.

No camera/serial actuation. Synthetic success is software evidence, not accuracy.
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import subprocess
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", default="outputs/GRIP-desktop")
    parser.add_argument("--output", default="outputs/native-package-verification")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    bundle = Path(args.bundle).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    workdir = output / "workspace"
    workdir.mkdir(exist_ok=True)
    make_video = runpy.run_path(str(root / "tools/generate_synthetic_videos.py"))["make_video"]
    for label in ("left", "right"):
        for variant in (1, 2):
            make_video(workdir / label / f"{label}_{variant}.avi", label, variant, events=2)
    config = workdir / "extract.yaml"
    config.write_text("detector:\n  backend: belt_foreground\n  roi: [0, 0, 1, 1]\n  trigger_zone: [0, 0, 1, 1]\nevent:\n  min_detected_frames: 1\n  exit_missing_frames: 2\n  crop_padding: 0\n  output_size: 32\n", encoding="utf-8")
    environment = os.environ.copy()
    environment["PATH"] = r"C:\Windows\System32;C:\Windows"
    for key in ("PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "QT_PLUGIN_PATH", "QML_IMPORT_PATH", "QML2_IMPORT_PATH"):
        environment.pop(key, None)
    worker = bundle / "worker/grip-worker.exe"
    ready = output / "ready.json"
    ready.unlink(missing_ok=True)
    log = (output / "worker.log").open("w", encoding="utf-8")
    process = subprocess.Popen([str(worker), "serve", "--workdir", str(workdir), "--ready-file", str(ready)], env=environment, stdout=log, stderr=subprocess.STDOUT)
    outcomes = []
    try:
        deadline = time.monotonic() + 60
        while not ready.exists():
            if process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError("Frozen worker did not become ready; inspect worker.log")
            time.sleep(.1)
        # Capability stays local; never print it or include it in the report.
        connection = json.loads(ready.read_text(encoding="utf-8"))

        def api(path, body=None, method=None):
            request = Request(connection["url"] + path, data=None if body is None else json.dumps(body).encode(),
                              headers={"X-GRIP-Desktop": connection["token"], "Content-Type": "application/json"},
                              method=method or ("GET" if body is None else "POST"))
            with urlopen(request, timeout=30) as response:
                raw = response.read()
                return json.loads(raw) if "json" in response.headers.get("Content-Type", "") else raw

        def run(action, payload):
            started = time.monotonic()
            job = api("/api/run", {"action": action, **payload})
            deadline = time.monotonic() + 150
            while True:
                result = api(f"/api/jobs/{job['job_id']}")
                if result["status"] in {"succeeded", "failed", "cancelled"}:
                    break
                if time.monotonic() > deadline:
                    api(f"/api/jobs/{job['job_id']}/stop", {})
                    raise RuntimeError(f"{action} timed out")
                time.sleep(.15)
            (output / f"{action}.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
            if result["status"] != "succeeded":
                raise RuntimeError(f"Frozen {action} failed; inspect {action}.json and worker.log")
            outcomes.append({"action": action, "status": result["status"], "elapsed_seconds": round(time.monotonic()-started, 3)})
            return result

        schema = api("/api/desktop/schema")
        assert len(schema["forms"]) == 10
        run("extract_dataset", {"left": "left", "right": "right", "output": "dataset", "config": str(config)})
        run("audit_dataset", {"manifest": "dataset/manifest.csv", "output": "audit.json"})
        run("preview", {"video": "right/right_1.avi", "output": "preview.png", "config": str(config), "seconds": 0, "warmup_seconds": 0})
        checkpoint = workdir / "candidate.pt"
        training = run("train", {"manifest": "dataset/manifest.csv", "output": str(checkpoint), "model": "tiny_cnn", "epochs": 2, "image_size": 32,
                                 "batch_size": 4, "workers": 0, "device": "cpu", "augmentation": "none", "validation_fraction": .5})
        assert checkpoint.exists()
        run("infer_images", {"input": "dataset/images", "checkpoint": str(checkpoint), "output": "images.csv", "device": "cpu"})
        run("infer_video", {"video": "right/right_1.avi", "checkpoint": str(checkpoint), "output": "video-results", "config": str(config), "device": "cpu"})
        first_image = next((workdir / "dataset/images").rglob("*.jpg"))
        run("explain", {"image": str(first_image), "checkpoint": str(checkpoint), "output": "explanation.png", "device": "cpu", "method": "occlusion"})
        run("explain", {"image": str(first_image), "checkpoint": str(checkpoint), "output": "smoothgrad.png", "device": "cpu", "method": "smoothgrad"})
        artifacts = api(f"/api/jobs/{training['job_id']}/artifacts")
        assert artifacts
        api("/api/inference/start", {"source_type": "video", "source": str(workdir / "right/right_1.avi"), "config": str(config), "checkpoint": str(checkpoint),
                                     "device": "cpu", "mode": "armed", "confirm_armed": True, "decision_class": "right", "decision_threshold": 0})
        deadline = time.monotonic() + 60
        saw_frame = False
        while True:
            status = api("/api/inference/status")
            try:
                saw_frame = bool(api("/api/inference/frame.jpg")) or saw_frame
            except HTTPError:
                pass
            if not status["running"]:
                break
            if time.monotonic() > deadline:
                raise RuntimeError("Frozen video session did not finish")
            time.sleep(.1)
        assert not status.get("fault"), status
        assert status["mode"] == "shadow" and status["counters"]["commands_sent"] == 0
        assert saw_frame and status["counters"]["right_reject"] > 0
        outcomes.append({"action": "shadow_video", "status": "succeeded", "no_actuation": True})
        api("/api/desktop/shutdown", {})
        process.wait(timeout=20)
        assert process.returncode == 0
        report = {"status": "PASS", "sanitized_environment": True, "synthetic_only": True, "real_accuracy_claim": False,
                  "cuda_tested": False, "jobs": outcomes, "shutdown_exit_code": process.returncode}
        (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait(timeout=10)
        ready.unlink(missing_ok=True)
        log.close()


if __name__ == "__main__":
    main()
