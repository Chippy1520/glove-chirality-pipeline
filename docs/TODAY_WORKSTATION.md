# TODAY — GRIP workstation

This is an incremental upgrade of the existing Flask/Waitress application. The production CLI, detector, `PassageProcessor`, canonical crop, classifier transform and Arduino protocol remain the source of truth.

## Install/update on the separate training and factory device

This guide is for the device that pulls the repository and owns the recordings,
checkpoints, GPU, camera and Arduino. Development-machine test outputs are not
deployment inputs. Do not copy its `.venv`, `outputs/`, absolute paths or camera/COM
settings to the factory device.

1. Preserve any local changes on that device, then update with `git pull --ff-only`.
2. Activate that device's existing project virtual environment. If it does not
   have one, create it with `python -m venv .venv` and activate it as shown below.
   On Linux/macOS use `source .venv/bin/activate` instead.
3. Install the repository and optional training, detector and serial packages:

   ```bash
   python -m pip install -e ".[ml,yolo,factory,dev]"
   ```

4. For GPU training, retain/install a matching CUDA-enabled PyTorch/torchvision
   build using the official PyTorch installation selector. The development
   machine's CPU-only installation does not establish the target's GPU support.
   Verify the target before selecting CUDA:

   ```bash
   python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
   python -m pytest
   python -m ruff check src tests tools
   ```

5. Select paths that exist on **that device** for videos, output directories,
   Layer-1/Layer-2 checkpoints and YAML. Confirm the YAML's detector checkpoint
   resolves there. Scan its camera and serial ports rather than reusing indices
   or COM names from another machine. Do not commit private recordings, weights,
   tokens or generated outputs.
6. Perform recorded-video testing first, then real-camera SHADOW, and only then
   attended ARMED testing. Local software tests do not replace this acceptance.

## Start

From the repository in Git Bash:

```bash
source .venv/Scripts/activate
python -m glove_chirality.web_app --no-browser
```

Open **http://127.0.0.1:8765**. Click **Enable alerts** once in each browser that should play audio. Audio is browser-side; the PC does not play audio on behalf of a phone. A phone needs its own click and an active page.

For the separate authenticated read-only viewer:

```bash
python -m glove_chirality.web_app --lan --no-browser
```

Use the viewer URL / QR shown on the host. Tunnel **the viewer port 8766**, not the control port 8765. Keep the generated token private. The viewer has no training/start/stop, filesystem, serial or actuator mutation endpoints.

## Today's sequence

1. Finish the recordings. Use distinct source/session names and separate LEFT and RIGHT directories.
2. In **Layer 1**, retain the approved extraction YAML and custom detector checkpoint. Do not retune ROI just to fit the browser image.
3. In **Extract**, run the calibration preview on a representative recorded frame. Inspect ROI, trigger zone and the detector; approve geometry only after visual review.
4. Enter LEFT directory, RIGHT directory, YAML and a **new** dataset output directory. Start extraction; observe PREFLIGHT, actual frame/video counts and the log.
5. Review the automatic dataset audit. Zero crops, missing/corrupt samples, conflicting labels and cross-source duplicate content must be addressed rather than ignored. Keep the report with the dataset.
6. Use the dataset manifest in **Train**. Check class counts and the source-group split; validation must contain both classes. Each class needs at least two distinct source videos for this split.
7. Select the candidate architecture and device. Retain existing fine-tuning, augmentation, loss, recall and TensorBoard controls. Prefer right recall / macro recall / macro F1, not accuracy alone. Use a new checkpoint path per run.
8. Start training. The GUI performs server validation before the subprocess starts, shows live CLI telemetry and retains stdout/stderr. The exact training configuration and manifest hash are saved.
9. On success, inspect the actual retained checkpoint and class recalls. Click **Run on test video**, choose **Video file**, select a representative factory recording and start visualization. This workflow has **no physical actuator**.
10. Verify accepted passage crops, LEFT/RIGHT decisions, threshold and event-level RIGHT alarms. Test pause and playback speed separately from production frame resolution.
11. In **Factory Live**, select the real camera and reviewed resources. Start **SHADOW**, observe the startup checks and actual negotiated camera properties. Verify both glove classes under current lighting.
12. Only after SHADOW succeeds, connect serial and verify the port/baud/state. Clear the test area and verify the physical emergency stop before confirming **ARMED**.
13. Use the confirmed manual trigger only in a running, connected ARMED camera/stream session. Check the command and ACK. Then test accepted RIGHT → configured delay → firmware actuation.
14. Stop/disarm on any hardware warning. Reconnection never silently rearms.

## Reproducible CLI alternatives

Replace the example paths with today's actual paths. They are placeholders, not directories this upgrade assumes exist.

```bash
glove-pipeline extract-dataset --left "D:/recordings/today/left" --right "D:/recordings/today/right" --config "configs/production.yaml" --output "D:/datasets/today"
glove-pipeline audit-dataset --manifest "D:/datasets/today/manifest.csv" --output "D:/datasets/today/dataset_report.json" --validation-fraction 0.2 --seed 42
glove-pipeline train --manifest "D:/datasets/today/manifest.csv" --output "D:/models/today/candidate.pt" --model resnet18 --epochs 30 --head-only-epochs 3 --batch-size 32 --image-size 224 --learning-rate 0.001 --backbone-learning-rate 0.0001 --validation-fraction 0.2 --seed 42 --device auto --workers 0 --loss weighted_cross_entropy --recall-target right --selection-metric macro_recall --augmentation anti_spurious --tensorboard-logdir "D:/models/today/tensorboard"
glove-pipeline infer-video --video "D:/recordings/today/test.avi" --checkpoint "D:/models/today/candidate.pt" --config "configs/production.yaml" --output "D:/models/today/video-test" --device auto --decision-class right --decision-threshold 0.5
```

`infer-video --output` is a **directory**, not a CSV filename. Factory SHADOW/ARMED is controlled through the existing dedicated GUI, not generic CLI inference. Generic `infer-live` remains non-actuating.

## Important hardware limitation

The protocol remains `REJECT|event_id|delay_ms\n`. Delay is executed by the firmware. Stopping/disarming cancels **unsent host commands**, but cannot recall a delayed command already accepted by the Arduino. An ACK proves protocol acknowledgement, not that the mechanism physically moved. Hardware acceptance requires an attended test and emergency stop; software tests cannot establish pneumatic/robot safety.

## Verification boundary

Automated tests use generated videos, CPU checkpoints and serial/camera mocks where explicitly indicated. They prove orchestration and safety invariants, **not real-glove accuracy**. Real camera throughput, lighting robustness, GPU performance, audio audibility on a particular phone and physical actuator timing require today's attended checks.
