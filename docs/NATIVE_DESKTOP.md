# Native GRIP workstation (Windows)

The native application is a **C++/Qt Quick desktop client**, not an embedded browser.
It supervises a separate Python processing worker. Both native and browser clients
use the same validated extraction, training, inference, comparison and factory
services. The live runtime is shared:

1. **Layer 1:** segmentation and full-source masks/geometry.
2. **Layer 2:** ordered passage tracking and one canonical crop per accepted passage.
3. **Layer 3:** chirality classification and event/output delivery.

Presentation is latest-only and cannot stall ordered tracking. FP32, crop semantics,
ROI and trigger settings remain unchanged. See [three-stage runtime](THREE_STAGE_PIPELINE.md).

## Build from a GitHub checkout

Requirements: Windows x64, Python 3.10+, CMake 3.24+, **Qt 6.8+ MinGW desktop**
(including Multimedia), and the matching MinGW compiler. Qt and compiler locations
are explicit; no workstation-specific paths are embedded in the source.

In Git Bash at the repository root:

```bash
python -m venv .venv
source .venv/Scripts/activate
python -m pip install -e '.[all]'
python -m pip install pyinstaller
# Replace these two paths with your actual Qt and matching compiler installation.
python tools/build_native.py \
  --qt-prefix 'C:/Qt/6.10.3/mingw_64' \
  --compiler-bin 'C:/Qt/Tools/mingw1310_64/bin'
python tools/verify_native_package.py
outputs/GRIP-desktop/GRIP.exe --workspace 'D:/GRIP-workspace'
```

Use the official matching CUDA PyTorch build **before freezing the worker** if the
bundle must run CUDA. A CPU-frozen worker is not a GPU deployment. Frozen builds
include installed ML dependencies, not your trained weights, recordings or datasets.
These assets remain external and must be selected in Inspection setup.

`tools/build_native.py` compiles the client, runs `windeployqt`, freezes the worker,
and copies YAML presets into `outputs/GRIP-desktop`. Keep the complete bundle:
`GRIP.exe`, Qt DLLs/plugins/QML, `worker/` and `configs/`. Copying the executable alone
is insufficient. Builds and verification artifacts are ignored by Git; a source
commit does **not** create a downloadable GitHub release/installer.

For development without freezing, compile with CMake and launch:

```bash
outputs/native-build/GRIP.exe \
  --backend-python "$PWD/.venv/Scripts/python.exe" --workspace "$PWD"
```

The development process needs the installed editable Python package and matching
Qt runtime DLLs on `PATH`. The deployed bundle does not need system Python.

## Workflows and safety

The client provides Inspection, dataset/extraction/training workflows, model
comparison and tools. Advanced fields remain available behind advanced controls.
**Advanced → Explain a prediction** runs SmoothGrad or occlusion on a saved crop
and checkpoint for later diagnosis. These inspector jobs are not executed in the
live segmentation/tracking/classification path.
Inspection supports camera/video, source/model/config selection, PREVIEW/SHADOW/ARMED,
performance diagnostics, preview/crop, exports, policy controls and optional serial.
Generic inference is shadow-only and does not register hardware mutation routes.

The worker binds an ephemeral **loopback-only** port with a per-process capability;
it is not an unauthenticated LAN controller. The client owns its lifetime. Owner
loss triggers the worker watchdog, and normal close stops sessions and owned jobs.
An explicitly enabled LAN viewer is authenticated and read-only; use only on a
trusted private network (HTTP is not encrypted).

Startup defaults to SHADOW. ARMED requires explicit attended confirmation and
readiness checks. Stop, fault and normal capture completion inhibit new actuator
submissions **before final events drain**; final decisions remain available for
logging. Unexpected camera EOF remains a fault, unlike normal video EOF. Already
sent firmware-delayed commands cannot be recalled.

## Verification and limits

```bash
pytest
ruff check src tests tools
npm run test:frontend
python tools/verify_native_package.py
```

The package verifier launches the frozen worker with Python/Qt environment overrides
removed and exercises synthetic extraction/audit, preview, CPU training, image/video
inference, both explanation methods, shadow-only playback and shutdown. It then launches
GRIP.exe, captures all four workspaces, checks native shutdown and records hashes of
the exact frontend/worker binaries exercised. It writes a report under
`outputs/native-package-verification`. Synthetic results establish software behavior,
**not real chirality accuracy or factory throughput**. Camera, CUDA, production
weights, real-time deadlines and physical actuator acceptance remain target-device
checks. No camera or actuator is used by this verification.

Run native export regressions with `ctest --test-dir outputs/native-build --output-on-failure`
after the CMake build (Qt/toolchain runtime DLLs must be on PATH). These exercise
interrupted-export preservation and complete-response atomic commit. Python source
contract tests supplement these checks; they do not replace interactive GUI testing.

Before distributing a release, include applicable Qt and Python-package licenses/notices
and satisfy their redistribution terms. Validate the bundle on a clean Windows machine
without developer dependencies. Manually accept file dialogs, ARMED confirmation and
cancellation, delayed startup field errors, counters, session switches, exports and
connection-loss behaviour using disconnected hardware or a controlled acceptance fixture,
never an unguarded live actuator.
