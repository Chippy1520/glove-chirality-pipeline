"""Build the Windows Qt client and frozen Python worker without bundling private assets."""
from __future__ import annotations

import argparse
import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path


def short_path(path: Path) -> str:
    """MinGW Makefiles need a make executable path without spaces on Windows."""
    buffer = ctypes.create_unicode_buffer(32768)
    if not ctypes.windll.kernel32.GetShortPathNameW(str(path), buffer, len(buffer)):
        raise OSError(f"Cannot resolve build-tool path: {path}")
    return buffer.value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qt-prefix", required=True, type=Path,
                        help="Qt 6.8+ mingw_64 installation folder")
    parser.add_argument("--compiler-bin", required=True, type=Path,
                        help="Matching MinGW bin folder")
    parser.add_argument("--bundle", default="outputs/GRIP-desktop", type=Path)
    parser.add_argument("--build-dir", default="outputs/native-build", type=Path)
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("This deployment builder currently supports Windows only")
    root = Path(__file__).resolve().parents[1]
    qt = args.qt_prefix.resolve()
    compiler = args.compiler_bin.resolve()
    for executable in (qt / "bin/windeployqt.exe", compiler / "g++.exe", compiler / "mingw32-make.exe"):
        if not executable.is_file():
            parser.error(f"Missing build prerequisite: {executable}")
    bundle = (root / args.bundle).resolve()
    build = (root / args.build_dir).resolve()
    environment = os.environ.copy()
    environment["PATH"] = os.pathsep.join([str(compiler), str(qt / "bin"), environment.get("PATH", "")])

    def run(command):
        subprocess.run([str(part) for part in command], cwd=root, env=environment, check=True)

    run(["cmake", "-S", root / "desktop", "-B", build, "-G", "MinGW Makefiles",
         "-DCMAKE_BUILD_TYPE=Release", f"-DCMAKE_PREFIX_PATH={qt}",
         f"-DCMAKE_CXX_COMPILER={compiler / 'g++.exe'}",
         f"-DCMAKE_MAKE_PROGRAM={short_path(compiler / 'mingw32-make.exe')}"])
    run(["cmake", "--build", build, "--parallel", "4"])
    bundle.mkdir(parents=True, exist_ok=True)
    shutil.copy2(build / "GRIP.exe", bundle / "GRIP.exe")
    run([qt / "bin/windeployqt.exe", "--release", "--qmldir", root / "desktop", bundle / "GRIP.exe"])
    run([sys.executable, "-m", "PyInstaller", "--noconfirm", "--name", "grip-worker",
         "--distpath", root / "outputs/native-worker", "--workpath", root / "outputs/native-worker-build/work",
         "--specpath", root / "outputs/native-worker-build", "--paths", root / "src",
         "--collect-data", "tensorboard", "--collect-data", "glove_chirality",
         "--collect-all", "ultralytics", "--collect-all", "timm", "--collect-all", "torchvision",
         "--hidden-import", "tensorboard.main", "--hidden-import", "serial",
         "--hidden-import", "tensorboard.compat.tensorflow_stub", root / "tools/desktop_worker_entry.py"])
    shutil.copytree(root / "outputs/native-worker/grip-worker", bundle / "worker", dirs_exist_ok=True)
    shutil.copytree(root / "configs", bundle / "configs", dirs_exist_ok=True)
    print(f"Native bundle ready: {bundle}. Run tools/verify_native_package.py before deployment.")


if __name__ == "__main__":
    main()
