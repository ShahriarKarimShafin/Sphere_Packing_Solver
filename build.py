"""Compile the optional C kernels with GCC or Clang; no downloads or fast-math."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys


def build(compiler=None):
    """Build shared libraries next to their sources in native/."""
    native_dir = Path(__file__).resolve().parent / "native"
    compiler = (
        compiler or os.environ.get("CC") or shutil.which("gcc")
        or shutil.which("clang") or shutil.which("cc")
    )
    if not compiler:
        raise RuntimeError("Install GCC/Clang or use --backend python")
    suffix = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
    for stem in ("energy", "moves"):
        target = native_dir / (stem + suffix)
        temporary = native_dir / (stem + ".pending" + suffix)
        flags = ["-O3", "-std=c99", "-shared"]
        if sys.platform != "win32":
            flags += ["-fPIC"]
        elif "gcc" in Path(compiler).name.lower():
            flags += ["-static-libgcc"]
        subprocess.run(
            [compiler, *flags, str(native_dir / (stem + ".c")),
             "-o", str(temporary), "-lm"],
            check=True,
        )
        temporary.replace(target)
        print("Built " + str(target), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compiler")
    build(parser.parse_args().compiler)
