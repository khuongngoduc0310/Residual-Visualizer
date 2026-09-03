"""Print and validate the runtime contract used by Circuit Tracer."""

import platform
import struct
import sys

import gradio
import h5py
import keras
import numpy
import plotly
import tensorflow as tf


EXPECTED = {
    "tensorflow": "2.20.0",
    "keras": "3.13.2",
    "numpy": "2.1.3",
    "h5py": "3.16.0",
    "gradio": "6.26.0",
    "plotly": "7.0.0",
}


def main() -> int:
    versions = {
        "tensorflow": tf.__version__,
        "keras": keras.__version__,
        "numpy": numpy.__version__,
        "h5py": h5py.__version__,
        "gradio": gradio.__version__,
        "plotly": plotly.__version__,
    }
    print(f"Python: {sys.version.split()[0]} ({struct.calcsize('P') * 8}-bit)")
    print(f"Platform: {platform.platform()}")
    for package, version in versions.items():
        expected = EXPECTED[package]
        print(f"{package}: {version} (expected {expected})")
    print(f"TensorFlow CUDA build: {tf.test.is_built_with_cuda()}")
    print(f"Visible GPUs: {tf.config.list_physical_devices('GPU')}")

    mismatches = [
        f"{package} {version} != {EXPECTED[package]}"
        for package, version in versions.items()
        if version != EXPECTED[package]
    ]
    if sys.version_info[:2] != (3, 13) or struct.calcsize("P") * 8 != 64:
        mismatches.append("Python must be 64-bit Python 3.13")
    if mismatches:
        print("Runtime contract mismatch:")
        for mismatch in mismatches:
            print(f"- {mismatch}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
