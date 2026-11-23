"""Build the Python wheel with vMAR's original C23 sources (no duplicated math)."""

from pathlib import Path
import sys
from setuptools import Extension, setup


ROOT = Path(__file__).resolve().parent
CORE_SOURCES = [
    "vmar_core", "vmar_state_layout", "vmar_neuron", "vmar_memory",
    "vmar_control", "vmar_custom_activation", "vmar_adaptive_control",
    "vmar_regime", "vmar_trace", "vmar_network", "vmar_network_system",
    "vmar_network_mechanism",
]

if sys.platform not in {"darwin", "linux"}:
    raise RuntimeError("G0.9-P.2 Python builds support Linux and macOS only")

setup(
    ext_modules=[
        Extension(
            "vmar._native",
            sources=["vmar/_native_module.c"] + [f"src/{name}.c" for name in CORE_SOURCES],
            include_dirs=[str(ROOT / "include")],
            extra_compile_args=["-std=c2x", "-O2"],
            libraries=["m"] if sys.platform == "linux" else [],
        )
    ],
)
