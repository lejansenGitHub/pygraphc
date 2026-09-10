import os

from setuptools import Extension, setup


def compile_args() -> list[str]:
    """Portable `-O3` by default; `PYGRAPHC_NATIVE=1` adds `-march=native` for a CPU-tuned local build."""
    args = ["-O3"]
    if os.environ.get("PYGRAPHC_NATIVE") == "1":
        args.append("-march=native")
    return args


setup(
    ext_modules=[
        Extension(
            "pygraphc._core",
            sources=["src/pygraphc/_core.c"],
            extra_compile_args=compile_args(),
        ),
        Extension(
            "pygraphc._dag_learn",
            sources=["src/pygraphc/_dag_learn.c"],
            extra_compile_args=compile_args(),
        ),
    ],
)
