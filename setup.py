import os
from glob import glob

from setuptools import Extension, setup

CORE_SOURCE_DIRECTORY = "src/pygraphc/_core_src"


def core_files(pattern: str) -> list[str]:
    """Files of the `_core` extension below its source directory, sorted so the build order is stable."""
    return sorted(glob(f"{CORE_SOURCE_DIRECTORY}/**/{pattern}", recursive=True))


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
            sources=["src/pygraphc/_core.c", *core_files("*.c")],
            depends=core_files("*.h"),
            extra_compile_args=compile_args(),
        ),
        Extension(
            "pygraphc._dag_learn",
            sources=["src/pygraphc/_dag_learn.c"],
            extra_compile_args=compile_args(),
        ),
    ],
)
