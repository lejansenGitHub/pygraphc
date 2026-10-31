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


LINK_TIME_OPTIMIZATION = ["-flto"]
"""``_core`` is split across translation units; link-time optimization lets the
compiler inline across them as it did when everything was one file."""


setup(
    ext_modules=[
        Extension(
            "pygraphc._core",
            sources=core_files("*.c"),
            depends=core_files("*.h"),
            extra_compile_args=[*compile_args(), *LINK_TIME_OPTIMIZATION],
            extra_link_args=LINK_TIME_OPTIMIZATION,
        ),
        Extension(
            "pygraphc._ports",
            sources=[
                *sorted(glob("src/pygraphc/_ports_src/*.c")),
                f"{CORE_SOURCE_DIRECTORY}/models.c",
                f"{CORE_SOURCE_DIRECTORY}/helpers/parse.c",
            ],
            depends=[*sorted(glob("src/pygraphc/_ports_src/*.h")), *core_files("*.h")],
            extra_compile_args=[*compile_args(), *LINK_TIME_OPTIMIZATION],
            extra_link_args=LINK_TIME_OPTIMIZATION,
        ),
        Extension(
            "pygraphc._dag_learn",
            sources=["src/pygraphc/_dag_learn.c"],
            extra_compile_args=compile_args(),
        ),
    ],
)
