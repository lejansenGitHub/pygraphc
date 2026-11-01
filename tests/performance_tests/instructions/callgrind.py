"""Deterministic instruction counts of scenarios under callgrind.

A tiny shared library exposes ``run_measured(callback)``, which switches
callgrind's collection on, calls the callback, switches it off and dumps a
profile, all through valgrind client requests. Collection starts off
(``--collect-atstart=no``), so one valgrind process measures many scenarios,
each counted without its setup, imports or interpreter start. Client requests
do not depend on callgrind recognising the call by symbol, which on x86-64
it did not for a function entered from a ctypes callback.

Every count is split into three buckets by the source file of each
instruction, inlined lines included (``fi=``/``fe=``): ``algorithm`` for
pygraphc's ``_core_src/algorithms/``, ``boundary`` for the rest of pygraphc's
C sources (parsing, result building, entry points), ``cpython`` for CPython,
the C library and anything without debug info. The extensions must be built
with ``CFLAGS=-g``, which changes no generated instruction.
"""

import os
import posixpath
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MARKER_SOURCE = """#include <valgrind/callgrind.h>

void start_measure(void) { CALLGRIND_TOGGLE_COLLECT; }

void stop_measure(void) {
    CALLGRIND_TOGGLE_COLLECT;
    CALLGRIND_DUMP_STATS;
}

void run_measured(void (*callback)(void)) {
    start_measure();
    callback();
    stop_measure();
}
"""


def valgrind_available() -> bool:
    """Whether valgrind is on the path."""
    return shutil.which("valgrind") is not None


def build_marker(directory: Path) -> Path:
    """Compile the marker library into the directory and return its path."""
    source = directory / "marker.c"
    library = directory / "libmarker.so"
    source.write_text(MARKER_SOURCE, encoding="utf-8")
    compiler = os.environ.get("CC", "cc")
    subprocess.run([compiler, "-O0", "-shared", "-fPIC", "-o", str(library), str(source)], check=True)
    return library


BUCKETS = ("algorithm", "boundary", "cpython")

BUCKET_THRESHOLDS = {"algorithm": 0.001, "boundary": 0.005, "cpython": 0.005}
"""The algorithm bucket is exact for unchanged code; the others sit below the layout noise of up to 1.1% on
purpose, since code generation changes of 0.5 to 1% matter."""


def bucket_rows(counts: dict[str, dict[str, int]]) -> dict[str, int]:
    """One count per scenario and bucket, named ``scenario [bucket]``."""
    return {f"{name} [{bucket}]": buckets[bucket] for name, buckets in counts.items() for bucket in BUCKETS}


def bucket_threshold(row_name: str) -> float:
    """The threshold of a ``scenario [bucket]`` row."""
    return BUCKET_THRESHOLDS[row_name.rsplit("[", 1)[1].rstrip("]")]


def thresholds_text() -> str:
    """The bucket thresholds as one phrase, for the heading and the failure message."""
    return ", ".join(f"{bucket} {threshold:.1%}" for bucket, threshold in BUCKET_THRESHOLDS.items())


NAME_DEFINITION = re.compile(r"^\((\d+)\)(?: (.*))?$")


def instruction_bucket(source_file: str) -> str:
    """The bucket of an instruction by the path of its source file, normalized: a header included as
    ``entry_points/../algorithms/x.h`` is an algorithm."""
    source_file = posixpath.normpath(source_file)
    if "_core_src/algorithms/" in source_file:
        return "algorithm"
    if "src/pygraphc/" in source_file:
        return "boundary"
    return "cpython"


def short_source_name(source_file: str) -> str:
    """A source file as reports name it: relative to ``_core_src`` inside it, otherwise its base name."""
    source_file = posixpath.normpath(source_file)
    if "_core_src/" in source_file:
        return source_file.split("_core_src/", 1)[1]
    return source_file.rsplit("/", 1)[-1]


def _resolve(names: dict[str, str], value: str) -> str:
    """A compressed name: ``(id) name`` defines it, ``(id)`` refers back to it."""
    match = NAME_DEFINITION.match(value)
    if not match:
        return value
    identifier, name = match.groups()
    if name is not None:
        names[identifier] = name
    return names.get(identifier, value)


def self_costs(dump: str) -> tuple[dict[str, int], dict[str, int]]:
    """Own instructions per function (``file:function``, file shortened) and per source file path of one dump.

    The cost line after ``calls=`` is the inclusive cost of that call, not
    the caller's own; ``callgrind_annotate`` misreads it for frames active
    across a client-request dump, so the dump is read directly. ``fi=`` and
    ``fe=`` name the file of inlined lines, so the per-file costs follow
    inlining where the per-function costs follow the enclosing function.
    """
    file_names: dict[str, str] = {}
    function_names: dict[str, str] = {}
    by_function: dict[str, int] = {}
    by_file: dict[str, int] = {}
    function = function_file = line_file = "???"
    skip_next_cost = False
    for line in dump.splitlines():
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator and key in {"fl", "fi", "fe", "cfi", "cfl"}:
            name = _resolve(file_names, value)
            if key == "fl":
                function_file = line_file = name
            elif key in {"fi", "fe"}:
                line_file = name
            continue
        if separator and key in {"fn", "cfn"}:
            name = re.sub(r"'\d+$", "", _resolve(function_names, value))
            if key == "fn":
                function = name
                line_file = function_file
            continue
        if separator and key == "calls":
            skip_next_cost = True
            continue
        if separator or not (line[0].isdigit() or line[0] in "+-*"):
            continue
        if skip_next_cost:
            skip_next_cost = False
            continue
        cost = int(line.split()[-1])
        function_key = f"{short_source_name(function_file)}:{function}"
        by_function[function_key] = by_function.get(function_key, 0) + cost
        by_file[line_file] = by_file.get(line_file, 0) + cost
    return by_function, by_file


def bucket_costs(by_file: dict[str, int]) -> dict[str, int]:
    """The per-file costs summed into the three buckets, each present."""
    buckets = dict.fromkeys(BUCKETS, 0)
    for source_file, cost in by_file.items():
        buckets[instruction_bucket(source_file)] += cost
    return buckets


def require_debug_info(counts: dict[str, dict[str, int]]) -> None:
    """Raise when no instruction of a whole run belongs to pygraphc: the extensions were built without ``-g``."""
    if counts and not any(buckets["algorithm"] or buckets["boundary"] for buckets in counts.values()):
        message = "no instruction is attributed to pygraphc's C sources; build the extensions with CFLAGS=-g"
        raise RuntimeError(message)


def _dump_number(profile: Path) -> int:
    """Part number callgrind appends to every dump file."""
    return int(profile.name.rsplit(".", 1)[1])


def count_instructions(scenario_names: list[str], python: str = sys.executable) -> dict[str, dict[str, int]]:
    """Instruction counts per bucket of every scenario the given interpreter can prepare; others are left out.

    A ``workflow/<name>`` entry expands to one count per phase of that
    workflow; a phase entered several times is the sum of its dumps.
    """
    with tempfile.TemporaryDirectory() as directory_name:
        directory = Path(directory_name)
        library = build_marker(directory)
        manifest = directory / "measured.txt"
        environment = {
            **os.environ,
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PERFORMANCE_MARKER_LIBRARY": str(library),
            "PERFORMANCE_MANIFEST": str(manifest),
        }
        tests_directory = Path(__file__).resolve().parents[2]
        completed = subprocess.run(
            [
                "valgrind",
                "--tool=callgrind",
                "--collect-atstart=no",
                f"--callgrind-out-file={directory / 'callgrind.out'}",
                python,
                "-m",
                "performance_tests.instructions.measure",
                *scenario_names,
            ],
            check=True,
            cwd=tests_directory,
            env=environment,
            capture_output=True,
            text=True,
        )
        measured_names = manifest.read_text(encoding="utf-8").splitlines()
        profiles = sorted(directory.glob("callgrind.out.*"), key=_dump_number)
        if len(profiles) != len(measured_names):
            message = (
                f"{len(measured_names)} scenarios measured but {len(profiles)} callgrind dumps; "
                f"files: {sorted(path.name for path in directory.iterdir())}; "
                f"valgrind stderr: {completed.stderr[-3000:]}"
            )
            raise RuntimeError(message)
        counts: dict[str, dict[str, int]] = {}
        for name, profile in zip(measured_names, profiles, strict=True):
            _by_function, by_file = self_costs(profile.read_text(encoding="utf-8"))
            merged = counts.setdefault(name, dict.fromkeys(BUCKETS, 0))
            for bucket, cost in bucket_costs(by_file).items():
                merged[bucket] += cost
        require_debug_info(counts)
        return counts
