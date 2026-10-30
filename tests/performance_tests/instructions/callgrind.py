"""Deterministic instruction counts of scenarios under callgrind.

A tiny shared library exposes ``run_measured(callback)``, which only calls the
callback. Callgrind collects inside that function alone
(``--toggle-collect``) and dumps a profile each time it returns
(``--dump-after``), so one valgrind process measures many scenarios, each
counted without its setup, imports or interpreter start.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

MARKER_SOURCE = "void run_measured(void (*callback)(void)) { callback(); }\n"
MARKER_FUNCTION = "run_measured"


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


def _total_instructions(profile: Path) -> int:
    """The instruction total of one callgrind dump."""
    for line in profile.read_text(encoding="utf-8").splitlines():
        if line.startswith(("totals:", "summary:")):
            return int(line.split()[1])
    message = f"no instruction total in {profile}"
    raise ValueError(message)


def _dump_number(profile: Path) -> int:
    """Part number callgrind appends to every dump file."""
    return int(profile.name.rsplit(".", 1)[1])


def count_instructions(scenario_names: list[str], python: str = sys.executable) -> dict[str, int]:
    """Instruction count of every scenario that the given interpreter can prepare; others are left out."""
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
                f"--toggle-collect={MARKER_FUNCTION}",
                f"--dump-after={MARKER_FUNCTION}",
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
        measured_names = manifest.read_text(encoding="utf-8").split()
        profiles = sorted(directory.glob("callgrind.out.*"), key=_dump_number)
        if len(profiles) != len(measured_names):
            message = (
                f"{len(measured_names)} scenarios measured but {len(profiles)} callgrind dumps; "
                f"files: {sorted(path.name for path in directory.iterdir())}; "
                f"valgrind stderr: {completed.stderr[-3000:]}"
            )
            raise RuntimeError(message)
        return {name: _total_instructions(profile) for name, profile in zip(measured_names, profiles, strict=True)}
