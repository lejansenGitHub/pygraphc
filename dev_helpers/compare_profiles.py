"""Compare callgrind profiles of the working tree against a base revision, in a Linux container.

CI's instruction tier says that a count moved; this says where. Both trees are
built with debug info and measured by the same harness as CI
(``tests/performance_tests/instructions``): the working tree's scenarios, run
once by each build. The report lists every count, and for each one beyond the
threshold the functions whose own instructions changed most. The full
per-line annotation of every dump stays next to the report.

    python dev_helpers/compare_profiles.py --only bridges,articulation_points [--base origin/main]

Needs podman or docker; valgrind runs inside the container, so this works on
macOS too. The container is arm64 on Apple silicon, CI also measures x86-64
(``--platform linux/amd64`` emulates it, slowly). Uncommitted changes are
included; untracked files are included unless git ignores them.
"""

import argparse
import os
import platform
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path

IMAGE = "python:3.13-slim"
CONTAINER_DIRECTORY = Path("/comparison")
TREES = ("base", "head")
DEFAULT_THRESHOLD = 0.005
"""The threshold of CI's instruction tier."""
FUNCTION_ROWS = 15

MARKER_SOURCE = """#include <valgrind/callgrind.h>
void run_measured(void (*callback)(void)) {
    CALLGRIND_ZERO_STATS;
    CALLGRIND_TOGGLE_COLLECT;
    callback();
    CALLGRIND_TOGGLE_COLLECT;
    CALLGRIND_DUMP_STATS;
}
"""

SELECT_SCENARIOS = """
import sys
from performance_tests.instructions.compare import selected_names
from performance_tests.instructions.scenarios import SCENARIOS
from performance_tests.instructions.workflows import workflow_scenario_names
print("\\n".join(selected_names([*SCENARIOS, *workflow_scenario_names()], sys.argv[1])))
"""


# ── Report: pure functions over callgrind dumps ──

NAME_DEFINITION = re.compile(r"^\((\d+)\)(?: (.*))?$")


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
    """Own instructions per function (``file:function``) and per source file of one callgrind dump.

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
            name = _resolve(file_names, value).rsplit("/", 1)[-1]
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
        function_key = f"{function_file}:{function}"
        by_function[function_key] = by_function.get(function_key, 0) + cost
        by_file[line_file] = by_file.get(line_file, 0) + cost
    return by_function, by_file


def relative_change(base: int, head: int) -> float:
    """Head over base minus one; a count appearing from zero is a full change."""
    if base == 0:
        return float(head != 0)
    return head / base - 1


def totals_table(base: dict[str, int], head: dict[str, int], threshold: float) -> tuple[list[str], list[str]]:
    """Markdown rows of every scenario, largest change first, and the names beyond the threshold."""
    rows: list[tuple[float, str]] = []
    changed: list[str] = []
    for name in dict.fromkeys([*head, *base]):
        if name not in base or name not in head:
            base_cell = f"{base[name]:,}" if name in base else "not measured"
            head_cell = f"{head[name]:,}" if name in head else "not measured"
            rows.append((float("inf"), f"| {name} | {base_cell} | {head_cell} | |"))
            continue
        change = relative_change(base[name], head[name])
        marker = ""
        if abs(change) > threshold:
            changed.append(name)
            marker = " **changed**"
        rows.append((abs(change), f"| {name} | {base[name]:,} | {head[name]:,} | {change:+.2%}{marker} |"))
    header = ["| scenario | base | head | change |", "|---|---:|---:|---:|"]
    return header + [row for _size, row in sorted(rows, key=lambda size_and_row: -size_and_row[0])], changed


def difference_table(base: dict[str, int], head: dict[str, int], heading: str, limit: int = FUNCTION_ROWS) -> list[str]:
    """Markdown rows of the entries whose own instructions moved most, by absolute difference."""
    differences = sorted(
        ((head.get(entry, 0) - base.get(entry, 0), entry) for entry in {*base, *head}),
        key=lambda difference_and_entry: (-abs(difference_and_entry[0]), difference_and_entry[1]),
    )
    rows = [f"| {heading} | base | head | difference |", "|---|---:|---:|---:|"]
    for difference, entry in differences[:limit]:
        if difference == 0:
            break
        rows.append(f"| `{entry}` | {base.get(entry, 0):,} | {head.get(entry, 0):,} | {difference:+,} |")
    return rows


# ── Host: prepare both trees and run the container ──


def _run(
    command: list[str],
    *,
    capture_output: bool = False,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    quiet: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a command that must succeed; ``quiet`` drops its standard output but keeps its errors."""
    output = subprocess.DEVNULL if quiet else None
    if capture_output:
        return subprocess.run(command, check=True, text=True, capture_output=True, cwd=cwd, env=env)
    return subprocess.run(command, check=True, text=True, cwd=cwd, env=env, stdout=output)


def _export_base(repository: Path, revision: str, target: Path) -> None:
    """The tree of a revision, as git stores it."""
    archive = subprocess.run(
        ["git", "-C", str(repository), "archive", "--format=tar", revision], check=True, capture_output=True
    )
    with tempfile.TemporaryFile() as archive_file:
        archive_file.write(archive.stdout)
        archive_file.seek(0)
        with tarfile.open(fileobj=archive_file) as tar:
            tar.extractall(target, filter="data")


def _export_working_tree(repository: Path, target: Path) -> None:
    """Tracked files as they are on disk, plus untracked files git does not ignore."""
    listing = _run(
        ["git", "-C", str(repository), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
    ).stdout
    for relative_path in filter(None, listing.split("\0")):
        source = repository / relative_path
        if not source.is_file():
            continue
        destination = target / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)


def native_platform() -> str:
    """The container platform matching this machine, so no image runs emulated by accident."""
    return "linux/arm64" if platform.machine().lower() in {"arm64", "aarch64"} else "linux/amd64"


def _container_engine() -> str:
    for engine in ("podman", "docker"):
        if shutil.which(engine):
            return engine
    message = "needs podman or docker"
    raise SystemExit(message)


def run_on_host(arguments: argparse.Namespace) -> int:
    """Export both trees into the output directory and run the comparison in a container."""
    repository = Path(_run(["git", "rev-parse", "--show-toplevel"], capture_output=True).stdout.strip())
    output = repository / arguments.output
    if output.exists():
        shutil.rmtree(output)
    _export_base(repository, arguments.base, output / "base")
    _export_working_tree(repository, output / "head")
    shutil.copy2(Path(__file__), output / "compare_profiles.py")
    container_platform = arguments.platform or native_platform()
    command = [_container_engine(), "run", "--rm", "--platform", container_platform]
    command += ["-v", f"{output}:{CONTAINER_DIRECTORY}", IMAGE]
    command += [
        "python",
        str(CONTAINER_DIRECTORY / "compare_profiles.py"),
        "--inside-container",
        "--only",
        arguments.only,
        "--threshold",
        str(arguments.threshold),
        "--platform",
        container_platform,
    ]
    completed = subprocess.run(command, check=False)
    print(f"\nreport and per-line annotations: {output}")  # noqa: T201 — where to look next
    return completed.returncode


# ── Container: build, measure, annotate, report ──


def _build(tree: str) -> Path:
    """A virtual environment with the tree's pygraphc, compiled with debug info; outside the mount, where venv works."""
    environment = Path("/venvs") / tree
    _run([sys.executable, "-m", "venv", str(environment)])
    _run(
        [str(environment / "bin" / "pip"), "install", "-q", str(CONTAINER_DIRECTORY / tree)],
        env={**os.environ, "CFLAGS": "-g"},
        quiet=True,
    )
    return environment / "bin" / "python"


def _measure(tree: str, python: Path, names: list[str], marker: Path) -> list[tuple[str, Path]]:
    """Run the head tree's scenarios with one build under callgrind; one dump per measured scenario or phase."""
    profiles = CONTAINER_DIRECTORY / "profiles" / tree
    profiles.mkdir(parents=True)
    manifest = profiles / "measured.txt"
    completed = subprocess.run(
        [
            "valgrind",
            "--tool=callgrind",
            "--collect-atstart=no",
            f"--callgrind-out-file={profiles / 'callgrind.out'}",
            str(python),
            "-m",
            "performance_tests.instructions.measure",
            *names,
        ],
        cwd=CONTAINER_DIRECTORY / "head" / "tests",
        env={
            **os.environ,
            "PYTHONHASHSEED": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PERFORMANCE_MARKER_LIBRARY": str(marker),
            "PERFORMANCE_MANIFEST": str(manifest),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        message = f"measuring {tree} failed:\n{completed.stderr[-3000:]}"
        raise SystemExit(message)
    dumps = sorted(profiles.glob("callgrind.out.*"), key=lambda dump: int(dump.name.rsplit(".", 1)[1]))
    return list(zip(manifest.read_text(encoding="utf-8").splitlines(), dumps, strict=True))


def _total(dump: Path) -> int:
    for line in dump.read_text(encoding="utf-8").splitlines():
        if line.startswith(("totals:", "summary:")):
            return int(line.split()[1])
    message = f"no instruction total in {dump}"
    raise ValueError(message)


def _annotate_lines(tree: str, name: str, dump: Path) -> None:
    """Write the per-line annotation of the C core's sources next to the dump."""
    sources = CONTAINER_DIRECTORY / tree / "src" / "pygraphc" / "_core_src"
    annotation = _run(
        ["callgrind_annotate", "--threshold=100", "--inclusive=no", "--auto=yes", f"--include={sources}", str(dump)],
        capture_output=True,
    ).stdout
    (dump.parent / f"{name.replace('/', '__')}.{dump.name.rsplit('.', 1)[1]}.txt").write_text(
        annotation, encoding="utf-8"
    )


def run_in_container(arguments: argparse.Namespace) -> int:
    """Measure both builds and write report.md; return 1 when a count moved beyond the threshold."""
    _run(["apt-get", "-qq", "update"], quiet=True)
    _run(["apt-get", "-qq", "install", "-y", "gcc", "valgrind"], quiet=True)
    pythons = {tree: _build(tree) for tree in TREES}
    marker = CONTAINER_DIRECTORY / "libmarker.so"
    (CONTAINER_DIRECTORY / "marker.c").write_text(MARKER_SOURCE, encoding="utf-8")
    _run(["gcc", "-O0", "-shared", "-fPIC", "-o", str(marker), str(CONTAINER_DIRECTORY / "marker.c")])
    names = _run(
        [str(pythons["head"]), "-c", SELECT_SCENARIOS, arguments.only],
        cwd=CONTAINER_DIRECTORY / "head" / "tests",
        capture_output=True,
    ).stdout.split()

    totals: dict[str, dict[str, int]] = {tree: {} for tree in TREES}
    functions: dict[str, dict[str, dict[str, int]]] = {tree: {} for tree in TREES}
    files: dict[str, dict[str, dict[str, int]]] = {tree: {} for tree in TREES}
    for tree in TREES:
        for name, dump in _measure(tree, pythons[tree], names, marker):
            totals[tree][name] = totals[tree].get(name, 0) + _total(dump)
            _annotate_lines(tree, name, dump)
            by_function, by_file = self_costs(dump.read_text(encoding="utf-8"))
            for merged, costs in (
                (functions[tree].setdefault(name, {}), by_function),
                (files[tree].setdefault(name, {}), by_file),
            ):
                for entry, cost in costs.items():
                    merged[entry] = merged.get(entry, 0) + cost

    rows, changed = totals_table(totals["base"], totals["head"], arguments.threshold)
    heading = f"## Instruction counts on {arguments.platform} ({len(changed)} beyond {arguments.threshold:.1%})"
    report = [heading, "", *rows]
    for name in changed:
        report += ["", f"### {name}", "", *difference_table(files["base"][name], files["head"][name], "file")]
        report += ["", *difference_table(functions["base"][name], functions["head"][name], "function")]
    report += ["", "Per-line annotations: `profiles/base/` and `profiles/head/`, one file per dump."]
    text = "\n".join(report) + "\n"
    (CONTAINER_DIRECTORY / "report.md").write_text(text, encoding="utf-8")
    print(text)  # noqa: T201 — the report is the command's output
    return 1 if changed else 0


def main() -> int:
    """Run on the host, or inside the container when called by the host run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--only", required=True, help="comma-separated substrings of scenario names, as compare.py")
    parser.add_argument("--base", default="origin/main", help="revision to compare against (default origin/main)")
    parser.add_argument("--output", default="profiles/comparison", help="directory for trees, dumps and report")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument(
        "--platform", default="", help="container platform; default this machine's, linux/amd64 emulates x86-64"
    )
    parser.add_argument("--inside-container", action="store_true", help=argparse.SUPPRESS)
    arguments = parser.parse_args()
    return run_in_container(arguments) if arguments.inside_container else run_on_host(arguments)


if __name__ == "__main__":
    sys.exit(main())
