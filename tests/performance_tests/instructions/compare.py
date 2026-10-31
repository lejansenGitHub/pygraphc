"""Compare the instruction counts of a branch build against its merge-base build.

Both interpreters run the scenarios and workflow phases of this checkout, each
importing its own installed pygraphc. Within one binary the counts are exact,
but a changed C extension shifts memory addresses, and CPython's allocator
then takes slightly different paths: work that did not change moves by up to
about 1.1% (measured on a pure rename with identical machine code). A count
that moves by more than ``DEFAULT_THRESHOLD`` fails the run, in either
direction, because an unexpected drop can mean skipped work as much as a rise
can mean added work, so a C change often needs a reviewer's label even when
nothing but layout moved; every change appears in the table, sorted by size. The reviewer accepts
a change by labelling the pull request; the table goes to stdout and, in CI,
to the job summary. Both builds run at the same time: an instruction count does
not depend on the load next to it.

    python -m performance_tests.instructions.compare --base-python BASE --head-python HEAD [--only bfs,port_]

``--only`` takes comma-separated substrings and measures just the scenario and
workflow names containing one of them, for a quick check of the scenarios a
change touches; without it everything is measured.
"""

import argparse
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from performance_tests.instructions.callgrind import count_instructions
from performance_tests.instructions.scenarios import SCENARIOS
from performance_tests.instructions.workflows import workflow_scenario_names

DEFAULT_THRESHOLD = 0.005
"""Below the layout noise of up to 1.1% on purpose: code generation changes of 0.5 to 1% matter."""


def comparison_table(base: dict[str, int], head: dict[str, int], threshold: float) -> tuple[str, list[str]]:
    """Markdown table of every measured name and the names whose count changed by more than the threshold."""
    rows: list[tuple[float, str]] = []
    changed: list[str] = []
    for name in dict.fromkeys([*head, *base]):
        if name not in head:
            changed.append(name)
            rows.append((float("inf"), f"| {name} | {base[name]:,} | not measured | **changed** |"))
            continue
        if name not in base:
            rows.append((float("inf"), f"| {name} | new | {head[name]:,} | |"))
            continue
        change = head[name] / base[name] - 1 if base[name] else float(head[name] != 0)
        marker = ""
        if abs(change) > threshold:
            changed.append(name)
            marker = " **changed**"
        rows.append((abs(change), f"| {name} | {base[name]:,} | {head[name]:,} | {change:+.2%}{marker} |"))
    lines = ["| scenario | base | head | change |", "|---|---:|---:|---:|"]
    lines += [row for _size, row in sorted(rows, key=lambda size_and_row: -size_and_row[0])]
    return "\n".join(lines), changed


def selected_names(names: list[str], only: str) -> list[str]:
    """The names containing one of the comma-separated substrings of ``only``; all names when it is empty."""
    substrings = [substring for substring in only.split(",") if substring]
    if not substrings:
        return names
    return [name for name in names if any(substring in name for substring in substrings)]


def main() -> int:
    """Run both builds, print the table and return 1 when a count moved beyond the threshold."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-python", required=True)
    parser.add_argument("--head-python", required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--only", default="", help="comma-separated substrings; measure only names containing one")
    arguments = parser.parse_args()
    names = selected_names([*SCENARIOS, *workflow_scenario_names()], arguments.only)
    with ThreadPoolExecutor(max_workers=2) as executor:
        base_counts = executor.submit(count_instructions, names, arguments.base_python)
        head_counts = executor.submit(count_instructions, names, arguments.head_python)
        base, head = base_counts.result(), head_counts.result()
    table, changed = comparison_table(base, head, arguments.threshold)
    heading = f"Instruction counts against the merge base ({len(changed)} beyond {arguments.threshold:.1%})"
    report = f"### {heading}\n\n{table}\n"
    print(report)  # noqa: T201 — the table is the command's output
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as summary_file:
            summary_file.write(report)
    if changed:
        print(  # noqa: T201 — the failure reason
            f"{len(changed)} count(s) changed: {', '.join(changed)}; "
            "review them and label the pull request accept-instruction-change",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
