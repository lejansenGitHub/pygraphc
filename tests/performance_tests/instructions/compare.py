"""Compare the instruction counts of a branch build against its merge-base build.

Both interpreters run the scenarios and workflow phases of this checkout, each
importing its own installed pygraphc. Within one binary the counts are exact,
but a changed C extension shifts memory addresses, and CPython's allocator
then takes slightly different paths: work that did not change moves by up to
about 1.1% (measured on a pure rename with identical machine code). A count
that moves by more than ``DEFAULT_THRESHOLD`` fails the run, in either
direction, because an unexpected drop can mean skipped work as much as a rise
can mean added work, so a C change often needs a reviewer's label even when
nothing but layout moved; every count that moved appears in the aligned table, largest
change first, and the counts that round to 0.00% are only counted below it. The reviewer accepts
a change by labelling the pull request; the table goes to stdout and, in CI,
to the job summary. Both builds run at the same time: an instruction count does
not depend on the load next to it.

    python -m performance_tests.instructions.compare --base-python BASE --head-python HEAD
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


def _aligned(header: tuple[str, ...], rows: list[tuple[str, ...]]) -> list[str]:
    """Markdown table lines padded so every column starts at the same character; the numbers right-aligned."""
    widths = [max(len(row[column]) for row in [header, *rows]) for column in range(len(header))]

    def line(row: tuple[str, ...]) -> str:
        numbers = (cell.rjust(width) for cell, width in zip(row[1:-1], widths[1:-1], strict=True))
        cells = [row[0].ljust(widths[0]), *numbers, row[-1].ljust(widths[-1])]
        return "| " + " | ".join(cells) + " |"

    right_aligned = ("-" * (width + 1) + ":" for width in widths[1:-1])
    separator = "|" + "|".join(["-" * (widths[0] + 2), *right_aligned, "-" * (widths[-1] + 2)]) + "|"
    return [line(header), separator, *(line(row) for row in rows)]


def comparison_table(base: dict[str, int], head: dict[str, int], threshold: float) -> tuple[str, list[str]]:
    """Aligned Markdown table of every count that moved, largest change first, and the names beyond the threshold.

    Counts that round to +-0.00% are left out and only counted below the table.
    """
    rows: list[tuple[float, tuple[str, str, str, str, str]]] = []
    changed: list[str] = []
    unchanged_count = 0
    for name in dict.fromkeys([*head, *base]):
        if name not in head:
            changed.append(name)
            rows.append((float("inf"), (name, f"{base[name]:,}", "not measured", "", "**changed**")))
            continue
        if name not in base:
            rows.append((float("inf"), (name, "new", f"{head[name]:,}", "", "")))
            continue
        change = head[name] / base[name] - 1 if base[name] else float(head[name] != 0)
        if f"{abs(change):.2%}" == "0.00%":
            unchanged_count += 1
            continue
        marker = ""
        if abs(change) > threshold:
            changed.append(name)
            marker = "**changed**"
        rows.append((abs(change), (name, f"{base[name]:,}", f"{head[name]:,}", f"{change:+.2%}", marker)))
    ordered = [row for _size, row in sorted(rows, key=lambda size_and_row: -size_and_row[0])]
    lines = _aligned(("scenario", "base", "head", "change", f"beyond {threshold:.1%}"), ordered)
    lines.append("")
    lines.append(f"{unchanged_count} more count(s) unchanged (rounded to 0.00%), not listed.")
    return "\n".join(lines), changed


def main() -> int:
    """Run both builds, print the table and return 1 when a count moved beyond the threshold."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-python", required=True)
    parser.add_argument("--head-python", required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    arguments = parser.parse_args()
    names = [*SCENARIOS, *workflow_scenario_names()]
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
