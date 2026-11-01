"""Compare the instruction counts of a branch build against its merge-base build.

Both interpreters run the scenarios and workflow phases of this checkout, each
importing its own installed pygraphc. Within one binary the counts are exact,
but a changed C extension shifts memory addresses, and CPython's allocator
then takes slightly different paths: work that did not change moves by up to
about 1.1% (measured on a pure rename with identical machine code). A count
that moves by more than ``DEFAULT_THRESHOLD`` is marked and warned about; the run
fails only on a rise beyond ``FAILING_RISE``, a drop beyond ``FAILING_DROP`` or a
scenario no longer measured, so layout drift stays a warning while added or
skipped work still stops the run. Every count that moved appears in one of two
aligned tables, more instructions and fewer, each largest change first, and the
counts that round to 0.00% are only counted below them. A reviewer accepts a
failing change by labelling the pull request; the table goes to stdout and, in
CI, to the job summary. Both builds run at the same time: an instruction count does
not depend on the load next to it.

    python -m performance_tests.instructions.compare --base-python BASE --head-python HEAD [--only bfs,terminal_]

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
FAILING_RISE = 0.015
"""A rise beyond the layout noise of up to 1.1% fails the run (user decision 2026-11-01)."""
FAILING_DROP = 0.10
"""A drop this large fails the run: it can mean skipped work (user decision 2026-11-01)."""


def _aligned(header: tuple[str, ...], rows: list[tuple[str, ...]], widths: list[int]) -> list[str]:
    """Markdown table lines padded to the given column widths; the numbers right-aligned."""

    def line(row: tuple[str, ...]) -> str:
        numbers = (cell.rjust(width) for cell, width in zip(row[1:-1], widths[1:-1], strict=True))
        cells = [row[0].ljust(widths[0]), *numbers, row[-1].ljust(widths[-1])]
        return "| " + " | ".join(cells) + " |"

    right_aligned = ("-" * (width + 1) + ":" for width in widths[1:-1])
    separator = "|" + "|".join(["-" * (widths[0] + 2), *right_aligned, "-" * (widths[-1] + 2)]) + "|"
    return [line(header), separator, *(line(row) for row in rows)]


def comparison_table(base: dict[str, int], head: dict[str, int], threshold: float) -> tuple[str, list[str]]:
    """Aligned Markdown tables of the counts that moved, and the names beyond the threshold.

    One table for more instructions (slower), one for fewer (faster), one for
    scenarios measured on one side only; each sorted by the size of the
    change, largest first, all with the same column widths. Counts that round
    to +-0.00% are left out and only counted below the tables.
    """
    slower: list[tuple[float, tuple[str, ...]]] = []
    faster: list[tuple[float, tuple[str, ...]]] = []
    one_sided: list[tuple[str, ...]] = []
    changed: list[str] = []
    unchanged_count = 0
    for name in dict.fromkeys([*head, *base]):
        if name not in head:
            changed.append(name)
            one_sided.append((name, f"{base[name]:,}", "not measured", "", "**fails**"))
            continue
        if name not in base:
            one_sided.append((name, "new", f"{head[name]:,}", "", ""))
            continue
        change = head[name] / base[name] - 1 if base[name] else float(head[name] != 0)
        if f"{abs(change):.2%}" == "0.00%":
            unchanged_count += 1
            continue
        marker = ""
        if abs(change) > threshold:
            changed.append(name)
            marker = "**fails**" if _fails(change) else "**changed**"
        row = (name, f"{base[name]:,}", f"{head[name]:,}", f"{change:+.2%}", marker)
        (slower if change > 0 else faster).append((abs(change), row))
    header = ("scenario", "base", "head", "change", f"beyond {threshold:.1%}")
    groups = [
        ("More instructions (slower)", _largest_first(slower), True),
        ("Fewer instructions (faster)", _largest_first(faster), True),
        ("Measured on one side only", one_sided, False),
    ]
    all_rows = [header, *(row for _title, rows, _shown_when_empty in groups for row in rows)]
    widths = [max(len(row[column]) for row in all_rows) for column in range(len(header))]
    lines: list[str] = []
    for title, rows, shown_when_empty in groups:
        if not rows and not shown_when_empty:
            continue
        lines += [f"**{title}: {len(rows)}**", ""]
        if rows:
            lines += [*_aligned(header, rows, widths), ""]
    lines.append(f"{unchanged_count} more count(s) unchanged (rounded to 0.00%), not listed.")
    return "\n".join(lines), changed


def _fails(change: float) -> bool:
    """Whether a relative change fails the run: a rise beyond ``FAILING_RISE`` or a drop beyond ``FAILING_DROP``."""
    return change > FAILING_RISE or change < -FAILING_DROP


def failing_names(base: dict[str, int], head: dict[str, int]) -> list[str]:
    """The counts that fail the run, the rows marked **fails**: beyond the failing bounds, or no longer measured."""
    return [
        name
        for name, base_count in base.items()
        if name not in head or _fails(head[name] / base_count - 1 if base_count else float(head[name] != 0))
    ]


def _largest_first(rows: list[tuple[float, tuple[str, ...]]]) -> list[tuple[str, ...]]:
    """The rows ordered by the size of their change, largest first."""
    return [row for _size, row in sorted(rows, key=lambda size_and_row: -size_and_row[0])]


def selected_names(names: list[str], only: str) -> list[str]:
    """The names containing one of the comma-separated substrings of ``only``; all names when it is empty."""
    substrings = [substring for substring in only.split(",") if substring]
    if not substrings:
        return names
    return [name for name in names if any(substring in name for substring in substrings)]


def main() -> int:
    """Run both builds, print the table, warn on a count beyond the threshold, return 1 on a failing one."""
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
    print(report, flush=True)  # noqa: T201 — the table is the command's output
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as summary_file:
            summary_file.write(report)
    failing = failing_names(base, head)
    if failing:
        print(failure_message(len(failing)), file=sys.stderr)  # noqa: T201 — the failure reason
        return 1
    if changed:
        print(warning_message(len(changed), arguments.threshold))  # noqa: T201 — a GitHub warning annotation
    return 0


def warning_message(changed_count: int, threshold: float) -> str:
    """A GitHub warning annotation: counts moved beyond the threshold but within the failing bounds."""
    return (
        f"::warning title=Instruction counts::{changed_count} count(s) moved by more than {threshold:.1%}, "
        f"none beyond +{FAILING_RISE:.1%} or -{FAILING_DROP:.0%}; see the tables in the job summary"
    )


def failure_message(failing_count: int) -> str:
    """One line after the tables: how many counts fail; their names are the rows marked **fails**."""
    return (
        f"{failing_count} count(s) rose by more than {FAILING_RISE:.1%}, fell by more than {FAILING_DROP:.0%} "
        "or are no longer measured, marked **fails** in the tables above; "
        "review them and label the pull request accept-instruction-change"
    )


if __name__ == "__main__":
    sys.exit(main())
