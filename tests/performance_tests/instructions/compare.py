"""Compare the instruction counts of a branch build against its merge-base build.

Both interpreters run the scenarios of this checkout, each importing its own
installed pygraphc. A scenario whose count grows by more than the threshold
is a regression and fails the run; scenarios the base cannot prepare are
listed as new. The table goes to stdout and, in CI, to the job summary.

    python -m performance_tests.instructions.compare --base-python BASE --head-python HEAD
"""

import argparse
import os
import sys
from pathlib import Path

from performance_tests.instructions.callgrind import count_instructions
from performance_tests.instructions.scenarios import SCENARIOS

DEFAULT_THRESHOLD = 0.02


def comparison_table(base: dict[str, int], head: dict[str, int], threshold: float) -> tuple[str, list[str]]:
    """Markdown table of every scenario and the names above the threshold."""
    lines = ["| scenario | base | head | change |", "|---|---:|---:|---:|"]
    regressions: list[str] = []
    for name in SCENARIOS:
        if name not in head:
            lines.append(f"| {name} | {base.get(name, 0):,} | not measured | |")
            continue
        if name not in base:
            lines.append(f"| {name} | new | {head[name]:,} | |")
            continue
        change = head[name] / base[name] - 1
        marker = ""
        if change > threshold:
            regressions.append(name)
            marker = " **regression**"
        lines.append(f"| {name} | {base[name]:,} | {head[name]:,} | {change:+.2%}{marker} |")
    return "\n".join(lines), regressions


def main() -> int:
    """Run both builds, print the table and return 1 on a regression."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-python", required=True)
    parser.add_argument("--head-python", required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    arguments = parser.parse_args()
    names = list(SCENARIOS)
    base = count_instructions(names, arguments.base_python)
    head = count_instructions(names, arguments.head_python)
    table, regressions = comparison_table(base, head, arguments.threshold)
    heading = f"Instruction counts against the merge base (threshold {arguments.threshold:+.0%})"
    report = f"### {heading}\n\n{table}\n"
    print(report)  # noqa: T201 — the table is the command's output
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with Path(summary).open("a", encoding="utf-8") as summary_file:
            summary_file.write(report)
    if regressions:
        print(f"{len(regressions)} regression(s): {', '.join(regressions)}", file=sys.stderr)  # noqa: T201 — the failure reason
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
