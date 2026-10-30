"""Compare the instruction counts of a branch build against its merge-base build.

Both interpreters run the scenarios and workflow phases of this checkout, each
importing its own installed pygraphc. Counts are deterministic, so any
difference is a real change: by default every changed count fails the run, in
either direction, because an unexpected drop can mean skipped work as much as
a rise can mean added work. The reviewer accepts a change by labelling the
pull request; the table goes to stdout and, in CI, to the job summary.

    python -m performance_tests.instructions.compare --base-python BASE --head-python HEAD
"""

import argparse
import os
import sys
from pathlib import Path

from performance_tests.instructions.callgrind import count_instructions
from performance_tests.instructions.scenarios import SCENARIOS
from performance_tests.instructions.workflows import workflow_scenario_names

DEFAULT_THRESHOLD = 0.0


def comparison_table(base: dict[str, int], head: dict[str, int], threshold: float) -> tuple[str, list[str]]:
    """Markdown table of every measured name and the names whose count changed by more than the threshold."""
    lines = ["| scenario | base | head | change |", "|---|---:|---:|---:|"]
    changed: list[str] = []
    for name in dict.fromkeys([*head, *base]):
        if name not in head:
            changed.append(name)
            lines.append(f"| {name} | {base[name]:,} | not measured | **changed** |")
            continue
        if name not in base:
            lines.append(f"| {name} | new | {head[name]:,} | |")
            continue
        change = head[name] / base[name] - 1 if base[name] else float(head[name] != 0)
        marker = ""
        if abs(change) > threshold:
            changed.append(name)
            marker = " **changed**"
        lines.append(f"| {name} | {base[name]:,} | {head[name]:,} | {change:+.2%}{marker} |")
    return "\n".join(lines), changed


def main() -> int:
    """Run both builds, print the table and return 1 when a count changed."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-python", required=True)
    parser.add_argument("--head-python", required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    arguments = parser.parse_args()
    names = [*SCENARIOS, *workflow_scenario_names()]
    base = count_instructions(names, arguments.base_python)
    head = count_instructions(names, arguments.head_python)
    table, changed = comparison_table(base, head, arguments.threshold)
    heading = f"Instruction counts against the merge base ({len(changed)} changed)"
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
