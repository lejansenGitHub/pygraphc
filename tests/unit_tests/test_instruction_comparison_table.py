"""The table the instruction tier prints: only counts that moved, largest change first, columns aligned."""

import importlib.util
import sys
from pathlib import Path

_TESTS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_TESTS))
_SPEC = importlib.util.spec_from_file_location(
    "instruction_compare", _TESTS / "performance_tests" / "instructions" / "compare.py"
)
assert _SPEC is not None
assert _SPEC.loader is not None
instruction_compare = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(instruction_compare)


def test_unchanged_counts_are_left_out_and_counted() -> None:
    """A count that rounds to 0.00% is not a row; the line below the table says how many there were."""
    table, changed = instruction_compare.comparison_table(
        {"steady": 1_000_000, "moved": 1_000_000}, {"steady": 1_000_003, "moved": 1_010_000}, 0.005
    )

    assert "steady" not in table
    assert table.endswith("1 more count(s) unchanged (rounded to 0.00%), not listed.")
    assert changed == ["moved"]


def _scenario_names(table_lines: list[str]) -> list[str]:
    """The scenario column of the data rows among the given lines."""
    return [line.split("|")[1].strip() for line in table_lines if line.startswith("| ") and "scenario" not in line]


def test_slower_and_faster_counts_are_separate_tables_largest_change_first() -> None:
    """More instructions in one table, fewer in the next, each ordered by the size of the change."""
    base = {"small_rise": 1000, "large_rise": 1000, "small_drop": 1000, "large_drop": 1000}
    head = {"small_rise": 1003, "large_rise": 1050, "small_drop": 998, "large_drop": 900}

    table, _changed = instruction_compare.comparison_table(base, head, 0.005)
    slower, faster = table.split("**Fewer instructions (faster): 2**")

    assert slower.startswith("**More instructions (slower): 2**")
    assert _scenario_names(slower.splitlines()) == ["large_rise", "small_rise"]
    assert _scenario_names(faster.splitlines()) == ["large_drop", "small_drop"]
    assert "Measured on one side only" not in table


def test_every_table_row_starts_its_columns_at_the_same_character() -> None:
    """All tables share their column widths, so every row of every table lines up."""
    base = {"rise": 1000, "long_scenario_name/size": 1000, "gone": 7}
    head = {"rise": 1010, "long_scenario_name/size": 900, "brand_new": 3}

    table, changed = instruction_compare.comparison_table(base, head, 0.005)
    rows = [line for line in table.splitlines() if line.startswith("|")]

    assert len({len(line) for line in rows}) == 1
    assert len({line.index("|", 2) for line in rows}) == 1
    assert "**Measured on one side only: 2**" in table
    assert sorted(changed) == ["gone", "long_scenario_name/size", "rise"]


def test_the_failure_message_is_one_line_without_the_scenario_names() -> None:
    """The names are in the tables, marked **fails**; the message after them only counts and says what to do."""
    message = instruction_compare.failure_message(3)

    assert message == (
        "3 count(s) rose by more than 1.5%, fell by more than 10% or are no longer measured, "
        "marked **fails** in the tables above; review them and label the pull request accept-instruction-change"
    )


def test_only_a_rise_beyond_one_and_a_half_percent_or_a_drop_beyond_ten_fails() -> None:
    """Layout drift up to about 1.1% only warns; added work, skipped work and a lost scenario fail."""
    base = {"drift": 1000, "rise": 1000, "edge_rise": 1000, "drop": 1000, "large_drop": 1000, "gone": 1000}
    head = {"drift": 1011, "rise": 1016, "edge_rise": 1015, "drop": 910, "large_drop": 899}

    assert instruction_compare.failing_names(base, head) == ["rise", "large_drop", "gone"]


def test_the_table_marks_failing_rows_apart_from_changed_ones() -> None:
    """A row beyond the threshold but within the bounds is **changed**, one beyond the bounds **fails**."""
    table, changed = instruction_compare.comparison_table(
        {"drift": 1000, "rise": 1000}, {"drift": 1011, "rise": 1020}, 0.005
    )
    rows = {line.split("|")[1].strip(): line for line in table.splitlines() if line.startswith("| ")}

    assert rows["drift"].rstrip(" |").endswith("**changed**")
    assert rows["rise"].rstrip(" |").endswith("**fails**")
    assert changed == ["drift", "rise"]


def test_a_warning_inside_the_bounds_is_a_github_annotation() -> None:
    """CI shows it on the run without failing it."""
    message = instruction_compare.warning_message(2, 0.005)

    assert message == (
        "::warning title=Instruction counts::2 count(s) moved by more than 0.5%, none beyond +1.5% or -10%; "
        "see the tables in the job summary"
    )
