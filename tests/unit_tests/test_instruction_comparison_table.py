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


def test_rows_are_sorted_by_the_size_of_the_change_and_aligned() -> None:
    """Largest change first in either direction; every column starts at the same character."""
    base = {"small_rise": 1000, "large_drop": 1000, "long_scenario_name/size": 1000}
    head = {"small_rise": 1003, "large_drop": 900, "long_scenario_name/size": 1050}

    table, _changed = instruction_compare.comparison_table(base, head, 0.005)
    lines = table.splitlines()[:5]

    assert [line.split("|")[1].strip() for line in lines[2:]] == ["large_drop", "long_scenario_name/size", "small_rise"]
    assert len({len(line) for line in lines}) == 1
    assert len({line.index("|", 2) for line in lines}) == 1
