"""The report half of dev_helpers/compare_profiles.py: the tables; reading dumps is test_instruction_buckets.py."""

import importlib.util
import sys
from pathlib import Path

_PATH = Path(__file__).resolve().parents[2] / "dev_helpers" / "compare_profiles.py"
_SPEC = importlib.util.spec_from_file_location("compare_profiles", _PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
compare_profiles = importlib.util.module_from_spec(_SPEC)
sys.modules["compare_profiles"] = compare_profiles
_SPEC.loader.exec_module(compare_profiles)


def test_totals_table_marks_changes_beyond_the_threshold_in_either_direction() -> None:
    """Largest change first; new and missing scenarios stay in the table without a change."""
    base = {"steady": 1000, "faster": 1000, "missing": 5}
    head = {"steady": 1004, "faster": 900, "new": 7}

    rows, changed = compare_profiles.totals_table(base, head, lambda _name: 0.005)

    assert changed == ["faster"]
    assert rows[2:] == [
        "| new | not measured | 7 | |",
        "| missing | 5 | not measured | |",
        "| faster | 1,000 | 900 | -10.00% **changed** |",
        "| steady | 1,000 | 1,004 | +0.40% |",
    ]


def test_difference_table_lists_the_largest_moves_and_stops_at_unchanged_entries() -> None:
    """Entries present on one side only count from zero; equal entries are left out."""
    rows = compare_profiles.difference_table(
        {"old_entry": 50, "same": 10, "grew": 5}, {"new_entry": 60, "same": 10, "grew": 6}, "function"
    )

    assert rows[2:] == [
        "| `new_entry` | 0 | 60 | +60 |",
        "| `old_entry` | 50 | 0 | -50 |",
        "| `grew` | 5 | 6 | +1 |",
    ]


def test_relative_change_from_zero_is_a_full_change() -> None:
    assert compare_profiles.relative_change(0, 0) == 0.0
    assert compare_profiles.relative_change(0, 3) == 1.0
    assert compare_profiles.relative_change(200, 150) == -0.25
