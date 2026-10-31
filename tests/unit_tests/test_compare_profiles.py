"""The report half of dev_helpers/compare_profiles.py: reading callgrind dumps and building the tables."""

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

_DUMP = """# callgrind format
version: 1
positions: line
events: Ir
summary: 160

ob=(6) /usr/lib/libpython.so
fl=(1) ???
fn=(7) PyLong_FromLong'2
cfn=(8) caller'2
calls=0 0
0 53479350093

ob=(9) /venv/_core.so
fl=(2) /build/src/biconnectivity.c
fn=(10) tarjan_entry
12 100
fi=(3) /build/src/terminals.h
40 30
fe=(2)
+1 10
cfn=(7)
calls=1 0
13 999
fn=(11) report_block
* 20
"""


def test_self_costs_skip_the_inclusive_cost_after_calls() -> None:
    """The cost line after calls= is the callee's inclusive cost, not the caller's own."""
    by_function, by_file = compare_profiles.self_costs(_DUMP)

    assert by_function == {"biconnectivity.c:tarjan_entry": 140, "biconnectivity.c:report_block": 20}
    assert sum(by_function.values()) == 160


def test_self_costs_follow_inlined_files() -> None:
    """fi= and fe= move the cost of inlined lines to their own file; fn= returns to the function's file."""
    _by_function, by_file = compare_profiles.self_costs(_DUMP)

    assert by_file == {"biconnectivity.c": 130, "terminals.h": 30}


def test_totals_table_marks_changes_beyond_the_threshold_in_either_direction() -> None:
    """Largest change first; new and missing scenarios stay in the table without a change."""
    base = {"steady": 1000, "faster": 1000, "missing": 5}
    head = {"steady": 1004, "faster": 900, "new": 7}

    rows, changed = compare_profiles.totals_table(base, head, 0.005)

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
