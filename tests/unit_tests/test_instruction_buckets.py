"""Reading callgrind dumps into the three instruction buckets the instruction tier compares."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from performance_tests.instructions.callgrind import (
    bucket_costs,
    bucket_rows,
    bucket_threshold,
    instruction_bucket,
    require_debug_info,
    self_costs,
    short_source_name,
)

_CORE = "/build/src/pygraphc/_core_src"

_DUMP = f"""# callgrind format
version: 1
positions: line
events: Ir
summary: 167

ob=(6) /usr/lib/libpython.so
fl=(1) ???
fn=(7) PyLong_FromLong'2
0 7
cfn=(8) caller'2
calls=0 0
0 53479350093

ob=(9) /venv/_core.so
fl=(2) {_CORE}/entry_points/biconnectivity.c
fn=(10) tarjan_entry
12 100
fi=(3) {_CORE}/algorithms/terminals.h
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
    by_function, _by_file = self_costs(_DUMP)

    assert by_function == {
        "???:PyLong_FromLong": 7,
        "entry_points/biconnectivity.c:tarjan_entry": 140,
        "entry_points/biconnectivity.c:report_block": 20,
    }
    assert sum(by_function.values()) == 167


def test_self_costs_follow_inlined_files_by_full_path() -> None:
    """fi= and fe= move the cost of inlined lines to their own file; fn= returns to the function's file."""
    _by_function, by_file = self_costs(_DUMP)

    assert by_file == {"???": 7, f"{_CORE}/entry_points/biconnectivity.c": 130, f"{_CORE}/algorithms/terminals.h": 30}


def test_inlined_algorithm_lines_count_as_algorithm_inside_a_boundary_function() -> None:
    _by_function, by_file = self_costs(_DUMP)

    assert bucket_costs(by_file) == {"algorithm": 30, "boundary": 130, "cpython": 7}


@pytest.mark.parametrize(
    ("source_file", "bucket"),
    [
        pytest.param(f"{_CORE}/algorithms/components.c", "algorithm", id="algorithm-source"),
        pytest.param(f"{_CORE}/algorithms/union_find.h", "algorithm", id="algorithm-header"),
        pytest.param(f"{_CORE}/entry_points/connectivity.c", "boundary", id="entry-point"),
        pytest.param(f"{_CORE}/conversion/parse.c", "boundary", id="conversion"),
        pytest.param(f"{_CORE}/models.h", "boundary", id="models"),
        pytest.param("/build/src/pygraphc/_dag_learn.c", "boundary", id="other-extension"),
        pytest.param("/usr/local/include/python3.13/object.h", "cpython", id="python-header"),
        pytest.param("./malloc/malloc.c", "cpython", id="libc"),
        pytest.param("???", "cpython", id="no-debug-info"),
    ],
)
def test_instruction_bucket_by_source_path(source_file: str, bucket: str) -> None:
    assert instruction_bucket(source_file) == bucket


def test_short_source_name_keeps_the_directory_inside_the_core() -> None:
    """Two traversal.c files exist, one in algorithms/, one in entry_points/; reports must tell them apart."""
    assert short_source_name(f"{_CORE}/algorithms/traversal.c") == "algorithms/traversal.c"
    assert short_source_name(f"{_CORE}/entry_points/traversal.c") == "entry_points/traversal.c"
    assert short_source_name("/usr/src/glibc/malloc/malloc.c") == "malloc.c"


def test_bucket_rows_name_every_bucket_of_every_scenario() -> None:
    rows = bucket_rows({"bfs/sparse": {"algorithm": 3, "boundary": 2, "cpython": 1}})

    assert rows == {"bfs/sparse [algorithm]": 3, "bfs/sparse [boundary]": 2, "bfs/sparse [cpython]": 1}


def test_the_algorithm_bucket_has_the_tighter_threshold() -> None:
    assert bucket_threshold("bfs/sparse [algorithm]") == 0.001
    assert bucket_threshold("bfs/sparse [boundary]") == 0.005
    assert bucket_threshold("bfs/sparse [cpython]") == 0.005


def test_a_run_without_pygraphc_instructions_needs_debug_info() -> None:
    """Without -g every pygraphc instruction lands in cpython, and the buckets would compare nothing."""
    with pytest.raises(RuntimeError, match="CFLAGS=-g"):
        require_debug_info({"bfs": {"algorithm": 0, "boundary": 0, "cpython": 10}})


def test_one_scenario_with_pygraphc_instructions_is_enough() -> None:
    """A pure-Python scenario has no pygraphc instruction of its own; the run as a whole does."""
    require_debug_info({
        "reduce_python": {"algorithm": 0, "boundary": 0, "cpython": 10},
        "bfs": {"algorithm": 5, "boundary": 1, "cpython": 10},
    })
