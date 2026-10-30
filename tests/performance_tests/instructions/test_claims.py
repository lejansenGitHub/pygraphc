"""Design claims of the form "A costs at most k times B", asserted on instruction counts.

The limits are the claims the walltime suite asserted before, now checked
deterministically: one callgrind run per session measures every pair, so a
claim fails when the design stops holding, never because a runner was busy.
Each docstring gives the ratio measured when the claim moved here.
"""

import pytest

from performance_tests.instructions.callgrind import count_instructions, valgrind_available

pytestmark = [
    pytest.mark.performance,
    pytest.mark.skipif(not valgrind_available(), reason="instruction counts need valgrind"),
]

CLAIMS: list[tuple[str, str, float]] = [
    ("connected_components_masked/sparse_random_20k", "connected_components/sparse_random_20k", 1.15),
    ("bridges_masked/sparse_random_20k", "bridges/sparse_random_20k", 1.15),
    ("articulation_points_masked/sparse_random_20k", "articulation_points/sparse_random_20k", 1.15),
    ("connected_components_node_masked/sparse_random_20k", "connected_components/sparse_random_20k", 1.6),
    ("bridges_node_masked/sparse_random_20k", "bridges/sparse_random_20k", 1.6),
    ("articulation_points_node_masked/sparse_random_20k", "articulation_points/sparse_random_20k", 1.6),
    (
        "connected_components_without_one_edge_by_mask/sparse_random_20k",
        "connected_components_without_one_edge_by_rebuild/sparse_random_20k",
        1 / 1.5,
    ),
    (
        "bridges_without_one_edge_by_mask/sparse_random_20k",
        "bridges_without_one_edge_by_rebuild/sparse_random_20k",
        1 / 1.5,
    ),
    (
        "connected_components_without_ten_nodes_by_mask/sparse_random_20k",
        "connected_components_without_ten_nodes_by_rebuild/sparse_random_20k",
        1 / 1.5,
    ),
    ("connected_components_with_branch_ids/sparse_random_20k", "connected_components/sparse_random_20k", 3.5),
    ("parse_directed/sparse_random_20k", "parse/sparse_random_20k", 1.5),
    ("reduce_c/meshed_feeders_5k", "reduce_python/meshed_feeders_5k", 0.5),
    ("shortest_path_float_buffer/sparse_random_20k", "shortest_path/sparse_random_20k", 1.0),
    ("workflow_components_with_branch_ids/branches_20k", "workflow_components/branches_20k", 1.5),
]


@pytest.fixture(scope="module")
def instruction_counts() -> dict[str, int]:
    """Every scenario of every claim, measured in one callgrind run."""
    names = sorted({name for claimed, reference, _limit in CLAIMS for name in (claimed, reference)})
    return count_instructions(names)


def _ratio(instruction_counts: dict[str, int], claimed: str, reference: str) -> float:
    return instruction_counts[claimed] / instruction_counts[reference]


def test_an_empty_edge_mask_costs_at_most_fifteen_percent(instruction_counts: dict[str, int]) -> None:
    """A view with nothing excluded only adds a mask lookup per edge (measured 1.01 to 1.05)."""
    for claimed, reference, limit in CLAIMS[:3]:
        assert _ratio(instruction_counts, claimed, reference) <= limit, claimed


def test_an_empty_node_mask_costs_at_most_sixty_percent(instruction_counts: dict[str, int]) -> None:
    """A node mask adds a lookup per visited neighbour (measured 1.05 to 1.11)."""
    for claimed, reference, limit in CLAIMS[3:6]:
        assert _ratio(instruction_counts, claimed, reference) <= limit, claimed


def test_masking_beats_rebuilding_by_half(instruction_counts: dict[str, int]) -> None:
    """Rebuilding filters the edge list and parses it again; masking does neither (measured 0.08 to 0.44)."""
    for claimed, reference, limit in CLAIMS[6:9]:
        assert _ratio(instruction_counts, claimed, reference) <= limit, claimed


def test_branch_ids_cost_at_most_three_and_a_half_times_plain_components(instruction_counts: dict[str, int]) -> None:
    """Collecting one branch set per component is the extra work (measured 2.01)."""
    claimed, reference, limit = CLAIMS[9]
    assert _ratio(instruction_counts, claimed, reference) <= limit


def test_directed_parsing_costs_at_most_half_again(instruction_counts: dict[str, int]) -> None:
    """The reverse adjacency is the only extra structure (measured 1.04)."""
    claimed, reference, limit = CLAIMS[10]
    assert _ratio(instruction_counts, claimed, reference) <= limit


def test_the_c_engine_needs_at_most_half_of_the_python_engine(instruction_counts: dict[str, int]) -> None:
    """Both build the same provenance trees; only the structural loop differs (measured 0.30)."""
    claimed, reference, limit = CLAIMS[11]
    assert _ratio(instruction_counts, claimed, reference) <= limit


def test_a_float_buffer_is_cheaper_than_a_list_of_weights(instruction_counts: dict[str, int]) -> None:
    """A list is converted element by element before the search, a buffer is read in place (measured 0.06)."""
    claimed, reference, limit = CLAIMS[12]
    assert _ratio(instruction_counts, claimed, reference) < limit


def test_branch_ids_add_at_most_half_to_the_whole_workflow(instruction_counts: dict[str, int]) -> None:
    """Gathering, parsing and labelling from domain objects dwarfs the branch sets (measured 1.28)."""
    claimed, reference, limit = CLAIMS[13]
    assert _ratio(instruction_counts, claimed, reference) <= limit
