"""Correctness tests for the C union-find connected components."""

import pytest

from cgraph import connected_components, connected_components_with_branch_ids


@pytest.mark.parametrize(
    ("node_ids", "edges", "expected"),
    [
        pytest.param([], [], [], id="no_nodes"),
        pytest.param([0], [], [{0}], id="single_node_no_edges"),
        pytest.param([0, 1, 2, 3], [], [{0}, {1}, {2}, {3}], id="all_nodes_isolated"),
        pytest.param(
            [0, 1, 2, 3],
            [(0, 1), (1, 2), (2, 3)],
            [{0, 1, 2, 3}],
            id="single_connected_component",
        ),
        pytest.param(
            [0, 1, 2],
            [(0, 1), (1, 2), (2, 0)],
            [{0, 1, 2}],
            id="cycle_graph",
        ),
        pytest.param(
            [0, 1, 2, 3, 4],
            [(0, 1), (0, 2), (0, 3), (0, 4)],
            [{0, 1, 2, 3, 4}],
            id="star_topology",
        ),
        pytest.param(
            [0, 1, 2, 3],
            [(0, 1), (2, 3)],
            [{0, 1}, {2, 3}],
            id="two_disconnected_components",
        ),
        pytest.param(
            [0, 1, 2, 3],
            [(0, 1), (0, 1), (2, 3)],
            [{0, 1}, {2, 3}],
            id="parallel_edges",
        ),
        pytest.param(
            [0, 1, 2],
            [(0, 0), (1, 2)],
            [{0}, {1, 2}],
            id="self_loop",
        ),
        pytest.param(
            [0, 1, 2, 3, 4],
            [(1, 3)],
            [{0}, {1, 3}, {2}, {4}],
            id="mixed_isolated_and_connected",
        ),
        pytest.param(
            [0, 1, 2, 3, 4],
            [(i, j) for i in range(5) for j in range(i + 1, 5)],
            [{0, 1, 2, 3, 4}],
            id="complete_graph",
        ),
    ],
)
def test_connected_components(
    node_ids: list[int],
    edges: list[tuple[int, int]],
    expected: list[set[int]],
) -> None:
    """
    Each parametrized case verifies a different graph topology.

    The expected components are unordered sets — we check that every expected
    component appears in the result and nothing extra is returned.
    """
    # --- Execute ---
    result = list(connected_components(node_ids, edges))

    # --- Assert ---
    assert len(result) == len(expected)
    for component in expected:
        assert component in result


@pytest.mark.parametrize(
    ("node_ids", "edges", "branch_ids", "expected"),
    [
        pytest.param([], [], [], [], id="no_nodes"),
        pytest.param([0], [], [], [({0}, set())], id="single_node_no_edges"),
        pytest.param(
            [0, 1, 2, 3],
            [],
            [],
            [({0}, set()), ({1}, set()), ({2}, set()), ({3}, set())],
            id="all_nodes_isolated",
        ),
        pytest.param(
            [0, 1, 2, 3],
            [(0, 1), (1, 2), (2, 3)],
            [10, 11, 12],
            [({0, 1, 2, 3}, {10, 11, 12})],
            id="single_connected_component",
        ),
        pytest.param(
            [0, 1, 2],
            [(0, 1), (1, 2), (2, 0)],
            [20, 21, 22],
            [({0, 1, 2}, {20, 21, 22})],
            id="cycle_graph",
        ),
        pytest.param(
            [0, 1, 2, 3, 4],
            [(0, 1), (0, 2), (0, 3), (0, 4)],
            [30, 31, 32, 33],
            [({0, 1, 2, 3, 4}, {30, 31, 32, 33})],
            id="star_topology",
        ),
        pytest.param(
            [0, 1, 2, 3],
            [(0, 1), (2, 3)],
            [40, 41],
            [({0, 1}, {40}), ({2, 3}, {41})],
            id="two_disconnected_components",
        ),
        pytest.param(
            [0, 1, 2, 3],
            [(0, 1), (0, 1), (2, 3)],
            [50, 51, 52],
            [({0, 1}, {50, 51}), ({2, 3}, {52})],
            id="parallel_edges",
        ),
        pytest.param(
            [0, 1, 2],
            [(0, 0), (1, 2)],
            [60, 61],
            [({0}, {60}), ({1, 2}, {61})],
            id="self_loop",
        ),
        pytest.param(
            [0, 1, 2, 3, 4],
            [(1, 3)],
            [70],
            [({0}, set()), ({1, 3}, {70}), ({2}, set()), ({4}, set())],
            id="mixed_isolated_and_connected",
        ),
        pytest.param(
            [0, 1, 2, 3, 4],
            [(i, j) for i in range(5) for j in range(i + 1, 5)],
            list(range(100, 110)),
            [({0, 1, 2, 3, 4}, set(range(100, 110)))],
            id="complete_graph",
        ),
    ],
)
def test_connected_components_with_branch_ids(
    node_ids: list[int],
    edges: list[tuple[int, int]],
    branch_ids: list[int],
    expected: list[tuple[set[int], set[int]]],
) -> None:
    """
    Verifies that branch IDs are correctly grouped alongside their nodes.

    Each edge has a corresponding branch_id — branches must appear in the same
    component as the nodes they connect.
    """
    # --- Execute ---
    result = list(connected_components_with_branch_ids(node_ids, edges, branch_ids))

    # --- Assert ---
    assert len(result) == len(expected)
    for expected_nodes, expected_branches in expected:
        assert (expected_nodes, expected_branches) in result


def test_node_coverage() -> None:
    """
    Every node must appear in exactly one component.

    Pairing 1000 nodes into 500 two-node components (edges 0-1, 2-3, ...)
    ensures the union of all components equals the full node set with no overlap.
    """
    # --- Input ---
    node_ids = list(range(1000))
    edges = [(i, i + 1) for i in range(0, 999, 2)]

    # --- Execute ---
    components = list(connected_components(node_ids, edges))

    # --- Assert ---
    all_nodes: set[int] = set()
    for component in components:
        assert not all_nodes & component, "overlap"
        all_nodes |= component
    assert all_nodes == set(range(1000))


# ── Remapped variants (non-contiguous node IDs) ──


def test_remapped_basic() -> None:
    """
    Non-contiguous node IDs (100..400) are returned directly in the component sets.

    Edges reference indices 0..3, but output uses the original node IDs.
    Two edges: 0-1 and 2-3 produce two components: {100, 200} and {300, 400}.
    """
    # --- Input ---
    node_ids = [100, 200, 300, 400]
    edges = [(0, 1), (2, 3)]

    # --- Execute ---
    result = list(connected_components(node_ids, edges))

    # --- Assert ---
    assert len(result) == 2
    assert {100, 200} in result
    assert {300, 400} in result


def test_remapped_no_edges() -> None:
    """
    Without edges, each node forms its own component.

    Three non-contiguous IDs produce three singleton components.
    """
    # --- Input ---
    node_ids = [10, 20, 30]

    # --- Execute ---
    result = list(connected_components(node_ids, []))

    # --- Assert ---
    assert len(result) == 3
    assert {10} in result
    assert {20} in result
    assert {30} in result


def test_remapped_single_component() -> None:
    """
    A chain 0->1->2 connects all three nodes into one component.

    Output uses original IDs {5, 10, 15}.
    """
    # --- Input ---
    node_ids = [5, 10, 15]
    edges = [(0, 1), (1, 2)]

    # --- Execute ---
    result = list(connected_components(node_ids, edges))

    # --- Assert ---
    assert len(result) == 1
    assert result[0] == {5, 10, 15}


def test_remapped_empty() -> None:
    """Empty input produces no components."""
    # --- Execute ---
    result = list(connected_components([], []))

    # --- Assert ---
    assert result == []


def test_remapped_with_branches_basic() -> None:
    """
    Branch IDs are grouped with their component's node IDs.

    Two edges create two components, each with its own branch.
    """
    # --- Input ---
    node_ids = [100, 200, 300, 400]
    edges = [(0, 1), (2, 3)]
    branch_ids = [901, 902]

    # --- Execute ---
    result = list(connected_components_with_branch_ids(node_ids, edges, branch_ids))

    # --- Assert ---
    assert len(result) == 2
    result_map = {frozenset(nodes): branches for nodes, branches in result}
    assert result_map[frozenset({100, 200})] == {901}
    assert result_map[frozenset({300, 400})] == {902}


def test_remapped_with_branches_merged() -> None:
    """
    A chain merges all nodes and their branches into one component.

    Edges 0->1 and 1->2 unify {10, 20, 30} with branches {500, 600}.
    """
    # --- Input ---
    node_ids = [10, 20, 30]
    edges = [(0, 1), (1, 2)]
    branch_ids = [500, 600]

    # --- Execute ---
    result = list(connected_components_with_branch_ids(node_ids, edges, branch_ids))

    # --- Assert ---
    assert len(result) == 1
    nodes, branches = result[0]
    assert nodes == {10, 20, 30}
    assert branches == {500, 600}


def test_remapped_with_branches_isolated() -> None:
    """
    Without edges, each node is isolated with an empty branch set.
    """
    # --- Input ---
    node_ids = [7, 8, 9]

    # --- Execute ---
    result = list(connected_components_with_branch_ids(node_ids, [], []))

    # --- Assert ---
    assert len(result) == 3
    for nodes, branches in result:
        assert len(nodes) == 1
        assert branches == set()
