"""Correctness tests for the C union-find connected components."""

import pytest

from connected_component import (
    igp_connected_components,
    igp_connected_components_remapped,
    igp_connected_components_with_branch_ids,
    igp_connected_components_with_branch_ids_remapped,
)


@pytest.mark.parametrize(
    ("num_nodes", "edges", "expected"),
    [
        pytest.param(0, [], [], id="no_nodes"),
        pytest.param(1, [], [{0}], id="single_node_no_edges"),
        pytest.param(4, [], [{0}, {1}, {2}, {3}], id="all_nodes_isolated"),
        pytest.param(
            4, [(0, 1), (1, 2), (2, 3)], [{0, 1, 2, 3}], id="single_connected_component",
        ),
        pytest.param(3, [(0, 1), (1, 2), (2, 0)], [{0, 1, 2}], id="cycle_graph"),
        pytest.param(
            5, [(0, 1), (0, 2), (0, 3), (0, 4)], [{0, 1, 2, 3, 4}], id="star_topology",
        ),
        pytest.param(
            4, [(0, 1), (2, 3)], [{0, 1}, {2, 3}], id="two_disconnected_components",
        ),
        pytest.param(4, [(0, 1), (0, 1), (2, 3)], [{0, 1}, {2, 3}], id="parallel_edges"),
        pytest.param(3, [(0, 0), (1, 2)], [{0}, {1, 2}], id="self_loop"),
        pytest.param(
            5, [(1, 3)], [{0}, {1, 3}, {2}, {4}], id="mixed_isolated_and_connected",
        ),
        pytest.param(
            5,
            [(i, j) for i in range(5) for j in range(i + 1, 5)],
            [{0, 1, 2, 3, 4}],
            id="complete_graph",
        ),
    ],
)
def test_connected_components(
    num_nodes: int,
    edges: list[tuple[int, int]],
    expected: list[set[int]],
) -> None:
    result = list(igp_connected_components(num_nodes, edges))
    assert len(result) == len(expected)
    for comp in expected:
        assert comp in result


@pytest.mark.parametrize(
    ("num_nodes", "edges", "branch_ids", "expected"),
    [
        pytest.param(0, [], [], [], id="no_nodes"),
        pytest.param(1, [], [], [({0}, set())], id="single_node_no_edges"),
        pytest.param(
            4,
            [],
            [],
            [({0}, set()), ({1}, set()), ({2}, set()), ({3}, set())],
            id="all_nodes_isolated",
        ),
        pytest.param(
            4,
            [(0, 1), (1, 2), (2, 3)],
            [10, 11, 12],
            [({0, 1, 2, 3}, {10, 11, 12})],
            id="single_connected_component",
        ),
        pytest.param(
            3,
            [(0, 1), (1, 2), (2, 0)],
            [20, 21, 22],
            [({0, 1, 2}, {20, 21, 22})],
            id="cycle_graph",
        ),
        pytest.param(
            5,
            [(0, 1), (0, 2), (0, 3), (0, 4)],
            [30, 31, 32, 33],
            [({0, 1, 2, 3, 4}, {30, 31, 32, 33})],
            id="star_topology",
        ),
        pytest.param(
            4,
            [(0, 1), (2, 3)],
            [40, 41],
            [({0, 1}, {40}), ({2, 3}, {41})],
            id="two_disconnected_components",
        ),
        pytest.param(
            4,
            [(0, 1), (0, 1), (2, 3)],
            [50, 51, 52],
            [({0, 1}, {50, 51}), ({2, 3}, {52})],
            id="parallel_edges",
        ),
        pytest.param(
            3,
            [(0, 0), (1, 2)],
            [60, 61],
            [({0}, {60}), ({1, 2}, {61})],
            id="self_loop",
        ),
        pytest.param(
            5,
            [(1, 3)],
            [70],
            [({0}, set()), ({1, 3}, {70}), ({2}, set()), ({4}, set())],
            id="mixed_isolated_and_connected",
        ),
        pytest.param(
            5,
            [(i, j) for i in range(5) for j in range(i + 1, 5)],
            list(range(100, 110)),
            [({0, 1, 2, 3, 4}, set(range(100, 110)))],
            id="complete_graph",
        ),
    ],
)
def test_connected_components_with_branch_ids(
    num_nodes: int,
    edges: list[tuple[int, int]],
    branch_ids: list[int],
    expected: list[tuple[set[int], set[int]]],
) -> None:
    result = list(igp_connected_components_with_branch_ids(num_nodes, edges, branch_ids))
    assert len(result) == len(expected)
    for exp_nodes, exp_branches in expected:
        assert (exp_nodes, exp_branches) in result


def test_node_coverage():
    """Every node must appear in exactly one component."""
    n = 1000
    edges = [(i, i + 1) for i in range(0, n - 1, 2)]
    components = list(igp_connected_components(n, edges))
    all_nodes: set[int] = set()
    for c in components:
        assert not all_nodes & c, "overlap"
        all_nodes |= c
    assert all_nodes == set(range(n))


# ── Remapped variants (node_ids maps index -> original ID) ──


def test_remapped_basic():
    """Simulates the real use case: non-contiguous node IDs mapped to indices."""
    # Original node IDs: 100, 200, 300, 400
    node_ids = [100, 200, 300, 400]
    # Edges use indices: 0-1 connected, 2-3 connected
    edges = [(0, 1), (2, 3)]
    result = list(igp_connected_components_remapped(node_ids, edges))
    assert len(result) == 2
    assert {100, 200} in result
    assert {300, 400} in result


def test_remapped_no_edges():
    node_ids = [10, 20, 30]
    result = list(igp_connected_components_remapped(node_ids, []))
    assert len(result) == 3
    assert {10} in result
    assert {20} in result
    assert {30} in result


def test_remapped_single_component():
    node_ids = [5, 10, 15]
    edges = [(0, 1), (1, 2)]
    result = list(igp_connected_components_remapped(node_ids, edges))
    assert len(result) == 1
    assert result[0] == {5, 10, 15}


def test_remapped_empty():
    result = list(igp_connected_components_remapped([], []))
    assert result == []


def test_remapped_with_branches_basic():
    node_ids = [100, 200, 300, 400]
    edges = [(0, 1), (2, 3)]
    branch_ids = [901, 902]
    result = list(igp_connected_components_with_branch_ids_remapped(node_ids, edges, branch_ids))
    assert len(result) == 2
    result_map = {frozenset(nodes): branches for nodes, branches in result}
    assert result_map[frozenset({100, 200})] == {901}
    assert result_map[frozenset({300, 400})] == {902}


def test_remapped_with_branches_merged():
    node_ids = [10, 20, 30]
    edges = [(0, 1), (1, 2)]
    branch_ids = [500, 600]
    result = list(igp_connected_components_with_branch_ids_remapped(node_ids, edges, branch_ids))
    assert len(result) == 1
    nodes, branches = result[0]
    assert nodes == {10, 20, 30}
    assert branches == {500, 600}


def test_remapped_with_branches_isolated():
    node_ids = [7, 8, 9]
    result = list(igp_connected_components_with_branch_ids_remapped(node_ids, [], []))
    assert len(result) == 3
    for nodes, branches in result:
        assert len(nodes) == 1
        assert branches == set()
