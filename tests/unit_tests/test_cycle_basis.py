"""Tests for cycle_basis algorithm."""

import pytest

import pygraphc
from pygraphc import Graph


def _cycles_as_frozensets(cycles: list[list[int]]) -> set[frozenset[int]]:
    """Convert cycles to a set of frozensets for order-independent comparison."""
    return {frozenset(cycle) for cycle in cycles}


# ── Correctness ──


@pytest.mark.parametrize(
    ("node_ids", "edges", "expected_count", "expected_node_sets"),
    [
        pytest.param([], [], 0, set(), id="empty_graph"),
        pytest.param([1], [], 0, set(), id="single_node"),
        pytest.param([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4)], 0, set(), id="tree_no_cycles"),
        pytest.param(
            [1, 2, 3],
            [(1, 2), (2, 3), (3, 1)],
            1,
            {frozenset({1, 2, 3})},
            id="triangle",
        ),
        pytest.param(
            [1, 2, 3, 4],
            [(1, 2), (2, 3), (3, 4), (4, 1)],
            1,
            {frozenset({1, 2, 3, 4})},
            id="square_cycle",
        ),
        pytest.param(
            [1, 2, 3, 4, 5, 6],
            [(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4)],
            2,
            {frozenset({1, 2, 3}), frozenset({4, 5, 6})},
            id="two_independent_cycles",
        ),
        pytest.param(
            [1, 2, 3, 4, 5],
            [(1, 2), (2, 3), (3, 1), (3, 4), (4, 5), (5, 3)],
            2,
            {frozenset({1, 2, 3}), frozenset({3, 4, 5})},
            id="figure_eight",
        ),
        pytest.param(
            [1, 2, 3, 4, 5],
            [(1, 2), (2, 3), (3, 4), (4, 5)],
            0,
            set(),
            id="chain_no_cycles",
        ),
    ],
)
def test_cycle_basis_correctness(node_ids, edges, expected_count, expected_node_sets):
    graph = Graph(node_ids, edges)
    result = graph.cycle_basis()
    assert len(result) == expected_count
    assert _cycles_as_frozensets(result) == expected_node_sets


def test_cycle_basis_complete_graph_k4():
    """K4 has circuit rank = 6 - 4 + 1 = 3 fundamental cycles."""
    node_ids = [0, 1, 2, 3]
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    graph = Graph(node_ids, edges)
    result = graph.cycle_basis()
    assert len(result) == 3  # m - n + c = 6 - 4 + 1


def test_cycle_basis_disconnected_one_has_cycle():
    """Only the component with a cycle contributes."""
    node_ids = [1, 2, 3, 4, 5]
    edges = [(1, 2), (2, 3), (3, 1), (4, 5)]
    graph = Graph(node_ids, edges)
    result = graph.cycle_basis()
    assert len(result) == 1
    assert frozenset(result[0]) == {1, 2, 3}


def test_cycle_basis_self_loop():
    """Self-loop creates a cycle of length 1."""
    node_ids = [1, 2]
    edges = [(1, 1), (1, 2)]
    graph = Graph(node_ids, edges)
    result = graph.cycle_basis()
    assert len(result) == 1
    assert frozenset(result[0]) == {1}


def test_cycle_basis_large_single_cycle():
    n = 100
    node_ids = list(range(n))
    edges = [(i, (i + 1) % n) for i in range(n)]
    graph = Graph(node_ids, edges)
    result = graph.cycle_basis()
    assert len(result) == 1
    assert frozenset(result[0]) == set(range(n))


def test_cycle_basis_circuit_rank():
    """Verify circuit rank = m - n + c for various graphs."""
    # Grid-like graph: 4 nodes, 5 edges, 1 component -> 5-4+1=2
    node_ids = [1, 2, 3, 4]
    edges = [(1, 2), (2, 3), (3, 4), (4, 1), (1, 3)]
    graph = Graph(node_ids, edges)
    result = graph.cycle_basis()
    assert len(result) == 2  # 5 - 4 + 1 = 2


# ── Type Guards ──


def test_cycle_basis_directed_raises():
    graph = Graph([1, 2, 3], [(1, 2), (2, 3), (3, 1)], directed=True)
    with pytest.raises(TypeError, match="not defined for directed"):
        graph.cycle_basis()


# ── GraphView ──


def test_cycle_basis_view_with_edge_mask():
    """Remove cycle-closing edge => no cycles."""
    graph = Graph([1, 2, 3], [(1, 2), (2, 3), (3, 1)])
    view = graph.without_edges([2])  # remove edge 3--1
    assert view.cycle_basis() == []


def test_cycle_basis_view_with_node_mask():
    """Remove a node from a cycle => cycle broken."""
    graph = Graph([1, 2, 3], [(1, 2), (2, 3), (3, 1)])
    view = graph.without_nodes([2])
    assert view.cycle_basis() == []


def test_cycle_basis_view_preserves_other_cycle():
    """Remove one cycle's edge, other cycle remains."""
    graph = Graph(
        [1, 2, 3, 4, 5, 6],
        [(1, 2), (2, 3), (3, 1), (4, 5), (5, 6), (6, 4)],
    )
    view = graph.without_edges([2])  # remove edge 3--1
    result = view.cycle_basis()
    assert len(result) == 1
    assert frozenset(result[0]) == {4, 5, 6}


# ── Free Function ──


def test_cycle_basis_free_function():
    result = pygraphc.cycle_basis([1, 2, 3], [(1, 2), (2, 3), (3, 1)])
    assert len(result) == 1
    assert frozenset(result[0]) == {1, 2, 3}


def test_cycle_basis_free_function_no_cycles():
    result = pygraphc.cycle_basis([1, 2, 3], [(1, 2), (2, 3)])
    assert result == []
