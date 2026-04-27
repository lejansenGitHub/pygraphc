"""Tests for dag_longest_path algorithm."""

import pytest

import networkc
from networkc import Graph

# ── Correctness (unweighted) ──


@pytest.mark.parametrize(
    ("node_ids", "edges", "expected_length"),
    [
        pytest.param([1], [], 1, id="single_node"),
        pytest.param([1, 2, 3], [(1, 2), (2, 3)], 3, id="chain_3"),
        pytest.param([1, 2, 3, 4, 5], [(1, 2), (2, 3), (3, 4), (4, 5)], 5, id="chain_5"),
        pytest.param([1, 2, 3], [(1, 2), (1, 3)], 2, id="fan_out"),
        pytest.param([1, 2, 3], [(2, 1), (3, 1)], 2, id="fan_in"),
    ],
)
def test_dag_longest_path_length(node_ids, edges, expected_length):
    graph = Graph(node_ids, edges, directed=True)
    result = graph.dag_longest_path()
    assert len(result) == expected_length


def test_dag_longest_path_diamond():
    """Diamond DAG: 1->2, 1->3, 2->4, 3->4. Longest = 3 nodes."""
    graph = Graph([1, 2, 3, 4], [(1, 2), (1, 3), (2, 4), (3, 4)], directed=True)
    result = graph.dag_longest_path()
    assert len(result) == 3
    assert result[0] == 1
    assert result[-1] == 4


def test_dag_longest_path_is_valid_path():
    """Verify the returned path is a valid directed path."""
    node_ids = [1, 2, 3, 4, 5]
    edges = [(1, 2), (2, 3), (1, 4), (4, 5), (5, 3)]
    graph = Graph(node_ids, edges, directed=True)
    result = graph.dag_longest_path()
    # Check consecutive nodes are connected by an edge
    edge_set = set(edges)
    for i in range(len(result) - 1):
        assert (result[i], result[i + 1]) in edge_set


def test_dag_longest_path_disconnected():
    """Disconnected DAG: longest path is within one component."""
    graph = Graph([1, 2, 3, 4, 5, 6], [(1, 2), (2, 3), (4, 5)], directed=True)
    result = graph.dag_longest_path()
    assert len(result) == 3  # 1->2->3 is longer than 4->5


def test_dag_longest_path_no_edges():
    """Graph with no edges: longest path is a single node."""
    graph = Graph([1, 2, 3], [], directed=True)
    result = graph.dag_longest_path()
    assert len(result) == 1


def test_dag_longest_path_empty_graph():
    graph = Graph([], [], directed=True)
    result = graph.dag_longest_path()
    assert result == []


# ── Weighted ──


def test_dag_longest_path_weighted():
    # 1 --(1)--> 2 --(1)--> 3
    # 1 --(10)--> 3
    graph = Graph([1, 2, 3], [(1, 2), (2, 3), (1, 3)], directed=True)
    result_unweighted = graph.dag_longest_path()
    assert len(result_unweighted) == 3  # 1->2->3 (2 hops)

    result_weighted = graph.dag_longest_path(weights=[1.0, 1.0, 10.0])
    assert result_weighted == [1, 3]  # weight 10 > weight 1+1


def test_dag_longest_path_weighted_chain():
    # Chain with varying weights
    graph = Graph([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4)], directed=True)
    result = graph.dag_longest_path(weights=[5.0, 1.0, 1.0])
    assert result == [1, 2, 3, 4]  # only one path


def test_dag_longest_path_weighted_picks_heavier_branch():
    # 1->2 (w=1), 1->3 (w=100)
    graph = Graph([1, 2, 3], [(1, 2), (1, 3)], directed=True)
    result = graph.dag_longest_path(weights=[1.0, 100.0])
    assert result == [1, 3]


# ── Error Cases ──


def test_dag_longest_path_cycle_raises():
    graph = Graph([1, 2, 3], [(1, 2), (2, 3), (3, 1)], directed=True)
    with pytest.raises(ValueError, match="cycle"):
        graph.dag_longest_path()


def test_dag_longest_path_undirected_raises():
    graph = Graph([1, 2, 3], [(1, 2), (2, 3)])
    with pytest.raises(TypeError, match="requires a directed"):
        graph.dag_longest_path()


# ── GraphView ──


def test_dag_longest_path_view_with_edge_mask():
    """Remove an edge to shorten the longest path."""
    graph = Graph([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4)], directed=True)
    view = graph.without_edges([2])  # remove 3->4
    result = view.dag_longest_path()
    assert len(result) == 3  # 1->2->3 (was 4)


def test_dag_longest_path_view_with_node_mask():
    """Remove a node to split the path."""
    graph = Graph([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4)], directed=True)
    view = graph.without_nodes([2])
    result = view.dag_longest_path()
    assert len(result) == 2  # 3->4 is the longest remaining path


def test_dag_longest_path_view_break_cycle():
    """Remove edge that creates cycle => longest path works."""
    graph = Graph([1, 2, 3], [(1, 2), (2, 3), (3, 1)], directed=True)
    view = graph.without_edges([2])  # remove 3->1
    result = view.dag_longest_path()
    assert len(result) == 3  # 1->2->3


def test_dag_longest_path_view_weighted():
    graph = Graph([1, 2, 3], [(1, 2), (1, 3)], directed=True)
    view = graph.without_edges([])
    result = view.dag_longest_path(weights=[1.0, 100.0])
    assert result == [1, 3]


# ── Free Function ──


def test_dag_longest_path_free_function():
    result = networkc.dag_longest_path([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4)])
    assert len(result) == 4


def test_dag_longest_path_free_function_weighted():
    result = networkc.dag_longest_path(
        [1, 2, 3],
        [(1, 2), (1, 3)],
        weights=[1.0, 100.0],
    )
    assert result == [1, 3]


def test_dag_longest_path_free_function_cycle_raises():
    with pytest.raises(ValueError, match="cycle"):
        networkc.dag_longest_path([1, 2, 3], [(1, 2), (2, 3), (3, 1)])
