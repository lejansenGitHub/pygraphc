"""Tests for Phase 6a: MultiGraph support (parallel edges)."""

import pytest

from pygraphc import Graph


@pytest.fixture
def simple_multigraph():
    """Graph with parallel edges between 2-3: edges 1 and 2."""
    nodes = [1, 2, 3, 4]
    edges = [(1, 2), (2, 3), (2, 3), (3, 4)]
    return Graph(nodes, edges)


@pytest.fixture
def non_multigraph():
    """Simple graph with no parallel edges."""
    return Graph([1, 2, 3], [(1, 2), (2, 3), (1, 3)])


# ── is_multigraph property ──


def test_is_multigraph_true(simple_multigraph):
    assert simple_multigraph.is_multigraph is True


def test_is_multigraph_false(non_multigraph):
    assert non_multigraph.is_multigraph is False


def test_is_multigraph_empty():
    g = Graph([], [])
    assert g.is_multigraph is False


def test_is_multigraph_no_edges():
    g = Graph([1, 2], [])
    assert g.is_multigraph is False


# ── edge_indices with parallel edges ──


def test_edge_indices_returns_all_parallel(simple_multigraph):
    indices = simple_multigraph.edge_indices(2, 3)
    assert sorted(indices) == [1, 2]


def test_edge_indices_reverse_order(simple_multigraph):
    indices = simple_multigraph.edge_indices(3, 2)
    assert sorted(indices) == [1, 2]


def test_edge_indices_single(simple_multigraph):
    assert simple_multigraph.edge_indices(1, 2) == [0]


def test_edge_indices_nonexistent(simple_multigraph):
    assert simple_multigraph.edge_indices(1, 4) == []


# ── Connected components with parallel edges ──


def test_cc_parallel_edges(simple_multigraph):
    comps = list(simple_multigraph.connected_components())
    assert comps == [{1, 2, 3, 4}]


# ── Bridges: parallel edges prevent bridge ──


def test_bridges_parallel_not_bridge(simple_multigraph):
    """Edge between 2-3 is NOT a bridge because there are parallel edges."""
    bridge_list = simple_multigraph.bridges()
    bridge_set = {(min(u, v), max(u, v)) for u, v in bridge_list}
    # 1-2 and 3-4 are bridges, but 2-3 is not (parallel edges)
    assert (2, 3) not in bridge_set
    assert (1, 2) in bridge_set
    assert (3, 4) in bridge_set


# ── BFS visits each node once ──


def test_bfs_visits_once(simple_multigraph):
    visited = simple_multigraph.bfs(1)
    assert len(visited) == 4
    assert len(set(visited)) == 4  # no duplicates


# ── Dijkstra picks lighter parallel edge ──


def test_dijkstra_picks_lighter_parallel():
    g = Graph([1, 2, 3], [(1, 2), (1, 2), (2, 3)])
    weights = [10.0, 1.0, 1.0]
    path = g.shortest_path(weights, 1, 3)
    assert path == [1, 2, 3]
    lengths = g.shortest_path_lengths(weights, 1)
    assert lengths[2] == 1.0  # picks the lighter edge
    assert lengths[3] == 2.0


# ── Edge count includes parallel edges ──


def test_edge_count_includes_parallel(simple_multigraph):
    assert simple_multigraph.edge_count == 4


# ── Masking one parallel edge leaves others ──


def test_mask_one_parallel_leaves_other(simple_multigraph):
    """Masking edge index 1 (one of the 2-3 edges) still leaves edge 2."""
    view = simple_multigraph.without_edges([1])
    comps = list(view.connected_components())
    assert comps == [{1, 2, 3, 4}]  # still connected


def test_mask_both_parallel_disconnects(simple_multigraph):
    """Masking both parallel edges between 2-3 disconnects the graph."""
    view = simple_multigraph.without_edges([1, 2])
    comps = sorted(view.connected_components(), key=min)
    assert comps == [{1, 2}, {3, 4}]


# ── Articulation points ──


def test_ap_with_parallel(simple_multigraph):
    aps = simple_multigraph.articulation_points()
    # Node 2 connects {1} to {3,4} via bridge 1-2, so AP
    # Node 3 connects {1,2} to {4} via bridge 3-4, so AP
    # But 2-3 are parallel so removing either doesn't disconnect
    assert 2 in aps
    assert 3 in aps
