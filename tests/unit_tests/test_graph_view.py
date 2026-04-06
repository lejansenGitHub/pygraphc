"""Tests for GraphView (edge-masked views) and for_each_edge_excluded."""

import pytest

from cgraph import Graph, GraphView, for_each_edge_excluded


# ── Fixtures ──


@pytest.fixture()
def bridge_graph():
    """Graph: (0-1-2-0) --bridge(2,3)-- (3-4-5-3).

    Edges by index:
        0: (0,1), 1: (1,2), 2: (2,0), 3: (2,3), 4: (3,4), 5: (4,5), 6: (5,3)
    """
    nodes = [0, 1, 2, 3, 4, 5]
    edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)]
    return Graph(nodes, edges)


@pytest.fixture()
def weighted_diamond():
    """Diamond: 0-1(1), 0-2(4), 1-3(2), 2-3(1). Edges indexed 0-3."""
    nodes = [0, 1, 2, 3]
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    weights = [1.0, 4.0, 2.0, 1.0]
    return Graph(nodes, edges), weights


# ── GraphView creation ──


def test_without_edges_returns_view(bridge_graph):
    view = bridge_graph.without_edges([3])
    assert isinstance(view, GraphView)


def test_edge_count(bridge_graph):
    assert bridge_graph.edge_count == 7


def test_node_count(bridge_graph):
    assert bridge_graph.node_count == 6


def test_edge_indices(bridge_graph):
    assert bridge_graph.edge_indices(2, 3) == [3]
    assert bridge_graph.edge_indices(3, 2) == [3]  # undirected
    assert bridge_graph.edge_indices(0, 1) == [0]
    assert bridge_graph.edge_indices(99, 100) == []


# ── Connected components with mask ──


def test_cc_mask_bridge_removed(bridge_graph):
    """Removing the bridge (2,3) splits the graph into two components."""
    view = bridge_graph.without_edges([3])
    comps = sorted(list(view.connected_components()), key=min)
    assert comps == [{0, 1, 2}, {3, 4, 5}]


def test_cc_no_mask_single_component(bridge_graph):
    """Without mask, full graph is one component."""
    comps = list(bridge_graph.connected_components())
    assert len(comps) == 1


# ── Bridges with mask ──


def test_bridges_mask_edge_removed(bridge_graph):
    """Remove edge (0,1) idx=0. Now edge (1,2) becomes a bridge too."""
    view = bridge_graph.without_edges([0])
    bridge_list = view.bridges()
    normed = {(min(u, v), max(u, v)) for u, v in bridge_list}
    assert (1, 2) in normed
    assert (2, 3) in normed


def test_bridges_mask_bridge_itself(bridge_graph):
    """Remove the bridge itself — remaining triangles have no bridges."""
    view = bridge_graph.without_edges([3])
    assert view.bridges() == []


# ── Articulation points with mask ──


def test_ap_mask(bridge_graph):
    """Remove bridge (2,3). No articulation points in either triangle."""
    view = bridge_graph.without_edges([3])
    assert view.articulation_points() == set()


# ── BCC with mask ──


def test_bcc_mask(bridge_graph):
    """Remove bridge, each triangle is its own biconnected component."""
    view = bridge_graph.without_edges([3])
    blocks = [frozenset(b) for b in view.biconnected_components()]
    assert frozenset({0, 1, 2}) in blocks
    assert frozenset({3, 4, 5}) in blocks


# ── BFS with mask ──


def test_bfs_mask(bridge_graph):
    """BFS from 0 with bridge removed — only reaches {0, 1, 2}."""
    view = bridge_graph.without_edges([3])
    visited = view.bfs(0)
    assert set(visited) == {0, 1, 2}


def test_bfs_mask_other_side(bridge_graph):
    """BFS from 3 with bridge removed — only reaches {3, 4, 5}."""
    view = bridge_graph.without_edges([3])
    visited = view.bfs(3)
    assert set(visited) == {3, 4, 5}


# ── Shortest path with mask ──


def test_shortest_path_mask(weighted_diamond):
    """Diamond: remove direct edge 0-1, path must go 0->2->3 (cost 5)."""
    g, w = weighted_diamond
    view = g.without_edges([0])  # remove edge (0,1)
    path = view.shortest_path(w, 0, 3)
    assert path == [0, 2, 3]


def test_shortest_path_unreachable(weighted_diamond):
    """Remove all edges from 0 — target unreachable."""
    g, w = weighted_diamond
    view = g.without_edges([0, 1])  # remove both edges from node 0
    path = view.shortest_path(w, 0, 3)
    assert path == []


# ── SSSP with mask ──


def test_sssp_mask(weighted_diamond):
    g, w = weighted_diamond
    view = g.without_edges([0])  # remove edge (0,1)
    lengths = view.shortest_path_lengths(w, 0)
    assert lengths[0] == 0.0
    assert lengths[2] == 4.0
    assert lengths[3] == 5.0  # 0->2->3
    assert 1 not in lengths or lengths[1] > 1.0  # can't use direct edge


# ── Multi-source Dijkstra with mask ──


def test_multi_source_mask(weighted_diamond):
    g, w = weighted_diamond
    view = g.without_edges([0])  # remove edge (0,1)
    lengths = view.multi_source_shortest_path_lengths(w, [0, 3])
    assert lengths[0] == 0.0
    assert lengths[3] == 0.0


# ── for_each_edge_excluded ──


def test_for_each_edge_excluded_cc():
    """Simple chain: 0-1-2. Removing any edge splits into 2 components."""
    g = Graph([0, 1, 2], [(0, 1), (1, 2)])
    results = dict(for_each_edge_excluded(g, "connected_components"))
    # edge 0 removed: {0}, {1,2}
    assert len(list(results[0])) == 2
    # edge 1 removed: {0,1}, {2}
    assert len(list(results[1])) == 2


def test_for_each_edge_excluded_bridges(bridge_graph):
    """For each edge excluded, count remaining bridges."""
    results = {}
    for idx, bridge_list in for_each_edge_excluded(bridge_graph, "bridges"):
        results[idx] = len(bridge_list)
    # Removing the bridge itself (idx=3) should leave 0 bridges
    assert results[3] == 0


def test_for_each_edge_excluded_subset():
    """Only iterate over a subset of edge indices."""
    g = Graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
    results = dict(
        for_each_edge_excluded(g, "connected_components", edge_indices=[1])
    )
    assert 1 in results
    assert 0 not in results


# ── Mask doesn't affect base graph ──


def test_view_does_not_mutate_base(bridge_graph):
    """Creating and using a view must not change the base graph's results."""
    comps_before = sorted(list(bridge_graph.connected_components()), key=min)

    view = bridge_graph.without_edges([3])
    _ = list(view.connected_components())

    comps_after = sorted(list(bridge_graph.connected_components()), key=min)
    assert comps_before == comps_after


def test_multiple_views_independent(bridge_graph):
    """Two views from the same graph are independent."""
    view_a = bridge_graph.without_edges([3])  # remove bridge
    view_b = bridge_graph.without_edges([0])  # remove edge (0,1)

    comps_a = sorted(list(view_a.connected_components()), key=min)
    comps_b = sorted(list(view_b.connected_components()), key=min)

    assert comps_a == [{0, 1, 2}, {3, 4, 5}]
    assert len(comps_b) == 1  # still connected
