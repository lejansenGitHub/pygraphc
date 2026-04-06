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


# ── Edge-case: empty graph ──


def test_mask_empty_graph():
    """Empty graph (no nodes, no edges) — views should work without error."""
    g = Graph([], [])
    view = g.without_edges([])
    assert list(view.connected_components()) == []
    assert view.bridges() == []
    assert view.articulation_points() == set()
    assert list(view.biconnected_components()) == []


# ── Edge-case: nodes but no edges ──


def test_mask_no_edges():
    """Graph with nodes but no edges — mask is empty, all nodes isolated."""
    g = Graph([10, 20, 30], [])
    assert g.edge_count == 0
    view = g.without_edges([])
    comps = sorted(list(view.connected_components()), key=min)
    assert comps == [{10}, {20}, {30}]
    assert view.bridges() == []
    assert view.articulation_points() == set()


# ── Edge-case: single edge graph ──


def test_mask_single_edge():
    """Graph with one edge — masking it isolates both nodes."""
    g = Graph([0, 1], [(0, 1)])
    view = g.without_edges([0])
    comps = sorted(list(view.connected_components()), key=min)
    assert comps == [{0}, {1}]
    assert view.bridges() == []


def test_mask_single_edge_bfs():
    """BFS on single-edge graph with edge masked — only source visited."""
    g = Graph([0, 1], [(0, 1)])
    view = g.without_edges([0])
    assert view.bfs(0) == [0]
    assert view.bfs(1) == [1]


# ── Edge-case: mask ALL edges ──


def test_mask_all_edges():
    """Masking every edge makes all nodes isolated."""
    g = Graph([0, 1, 2], [(0, 1), (1, 2), (2, 0)])
    view = g.without_edges([0, 1, 2])
    comps = sorted(list(view.connected_components()), key=min)
    assert comps == [{0}, {1}, {2}]
    assert view.bridges() == []
    assert view.articulation_points() == set()
    assert list(view.biconnected_components()) == []
    assert view.bfs(0) == [0]


def test_mask_all_edges_weighted():
    """All edges masked — shortest path unreachable, SSSP only has source."""
    g = Graph([0, 1, 2], [(0, 1), (1, 2)])
    w = [1.0, 2.0]
    view = g.without_edges([0, 1])
    assert view.shortest_path(w, 0, 2) == []
    lengths = view.shortest_path_lengths(w, 0)
    assert lengths == {0: 0.0}


# ── Edge-case: self-loops ──


def test_mask_self_loop():
    """Self-loop masked — should not affect connectivity."""
    g = Graph([0, 1], [(0, 0), (0, 1)])
    # Without mask: 1 component
    assert len(list(g.connected_components())) == 1
    # Mask the self-loop (idx 0) — still connected via edge 1
    view = g.without_edges([0])
    comps = list(view.connected_components())
    assert len(comps) == 1
    assert comps[0] == {0, 1}


def test_mask_self_loop_only():
    """Graph with only a self-loop, masked — node becomes isolated."""
    g = Graph([0], [(0, 0)])
    view = g.without_edges([0])
    comps = list(view.connected_components())
    assert comps == [{0}]


# ── Edge-case: parallel (multi) edges ──


def test_mask_one_of_parallel_edges():
    """Two edges between same nodes — masking one still leaves them connected."""
    g = Graph([0, 1], [(0, 1), (0, 1)])
    view = g.without_edges([0])  # remove first copy
    comps = list(view.connected_components())
    assert comps == [{0, 1}]
    # Both masked — disconnected
    view2 = g.without_edges([0, 1])
    comps2 = sorted(list(view2.connected_components()), key=min)
    assert comps2 == [{0}, {1}]


def test_mask_parallel_edges_bridges():
    """Parallel edges: neither is a bridge. Mask one: the other becomes a bridge."""
    g = Graph([0, 1], [(0, 1), (0, 1)])
    assert g.bridges() == []  # parallel edges = no bridge
    view = g.without_edges([0])
    bridge_list = view.bridges()
    normed = {(min(u, v), max(u, v)) for u, v in bridge_list}
    assert normed == {(0, 1)}


# ── Edge-case: mask creates isolated node ──


def test_mask_isolates_node():
    """Star graph: mask all edges from center — center becomes isolated."""
    # 0 is center, connected to 1, 2, 3
    g = Graph([0, 1, 2, 3], [(0, 1), (0, 2), (0, 3)])
    view = g.without_edges([0, 1, 2])  # remove all edges from node 0
    comps = sorted(list(view.connected_components()), key=min)
    assert comps == [{0}, {1}, {2}, {3}]
    assert view.bfs(0) == [0]


def test_mask_isolates_leaf():
    """Chain 0-1-2: mask edge (1,2) — node 2 becomes isolated."""
    g = Graph([0, 1, 2], [(0, 1), (1, 2)])
    view = g.without_edges([1])
    comps = sorted(list(view.connected_components()), key=min)
    assert comps == [{0, 1}, {2}]


# ── Edge-case: boundary indices (first and last edge) ──


def test_mask_first_edge():
    """Mask only the first edge in the list."""
    g = Graph([0, 1, 2], [(0, 1), (1, 2)])
    view = g.without_edges([0])
    comps = sorted(list(view.connected_components()), key=min)
    assert comps == [{0}, {1, 2}]


def test_mask_last_edge():
    """Mask only the last edge in the list."""
    g = Graph([0, 1, 2], [(0, 1), (1, 2)])
    view = g.without_edges([1])
    comps = sorted(list(view.connected_components()), key=min)
    assert comps == [{0, 1}, {2}]


# ── Edge-case: mask on disconnected graph ──


def test_mask_on_disconnected_graph():
    """Already disconnected graph — masking an edge within a component splits it."""
    g = Graph([0, 1, 2, 3], [(0, 1), (2, 3)])
    view = g.without_edges([0])  # remove edge in first component
    comps = sorted(list(view.connected_components()), key=min)
    assert comps == [{0}, {1}, {2, 3}]


# ── Edge-case: articulation points appear/disappear with mask ──


def test_ap_appears_with_mask():
    """Removing an edge can create a new articulation point.

    Graph: 0-1-2-3 with extra edge 0-2.
    Edges: 0:(0,1), 1:(1,2), 2:(2,3), 3:(0,2)
    Unmasked: node 2 is AP (removing it disconnects 3). Node 0 is not AP
    because 0-1-2 and 0-2 form a cycle.
    Mask edge (0,2) idx=3: now 0-1-2-3 is a chain, nodes 1 and 2 are APs.
    """
    g = Graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3), (0, 2)])
    ap_base = g.articulation_points()
    assert 2 in ap_base

    view = g.without_edges([3])  # remove shortcut (0,2)
    ap_masked = view.articulation_points()
    assert ap_masked == {1, 2}


# ── Edge-case: biconnected components change with mask ──


def test_bcc_mask_breaks_biconnectivity():
    """Cycle 0-1-2-3-0 is one biconnected component.
    Mask one edge: it splits into a chain (no biconnected component of size > 2
    unless there are cycles).
    """
    g = Graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3), (3, 0)])
    blocks = list(g.biconnected_components())
    assert len(blocks) == 1
    assert blocks[0] == {0, 1, 2, 3}

    view = g.without_edges([3])  # remove edge (3,0)
    blocks_masked = list(view.biconnected_components())
    # Chain 0-1-2-3: each edge is its own biconnected component
    assert len(blocks_masked) == 3


# ── Edge-case: weighted algorithms with mask forcing longer paths ──


def test_dijkstra_mask_forces_longer_path():
    """Triangle 0-1-2 with weights. Mask cheap edge, force expensive route."""
    g = Graph([0, 1, 2], [(0, 1), (1, 2), (0, 2)])
    w = [1.0, 1.0, 100.0]
    # Unmasked: 0->1->2 costs 2.0
    assert g.shortest_path(w, 0, 2) == [0, 1, 2]
    # Mask edge (0,1): must use 0->2 directly, cost 100.0
    view = g.without_edges([0])
    path = view.shortest_path(w, 0, 2)
    assert path == [0, 2]


def test_sssp_mask_with_cutoff():
    """SSSP with cutoff + mask interaction: mask cheap edge, cutoff blocks long path."""
    g = Graph([0, 1, 2], [(0, 1), (1, 2), (0, 2)])
    w = [1.0, 1.0, 100.0]
    # Mask (0,1): path to 2 costs 100.0, path to 1 costs 101.0
    view = g.without_edges([0])
    lengths = view.shortest_path_lengths(w, 0, cutoff=50.0)
    assert 0 in lengths  # source always reachable
    assert 2 not in lengths  # cost 100 > cutoff 50
    assert 1 not in lengths  # cost 101 > cutoff 50


# ── Edge-case: multi-source Dijkstra with mask ──


def test_multi_source_mask_isolates_source():
    """Mask all edges from one source — other source still reaches nodes."""
    # 0-1-2, 3-4
    g = Graph([0, 1, 2, 3, 4], [(0, 1), (1, 2), (3, 4)])
    w = [1.0, 1.0, 1.0]
    view = g.without_edges([0, 1])  # isolate node 0
    lengths = view.multi_source_shortest_path_lengths(w, [0, 3])
    assert lengths[0] == 0.0  # source itself
    assert lengths[3] == 0.0
    assert lengths[4] == 1.0
    assert 1 not in lengths  # can't reach from 0 (isolated) or 3
    assert 2 not in lengths


# ── Edge-case: eccentricity with mask ──


def test_eccentricity_mask():
    """Chain 0-1-2-3. Mask middle edge — eccentricity shrinks."""
    g = Graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
    w = [1.0, 1.0, 1.0]
    # Unmasked: eccentricity of 0 is 3.0 (0->1->2->3)
    assert g.eccentricity(w, 0) == 3.0
    # Mask edge (1,2): 0 can only reach 1, eccentricity is 1.0
    view = g.without_edges([1])
    assert view.eccentricity(w, 0) == 1.0


# ── Edge-case: for_each_edge_excluded on empty/single-edge graphs ──


def test_for_each_excluded_empty_graph():
    """Empty graph — no edges to iterate over."""
    g = Graph([0], [])
    results = list(for_each_edge_excluded(g, "connected_components"))
    assert results == []


def test_for_each_excluded_single_edge():
    """Single edge — one iteration, removing it."""
    g = Graph([0, 1], [(0, 1)])
    results = list(for_each_edge_excluded(g, "connected_components"))
    assert len(results) == 1
    idx, comps = results[0]
    assert idx == 0
    comps_sorted = sorted(comps, key=min)
    assert comps_sorted == [{0}, {1}]


def test_for_each_excluded_mask_resets():
    """Verify mask is properly reset between iterations — base graph unchanged."""
    g = Graph([0, 1, 2], [(0, 1), (1, 2)])
    # After full iteration, base graph should still be fully connected
    for _ in for_each_edge_excluded(g, "connected_components"):
        pass
    comps = list(g.connected_components())
    assert len(comps) == 1


# ── Edge-case: large node IDs with mask ──


def test_mask_large_node_ids():
    """Non-contiguous large node IDs with mask."""
    nids = [1000, 2000, 3000, 4000]
    edges = [(1000, 2000), (2000, 3000), (3000, 4000)]
    g = Graph(nids, edges)
    view = g.without_edges([1])  # remove (2000, 3000)
    comps = sorted(list(view.connected_components()), key=min)
    assert comps == [{1000, 2000}, {3000, 4000}]
    assert view.bfs(1000) == [1000, 2000]
