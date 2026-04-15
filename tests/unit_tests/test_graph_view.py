"""Tests for GraphView (edge-masked views) and for_each_edge_excluded."""

import pytest

from cgraph import Graph, GraphView, for_each_edge_excluded

# ── Fixtures ──


@pytest.fixture
def bridge_graph():
    """Graph: (0-1-2-0) --bridge(2,3)-- (3-4-5-3).

    Edges by index:
        0: (0,1), 1: (1,2), 2: (2,0), 3: (2,3), 4: (3,4), 5: (4,5), 6: (5,3)
    """
    nodes = [0, 1, 2, 3, 4, 5]
    edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)]
    return Graph(nodes, edges)


@pytest.fixture
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
    comps = sorted(view.connected_components(), key=min)
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
    results = dict(for_each_edge_excluded(g, "connected_components", edge_indices=[1]))
    assert 1 in results
    assert 0 not in results


# ── Mask doesn't affect base graph ──


def test_view_does_not_mutate_base(bridge_graph):
    """Creating and using a view must not change the base graph's results."""
    comps_before = sorted(bridge_graph.connected_components(), key=min)

    view = bridge_graph.without_edges([3])
    _ = list(view.connected_components())

    comps_after = sorted(bridge_graph.connected_components(), key=min)
    assert comps_before == comps_after


def test_multiple_views_independent(bridge_graph):
    """Two views from the same graph are independent."""
    view_a = bridge_graph.without_edges([3])  # remove bridge
    view_b = bridge_graph.without_edges([0])  # remove edge (0,1)

    comps_a = sorted(view_a.connected_components(), key=min)
    comps_b = sorted(view_b.connected_components(), key=min)

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
    comps = sorted(view.connected_components(), key=min)
    assert comps == [{10}, {20}, {30}]
    assert view.bridges() == []
    assert view.articulation_points() == set()


# ── Edge-case: single edge graph ──


def test_mask_single_edge():
    """Graph with one edge — masking it isolates both nodes."""
    g = Graph([0, 1], [(0, 1)])
    view = g.without_edges([0])
    comps = sorted(view.connected_components(), key=min)
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
    comps = sorted(view.connected_components(), key=min)
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
    comps2 = sorted(view2.connected_components(), key=min)
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
    comps = sorted(view.connected_components(), key=min)
    assert comps == [{0}, {1}, {2}, {3}]
    assert view.bfs(0) == [0]


def test_mask_isolates_leaf():
    """Chain 0-1-2: mask edge (1,2) — node 2 becomes isolated."""
    g = Graph([0, 1, 2], [(0, 1), (1, 2)])
    view = g.without_edges([1])
    comps = sorted(view.connected_components(), key=min)
    assert comps == [{0, 1}, {2}]


# ── Edge-case: boundary indices (first and last edge) ──


def test_mask_first_edge():
    """Mask only the first edge in the list."""
    g = Graph([0, 1, 2], [(0, 1), (1, 2)])
    view = g.without_edges([0])
    comps = sorted(view.connected_components(), key=min)
    assert comps == [{0}, {1, 2}]


def test_mask_last_edge():
    """Mask only the last edge in the list."""
    g = Graph([0, 1, 2], [(0, 1), (1, 2)])
    view = g.without_edges([1])
    comps = sorted(view.connected_components(), key=min)
    assert comps == [{0, 1}, {2}]


# ── Edge-case: mask on disconnected graph ──


def test_mask_on_disconnected_graph():
    """Already disconnected graph — masking an edge within a component splits it."""
    g = Graph([0, 1, 2, 3], [(0, 1), (2, 3)])
    view = g.without_edges([0])  # remove edge in first component
    comps = sorted(view.connected_components(), key=min)
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
    comps = sorted(view.connected_components(), key=min)
    assert comps == [{1000, 2000}, {3000, 4000}]
    assert view.bfs(1000) == [1000, 2000]


# ── Phase 5b: Edge-Addition Views ──


class TestWithEdgesBasic:
    """Tests for Graph.with_edges() and GraphView.with_edges()."""

    def test_with_edges_returns_view(self, bridge_graph):
        view = bridge_graph.with_edges([(10, 20)])
        assert isinstance(view, GraphView)

    def test_with_edges_adds_connectivity(self):
        """Two disconnected nodes — adding an edge connects them."""
        g = Graph([0, 1], [])
        view = g.with_edges([(0, 1)])
        comps = list(view.connected_components())
        assert len(comps) == 1
        assert comps[0] == {0, 1}

    def test_with_edges_introduces_new_node(self):
        """Added edge references a node not in the base graph."""
        g = Graph([0, 1], [(0, 1)])
        view = g.with_edges([(1, 2)])
        comps = list(view.connected_components())
        assert len(comps) == 1
        assert comps[0] == {0, 1, 2}

    def test_with_edges_preserves_base(self):
        """Base graph remains unchanged after creating a with_edges view."""
        g = Graph([0, 1], [])
        _ = g.with_edges([(0, 1)])
        comps = sorted(g.connected_components(), key=min)
        assert comps == [{0}, {1}]

    def test_with_edges_multiple(self):
        """Add multiple edges at once."""
        g = Graph([0, 1, 2, 3], [])
        view = g.with_edges([(0, 1), (2, 3)])
        comps = sorted(view.connected_components(), key=min)
        assert comps == [{0, 1}, {2, 3}]

    def test_with_edges_bridges(self):
        """Added edge creates a bridge in the new view."""
        g = Graph([0, 1, 2], [(0, 1)])
        view = g.with_edges([(1, 2)])
        bridge_list = view.bridges()
        normed = {(min(u, v), max(u, v)) for u, v in bridge_list}
        assert (0, 1) in normed
        assert (1, 2) in normed

    def test_with_edges_bfs(self):
        """BFS traverses added edges."""
        g = Graph([0, 1, 2], [(0, 1)])
        view = g.with_edges([(1, 2)])
        visited = set(view.bfs(0))
        assert visited == {0, 1, 2}

    def test_with_edges_shortest_path(self):
        """Shortest path uses added edge."""
        g = Graph([0, 1, 2], [(0, 1)])
        view = g.with_edges([(1, 2)])
        # 3 edges in rebuilt graph: (0,1), (1,2)
        w = [1.0, 1.0]
        path = view.shortest_path(w, 0, 2)
        assert path == [0, 1, 2]

    def test_with_edges_empty_additions(self):
        """Adding no edges returns a view equivalent to the base graph."""
        g = Graph([0, 1], [(0, 1)])
        view = g.with_edges([])
        comps = list(view.connected_components())
        assert len(comps) == 1


class TestWithEdgesFromView:
    """Tests for chaining: GraphView.with_edges()."""

    def test_without_then_with(self):
        """Exclude an edge, then add a different one."""
        # Triangle 0-1-2
        g = Graph([0, 1, 2], [(0, 1), (1, 2), (2, 0)])
        view = g.without_edges([2])  # remove (2,0)
        view2 = view.with_edges([(0, 3)])  # add edge to new node
        comps = list(view2.connected_components())
        assert len(comps) == 1
        assert {0, 1, 2, 3} == comps[0]

    def test_with_edges_on_addition_view(self):
        """Chaining with_edges on an already-added view."""
        g = Graph([0, 1], [])
        v1 = g.with_edges([(0, 1)])
        v2 = v1.with_edges([(1, 2)])
        comps = list(v2.connected_components())
        assert len(comps) == 1
        assert comps[0] == {0, 1, 2}


class TestWithEdgesEdgeCases:
    """Edge cases for edge-addition views."""

    def test_with_edges_empty_graph(self):
        """Start with empty graph, add edges."""
        g = Graph([], [])
        view = g.with_edges([(0, 1)])
        comps = list(view.connected_components())
        assert len(comps) == 1
        assert comps[0] == {0, 1}

    def test_with_edges_self_loop(self):
        """Adding a self-loop."""
        g = Graph([0, 1], [(0, 1)])
        view = g.with_edges([(0, 0)])
        comps = list(view.connected_components())
        assert len(comps) == 1

    def test_with_edges_parallel(self):
        """Adding a parallel edge (duplicate)."""
        g = Graph([0, 1], [(0, 1)])
        view = g.with_edges([(0, 1)])
        comps = list(view.connected_components())
        assert len(comps) == 1
        # Parallel edges: no bridges
        assert view.bridges() == []

    def test_with_edges_large_node_ids(self):
        """Non-contiguous large node IDs."""
        g = Graph([1000, 2000], [(1000, 2000)])
        view = g.with_edges([(2000, 3000)])
        comps = list(view.connected_components())
        assert len(comps) == 1
        assert comps[0] == {1000, 2000, 3000}

    def test_with_edges_articulation_points(self):
        """Added edge eliminates an articulation point."""
        # Chain: 0-1-2 — node 1 is AP
        g = Graph([0, 1, 2], [(0, 1), (1, 2)])
        assert 1 in g.articulation_points()
        # Add shortcut 0-2 — no more AP
        view = g.with_edges([(0, 2)])
        assert view.articulation_points() == set()

    def test_with_edges_biconnected_components(self):
        """Added edge merges biconnected components."""
        # Chain 0-1-2: three separate BCCs (each edge is one)
        g = Graph([0, 1, 2], [(0, 1), (1, 2)])
        blocks = list(g.biconnected_components())
        assert len(blocks) == 2
        # Add 0-2: entire graph becomes one BCC
        view = g.with_edges([(0, 2)])
        blocks2 = list(view.biconnected_components())
        assert len(blocks2) == 1
        assert blocks2[0] == {0, 1, 2}


# ── Connected Components with Branch IDs ──


def test_graph_cc_with_branch_ids_no_exclusions():
    """Graph method: two components, branch IDs correctly assigned."""
    # --- Input ---
    graph = Graph([1, 2, 3, 4], [(1, 2), (3, 4)], branch_ids=[100, 200])

    # --- Expected ---
    expected = {frozenset({1, 2}): {100}, frozenset({3, 4}): {200}}

    # --- Execute ---
    result = list(graph.connected_components_with_branch_ids())

    # --- Assert ---
    result_map = {frozenset(nodes): branches for nodes, branches in result}
    assert result_map == expected


def test_graph_cc_with_branch_ids_single_component():
    """All nodes connected — single component gets all branch IDs."""
    # --- Input ---
    graph = Graph([10, 20, 30], [(10, 20), (20, 30)], branch_ids=[500, 600])

    # --- Execute ---
    result = list(graph.connected_components_with_branch_ids())

    # --- Assert ---
    assert len(result) == 1
    nodes, branches = result[0]
    assert nodes == {10, 20, 30}
    assert branches == {500, 600}


def test_view_cc_with_branch_ids_exclude_branch():
    """Excluding a branch by ID splits the component and drops its branch ID.

    Graph: 1--2--3 with branch IDs [100, 200].
    Excluding branch 200 (2--3) splits into {1,2} with {100} and {3} with {}.
    """
    # --- Input ---
    graph = Graph([1, 2, 3], [(1, 2), (2, 3)], branch_ids=[100, 200])
    view = graph.without_branches([200])

    # --- Expected ---
    expected = {frozenset({1, 2}): {100}, frozenset({3}): set()}

    # --- Execute ---
    result = list(view.connected_components_with_branch_ids())

    # --- Assert ---
    result_map = {frozenset(nodes): branches for nodes, branches in result}
    assert result_map == expected


def test_view_cc_with_branch_ids_exclude_one_node():
    """Excluding a node removes it from output but edges still connect.

    Graph: A(1)--B(2) with branch_id 100.
    Excluding node 1: component has nodes={2}, branches={100}.
    """
    # --- Input ---
    graph = Graph([1, 2], [(1, 2)], branch_ids=[100])
    view = graph.without_nodes([1])

    # --- Execute ---
    result = list(view.connected_components_with_branch_ids())

    # --- Assert ---
    assert len(result) == 1
    nodes, branches = result[0]
    assert nodes == {2}
    assert branches == {100}


def test_view_cc_with_branch_ids_exclude_both_nodes():
    """Excluding all nodes leaves a component with only the branch ID.

    Graph: A(1)--B(2) with branch_id 100.
    Excluding both nodes: component has nodes={}, branches={100}.
    """
    # --- Input ---
    graph = Graph([1, 2], [(1, 2)], branch_ids=[100])
    view = graph.without_nodes([1, 2])

    # --- Execute ---
    result = list(view.connected_components_with_branch_ids())

    # --- Assert ---
    assert len(result) == 1
    nodes, branches = result[0]
    assert nodes == set()
    assert branches == {100}


def test_view_cc_with_branch_ids_exclude_branch_and_node():
    """Combining branch and node exclusion.

    Graph: 1--2--3--4 with branch IDs [100, 200, 300].
    Exclude branch 200 (2--3), exclude node 1.
    Branch exclusion splits connectivity: {1,2} and {3,4}.
    Node exclusion removes node 1 from output.
    Result: ({2}, {100}) and ({3, 4}, {300}).
    """
    # --- Input ---
    graph = Graph([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4)], branch_ids=[100, 200, 300])
    view = graph.without_branches([200]).without_nodes([1])

    # --- Expected ---
    expected = {frozenset({2}): {100}, frozenset({3, 4}): {300}}

    # --- Execute ---
    result = list(view.connected_components_with_branch_ids())

    # --- Assert ---
    result_map = {frozenset(nodes): branches for nodes, branches in result}
    assert result_map == expected


def test_view_cc_with_branch_ids_excluded_node_breaks_connectivity():
    """
    Excluding a middle node breaks connectivity — traversal stops at it.

    Graph: 1--2--3 with branch IDs [100, 200].
    Exclude node 2: nodes 1 and 3 are in separate components because
    traversal does not pass through excluded nodes. Each CC collects the
    branch_id of its edge incident to the excluded node.
    """
    # --- Input ---
    graph = Graph([1, 2, 3], [(1, 2), (2, 3)], branch_ids=[100, 200])
    view = graph.without_nodes([2])

    # --- Execute ---
    result = list(view.connected_components_with_branch_ids())

    # --- Assert ---
    result_map = {frozenset(nodes): branches for nodes, branches in result}
    assert frozenset({1}) in result_map
    assert frozenset({3}) in result_map
    assert result_map[frozenset({1})] == {100}
    assert result_map[frozenset({3})] == {200}


def test_view_cc_with_branch_ids_no_edges():
    """Isolated nodes, no edges — each node is its own component."""
    # --- Input ---
    graph = Graph([1, 2, 3], [], branch_ids=[])
    view = graph.without_nodes([2])

    # --- Execute ---
    result = list(view.connected_components_with_branch_ids())

    # --- Assert ---
    result_nodes = [nodes for nodes, _branches in result]
    assert {1} in result_nodes
    assert {3} in result_nodes


def test_view_cc_with_branch_ids_large_node_ids():
    """Non-contiguous large node IDs with exclusions."""
    # --- Input ---
    graph = Graph(
        [1000, 2000, 3000, 4000],
        [(1000, 2000), (2000, 3000), (3000, 4000)],
        branch_ids=[10, 20, 30],
    )
    view = graph.without_branches([20]).without_nodes([1000])

    # --- Expected ---
    # Branch 20 (2000--3000) excluded: splits into {1000,2000} and {3000,4000}
    # Node 1000 excluded from output: {2000} and {3000,4000}
    expected = {frozenset({2000}): {10}, frozenset({3000, 4000}): {30}}

    # --- Execute ---
    result = list(view.connected_components_with_branch_ids())

    # --- Assert ---
    result_map = {frozenset(nodes): branches for nodes, branches in result}
    assert result_map == expected


def test_graph_cc_with_branch_ids_empty_graph():
    """Empty graph — no components."""
    # --- Input ---
    graph = Graph([], [], branch_ids=[])

    # --- Execute ---
    result = list(graph.connected_components_with_branch_ids())

    # --- Assert ---
    assert result == []


def test_view_cc_with_branch_ids_exclude_only_branch():
    """Excluding the only branch gives two isolated nodes, no branches.

    Graph: 1--2 with branch_id 100. Exclude branch 100.
    Result: ({1}, {}) and ({2}, {}).
    """
    # --- Input ---
    graph = Graph([1, 2], [(1, 2)], branch_ids=[100])
    view = graph.without_branches([100])

    # --- Execute ---
    result = list(view.connected_components_with_branch_ids())

    # --- Assert ---
    result_map = {frozenset(nodes): branches for nodes, branches in result}
    assert result_map == {frozenset({1}): set(), frozenset({2}): set()}


def test_view_cc_with_branch_ids_exclude_all_branches():
    """Excluding all branches gives isolated nodes, no branches.

    Graph: 1--2--3 with branch IDs [100, 200]. Exclude both.
    Result: three isolated components, all with empty branch sets.
    """
    # --- Input ---
    graph = Graph([1, 2, 3], [(1, 2), (2, 3)], branch_ids=[100, 200])
    view = graph.without_branches([100, 200])

    # --- Execute ---
    result = list(view.connected_components_with_branch_ids())

    # --- Assert ---
    assert len(result) == 3
    for nodes, branches in result:
        assert len(nodes) == 1
        assert branches == set()


def test_view_cc_with_branch_ids_parallel_edges():
    """Multigraph: parallel edges between same nodes.

    Graph: 1==2 (two parallel edges, branch IDs 100 and 200).
    Exclude branch 100: remaining edge still connects, branch 200 survives.
    """
    # --- Input ---
    graph = Graph([1, 2], [(1, 2), (1, 2)], branch_ids=[100, 200])
    view = graph.without_branches([100])

    # --- Execute ---
    result = list(view.connected_components_with_branch_ids())

    # --- Assert ---
    assert len(result) == 1
    nodes, branches = result[0]
    assert nodes == {1, 2}
    assert branches == {200}


def test_view_cc_with_branch_ids_exclude_node_isolated():
    """Excluding an isolated node removes it entirely.

    Graph: 1  2  3 (no edges). Exclude node 2.
    Result: two components ({1}, {}) and ({3}, {}). Node 2 gone.
    """
    # --- Input ---
    graph = Graph([1, 2, 3], [], branch_ids=[])
    view = graph.without_nodes([2])

    # --- Execute ---
    result = list(view.connected_components_with_branch_ids())

    # --- Assert ---
    result_nodes = sorted([frozenset(nodes) for nodes, _branches in result])
    assert frozenset({1}) in result_nodes
    assert frozenset({3}) in result_nodes
    assert frozenset({2}) not in result_nodes


def test_view_cc_with_branch_ids_hub_excluded_breaks_connectivity():
    """
    Excluding a hub node breaks connectivity — each leaf becomes its own CC.

    Graph: hub(10) connects to leaf nodes 1, 2, 3.
    Excluding hub(10): traversal does not pass through it, so each leaf
    is isolated. Each leaf's CC collects the branch_id of its edge to
    the hub.
    """
    # --- Input ---
    graph = Graph(
        [1, 2, 3, 10],
        [(1, 10), (2, 10), (3, 10)],
        branch_ids=[101, 102, 103],
    )
    view = graph.without_nodes([10])

    # --- Execute ---
    result = list(view.connected_components_with_branch_ids())

    # --- Assert ---
    result_map = {frozenset(nodes): branches for nodes, branches in result}
    assert frozenset({1}) in result_map
    assert frozenset({2}) in result_map
    assert frozenset({3}) in result_map
    assert result_map[frozenset({1})] == {101}
    assert result_map[frozenset({2})] == {102}
    assert result_map[frozenset({3})] == {103}


def test_view_cc_with_branch_ids_excluded_node_triangle_keeps_alternate_path():
    """
    Excluding a node only breaks connectivity THROUGH that node.
    Alternative paths that bypass the excluded node are unaffected.

    Graph: 1--2--3, 1--3 (triangle). Branch IDs [100, 200, 300].
    Exclude node 2: nodes 1 and 3 stay connected via edge 1--3 (branch 300).
    Edges 100 (1-2) and 200 (2-3) are incident to non-excluded nodes
    1 and 3, so their branch_ids are still collected in the CC.
    """
    # --- Input ---
    graph = Graph(
        [1, 2, 3],
        [(1, 2), (2, 3), (1, 3)],
        branch_ids=[100, 200, 300],
    )
    view = graph.without_nodes([2])

    # --- Execute ---
    result = list(view.connected_components_with_branch_ids())

    # --- Assert ---
    assert len(result) == 1
    nodes, branches = result[0]
    assert nodes == {1, 3}
    assert branches == {100, 200, 300}


def test_graph_cc_with_branch_ids_raises_without_branch_ids():
    """Graph.connected_components_with_branch_ids raises if no branch_ids."""
    graph = Graph([1, 2], [(1, 2)])
    with pytest.raises(ValueError, match="no branch_ids"):
        list(graph.connected_components_with_branch_ids())


def test_graph_without_branches_raises_without_branch_ids():
    """Graph.without_branches raises if no branch_ids."""
    graph = Graph([1, 2], [(1, 2)])
    with pytest.raises(ValueError, match="no branch_ids"):
        graph.without_branches([100])
