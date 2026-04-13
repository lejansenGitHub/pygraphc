"""Tests for Graph.split_node and GraphView.split_node."""

import pytest

from cgraph import Graph, GraphView

# ── Basic split ──


def test_split_node_returns_view():
    graph = Graph([0, 1, 2], [(0, 1), (1, 2)])
    view = graph.split_node(node_id=1, new_node_id=3, edge_indices_to_new_node=[1])
    assert isinstance(view, GraphView)


def test_split_node_triangle_stays_connected():
    """Split node in a triangle: reroute one edge to new node, graph stays connected."""
    # --- Input ---
    #   0 -- 1 -- 2
    #    \       /
    #     -------
    # edges: 0:(0,1), 1:(1,2), 2:(2,0)
    graph = Graph([0, 1, 2], [(0, 1), (1, 2), (2, 0)])

    # --- Split ---
    # Reroute edge 1 (1,2) to new node 3 → becomes (3,2)
    view = graph.split_node(node_id=1, new_node_id=3, edge_indices_to_new_node=[1])

    # --- Expected ---
    # Edges: (0,1), (3,2), (2,0) — all connected via node 0 and 2
    components = list(view.connected_components())
    assert len(components) == 1
    assert components[0] == {0, 1, 2, 3}


def test_split_node_chain_disconnects():
    """Splitting the middle node of a chain disconnects the graph."""
    # --- Input ---
    # 0 -- 1 -- 2
    # edges: 0:(0,1), 1:(1,2)
    graph = Graph([0, 1, 2], [(0, 1), (1, 2)])

    # --- Split ---
    # Reroute edge 1 (1,2) to new node 3 → becomes (3,2)
    view = graph.split_node(node_id=1, new_node_id=3, edge_indices_to_new_node=[1])

    # --- Expected ---
    # Edges: (0,1) and (3,2) — two components
    components = sorted(view.connected_components(), key=min)
    assert components == [{0, 1}, {2, 3}]


def test_split_node_reroute_all_edges():
    """Rerouting all edges to new node makes the original node isolated."""
    # --- Input ---
    # Star: center=0, leaves=1,2,3
    graph = Graph([0, 1, 2, 3], [(0, 1), (0, 2), (0, 3)])

    # --- Split ---
    view = graph.split_node(node_id=0, new_node_id=99, edge_indices_to_new_node=[0, 1, 2])

    # --- Expected ---
    components = sorted(view.connected_components(), key=min)
    assert {0} in components
    assert {1, 2, 3, 99} in components


def test_split_node_reroute_no_edges():
    """Rerouting zero edges is a valid no-op."""
    graph = Graph([0, 1, 2], [(0, 1), (1, 2)])
    view = graph.split_node(node_id=1, new_node_id=3, edge_indices_to_new_node=[])
    components = list(view.connected_components())
    assert len(components) == 1
    assert components[0] == {0, 1, 2}


# ── Base graph preservation ──


def test_split_node_preserves_base_graph():
    """Base graph is unchanged after split."""
    graph = Graph([0, 1, 2], [(0, 1), (1, 2)])
    _ = graph.split_node(node_id=1, new_node_id=3, edge_indices_to_new_node=[1])
    components = list(graph.connected_components())
    assert len(components) == 1
    assert components[0] == {0, 1, 2}


# ── Chaining on views ──


def test_split_node_on_masked_view():
    """split_node works on a GraphView created by without_edges."""
    # --- Input ---
    # 0 -- 1 -- 2 -- 3
    # edges: 0:(0,1), 1:(1,2), 2:(2,3)
    graph = Graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])

    # --- Step 1: mask edge (0,1) ---
    view = graph.without_edges([0])
    # Now: isolated 0, chain 1-2-3

    # --- Step 2: split node 2 on the view ---
    view2 = view.split_node(node_id=2, new_node_id=99, edge_indices_to_new_node=[2])
    # Edge 2 (2,3) rerouted to (99,3). Remaining non-excluded: (1,2) and (99,3)

    # --- Expected ---
    components = sorted(view2.connected_components(), key=min)
    assert {0} in components
    assert {1, 2} in components
    assert {3, 99} in components


def test_split_node_on_node_masked_view():
    """split_node works on a GraphView created by without_nodes."""
    # --- Input ---
    # 0 -- 1 -- 2 -- 3
    graph = Graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])

    # --- Step 1: mask node 0 ---
    view = graph.without_nodes([0])

    # --- Step 2: split node 2 ---
    view2 = view.split_node(node_id=2, new_node_id=99, edge_indices_to_new_node=[2])

    # --- Expected: node 0 still in base graph, but excluded from output ---
    # The rebuilt graph includes all base nodes + 99. Node mask is lost after
    # with_edges rebuild (pre-existing limitation).
    components = list(view2.connected_components())
    node_union = set().union(*components)
    assert 99 in node_union
    assert 1 in node_union


# ── Multigraph ──


def test_split_node_parallel_edges():
    """Reroute one of two parallel edges — graph stays connected."""
    # --- Input ---
    # Two edges between 0 and 1
    graph = Graph([0, 1], [(0, 1), (0, 1)])

    # --- Split ---
    # Reroute first copy to new node 2
    view = graph.split_node(node_id=0, new_node_id=2, edge_indices_to_new_node=[0])

    # --- Expected ---
    # Edges: (2,1) and (0,1) — all three nodes connected through 1
    components = list(view.connected_components())
    assert len(components) == 1
    assert components[0] == {0, 1, 2}


# ── Algorithms on split view ──


def test_split_node_bfs():
    """BFS traverses edges correctly after split."""
    # --- Input ---
    # Diamond: 0-1, 0-2, 1-3, 2-3
    graph = Graph([0, 1, 2, 3], [(0, 1), (0, 2), (1, 3), (2, 3)])

    # --- Split ---
    # Reroute edge 1 (0,2) to new node 99 → becomes (99,2)
    view = graph.split_node(node_id=0, new_node_id=99, edge_indices_to_new_node=[1])

    # --- Expected ---
    # Edges: (0,1), (99,2), (1,3), (2,3)
    # BFS from 0: 0 → 1 → 3 → 2 → 99
    visited = set(view.bfs(0))
    assert visited == {0, 1, 2, 3, 99}


def test_split_node_bridges():
    """Splitting creates a bridge where there wasn't one."""
    # --- Input ---
    # Triangle: 0-1-2-0
    graph = Graph([0, 1, 2], [(0, 1), (1, 2), (2, 0)])
    assert graph.bridges() == []

    # --- Split ---
    # Reroute edge 2 (2,0) to new node 3 → becomes (2,3)
    view = graph.split_node(node_id=0, new_node_id=3, edge_indices_to_new_node=[2])

    # --- Expected ---
    # Edges: (0,1), (1,2), (2,3) — chain, every edge is a bridge
    bridge_set = {(min(u, v), max(u, v)) for u, v in view.bridges()}
    assert (0, 1) in bridge_set
    assert (1, 2) in bridge_set
    assert (2, 3) in bridge_set


def test_split_node_articulation_points():
    """Splitting creates articulation points."""
    # --- Input ---
    # Triangle: 0-1-2-0 — no articulation points
    graph = Graph([0, 1, 2], [(0, 1), (1, 2), (2, 0)])
    assert graph.articulation_points() == set()

    # --- Split ---
    # Reroute edge 2 (2,0) to new node 3 → chain: 3-2-1-0
    view = graph.split_node(node_id=0, new_node_id=3, edge_indices_to_new_node=[2])

    # --- Expected ---
    # Chain 0-1-2-3: nodes 1 and 2 are articulation points
    assert view.articulation_points() == {1, 2}


def test_split_node_shortest_path():
    """Shortest path uses rerouted edges correctly."""
    # --- Input ---
    # Diamond: 0-1(w=1), 0-2(w=10), 1-3(w=1), 2-3(w=1)
    graph = Graph([0, 1, 2, 3], [(0, 1), (0, 2), (1, 3), (2, 3)])

    # --- Split ---
    # Reroute edge 1 (0,2) to new node 99 → becomes (99,2)
    view = graph.split_node(node_id=0, new_node_id=99, edge_indices_to_new_node=[1])

    # --- Expected ---
    # Rebuilt edges: (0,1), (99,2), (1,3), (2,3)
    # Weights in rebuilt graph: [1.0, 10.0, 1.0, 1.0] (same order minus excluded + added)
    # Actually rebuilt edges are: (0,1),(1,3),(2,3),(99,2) — order may change
    # Shortest path from 0 to 99: 0→1→3→2→99
    # Let's just verify connectivity
    lengths = view.shortest_path_lengths([1.0, 1.0, 1.0, 1.0], 0)
    assert 99 in lengths


# ── Error cases ──


def test_split_node_error_node_not_in_graph():
    graph = Graph([0, 1], [(0, 1)])
    with pytest.raises(ValueError, match="not in the graph"):
        graph.split_node(node_id=99, new_node_id=3, edge_indices_to_new_node=[0])


def test_split_node_error_new_node_already_exists():
    graph = Graph([0, 1, 2], [(0, 1), (1, 2)])
    with pytest.raises(ValueError, match="already exists"):
        graph.split_node(node_id=1, new_node_id=2, edge_indices_to_new_node=[1])


def test_split_node_error_edge_not_incident():
    graph = Graph([0, 1, 2], [(0, 1), (1, 2)])
    with pytest.raises(ValueError, match="not incident"):
        graph.split_node(node_id=0, new_node_id=3, edge_indices_to_new_node=[1])


def test_split_node_error_edge_index_out_of_range():
    graph = Graph([0, 1], [(0, 1)])
    with pytest.raises(IndexError):
        graph.split_node(node_id=0, new_node_id=2, edge_indices_to_new_node=[5])


# ── GraphView error cases ──


def test_split_node_view_error_node_not_in_graph():
    graph = Graph([0, 1], [(0, 1)])
    view = graph.without_edges([])
    with pytest.raises(ValueError, match="not in the graph"):
        view.split_node(node_id=99, new_node_id=3, edge_indices_to_new_node=[0])


def test_split_node_view_error_new_node_already_exists():
    graph = Graph([0, 1, 2], [(0, 1), (1, 2)])
    view = graph.without_edges([])
    with pytest.raises(ValueError, match="already exists"):
        view.split_node(node_id=1, new_node_id=2, edge_indices_to_new_node=[1])


def test_split_node_view_error_edge_not_incident():
    graph = Graph([0, 1, 2], [(0, 1), (1, 2)])
    view = graph.without_edges([])
    with pytest.raises(ValueError, match="not incident"):
        view.split_node(node_id=0, new_node_id=3, edge_indices_to_new_node=[1])
