"""Tests for Phase 6b: Node-masked views."""

import pytest

from cgraph import Graph, GraphView


@pytest.fixture
def chain():
    """Linear chain: 1-2-3-4-5."""
    nodes = [1, 2, 3, 4, 5]
    edges = [(1, 2), (2, 3), (3, 4), (4, 5)]
    return Graph(nodes, edges)


@pytest.fixture
def diamond():
    """Diamond: 0-1, 0-2, 1-3, 2-3."""
    nodes = [0, 1, 2, 3]
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    weights = [1.0, 4.0, 2.0, 1.0]
    return Graph(nodes, edges), weights


@pytest.fixture
def bridge_graph():
    """(0-1-2-0) --bridge(2,3)-- (3-4-5-3)."""
    nodes = [0, 1, 2, 3, 4, 5]
    edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)]
    return Graph(nodes, edges)


# ── without_nodes returns GraphView ──


def test_without_nodes_returns_view(chain):
    view = chain.without_nodes([3])
    assert isinstance(view, GraphView)


# ── Connected components ──


def test_cc_exclude_middle_splits(chain):
    """Excluding node 3 splits chain into {1,2} and {4,5}."""
    view = chain.without_nodes([3])
    comps = sorted(view.connected_components(), key=min)
    assert comps == [{1, 2}, {4, 5}]


def test_cc_exclude_endpoint(chain):
    """Excluding node 1 leaves {2,3,4,5}."""
    view = chain.without_nodes([1])
    comps = list(view.connected_components())
    assert comps == [{2, 3, 4, 5}]


def test_cc_exclude_all_nodes(chain):
    """Excluding all nodes yields empty result."""
    view = chain.without_nodes([1, 2, 3, 4, 5])
    comps = list(view.connected_components())
    assert comps == []


def test_cc_exclude_nonexistent_node(chain):
    """Excluding a node not in the graph has no effect."""
    view = chain.without_nodes([99])
    comps = list(view.connected_components())
    assert comps == [{1, 2, 3, 4, 5}]


# ── Bridges ──


def test_bridges_with_node_excluded(bridge_graph):
    """Excluding node 3 means bridge (2,3) disappears."""
    view = bridge_graph.without_nodes([3])
    bridge_list = view.bridges()
    bridge_set = {(min(u, v), max(u, v)) for u, v in bridge_list}
    assert (2, 3) not in bridge_set


# ── Articulation points ──


def test_ap_with_node_excluded(bridge_graph):
    """Excluding node 2 removes it from AP set."""
    view = bridge_graph.without_nodes([2])
    aps = view.articulation_points()
    assert 2 not in aps


# ── BFS ──


def test_bfs_skips_excluded_node(chain):
    """BFS from 1, excluding 3: should only reach 1,2."""
    view = chain.without_nodes([3])
    visited = view.bfs(1)
    assert set(visited) == {1, 2}


def test_bfs_excluded_source_raises(chain):
    """BFS from an excluded source should raise ValueError."""
    view = chain.without_nodes([1])
    with pytest.raises(ValueError, match="excluded"):
        view.bfs(1)


# ── Shortest path ──


def test_shortest_path_avoids_excluded_node(diamond):
    """Excluding node 1 forces path through node 2."""
    g, weights = diamond
    view = g.without_nodes([1])
    path = view.shortest_path(weights, 0, 3)
    assert 1 not in path
    assert path == [0, 2, 3]


def test_shortest_path_excluded_target_returns_empty(diamond):
    """Excluding the target returns empty path."""
    g, weights = diamond
    view = g.without_nodes([3])
    path = view.shortest_path(weights, 0, 3)
    assert path == []


# ── SSSP ──


def test_sssp_excludes_masked_nodes(diamond):
    g, weights = diamond
    view = g.without_nodes([1])
    lengths = view.shortest_path_lengths(weights, 0)
    assert 1 not in lengths
    assert 0 in lengths
    assert 2 in lengths
    assert 3 in lengths


# ── Multi-source Dijkstra ──


def test_msdijk_masked_source_ignored(diamond):
    g, weights = diamond
    view = g.without_nodes([0])
    lengths = view.multi_source_shortest_path_lengths(weights, [0, 1])
    assert 0 not in lengths
    assert 1 in lengths


# ── Chaining: edge mask + node mask ──


def test_combined_edge_and_node_mask(bridge_graph):
    """Exclude edge 3 (bridge 2-3) and node 4."""
    view = bridge_graph.without_edges([3]).without_nodes([4])
    comps = sorted(view.connected_components(), key=min)
    # Node 4 is excluded, edge 2-3 is excluded
    # Left: {0,1,2}, Right: {3,5} (connected via 5-3)
    assert {0, 1, 2} in comps
    assert {3, 5} in comps
    assert 4 not in {n for c in comps for n in c}


def test_chain_without_nodes_then_edges(chain):
    """Chainable: without_nodes then without_edges on the resulting view."""
    view = chain.without_nodes([3]).without_edges([0])
    # Exclude node 3 and edge 0 (1-2)
    comps = sorted(view.connected_components(), key=min)
    # Node 1 isolated, Node 2 isolated, {4,5} connected
    assert {1} in comps
    assert {2} in comps
    assert {4, 5} in comps


# ── Biconnected components ──


def test_bcc_with_node_excluded(bridge_graph):
    view = bridge_graph.without_nodes([2])
    bccs = list(view.biconnected_components())
    # Node 2 excluded: nodes 0,1 are connected only via edges through 2 which is masked
    # Left side: only edge 0-1 remains. Right side: 3-4-5-3 triangle
    node_sets = [frozenset(b) for b in bccs]
    assert frozenset({3, 4, 5}) in node_sets
