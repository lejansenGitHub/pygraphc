"""Tests for the Graph class — ensures parity with free-function API."""

import pytest

from pygraphc import Graph

# ── Shared fixtures ──


@pytest.fixture
def triangle():
    """Simple triangle graph: 0-1-2-0."""
    return Graph([0, 1, 2], [(0, 1), (1, 2), (2, 0)])


@pytest.fixture
def bridge_graph():
    """Graph with a bridge: (0-1-2) -- bridge(2,3) -- (3-4-5)."""
    nodes = [0, 1, 2, 3, 4, 5]
    edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)]
    return Graph(nodes, edges)


@pytest.fixture
def weighted_graph():
    """Weighted diamond: 0-1(1), 0-2(4), 1-3(2), 2-3(1)."""
    nodes = [0, 1, 2, 3]
    edges = [(0, 1), (0, 2), (1, 3), (2, 3)]
    weights = [1.0, 4.0, 2.0, 1.0]
    return Graph(nodes, edges), weights


# ── Connected Components ──


def test_cc_single_component(triangle):
    comps = list(triangle.connected_components())
    assert comps == [{0, 1, 2}]


def test_cc_two_components():
    g = Graph([0, 1, 2, 3], [(0, 1), (2, 3)])
    comps = sorted(g.connected_components(), key=min)
    assert comps == [{0, 1}, {2, 3}]


def test_cc_empty():
    g = Graph([], [])
    assert list(g.connected_components()) == []


def test_cc_isolated_nodes():
    g = Graph([0, 1, 2], [])
    comps = sorted(g.connected_components(), key=min)
    assert comps == [{0}, {1}, {2}]


# ── Bridges ──


def test_bridges(bridge_graph):
    b = bridge_graph.bridges()
    normed = {(min(u, v), max(u, v)) for u, v in b}
    assert normed == {(2, 3)}


def test_bridges_none(triangle):
    assert triangle.bridges() == []


# ── Articulation Points ──


def test_ap(bridge_graph):
    ap = bridge_graph.articulation_points()
    assert ap == {2, 3}


def test_ap_none(triangle):
    assert triangle.articulation_points() == set()


# ── Biconnected Components ──


def test_bcc(bridge_graph):
    blocks = [frozenset(b) for b in bridge_graph.biconnected_components()]
    assert frozenset({0, 1, 2}) in blocks
    assert frozenset({3, 4, 5}) in blocks


def test_bcc_triangle(triangle):
    blocks = list(triangle.biconnected_components())
    assert len(blocks) == 1
    assert blocks[0] == {0, 1, 2}


# ── BFS ──


def test_bfs(triangle):
    visited = triangle.bfs(0)
    assert set(visited) == {0, 1, 2}
    assert visited[0] == 0


def test_bfs_disconnected():
    g = Graph([0, 1, 2, 3], [(0, 1), (2, 3)])
    visited = g.bfs(0)
    assert set(visited) == {0, 1}


# ── Shortest Path ──


def test_shortest_path(weighted_graph):
    g, w = weighted_graph
    path = g.shortest_path(w, 0, 3)
    assert path == [0, 1, 3]


def test_shortest_path_same_node(weighted_graph):
    g, w = weighted_graph
    path = g.shortest_path(w, 0, 0)
    assert path == [0]


# ── SSSP Lengths ──


def test_sssp(weighted_graph):
    g, w = weighted_graph
    lengths = g.shortest_path_lengths(w, 0)
    assert lengths[0] == 0.0
    assert lengths[1] == 1.0
    assert lengths[3] == 3.0  # 0->1->3


def test_sssp_cutoff(weighted_graph):
    g, w = weighted_graph
    lengths = g.shortest_path_lengths(w, 0, cutoff=2.0)
    assert 0 in lengths
    assert 1 in lengths
    assert 3 not in lengths  # distance 3.0 > cutoff 2.0


# ── Multi-Source Dijkstra ──


def test_multi_source(weighted_graph):
    g, w = weighted_graph
    lengths = g.multi_source_shortest_path_lengths(w, [0, 3])
    assert lengths[0] == 0.0
    assert lengths[3] == 0.0
    assert lengths[1] == min(1.0, 2.0)  # min(0->1, 3->1)


# ── Eccentricity ──


def test_eccentricity(weighted_graph):
    g, w = weighted_graph
    e = g.eccentricity(w, 0)
    assert e == pytest.approx(4.0)  # 0->2 costs 4.0


# ── Two-Edge-Connected Components ──


def test_two_edge_cc(bridge_graph):
    comps = sorted(bridge_graph.two_edge_connected_components(), key=min)
    normed = [frozenset(c) for c in comps]
    assert frozenset({0, 1, 2}) in normed
    assert frozenset({3, 4, 5}) in normed


# ── Nodes on Simple Paths ──


def test_nodes_on_simple_paths(bridge_graph):
    result = bridge_graph.nodes_on_simple_paths(0, [5])
    assert result == {0, 1, 2, 3, 4, 5}


def test_nodes_on_simple_paths_same_component(triangle):
    result = triangle.nodes_on_simple_paths(0, [2])
    assert result == {0, 1, 2}


# ── Split-list constructor ──


def test_split_list_constructor():
    g = Graph([0, 1, 2, 3], [0, 2], [1, 3])
    comps = sorted(g.connected_components(), key=min)
    assert comps == [{0, 1}, {2, 3}]


# ── Multiple algorithms on same graph (the key use case) ──


def test_multiple_algorithms_same_graph(bridge_graph):
    """The main value prop: parse once, run many algorithms."""
    bridges_result = bridge_graph.bridges()
    ap_result = bridge_graph.articulation_points()
    comps = list(bridge_graph.connected_components())
    bfs_result = bridge_graph.bfs(0)

    assert len(bridges_result) == 1
    assert ap_result == {2, 3}
    assert len(comps) == 1
    assert set(bfs_result) == {0, 1, 2, 3, 4, 5}


def test_large_node_ids():
    """Ensure arbitrary (non-contiguous) node IDs work."""
    nids = [100, 200, 300, 400]
    edges = [(100, 200), (300, 400)]
    g = Graph(nids, edges)
    comps = sorted(g.connected_components(), key=min)
    assert comps == [{100, 200}, {300, 400}]
    assert g.bridges() == [(100, 200), (300, 400)] or {(min(u, v), max(u, v)) for u, v in g.bridges()} == {
        (100, 200),
        (300, 400),
    }
