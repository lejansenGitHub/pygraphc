"""Tests for Phase 6c: All-edge-paths enumeration."""

import pytest

from cgraph import Graph


@pytest.fixture
def simple_chain():
    """1-2-3-4, edges: 0:(1,2), 1:(2,3), 2:(3,4)."""
    return Graph([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4)])


@pytest.fixture
def diamond():
    """Diamond: 0-1, 0-2, 1-3, 2-3. Edges 0-3."""
    return Graph([0, 1, 2, 3], [(0, 1), (0, 2), (1, 3), (2, 3)])


@pytest.fixture
def parallel_edges():
    """Multigraph: 1-2 (e0), 2-3 (e1), 2-3 (e2), 3-4 (e3)."""
    return Graph([1, 2, 3, 4], [(1, 2), (2, 3), (2, 3), (3, 4)])


# ── Basic functionality ──


def test_single_path(simple_chain):
    paths = simple_chain.all_edge_paths(1, 4)
    assert paths == [[0, 1, 2]]


def test_no_path():
    g = Graph([1, 2, 3], [(1, 2)])
    paths = g.all_edge_paths(1, 3)
    assert paths == []


def test_source_is_target():
    """Source == target: no edges traversed, no paths returned."""
    g = Graph([1, 2], [(1, 2)])
    paths = g.all_edge_paths(1, 1)
    # Source == target with no edges used means nothing to yield
    assert paths == []


def test_diamond_two_paths(diamond):
    paths = diamond.all_edge_paths(0, 3)
    path_sets = [tuple(p) for p in paths]
    assert sorted(path_sets) == [(0, 2), (1, 3)]


def test_single_target_as_int(diamond):
    """Target can be a single int instead of a collection."""
    paths = diamond.all_edge_paths(0, 3)
    assert len(paths) == 2


def test_multiple_targets(diamond):
    """Paths to either target 1 or 3."""
    paths = diamond.all_edge_paths(0, [1, 3])
    # Should include paths to node 1 (edge 0) and paths to node 3 (edges 0+2, 1+3)
    assert len(paths) >= 3


# ── Cutoff ──


def test_cutoff_limits_path_length(diamond):
    paths = diamond.all_edge_paths(0, 3, cutoff=1)
    # No path of length 1 reaches node 3 from 0
    assert paths == []


def test_cutoff_exact_length(diamond):
    paths = diamond.all_edge_paths(0, 3, cutoff=2)
    # Both paths are length 2, so they should be found
    assert len(paths) == 2


# ── Multigraph paths ──


def test_parallel_edges_give_distinct_paths(parallel_edges):
    paths = parallel_edges.all_edge_paths(1, 4)
    # Two paths: e0-e1-e3 and e0-e2-e3 (using different parallel edges)
    assert len(paths) == 2
    path_tuples = sorted(tuple(p) for p in paths)
    assert path_tuples == [(0, 1, 3), (0, 2, 3)]


# ── Edge masks ──


def test_edge_mask_blocks_path(simple_chain):
    view = simple_chain.without_edges([1])  # remove edge 2-3
    paths = view.all_edge_paths(1, 4)
    assert paths == []


def test_edge_mask_one_parallel(parallel_edges):
    """Mask one parallel edge, only one path remains."""
    view = parallel_edges.without_edges([1])
    paths = view.all_edge_paths(1, 4)
    assert len(paths) == 1
    assert paths[0] == [0, 2, 3]


# ── Node masks ──


def test_node_mask_blocks_path(simple_chain):
    view = simple_chain.without_nodes([3])  # remove node 3
    paths = view.all_edge_paths(1, 4)
    assert paths == []


def test_node_mask_forces_alternate(diamond):
    """Excluding node 1 forces path through node 2."""
    view = diamond.without_nodes([1])
    paths = view.all_edge_paths(0, 3)
    assert len(paths) == 1
    assert paths[0] == [1, 3]  # edges (0,2) and (2,3)


def test_node_mask_excluded_source_raises(simple_chain):
    view = simple_chain.without_nodes([1])
    with pytest.raises(ValueError, match="excluded"):
        view.all_edge_paths(1, 4)


# ── Edge-disjoint: node revisits allowed ──


def test_node_revisit_via_different_edges():
    """Triangle 1-2-3-1 + edge 3-4. Path 1->2->3->1->... uses node 1 twice."""
    g = Graph([1, 2, 3, 4], [(1, 2), (2, 3), (3, 1), (1, 4)])
    # Path using edges: 0(1-2), 1(2-3), 2(3-1), 3(1-4)
    paths = g.all_edge_paths(1, 4)
    # Direct path: [3] (edge 1-4)
    # Long path: [0, 1, 2, 3] (1->2->3->1->4)
    assert [3] in paths
    assert [0, 1, 2, 3] in paths


# ── Empty and edge cases ──


def test_empty_graph():
    g = Graph([], [])
    with pytest.raises(ValueError):
        g.all_edge_paths(1, 2)


def test_single_node_no_edges():
    g = Graph([1], [])
    paths = g.all_edge_paths(1, 1)
    assert paths == []


def test_cycle_graph():
    """Cycle: 1-2-3-4-1. Paths from 1 to 3."""
    g = Graph([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4), (4, 1)])
    paths = g.all_edge_paths(1, 3)
    path_tuples = sorted(tuple(p) for p in paths)
    # Path via 1->2->3: [0, 1]
    # Path via 1->4->3: [3, 2]
    assert len(path_tuples) == 2


# ── Combined masks ──


def test_combined_edge_and_node_mask(diamond):
    """Exclude node 2 and edge 0 (0-1)."""
    view = diamond.without_nodes([2]).without_edges([0])
    paths = view.all_edge_paths(0, 3)
    assert paths == []  # no way to reach 3
