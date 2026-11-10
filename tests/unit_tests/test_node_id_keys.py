"""Every 64-bit integer is a valid node id, and ids are found whatever their bit pattern.

The C core maps node ids to node indices through a hash table. These tests pin
the ids its empty-slot marker or its slot function could get wrong.
"""

import itertools
import sys

import pytest

import pygraphc
from pygraphc import Graph


def test_node_id_minus_one_is_a_node():
    """-1 used to be the empty-slot marker, so an edge to it raised "unknown node ID"."""
    # --- Input ---
    # -1 -- 0 -- 1
    graph = Graph([-1, 0, 1], [(-1, 0), (0, 1)])

    # --- Assert ---
    assert list(graph.connected_components()) == [{-1, 0, 1}]
    assert graph.shortest_path([1.0, 1.0], -1, 1) == [-1, 0, 1]


def test_node_id_minus_one_in_the_free_functions():
    """The free functions parse node ids with the same table."""
    # --- Assert ---
    assert [sorted(component) for component in pygraphc.connected_components([5, -1], [(5, -1)])] == [[-1, 5]]


def test_duplicate_node_id_minus_one_is_rejected():
    """A stored -1 used to look like an empty slot, so its duplicate went unnoticed."""
    # --- Assert ---
    with pytest.raises(ValueError, match="duplicate node id -1"):
        Graph([-1, -1], [])


def test_duplicate_node_id_minus_one_is_rejected_for_split_lists():
    """The split-list parser shares the check."""
    # --- Assert ---
    with pytest.raises(ValueError, match="duplicate node id -1"):
        Graph([-1, 2, -1], [-1], [2])


def test_extreme_node_ids_are_nodes():
    """The smallest and the largest 64-bit id are keys like any other."""
    # --- Input ---
    smallest, largest = -sys.maxsize - 1, sys.maxsize
    graph = Graph([smallest, 0, largest], [(smallest, 0), (0, largest)])

    # --- Assert ---
    assert graph.bfs(smallest) == [smallest, 0, largest]


def test_node_ids_differing_only_in_high_bits_are_all_found():
    """Multiples of 2^20 share their low 20 bits; each one maps to its own node."""
    # --- Input ---
    # 0 -- 2^20 -- 2 * 2^20 -- ... -- 999 * 2^20
    node_ids = [index << 20 for index in range(1000)]
    graph = Graph(node_ids, list(itertools.pairwise(node_ids)))

    # --- Assert ---
    assert graph.bfs(node_ids[-1]) == node_ids[::-1]
