"""Tests for ``connected_component``, the single-component accessor on ``Graph`` and ``GraphView``.

The accessor is checked against the ``connected_components`` generator on
random multigraphs and masked views, and on the degenerate shapes: the empty
graph, a single node, one giant component and all singletons.
"""

import random

import pytest

from pygraphc import Graph

SEED = 20260911
PROPERTY_TRIALS = 300


def random_graph(rng: random.Random, node_count: int, edge_count: int) -> tuple[list[int], list[tuple[int, int]]]:
    """Node ids in random order and edges with self-loops and parallel edges."""
    nodes = rng.sample(range(1000), node_count)
    edges: list[tuple[int, int]] = []
    for _ in range(edge_count):
        from_node = rng.choice(nodes)
        roll = rng.random()
        if roll < 0.08:
            to_node = from_node
        elif roll < 0.25 and edges:
            from_node, to_node = edges[rng.randrange(len(edges))]
        else:
            to_node = rng.choice(nodes)
        edges.append((from_node, to_node))
    return nodes, edges


def random_view(rng: random.Random, nodes: list[int], edges: list[tuple[int, int]]):
    """The parsed graph, or a view of it with random excluded edges and nodes."""
    graph = Graph(nodes, edges)
    excluded_edges = [index for index in range(len(edges)) if rng.random() < 0.2]
    excluded_nodes = [node_id for node_id in nodes if rng.random() < 0.1]
    if not excluded_edges and not excluded_nodes:
        return graph
    return graph.without_edges(excluded_edges).without_nodes(excluded_nodes)


def test_connected_component_returns_the_component_of_the_node():
    """The accessor answers with the whole component, whichever member is asked."""
    # --- Input ---
    #  10 -- 20 -- 30    40 -- 50    60
    graph = Graph([10, 20, 30, 40, 50, 60], [(10, 20), (20, 30), (40, 50)])

    # --- Assert ---
    assert graph.connected_component(10) == {10, 20, 30}
    assert graph.connected_component(30) == {10, 20, 30}
    assert graph.connected_component(50) == {40, 50}
    assert graph.connected_component(60) == {60}


def test_connected_component_of_the_degenerate_shapes():
    """A single node is its own component; one giant component and all singletons
    are the two extremes of how much one answer costs."""
    # --- Input ---
    giant = Graph(list(range(50)), [(index, index + 1) for index in range(49)])
    singletons = Graph(list(range(50)), [])

    # --- Assert ---
    assert Graph([7], []).connected_component(7) == {7}
    assert giant.connected_component(25) == set(range(50))
    assert singletons.connected_component(25) == {25}


def test_connected_component_rejects_an_unknown_node_and_the_empty_graph():
    """A node that is not in the graph has no component, so the answer is an error, not an empty set."""
    # --- Assert ---
    with pytest.raises(ValueError, match="node 99 is not in the graph"):
        Graph([1, 2], [(1, 2)]).connected_component(99)
    with pytest.raises(ValueError, match="node 0 is not in the graph"):
        Graph([], []).connected_component(0)


def test_connected_component_rejects_a_directed_graph():
    """Like ``connected_components``, the accessor is undirected only."""
    # --- Assert ---
    with pytest.raises(TypeError, match="connected_component is not defined for directed graphs"):
        Graph([0, 1], [(0, 1)], directed=True).connected_component(0)


def test_connected_component_on_a_view_respects_both_masks():
    """An excluded edge stops connecting and an excluded node has no component at all."""
    # --- Input ---
    #  1 -- 2 -- 3 -- 4, edge index 1 (2--3) excluded, node 4 excluded
    graph = Graph([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4)])
    view = graph.without_edges([1]).without_nodes([4])

    # --- Assert ---
    assert view.connected_component(1) == {1, 2}
    assert view.connected_component(3) == {3}
    with pytest.raises(ValueError, match="node 4 is excluded from this view"):
        view.connected_component(4)


def test_connected_component_agrees_with_connected_components_on_random_multigraphs():
    """The accessor must name exactly the component the generator puts the node in,
    for every node of every graph and masked view."""
    rng = random.Random(SEED)
    for _ in range(PROPERTY_TRIALS):
        # --- Input ---
        nodes, edges = random_graph(rng, rng.randrange(1, 12), rng.randrange(0, 20))
        view = random_view(rng, nodes, edges)

        # --- Execute ---
        components = list(view.connected_components())
        component_of = {node_id: component for component in components for node_id in component}

        # --- Assert ---
        for node_id in nodes:
            if node_id in component_of:
                assert view.connected_component(node_id) == component_of[node_id]
            else:
                with pytest.raises(ValueError, match="excluded from this view"):
                    view.connected_component(node_id)
