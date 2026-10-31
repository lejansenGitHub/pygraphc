"""Differential checks against networkx on seeded random graphs.

Moved out of the performance suite, where these equalities rode along with
walltime races: the race is gone, the cross-check against an independent
implementation stays. Sizes are moderate so the unit suite stays fast.
"""

import random
from collections.abc import Iterable
from itertools import pairwise

import pytest

from pygraphc import (
    Graph,
    articulation_points,
    biconnected_components,
    connected_components,
    eccentricity,
    multi_source_shortest_path_lengths,
    nodes_on_simple_paths,
    two_edge_connected_components,
)
from pygraphc.reduction import MultiGraph, Partition, reduce

networkx = pytest.importorskip("networkx")

NODE_COUNT = 5_000


def _sparse_graph(node_count: int, average_degree: int = 3, seed: int = 42) -> tuple[list[int], list[tuple[int, int]]]:
    """Random pairs without self-loops, parallel edges kept: not necessarily connected."""
    rng = random.Random(seed)
    edges = []
    for _ in range(node_count * average_degree // 2):
        first, second = rng.randrange(node_count), rng.randrange(node_count)
        if first != second:
            edges.append((first, second))
    return list(range(node_count)), edges


def _connected_simple_graph(
    node_count: int, average_degree: int = 3, seed: int = 42
) -> tuple[list[int], list[tuple[int, int]]]:
    """Random spanning tree plus extra edges, no parallel edges, so a networkx ``Graph`` holds the same edges."""
    rng = random.Random(seed)
    edges = [(node_id, rng.randrange(node_id)) for node_id in range(1, node_count)]
    seen_pairs = {frozenset(edge) for edge in edges}
    for _ in range(max(0, node_count * average_degree // 2 - len(edges))):
        first, second = rng.randrange(node_count), rng.randrange(node_count)
        if first != second and frozenset((first, second)) not in seen_pairs:
            edges.append((first, second))
            seen_pairs.add(frozenset((first, second)))
    return list(range(node_count)), edges


def _weights(edges: list[tuple[int, int]], seed: int = 7) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(0.1, 10.0) for _ in edges]


def _networkx_graph(
    node_ids: list[int], edges: list[tuple[int, int]], weights: list[float] | None = None
) -> "networkx.Graph":
    graph = networkx.Graph()
    graph.add_nodes_from(node_ids)
    for index, (first, second) in enumerate(edges):
        graph.add_edge(first, second, weight=1.0 if weights is None else weights[index])
    return graph


def _as_frozensets(components: Iterable[Iterable[int]]) -> set[frozenset[int]]:
    return {frozenset(component) for component in components}


# ── Connectivity ──


def test_connected_components_match_networkx() -> None:
    """Components depend only on which nodes are joined, so both libraries must agree exactly."""
    node_ids, edges = _sparse_graph(NODE_COUNT)

    expected = networkx.connected_components(_networkx_graph(node_ids, edges))

    assert _as_frozensets(connected_components(node_ids, edges)) == _as_frozensets(expected)


def test_masked_connected_components_match_networkx_without_the_masked_edges() -> None:
    """Masking edges on a parsed graph must equal building the graph without them."""
    node_ids, edges = _sparse_graph(NODE_COUNT)
    excluded = set(range(0, len(edges), len(edges) // 10))
    kept_edges = [edge for index, edge in enumerate(edges) if index not in excluded]

    masked = Graph(node_ids, edges).without_edges(sorted(excluded)).connected_components()

    assert _as_frozensets(masked) == _as_frozensets(
        networkx.connected_components(_networkx_graph(node_ids, kept_edges))
    )


def test_articulation_points_match_networkx() -> None:
    """Cut vertices are a property of the simple graph; parallel edges do not change them."""
    node_ids, edges = _sparse_graph(NODE_COUNT)

    expected = set(networkx.articulation_points(_networkx_graph(node_ids, edges)))

    assert articulation_points(node_ids, edges) == expected


def test_biconnected_components_match_networkx() -> None:
    """Blocks as node sets agree whenever the graph has no parallel edges that would form a block of their own."""
    node_ids, edges = _connected_simple_graph(NODE_COUNT)

    expected = networkx.biconnected_components(_networkx_graph(node_ids, edges))

    assert _as_frozensets(biconnected_components(node_ids, edges)) == _as_frozensets(expected)


def test_two_edge_connected_components_match_networkx() -> None:
    """Removing every bridge and taking components is what networkx computes as 2-edge components."""
    node_ids, edges = _connected_simple_graph(NODE_COUNT)

    expected = networkx.k_edge_components(_networkx_graph(node_ids, edges), 2)

    assert _as_frozensets(two_edge_connected_components(node_ids, edges)) == _as_frozensets(expected)


def test_nodes_on_simple_paths_match_the_union_of_networkx_simple_paths() -> None:
    """Small on purpose: the networkx reference enumerates every simple path, which is exponential."""
    node_ids, edges = _connected_simple_graph(24, seed=11)
    graph = _networkx_graph(node_ids, edges)
    targets = [23, 12]

    expected = {
        node_id for target in targets for path in networkx.all_simple_paths(graph, 0, target) for node_id in path
    }

    assert nodes_on_simple_paths(node_ids, edges, 0, targets) == expected


# ── Weighted paths ──


def test_shortest_path_has_the_networkx_shortest_weight() -> None:
    """Shortest paths may differ on ties, their total weight may not."""
    node_ids, edges = _connected_simple_graph(NODE_COUNT)
    weights = _weights(edges)
    edge_weight = {frozenset(edge): weight for edge, weight in zip(edges, weights, strict=True)}
    source, target = 0, NODE_COUNT - 1

    path = Graph(node_ids, edges).shortest_path(weights, source, target)
    expected = networkx.dijkstra_path_length(_networkx_graph(node_ids, edges, weights), source, target)

    # --- Assert ---
    assert path[0] == source
    assert path[-1] == target
    assert sum(edge_weight[frozenset(step)] for step in pairwise(path)) == pytest.approx(expected)


def test_multi_source_shortest_path_lengths_match_networkx() -> None:
    """Distance to the nearest source is unique even when paths tie."""
    node_ids, edges = _connected_simple_graph(NODE_COUNT)
    weights = _weights(edges)
    sources = [0, NODE_COUNT // 2, NODE_COUNT - 1]

    lengths = multi_source_shortest_path_lengths(node_ids, edges, weights, sources)
    expected = networkx.multi_source_dijkstra_path_length(_networkx_graph(node_ids, edges, weights), sources)

    # --- Assert ---
    assert set(lengths) == set(expected)
    assert all(length == pytest.approx(expected[node_id]) for node_id, length in lengths.items())


def test_eccentricity_matches_networkx() -> None:
    """Eccentricity is the largest shortest distance, unique for a connected graph."""
    node_ids, edges = _connected_simple_graph(NODE_COUNT)
    weights = _weights(edges)

    expected = networkx.eccentricity(_networkx_graph(node_ids, edges, weights), v=0, weight="weight")

    assert eccentricity(node_ids, edges, weights, 0) == pytest.approx(expected)


# ── Reduction ──


def _networkx_reduce(graph: MultiGraph[int], terminals: set[int]) -> set[int]:
    """The three moves as a networkx caller writes them: full sweeps to a fixpoint; returns the surviving nodes."""
    nx_graph = networkx.MultiGraph()
    nx_graph.add_nodes_from(graph.nodes)
    for edge_id, (from_node, to_node) in graph.endpoints.items():
        if from_node != to_node:
            nx_graph.add_edge(from_node, to_node, key=edge_id)
    for component in list(networkx.connected_components(nx_graph)):
        if not component & terminals:
            nx_graph.remove_nodes_from(component)
    changed = True
    while changed:
        changed = False
        for node_id in list(nx_graph.nodes):
            if node_id in terminals or node_id not in nx_graph:
                continue
            degree = nx_graph.degree(node_id)
            if degree == 1:
                nx_graph.remove_node(node_id)
                changed = True
            elif degree == 2:
                neighbours = list(nx_graph.neighbors(node_id))
                if len(neighbours) == 2:
                    networkx.contracted_nodes(nx_graph, neighbours[0], node_id, self_loops=False, copy=False)
                    changed = True
        for from_node, to_node in {(min(pair), max(pair)) for pair in nx_graph.edges(keys=False)}:
            for key in list(nx_graph[from_node][to_node])[1:]:
                nx_graph.remove_edge(from_node, to_node, key=key)
                changed = True
    return set(nx_graph.nodes)


def _sparse_multigraph(node_count: int, edge_count: int, seed: int) -> MultiGraph[int]:
    rng = random.Random(seed)
    endpoints = {edge_id: (rng.randrange(node_count), rng.randrange(node_count)) for edge_id in range(edge_count)}
    return MultiGraph(list(range(node_count)), endpoints)


def test_reduction_residual_matches_networkx_sweeps() -> None:
    """The closure of the three moves is unique without protected nodes, so any correct sweep reaches the same nodes."""
    graph = _sparse_multigraph(2_000, edge_count=2_500, seed=42)
    terminals = set(random.Random(7).sample(graph.nodes, 20))

    residual_nodes = set(reduce(graph, terminals).graph.nodes)

    # --- Assert ---
    assert terminals <= residual_nodes
    assert residual_nodes == _networkx_reduce(graph, terminals)


def test_label_partition_equals_the_set_partition() -> None:
    """The C label kernel and one Python set per component describe the same blocks."""
    graph = _sparse_multigraph(NODE_COUNT, edge_count=NODE_COUNT * 5 // 4, seed=42)
    active = set(graph.endpoints)
    kernel_graph = Graph(graph.nodes, list(graph.endpoints.values()))

    from_sets = Partition.from_groups(kernel_graph.connected_components())

    assert Partition.from_components(graph, active).block_of == from_sets.block_of
