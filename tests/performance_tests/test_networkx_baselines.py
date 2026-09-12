"""networkx baselines for the operations the other performance tests leave uncovered.

Each test builds a sparse random graph, times pygraphc and the closest networkx
equivalent best-of-3, prints the ratio, asserts identical results and asserts
pygraphc is faster. The measured ratios back the README benchmark table.
"""

import random
import time
from collections.abc import Callable, Iterable
from itertools import pairwise
from types import ModuleType
from typing import TypeVar

import pytest

from pygraphc import (
    Graph,
    articulation_points,
    biconnected_components,
    connected_components,
    eccentricity,
    multi_source_shortest_path_lengths,
    nodes_on_simple_paths,
    shortest_path,
    two_edge_connected_components,
)

pytestmark = pytest.mark.performance

ResultType = TypeVar("ResultType")

BEST_OF = 3


def _best_of(operation: Callable[[], ResultType], runs: int = BEST_OF) -> tuple[ResultType, float]:
    """Run the operation once to warm up, then return its result and fastest of `runs` timings."""
    result = operation()
    fastest_seconds = float("inf")
    for _ in range(runs):
        start = time.perf_counter()
        result = operation()
        fastest_seconds = min(fastest_seconds, time.perf_counter() - start)
    return result, fastest_seconds


def _report(operation_name: str, size_label: str, pygraphc_seconds: float, networkx_seconds: float) -> float:
    speedup = networkx_seconds / pygraphc_seconds if pygraphc_seconds > 0 else float("inf")
    print(  # noqa: T201 — benchmark output is intentional
        f"\n  {operation_name} ({size_label}): "
        f"nx={networkx_seconds:.4f}s | pygraphc={pygraphc_seconds:.4f}s ({speedup:.1f}x)"
    )
    return speedup


def _sparse_graph(
    number_of_nodes: int, average_degree: int = 3, seed: int = 42
) -> tuple[list[int], list[tuple[int, int]]]:
    """Random sparse graph, roughly `average_degree` edges per node, not necessarily connected."""
    rng = random.Random(seed)
    number_of_edges = (number_of_nodes * average_degree) // 2
    edges: list[tuple[int, int]] = []
    for _ in range(number_of_edges):
        first = rng.randint(0, number_of_nodes - 1)
        second = rng.randint(0, number_of_nodes - 1)
        if first != second:
            edges.append((first, second))
    return list(range(number_of_nodes)), edges


def _connected_sparse_graph(
    number_of_nodes: int,
    average_degree: int = 3,
    seed: int = 42,
) -> tuple[list[int], list[tuple[int, int]]]:
    """Random spanning tree plus extra edges, so every node is reachable from every other.

    Parallel edges are dropped, which keeps the graph comparable to a networkx
    `Graph`, where a repeated node pair collapses into a single edge.
    """
    rng = random.Random(seed)
    edges: list[tuple[int, int]] = [(node_id, rng.randint(0, node_id - 1)) for node_id in range(1, number_of_nodes)]
    seen_pairs = {frozenset(edge) for edge in edges}
    extra_edges = max(0, (number_of_nodes * average_degree) // 2 - len(edges))
    for _ in range(extra_edges):
        first = rng.randint(0, number_of_nodes - 1)
        second = rng.randint(0, number_of_nodes - 1)
        if first != second and frozenset((first, second)) not in seen_pairs:
            edges.append((first, second))
            seen_pairs.add(frozenset((first, second)))
    return list(range(number_of_nodes)), edges


def _weights_for(edges: list[tuple[int, int]], seed: int = 7) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(0.1, 10.0) for _ in edges]


def _networkx_graph(
    networkx_module: ModuleType,
    node_ids: list[int],
    edges: list[tuple[int, int]],
    weights: list[float] | None = None,
) -> object:
    graph = networkx_module.Graph()
    graph.add_nodes_from(node_ids)
    if weights is None:
        graph.add_edges_from(edges)
    else:
        for (first, second), weight in zip(edges, weights, strict=True):
            graph.add_edge(first, second, weight=weight)
    return graph


def _as_frozensets(components: Iterable[Iterable[int]]) -> set[frozenset[int]]:
    return {frozenset(component) for component in components}


def _path_weight(path: list[int], edge_weights: dict[frozenset[int], float]) -> float:
    return sum(edge_weights[frozenset(step)] for step in pairwise(path))


# ── connected_components: the headline README claim ──


@pytest.mark.parametrize(
    ("exponent", "size_label"),
    [(4, "10K"), (5, "100K"), (6, "1M")],
    ids=["10K", "100K", "1M"],
)
def test_connected_components_vs_networkx(exponent: int, size_label: str) -> None:
    networkx = pytest.importorskip("networkx")
    node_ids, edges = _sparse_graph(10**exponent)
    graph = _networkx_graph(networkx, node_ids, edges)

    pygraphc_result, pygraphc_seconds = _best_of(lambda: list(connected_components(node_ids, edges)))
    networkx_result, networkx_seconds = _best_of(lambda: list(networkx.connected_components(graph)))

    assert _as_frozensets(pygraphc_result) == _as_frozensets(networkx_result)
    speedup = _report("connected_components", size_label, pygraphc_seconds, networkx_seconds)
    assert speedup > 1.0, f"pygraphc is not faster than networkx ({speedup:.2f}x)"


# ── articulation_points ──


def test_articulation_points_vs_networkx() -> None:
    networkx = pytest.importorskip("networkx")
    node_ids, edges = _sparse_graph(10**5)
    graph = _networkx_graph(networkx, node_ids, edges)

    pygraphc_result, pygraphc_seconds = _best_of(lambda: articulation_points(node_ids, edges))
    networkx_result, networkx_seconds = _best_of(lambda: set(networkx.articulation_points(graph)))

    assert pygraphc_result == networkx_result
    speedup = _report("articulation_points", "100K", pygraphc_seconds, networkx_seconds)
    assert speedup > 1.0, f"pygraphc is not faster than networkx ({speedup:.2f}x)"


# ── biconnected_components ──


def test_biconnected_components_vs_networkx() -> None:
    networkx = pytest.importorskip("networkx")
    node_ids, edges = _sparse_graph(10**5)
    graph = _networkx_graph(networkx, node_ids, edges)

    pygraphc_result, pygraphc_seconds = _best_of(lambda: list(biconnected_components(node_ids, edges)))
    networkx_result, networkx_seconds = _best_of(lambda: list(networkx.biconnected_components(graph)))

    assert _as_frozensets(pygraphc_result) == _as_frozensets(networkx_result)
    speedup = _report("biconnected_components", "100K", pygraphc_seconds, networkx_seconds)
    assert speedup > 1.0, f"pygraphc is not faster than networkx ({speedup:.2f}x)"


# ── shortest_path: the path itself, not just its length ──


def test_shortest_path_vs_networkx() -> None:
    """The one operation where networkx wins: `nx.shortest_path` searches from both ends.

    Against `nx.dijkstra_path`, the same one-directional Dijkstra pygraphc runs,
    pygraphc is far ahead. Against `nx.shortest_path`, which dispatches to
    bidirectional Dijkstra for a single source-target pair and therefore settles
    a small fraction of the nodes, pygraphc is behind. Both ratios are printed;
    the assertion guards the like-for-like one, and the second assertion pins
    the bidirectional gap so a further regression fails.
    """
    networkx = pytest.importorskip("networkx")
    number_of_nodes = 10**5
    node_ids, edges = _connected_sparse_graph(number_of_nodes)
    weights = _weights_for(edges)
    graph = _networkx_graph(networkx, node_ids, edges, weights)
    source, target = 0, number_of_nodes - 1
    edge_weights = {frozenset(edge): weight for edge, weight in zip(edges, weights, strict=True)}

    pygraphc_result, pygraphc_seconds = _best_of(lambda: shortest_path(node_ids, edges, weights, source, target))
    unidirectional_result, unidirectional_seconds = _best_of(
        lambda: networkx.dijkstra_path(graph, source, target, weight="weight")
    )
    bidirectional_result, bidirectional_seconds = _best_of(
        lambda: networkx.shortest_path(graph, source, target, weight="weight")
    )

    assert pygraphc_result[0] == source
    assert pygraphc_result[-1] == target
    pygraphc_weight = _path_weight(pygraphc_result, edge_weights)
    assert pygraphc_weight == pytest.approx(_path_weight(unidirectional_result, edge_weights))
    assert pygraphc_weight == pytest.approx(_path_weight(bidirectional_result, edge_weights)), (
        "pygraphc path is not a shortest path"
    )

    speedup = _report("shortest_path vs nx.dijkstra_path", "100K", pygraphc_seconds, unidirectional_seconds)
    bidirectional_speedup = _report(
        "shortest_path vs nx.shortest_path (bidirectional)", "100K", pygraphc_seconds, bidirectional_seconds
    )
    assert speedup > 1.0, f"pygraphc is not faster than nx.dijkstra_path ({speedup:.2f}x)"
    assert bidirectional_speedup > 0.05, (
        f"pygraphc fell further behind bidirectional Dijkstra ({bidirectional_speedup:.2f}x)"
    )


# ── multi_source_shortest_path_lengths ──


def test_multi_source_shortest_path_lengths_vs_networkx() -> None:
    networkx = pytest.importorskip("networkx")
    number_of_nodes = 10**5
    node_ids, edges = _connected_sparse_graph(number_of_nodes)
    weights = _weights_for(edges)
    graph = _networkx_graph(networkx, node_ids, edges, weights)
    sources = [0, number_of_nodes // 2, number_of_nodes - 1]

    pygraphc_result, pygraphc_seconds = _best_of(
        lambda: multi_source_shortest_path_lengths(node_ids, edges, weights, sources)
    )
    networkx_result, networkx_seconds = _best_of(
        lambda: networkx.multi_source_dijkstra_path_length(graph, sources, weight="weight")
    )

    assert set(pygraphc_result) == set(networkx_result)
    for node_id, length in pygraphc_result.items():
        assert length == pytest.approx(networkx_result[node_id])
    speedup = _report("multi_source_shortest_path_lengths", "100K", pygraphc_seconds, networkx_seconds)
    assert speedup > 1.0, f"pygraphc is not faster than networkx ({speedup:.2f}x)"


# ── eccentricity ──


def test_eccentricity_vs_networkx() -> None:
    """networkx `eccentricity` needs a connected graph, so the graph here is connected by construction.

    `nx.eccentricity(G, v=source, weight=...)` is a single weighted single-source
    search, the same work pygraphc does, so it is the fair baseline and no
    max-of-single-source substitute is needed.
    """
    networkx = pytest.importorskip("networkx")
    number_of_nodes = 10**5
    node_ids, edges = _connected_sparse_graph(number_of_nodes)
    weights = _weights_for(edges)
    graph = _networkx_graph(networkx, node_ids, edges, weights)
    source = 0

    pygraphc_result, pygraphc_seconds = _best_of(lambda: eccentricity(node_ids, edges, weights, source))
    networkx_result, networkx_seconds = _best_of(lambda: networkx.eccentricity(graph, v=source, weight="weight"))

    assert pygraphc_result == pytest.approx(networkx_result)
    speedup = _report("eccentricity", "100K", pygraphc_seconds, networkx_seconds)
    assert speedup > 1.0, f"pygraphc is not faster than networkx ({speedup:.2f}x)"


# ── two_edge_connected_components ──


def test_two_edge_connected_components_vs_networkx() -> None:
    networkx = pytest.importorskip("networkx")
    node_ids, edges = _connected_sparse_graph(10**4)
    graph = _networkx_graph(networkx, node_ids, edges)

    pygraphc_result, pygraphc_seconds = _best_of(lambda: list(two_edge_connected_components(node_ids, edges)))
    networkx_result, networkx_seconds = _best_of(lambda: list(networkx.k_edge_components(graph, 2)))

    assert _as_frozensets(pygraphc_result) == _as_frozensets(networkx_result)
    speedup = _report("two_edge_connected_components", "10K", pygraphc_seconds, networkx_seconds)
    assert speedup > 1.0, f"pygraphc is not faster than networkx ({speedup:.2f}x)"


# ── nodes_on_simple_paths ──


def test_nodes_on_simple_paths_vs_networkx() -> None:
    """Small graph on purpose: the networkx baseline enumerates every simple path, which is exponential."""
    networkx = pytest.importorskip("networkx")
    number_of_nodes = 24
    node_ids, edges = _connected_sparse_graph(number_of_nodes, average_degree=3, seed=11)
    graph = _networkx_graph(networkx, node_ids, edges)
    source = 0
    targets = [number_of_nodes - 1, number_of_nodes // 2]

    def networkx_union() -> set[int]:
        reached: set[int] = set()
        for target in targets:
            for path in networkx.all_simple_paths(graph, source, target):
                reached.update(path)
        return reached

    pygraphc_result, pygraphc_seconds = _best_of(lambda: nodes_on_simple_paths(node_ids, edges, source, targets))
    networkx_result, networkx_seconds = _best_of(networkx_union)

    assert pygraphc_result == networkx_result
    speedup = _report("nodes_on_simple_paths", "24", pygraphc_seconds, networkx_seconds)
    assert speedup > 1.0, f"pygraphc is not faster than networkx ({speedup:.2f}x)"


# ── masked connected components: the components of a graph with some edges taken out ──


def test_masked_connected_components_vs_networkx() -> None:
    """pygraphc masks the excluded edges on an existing graph; networkx needs the graph built without them."""
    networkx = pytest.importorskip("networkx")
    node_ids, edges = _sparse_graph(10**5)
    excluded_edge_indices = list(range(0, len(edges), len(edges) // 10))
    kept_edges = [edge for index, edge in enumerate(edges) if index not in set(excluded_edge_indices)]
    graph = Graph(node_ids, edges)
    networkx_graph = _networkx_graph(networkx, node_ids, kept_edges)

    pygraphc_result, pygraphc_seconds = _best_of(
        lambda: list(graph.without_edges(excluded_edge_indices).connected_components())
    )
    networkx_result, networkx_seconds = _best_of(lambda: list(networkx.connected_components(networkx_graph)))

    assert _as_frozensets(pygraphc_result) == _as_frozensets(networkx_result)
    speedup = _report("connected_components (masked)", "100K", pygraphc_seconds, networkx_seconds)
    assert speedup > 1.0, f"pygraphc is not faster than networkx ({speedup:.2f}x)"
