"""Speedups over networkx for the README tables, measured and printed, never asserted.

Algorithm time only, both sides prepared before the timer, best of 3 after one
warm-up. Results are checked against networkx by the differential unit tests
(``tests/unit_tests/test_networkx_differential.py``); this script only reports.

    python benchmarks/networkx_baselines.py
"""

import array
import random
import time
from collections.abc import Callable

import networkx

from pygraphc import Graph

Edges = list[tuple[int, int]]


def sparse_graph(node_count: int, seed: int = 42) -> tuple[list[int], Edges]:
    """Random pairs without self-loops, about 1.5 edges per node."""
    rng = random.Random(seed)
    edges = []
    for _ in range(node_count * 3 // 2):
        first, second = rng.randrange(node_count), rng.randrange(node_count)
        if first != second:
            edges.append((first, second))
    return list(range(node_count)), edges


def connected_simple_graph(node_count: int, seed: int = 42) -> tuple[list[int], Edges]:
    """Random spanning tree plus extra edges without parallels, so a networkx ``Graph`` holds the same edges."""
    rng = random.Random(seed)
    edges = [(node_id, rng.randrange(node_id)) for node_id in range(1, node_count)]
    seen_pairs = {frozenset(edge) for edge in edges}
    for _ in range(max(0, node_count * 3 // 2 - len(edges))):
        first, second = rng.randrange(node_count), rng.randrange(node_count)
        if first != second and frozenset((first, second)) not in seen_pairs:
            edges.append((first, second))
            seen_pairs.add(frozenset((first, second)))
    return list(range(node_count)), edges


def directed_acyclic(node_count: int, seed: int = 42) -> tuple[list[int], Edges]:
    rng = random.Random(seed)
    edges = []
    for _ in range(node_count * 3):
        first, second = rng.randrange(node_count), rng.randrange(node_count)
        if first != second:
            edges.append((min(first, second), max(first, second)))
    return list(range(node_count)), edges


def weights_for(edges: Edges) -> list[float]:
    rng = random.Random(7)
    return [rng.uniform(0.1, 10.0) for _ in edges]


def networkx_graph(node_ids: list[int], edges: Edges, weights: list[float] | None = None, *, directed: bool = False):
    graph = networkx.DiGraph() if directed else networkx.Graph()
    graph.add_nodes_from(node_ids)
    for index, (first, second) in enumerate(edges):
        graph.add_edge(first, second, weight=1.0 if weights is None else weights[index])
    return graph


def best_of(operation: Callable[[], object], runs: int = 3) -> float:
    operation()
    fastest = float("inf")
    for _ in range(runs):
        start = time.perf_counter()
        operation()
        fastest = min(fastest, time.perf_counter() - start)
    return fastest


def report(
    name: str, size_label: str, pygraphc_operation: Callable[[], object], networkx_operation: Callable[[], object]
) -> None:
    pygraphc_seconds = best_of(pygraphc_operation)
    networkx_seconds = best_of(networkx_operation)
    speedup = networkx_seconds / pygraphc_seconds
    print(f"| {name} | {size_label} | {pygraphc_seconds:.4g}s | {networkx_seconds:.4g}s | **{speedup:.3g}x** |")  # noqa: T201 — the report is the script's output


def main() -> None:
    print("| Algorithm | Nodes | pygraphc | networkx | Speedup |")  # noqa: T201 — the report is the script's output
    print("|-----------|------:|-------:|---------:|--------:|")  # noqa: T201 — the report is the script's output
    for node_count, label in [(1_000, "1K"), (10_000, "10K"), (100_000, "100K"), (1_000_000, "1M")]:
        node_ids, edges = sparse_graph(node_count)
        graph, nx_graph = Graph(node_ids, edges), networkx_graph(node_ids, edges)
        report(
            "Connected Components",
            label,
            lambda graph=graph: list(graph.connected_components()),
            lambda nx_graph=nx_graph: list(networkx.connected_components(nx_graph)),
        )

    node_ids, edges = sparse_graph(1_000_000)
    graph, nx_graph = Graph(node_ids, edges), networkx_graph(node_ids, edges)
    weight_buffer = array.array("d", weights_for(edges))
    nx_weighted = networkx_graph(node_ids, edges, list(weight_buffer))
    report("Bridges", "1M", graph.bridges, lambda: list(networkx.bridges(nx_graph)))
    report("Articulation Points", "1M", graph.articulation_points, lambda: set(networkx.articulation_points(nx_graph)))
    report("BFS", "1M", lambda: graph.bfs(0), lambda: list(networkx.bfs_tree(nx_graph, 0)))
    report(
        "Dijkstra (single-source lengths)",
        "1M",
        lambda: graph.shortest_path_lengths(weight_buffer, 0),
        lambda: networkx.single_source_dijkstra_path_length(nx_weighted, 0),
    )

    node_ids, edges = directed_acyclic(1_000_000)
    directed, nx_directed = Graph(node_ids, edges, directed=True), networkx_graph(node_ids, edges, directed=True)
    report(
        "SCC (directed)",
        "1M",
        lambda: list(directed.strongly_connected_components()),
        lambda: list(networkx.kosaraju_strongly_connected_components(nx_directed)),
    )
    report(
        "WCC (directed)",
        "1M",
        lambda: list(directed.weakly_connected_components()),
        lambda: list(networkx.weakly_connected_components(nx_directed)),
    )
    report("Topological sort", "1M", directed.topological_sort, lambda: list(networkx.topological_sort(nx_directed)))
    report("DAG longest path", "1M", directed.dag_longest_path, lambda: networkx.dag_longest_path(nx_directed))

    node_ids, edges = sparse_graph(100_000)
    graph, nx_graph = Graph(node_ids, edges), networkx_graph(node_ids, edges)
    report("Cycle basis", "100K", graph.cycle_basis, lambda: networkx.cycle_basis(nx_graph))
    report(
        "Articulation Points", "100K", graph.articulation_points, lambda: set(networkx.articulation_points(nx_graph))
    )
    report(
        "Biconnected Components",
        "100K",
        lambda: list(graph.biconnected_components()),
        lambda: list(networkx.biconnected_components(nx_graph)),
    )
    excluded = list(range(0, len(edges), len(edges) // 10))
    kept = [edge for index, edge in enumerate(edges) if index not in set(excluded)]
    nx_kept = networkx_graph(node_ids, kept)
    report(
        "Connected Components, edge-masked view",
        "100K",
        lambda: list(graph.without_edges(excluded).connected_components()),
        lambda: list(networkx.connected_components(nx_kept)),
    )

    node_ids, edges = connected_simple_graph(100_000)
    weights = weights_for(edges)
    weight_buffer = array.array("d", weights)
    graph, nx_weighted = Graph(node_ids, edges), networkx_graph(node_ids, edges, weights)
    sources = [0, 50_000, 99_999]
    report(
        "Multi-source Dijkstra lengths",
        "100K",
        lambda: graph.multi_source_shortest_path_lengths(weight_buffer, sources),
        lambda: networkx.multi_source_dijkstra_path_length(nx_weighted, sources),
    )
    report(
        "Eccentricity (weighted)",
        "100K",
        lambda: graph.eccentricity(weight_buffer, 0),
        lambda: networkx.eccentricity(nx_weighted, v=0, weight="weight"),
    )
    report(
        "Single-pair `shortest_path` vs `nx.dijkstra_path`",
        "100K",
        lambda: graph.shortest_path(weight_buffer, 0, 99_999),
        lambda: networkx.dijkstra_path(nx_weighted, 0, 99_999),
    )
    report(
        "Single-pair `shortest_path` vs `nx.shortest_path`",
        "100K",
        lambda: graph.shortest_path(weight_buffer, 0, 99_999),
        lambda: networkx.shortest_path(nx_weighted, 0, 99_999, weight="weight"),
    )
    report(
        "Shortest path, single pair, `Graph` + list weights",
        "100K",
        lambda: graph.shortest_path(weights, 0, 99_999),
        lambda: networkx.shortest_path(nx_weighted, 0, 99_999, weight="weight"),
    )
    report(
        "the same, both sides building their graph inside the timer",
        "100K",
        lambda: Graph(node_ids, edges).shortest_path(weight_buffer, 0, 99_999),
        lambda: networkx.shortest_path(networkx_graph(node_ids, edges, weights), 0, 99_999, weight="weight"),
    )

    node_ids, edges = connected_simple_graph(10_000)
    graph, nx_graph = Graph(node_ids, edges), networkx_graph(node_ids, edges)
    report(
        "Two-edge-connected components",
        "10K",
        lambda: list(graph.two_edge_connected_components()),
        lambda: list(networkx.k_edge_components(nx_graph, 2)),
    )

    node_ids, edges = connected_simple_graph(24, seed=11)
    graph, nx_graph = Graph(node_ids, edges), networkx_graph(node_ids, edges)
    report(
        "`nodes_on_simple_paths`",
        "24",
        lambda: graph.nodes_on_simple_paths(0, [23, 12]),
        lambda: {
            node_id
            for target in (23, 12)
            for path in networkx.all_simple_paths(nx_graph, 0, target)
            for node_id in path
        },
    )


if __name__ == "__main__":
    main()
