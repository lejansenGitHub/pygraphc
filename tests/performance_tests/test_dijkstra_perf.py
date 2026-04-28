"""Performance tests for Dijkstra / shortest path algorithms."""

import random
import time

import pytest

from pygraphc import multi_source_shortest_path_lengths, shortest_path, shortest_path_lengths

pytestmark = pytest.mark.performance


def generate_weighted_graph(
    n: int,
    avg_degree: int = 3,
    seed: int = 42,
) -> tuple[list[tuple[int, int]], list[float]]:
    rng = random.Random(seed)
    m = (n * avg_degree) // 2
    edges: list[tuple[int, int]] = []
    weights: list[float] = []
    for _ in range(m):
        a = rng.randint(0, n - 1)
        b = rng.randint(0, n - 1)
        if a != b:
            edges.append((a, b))
            weights.append(rng.uniform(0.1, 10.0))
    return edges, weights


@pytest.mark.parametrize(
    ("exponent", "time_limit_seconds"),
    [(3, 0.005), (4, 0.03), (5, 0.5), (6, 6), (7, 80)],
    ids=["1K", "10K", "100K", "1M", "10M"],
)
def test_shortest_path_performance(exponent: int, time_limit_seconds: float) -> None:
    n = 10**exponent
    node_ids = list(range(n))
    edges, weights = generate_weighted_graph(n)

    start = time.perf_counter()
    shortest_path(node_ids, edges, weights, 0, n - 1)
    elapsed = time.perf_counter() - start

    assert elapsed < time_limit_seconds, f"took {elapsed:.3f}s, limit {time_limit_seconds}s"


@pytest.mark.parametrize(
    ("exponent", "time_limit_seconds"),
    [(3, 0.005), (4, 0.03), (5, 0.5), (6, 6), (7, 80)],
    ids=["1K", "10K", "100K", "1M", "10M"],
)
def test_sssp_lengths_performance(exponent: int, time_limit_seconds: float) -> None:
    n = 10**exponent
    node_ids = list(range(n))
    edges, weights = generate_weighted_graph(n)

    start = time.perf_counter()
    result = shortest_path_lengths(node_ids, edges, weights, 0)
    elapsed = time.perf_counter() - start

    assert len(result) > 0
    assert elapsed < time_limit_seconds, f"took {elapsed:.3f}s, limit {time_limit_seconds}s"


@pytest.mark.parametrize(
    ("exponent", "time_limit_seconds"),
    [(3, 0.005), (4, 0.03), (5, 0.5), (6, 6), (7, 80)],
    ids=["1K", "10K", "100K", "1M", "10M"],
)
def test_multi_source_dijkstra_performance(exponent: int, time_limit_seconds: float) -> None:
    n = 10**exponent
    node_ids = list(range(n))
    edges, weights = generate_weighted_graph(n)

    start = time.perf_counter()
    result = multi_source_shortest_path_lengths(
        node_ids,
        edges,
        weights,
        [0, n // 2, n - 1],
    )
    elapsed = time.perf_counter() - start

    assert len(result) > 0
    assert elapsed < time_limit_seconds, f"took {elapsed:.3f}s, limit {time_limit_seconds}s"


@pytest.mark.parametrize(
    "exponent",
    [3, 4, 5, 6],
    ids=["1K", "10K", "100K", "1M"],
)
def test_dijkstra_speedup_vs_networkx(exponent: int) -> None:
    nx = pytest.importorskip("networkx")

    n = 10**exponent
    node_ids = list(range(n))
    edges, weights = generate_weighted_graph(n)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for (u, v), w in zip(edges, weights, strict=True):
        G.add_edge(u, v, weight=w)

    # Warm up
    shortest_path_lengths(node_ids, edges, weights, 0)
    dict(nx.single_source_dijkstra_path_length(G, 0))

    start = time.perf_counter()
    dict(nx.single_source_dijkstra_path_length(G, 0))
    nx_time = time.perf_counter() - start

    start = time.perf_counter()
    shortest_path_lengths(node_ids, edges, weights, 0)
    c_time = time.perf_counter() - start

    speedup = nx_time / c_time if c_time > 0 else float("inf")
    print(f"\n  10^{exponent}: nx={nx_time:.4f}s | pygraphc={c_time:.4f}s ({speedup:.1f}x)")  # noqa: T201
