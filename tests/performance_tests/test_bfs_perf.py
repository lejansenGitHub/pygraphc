"""Performance tests for BFS."""

import random
import time

import pytest
from cgraph import bfs

pytestmark = pytest.mark.performance


def generate_sparse_graph(n: int, avg_degree: int = 3, seed: int = 42) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    m = (n * avg_degree) // 2
    edges: list[tuple[int, int]] = []
    for _ in range(m):
        a = rng.randint(0, n - 1)
        b = rng.randint(0, n - 1)
        if a != b:
            edges.append((a, b))
    return edges


@pytest.mark.parametrize(
    ("exponent", "time_limit_seconds"),
    [(3, 0.005), (4, 0.02), (5, 0.3), (6, 4), (7, 50)],
    ids=["1K", "10K", "100K", "1M", "10M"],
)
def test_bfs_performance(exponent: int, time_limit_seconds: float) -> None:
    n = 10**exponent
    node_ids = list(range(n))
    edges = generate_sparse_graph(n)

    start = time.perf_counter()
    result = bfs(node_ids, edges, 0)
    elapsed = time.perf_counter() - start

    assert len(result) > 0
    assert elapsed < time_limit_seconds, f"took {elapsed:.3f}s, limit {time_limit_seconds}s"


@pytest.mark.parametrize(
    "exponent",
    [3, 4, 5, 6],
    ids=["1K", "10K", "100K", "1M"],
)
def test_bfs_speedup_vs_networkx(exponent: int) -> None:
    nx = pytest.importorskip("networkx")

    n = 10**exponent
    node_ids = list(range(n))
    edges = generate_sparse_graph(n)

    G = nx.Graph()
    G.add_nodes_from(range(n))
    G.add_edges_from(edges)

    bfs(node_ids, edges, 0)
    list(nx.bfs_tree(G, 0))

    start = time.perf_counter()
    list(nx.bfs_tree(G, 0))
    nx_time = time.perf_counter() - start

    start = time.perf_counter()
    bfs(node_ids, edges, 0)
    c_time = time.perf_counter() - start

    speedup = nx_time / c_time if c_time > 0 else float("inf")
    print(f"\n  10^{exponent}: nx={nx_time:.4f}s | cgraph={c_time:.4f}s ({speedup:.1f}x)")  # noqa: T201
