"""Performance tests for cycle_basis and dag_longest_path."""

import random
import time

import pytest

from cgraph import Graph

pytestmark = pytest.mark.performance


def generate_sparse_graph_with_cycles(n: int, avg_degree: int = 3, seed: int = 42) -> list[tuple[int, int]]:
    """Generate random undirected graph with guaranteed cycles."""
    rng = random.Random(seed)
    m = (n * avg_degree) // 2
    edges: list[tuple[int, int]] = []
    for _ in range(m):
        a = rng.randint(0, n - 1)
        b = rng.randint(0, n - 1)
        if a != b:
            edges.append((a, b))
    return edges


def generate_dag(n: int, avg_out_degree: int = 3, seed: int = 42) -> list[tuple[int, int]]:
    """Generate a random DAG: edges go from lower to higher index only."""
    rng = random.Random(seed)
    m = n * avg_out_degree
    edges: list[tuple[int, int]] = []
    for _ in range(m):
        a = rng.randint(0, n - 2)
        b = rng.randint(a + 1, n - 1)
        edges.append((a, b))
    return edges


# ── cycle_basis Performance ──


@pytest.mark.parametrize(
    ("exponent", "time_limit_seconds"),
    [(3, 0.01), (4, 0.2), (5, 20)],
    ids=["1K", "10K", "100K"],
)
def test_cycle_basis_performance(exponent: int, time_limit_seconds: float) -> None:
    n = 10**exponent
    node_ids = list(range(n))
    edges = generate_sparse_graph_with_cycles(n)
    graph = Graph(node_ids, edges)

    # Warmup
    graph.cycle_basis()

    start = time.perf_counter()
    result = graph.cycle_basis()
    elapsed = time.perf_counter() - start

    assert isinstance(result, list)
    assert elapsed < time_limit_seconds, f"cycle_basis took {elapsed:.3f}s, limit {time_limit_seconds}s"


@pytest.mark.parametrize("exponent", [3, 4, 5], ids=["1K", "10K", "100K"])
def test_cycle_basis_speedup_vs_networkx(exponent: int) -> None:
    nx = pytest.importorskip("networkx")

    n = 10**exponent
    node_ids = list(range(n))
    edges = generate_sparse_graph_with_cycles(n)
    graph = Graph(node_ids, edges)

    nxg = nx.Graph()
    nxg.add_nodes_from(range(n))
    nxg.add_edges_from(edges)

    # Warmup
    graph.cycle_basis()
    nx.cycle_basis(nxg)

    start = time.perf_counter()
    nx.cycle_basis(nxg)
    nx_time = time.perf_counter() - start

    start = time.perf_counter()
    graph.cycle_basis()
    c_time = time.perf_counter() - start

    speedup = nx_time / c_time if c_time > 0 else float("inf")
    print(f"\n  cycle_basis 10^{exponent}: nx={nx_time:.4f}s | cgraph={c_time:.4f}s ({speedup:.1f}x)")  # noqa: T201


# ── dag_longest_path Performance ──


@pytest.mark.parametrize(
    ("exponent", "time_limit_seconds"),
    [(3, 0.01), (4, 0.05), (5, 0.5), (6, 5)],
    ids=["1K", "10K", "100K", "1M"],
)
def test_dag_longest_path_performance(exponent: int, time_limit_seconds: float) -> None:
    n = 10**exponent
    node_ids = list(range(n))
    edges = generate_dag(n)
    graph = Graph(node_ids, edges, directed=True)

    # Warmup
    graph.dag_longest_path()

    start = time.perf_counter()
    result = graph.dag_longest_path()
    elapsed = time.perf_counter() - start

    assert len(result) > 0
    assert elapsed < time_limit_seconds, f"dag_longest_path took {elapsed:.3f}s, limit {time_limit_seconds}s"


@pytest.mark.parametrize("exponent", [3, 4, 5, 6], ids=["1K", "10K", "100K", "1M"])
def test_dag_longest_path_speedup_vs_networkx(exponent: int) -> None:
    nx = pytest.importorskip("networkx")

    n = 10**exponent
    node_ids = list(range(n))
    edges = generate_dag(n)
    graph = Graph(node_ids, edges, directed=True)

    nxg = nx.DiGraph()
    nxg.add_nodes_from(range(n))
    nxg.add_edges_from(edges)

    # Warmup
    graph.dag_longest_path()
    nx.dag_longest_path(nxg)

    start = time.perf_counter()
    nx.dag_longest_path(nxg)
    nx_time = time.perf_counter() - start

    start = time.perf_counter()
    graph.dag_longest_path()
    c_time = time.perf_counter() - start

    speedup = nx_time / c_time if c_time > 0 else float("inf")
    print(f"\n  dag_longest_path 10^{exponent}: nx={nx_time:.4f}s | cgraph={c_time:.4f}s ({speedup:.1f}x)")  # noqa: T201


# ── dag_longest_path Weighted Performance ──


@pytest.mark.parametrize(
    ("exponent", "time_limit_seconds"),
    [(3, 0.01), (4, 0.05), (5, 0.5), (6, 5)],
    ids=["1K", "10K", "100K", "1M"],
)
def test_dag_longest_path_weighted_performance(exponent: int, time_limit_seconds: float) -> None:
    rng = random.Random(42)
    n = 10**exponent
    node_ids = list(range(n))
    edges = generate_dag(n)
    weights = [rng.uniform(0.1, 10.0) for _ in edges]
    graph = Graph(node_ids, edges, directed=True)

    # Warmup
    graph.dag_longest_path(weights=weights)

    start = time.perf_counter()
    result = graph.dag_longest_path(weights=weights)
    elapsed = time.perf_counter() - start

    assert len(result) > 0
    assert elapsed < time_limit_seconds, f"weighted dag_longest_path took {elapsed:.3f}s, limit {time_limit_seconds}s"
