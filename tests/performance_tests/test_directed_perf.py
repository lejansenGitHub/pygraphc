"""Performance tests for directed graph algorithms."""

import random
import time

import pytest

from networkc import Graph

pytestmark = pytest.mark.performance


def generate_sparse_directed_graph(n: int, avg_out_degree: int = 3, seed: int = 42) -> list[tuple[int, int]]:
    """Generate random directed edges for n nodes."""
    rng = random.Random(seed)
    m = n * avg_out_degree
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


def generate_weighted_directed_graph(
    n: int, avg_out_degree: int = 3, seed: int = 42
) -> tuple[list[tuple[int, int]], list[float]]:
    """Generate random directed edges with weights."""
    rng = random.Random(seed)
    edges = generate_sparse_directed_graph(n, avg_out_degree, seed)
    weights = [rng.uniform(0.1, 10.0) for _ in edges]
    return edges, weights


# ── SCC Performance ──


@pytest.mark.parametrize(
    ("exponent", "time_limit_seconds"),
    [(3, 0.01), (4, 0.05), (5, 0.5), (6, 5)],
    ids=["1K", "10K", "100K", "1M"],
)
def test_scc_performance(exponent: int, time_limit_seconds: float) -> None:
    n = 10**exponent
    node_ids = list(range(n))
    edges = generate_sparse_directed_graph(n)
    graph = Graph(node_ids, edges, directed=True)

    # Warmup
    list(graph.strongly_connected_components())

    start = time.perf_counter()
    result = list(graph.strongly_connected_components())
    elapsed = time.perf_counter() - start

    assert len(result) > 0
    assert elapsed < time_limit_seconds, f"SCC took {elapsed:.3f}s, limit {time_limit_seconds}s"


@pytest.mark.parametrize("exponent", [3, 4, 5, 6], ids=["1K", "10K", "100K", "1M"])
def test_scc_speedup_vs_networkx(exponent: int) -> None:
    nx = pytest.importorskip("networkx")

    n = 10**exponent
    node_ids = list(range(n))
    edges = generate_sparse_directed_graph(n)
    graph = Graph(node_ids, edges, directed=True)

    nxg = nx.DiGraph()
    nxg.add_nodes_from(range(n))
    nxg.add_edges_from(edges)

    # Warmup
    list(graph.strongly_connected_components())
    list(nx.kosaraju_strongly_connected_components(nxg))

    start = time.perf_counter()
    list(nx.kosaraju_strongly_connected_components(nxg))
    nx_time = time.perf_counter() - start

    start = time.perf_counter()
    list(graph.strongly_connected_components())
    c_time = time.perf_counter() - start

    speedup = nx_time / c_time if c_time > 0 else float("inf")
    print(f"\n  SCC 10^{exponent}: nx={nx_time:.4f}s | networkc={c_time:.4f}s ({speedup:.1f}x)")  # noqa: T201


# ── WCC Performance ──


@pytest.mark.parametrize(
    ("exponent", "time_limit_seconds"),
    [(3, 0.005), (4, 0.02), (5, 0.3), (6, 3)],
    ids=["1K", "10K", "100K", "1M"],
)
def test_wcc_performance(exponent: int, time_limit_seconds: float) -> None:
    n = 10**exponent
    node_ids = list(range(n))
    edges = generate_sparse_directed_graph(n)
    graph = Graph(node_ids, edges, directed=True)

    # Warmup
    list(graph.weakly_connected_components())

    start = time.perf_counter()
    result = list(graph.weakly_connected_components())
    elapsed = time.perf_counter() - start

    assert len(result) > 0
    assert elapsed < time_limit_seconds, f"WCC took {elapsed:.3f}s, limit {time_limit_seconds}s"


@pytest.mark.parametrize("exponent", [3, 4, 5, 6], ids=["1K", "10K", "100K", "1M"])
def test_wcc_speedup_vs_networkx(exponent: int) -> None:
    nx = pytest.importorskip("networkx")

    n = 10**exponent
    node_ids = list(range(n))
    edges = generate_sparse_directed_graph(n)
    graph = Graph(node_ids, edges, directed=True)

    nxg = nx.DiGraph()
    nxg.add_nodes_from(range(n))
    nxg.add_edges_from(edges)

    # Warmup
    list(graph.weakly_connected_components())
    list(nx.weakly_connected_components(nxg))

    start = time.perf_counter()
    list(nx.weakly_connected_components(nxg))
    nx_time = time.perf_counter() - start

    start = time.perf_counter()
    list(graph.weakly_connected_components())
    c_time = time.perf_counter() - start

    speedup = nx_time / c_time if c_time > 0 else float("inf")
    print(f"\n  WCC 10^{exponent}: nx={nx_time:.4f}s | networkc={c_time:.4f}s ({speedup:.1f}x)")  # noqa: T201


# ── Directed BFS Performance ──


@pytest.mark.parametrize(
    ("exponent", "time_limit_seconds"),
    [(3, 0.005), (4, 0.02), (5, 0.3), (6, 4)],
    ids=["1K", "10K", "100K", "1M"],
)
def test_directed_bfs_performance(exponent: int, time_limit_seconds: float) -> None:
    n = 10**exponent
    node_ids = list(range(n))
    edges = generate_sparse_directed_graph(n)
    graph = Graph(node_ids, edges, directed=True)

    # Warmup
    graph.bfs(source=0)

    start = time.perf_counter()
    result = graph.bfs(source=0)
    elapsed = time.perf_counter() - start

    assert len(result) > 0
    assert elapsed < time_limit_seconds, f"BFS took {elapsed:.3f}s, limit {time_limit_seconds}s"


# ── Directed Dijkstra Performance ──


@pytest.mark.parametrize(
    ("exponent", "time_limit_seconds"),
    [(3, 0.01), (4, 0.05), (5, 0.5), (6, 5)],
    ids=["1K", "10K", "100K", "1M"],
)
def test_directed_dijkstra_performance(exponent: int, time_limit_seconds: float) -> None:
    n = 10**exponent
    node_ids = list(range(n))
    edges, weights = generate_weighted_directed_graph(n)
    graph = Graph(node_ids, edges, directed=True)

    # Warmup
    graph.shortest_path_lengths(weights=weights, source=0)

    start = time.perf_counter()
    result = graph.shortest_path_lengths(weights=weights, source=0)
    elapsed = time.perf_counter() - start

    assert len(result) > 0
    assert elapsed < time_limit_seconds, f"Dijkstra took {elapsed:.3f}s, limit {time_limit_seconds}s"


# ── Topological Sort Performance ──


@pytest.mark.parametrize(
    ("exponent", "time_limit_seconds"),
    [(3, 0.01), (4, 0.05), (5, 0.5), (6, 5)],
    ids=["1K", "10K", "100K", "1M"],
)
def test_toposort_performance(exponent: int, time_limit_seconds: float) -> None:
    n = 10**exponent
    node_ids = list(range(n))
    edges = generate_dag(n)
    graph = Graph(node_ids, edges, directed=True)

    # Warmup
    graph.topological_sort()

    start = time.perf_counter()
    order = graph.topological_sort()
    elapsed = time.perf_counter() - start

    assert len(order) == n
    assert elapsed < time_limit_seconds, f"toposort took {elapsed:.3f}s, limit {time_limit_seconds}s"


@pytest.mark.parametrize("exponent", [3, 4, 5, 6], ids=["1K", "10K", "100K", "1M"])
def test_toposort_speedup_vs_networkx(exponent: int) -> None:
    nx = pytest.importorskip("networkx")

    n = 10**exponent
    node_ids = list(range(n))
    edges = generate_dag(n)
    graph = Graph(node_ids, edges, directed=True)

    nxg = nx.DiGraph()
    nxg.add_nodes_from(range(n))
    nxg.add_edges_from(edges)

    # Warmup
    graph.topological_sort()
    list(nx.topological_sort(nxg))

    start = time.perf_counter()
    list(nx.topological_sort(nxg))
    nx_time = time.perf_counter() - start

    start = time.perf_counter()
    graph.topological_sort()
    c_time = time.perf_counter() - start

    speedup = nx_time / c_time if c_time > 0 else float("inf")
    print(f"\n  toposort 10^{exponent}: nx={nx_time:.4f}s | networkc={c_time:.4f}s ({speedup:.1f}x)")  # noqa: T201


# ── Construction Overhead ──


@pytest.mark.parametrize(
    "exponent",
    [3, 4, 5, 6],
    ids=["1K", "10K", "100K", "1M"],
)
def test_directed_construction_overhead(exponent: int) -> None:
    """Directed graph construction should not be more than 1.5x slower than undirected."""
    n = 10**exponent
    node_ids = list(range(n))
    edges = generate_sparse_directed_graph(n)

    # Warmup
    Graph(node_ids, edges)
    Graph(node_ids, edges, directed=True)

    runs = 5
    start = time.perf_counter()
    for _ in range(runs):
        Graph(node_ids, edges)
    undirected_time = (time.perf_counter() - start) / runs

    start = time.perf_counter()
    for _ in range(runs):
        Graph(node_ids, edges, directed=True)
    directed_time = (time.perf_counter() - start) / runs

    ratio = directed_time / undirected_time if undirected_time > 0 else float("inf")
    print(f"\n  10^{exponent}: undirected={undirected_time:.4f}s | directed={directed_time:.4f}s (ratio={ratio:.2f})")  # noqa: T201
    assert ratio < 1.5, f"directed construction {ratio:.2f}x slower than undirected"
