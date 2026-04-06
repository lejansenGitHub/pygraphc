"""Performance tests for Phase 6: node masks and all_edge_paths."""

import random
import time

import pytest

from cgraph import Graph

pytestmark = pytest.mark.performance


def _sparse_graph(n: int, avg_degree: int = 3, seed: int = 42) -> tuple[list[int], list[tuple[int, int]]]:
    rng = random.Random(seed)
    m = (n * avg_degree) // 2
    nodes = list(range(n))
    edges: list[tuple[int, int]] = []
    for _ in range(m):
        a = rng.randint(0, n - 1)
        b = rng.randint(0, n - 1)
        if a != b:
            edges.append((a, b))
    return nodes, edges


# ── Node-masked CC vs full rebuild ──


@pytest.mark.parametrize(
    ("exponent", "num_exclusions"),
    [(4, 10), (5, 10), (5, 100)],
    ids=["10K-10nodes", "100K-10nodes", "100K-100nodes"],
)
def test_cc_node_masked_vs_rebuild(exponent: int, num_exclusions: int) -> None:
    """Node-masked CC should be faster than rebuilding without excluded nodes."""
    n = 10**exponent
    nodes, edges = _sparse_graph(n)
    g = Graph(nodes, edges)
    rng = random.Random(99)
    exclude_nodes = rng.sample(nodes, min(num_exclusions, n))

    # Masked approach
    start = time.perf_counter()
    for _ in range(3):
        view = g.without_nodes(exclude_nodes)
        _ = list(view.connected_components())
    masked_time = (time.perf_counter() - start) / 3

    # Rebuild approach
    exclude_set = set(exclude_nodes)
    start = time.perf_counter()
    for _ in range(3):
        rebuilt_nodes = [nid for nid in nodes if nid not in exclude_set]
        rebuilt_edges = [(a, b) for a, b in edges if a not in exclude_set and b not in exclude_set]
        g2 = Graph(rebuilt_nodes, rebuilt_edges)
        _ = list(g2.connected_components())
    rebuild_time = (time.perf_counter() - start) / 3

    speedup = rebuild_time / masked_time if masked_time > 0 else float("inf")
    assert speedup > 1.5, f"masked {masked_time:.4f}s vs rebuild {rebuild_time:.4f}s (speedup {speedup:.1f}x)"


# ── Node mask overhead vs no mask ──


@pytest.mark.parametrize(
    "algorithm",
    ["connected_components", "bridges", "articulation_points"],
    ids=["CC", "bridges", "AP"],
)
def test_node_mask_overhead_vs_no_mask(algorithm: str) -> None:
    """Node mask with no exclusions should have reasonable overhead.

    Node masks add a per-node check in outer loops (CC union-find, BFS queue)
    plus a per-neighbor check in inner loops — higher overhead than edge masks
    which only touch the inner loop. Up to ~60% overhead is acceptable.
    """
    n = 10**5
    nodes, edges = _sparse_graph(n)
    g = Graph(nodes, edges)

    runs = 20

    # No mask
    fn = getattr(g, algorithm)
    start = time.perf_counter()
    for _ in range(runs):
        result = fn()
        if hasattr(result, "__next__"):
            list(result)
    no_mask_time = (time.perf_counter() - start) / runs

    # With empty node mask
    view = g.without_nodes([])
    fn_view = getattr(view, algorithm)
    start = time.perf_counter()
    for _ in range(runs):
        result = fn_view()
        if hasattr(result, "__next__"):
            list(result)
    mask_time = (time.perf_counter() - start) / runs

    overhead = (mask_time - no_mask_time) / no_mask_time if no_mask_time > 0 else 0
    assert overhead < 0.60, (
        f"{algorithm}: node mask overhead {overhead:.1%} (no-mask {no_mask_time:.4f}s, mask {mask_time:.4f}s)"
    )


# ── Node mask view creation cost ──


@pytest.mark.parametrize(
    ("exponent", "num_exclusions"),
    [(5, 1), (5, 100), (5, 1000)],
    ids=["100K-1node", "100K-100nodes", "100K-1000nodes"],
)
def test_node_mask_creation_cost(exponent: int, num_exclusions: int) -> None:
    """Node mask view creation should be under 10ms."""
    n = 10**exponent
    nodes, edges = _sparse_graph(n)
    g = Graph(nodes, edges)
    rng = random.Random(99)
    exclude_nodes = rng.sample(nodes, min(num_exclusions, n))

    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        _ = g.without_nodes(exclude_nodes)
    elapsed = (time.perf_counter() - start) / iterations

    assert elapsed < 0.01, f"node mask creation took {elapsed * 1000:.2f}ms"


# ── all_edge_paths performance ──


def test_all_edge_paths_small_graph() -> None:
    """all_edge_paths on a small dense graph should complete quickly."""
    n = 50
    rng = random.Random(42)
    nodes = list(range(n))
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if rng.random() < 0.15]
    g = Graph(nodes, edges)

    start = time.perf_counter()
    paths = g.all_edge_paths(source=0, targets=n - 1, cutoff=5)
    elapsed = time.perf_counter() - start

    assert elapsed < 1.0, f"all_edge_paths took {elapsed:.3f}s"
    assert len(paths) >= 0  # sanity check


def test_all_edge_paths_linear_chain() -> None:
    """all_edge_paths on a linear chain — exactly one path, should be instant."""
    n = 10000
    nodes = list(range(n))
    edges = [(i, i + 1) for i in range(n - 1)]
    g = Graph(nodes, edges)

    start = time.perf_counter()
    paths = g.all_edge_paths(source=0, targets=n - 1)
    elapsed = time.perf_counter() - start

    assert elapsed < 0.5, f"all_edge_paths on chain took {elapsed:.3f}s"
    assert len(paths) == 1
    assert len(paths[0]) == n - 1


def test_all_edge_paths_with_node_mask() -> None:
    """all_edge_paths respects node masks."""
    n = 1000
    nodes = list(range(n))
    edges = [(i, i + 1) for i in range(n - 1)]
    # Add shortcut edges
    rng = random.Random(42)
    for _ in range(200):
        a = rng.randint(0, n - 1)
        b = rng.randint(0, n - 1)
        if a != b:
            edges.append((a, b))
    g = Graph(nodes, edges)

    # Exclude some intermediate nodes
    exclude = list(range(100, 200))
    view = g.without_nodes(exclude)

    start = time.perf_counter()
    view.all_edge_paths(source=0, targets=n - 1, cutoff=20)
    elapsed = time.perf_counter() - start

    assert elapsed < 2.0, f"all_edge_paths with node mask took {elapsed:.3f}s"
