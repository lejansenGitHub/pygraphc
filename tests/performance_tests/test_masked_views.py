"""Performance tests for edge-masked and edge-addition views (Phase 5c).

Compares masked view execution vs full graph rebuild, measures mask overhead
on unmasked graphs, and measures view creation cost.
"""

import random
import time

import pytest

from pygraphc import Graph

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


# ── Masked CC vs full rebuild ──


@pytest.mark.parametrize(
    ("exponent", "num_exclusions"),
    [(4, 1), (5, 1), (5, 10)],
    ids=["10K-1edge", "100K-1edge", "100K-10edges"],
)
def test_cc_masked_vs_rebuild(exponent: int, num_exclusions: int) -> None:
    """Masked CC should be faster than rebuilding the graph from scratch."""
    n = 10**exponent
    nodes, edges = _sparse_graph(n)
    g = Graph(nodes, edges)
    rng = random.Random(99)
    exclude_indices = rng.sample(range(len(edges)), min(num_exclusions, len(edges)))

    # Masked approach
    start = time.perf_counter()
    for _ in range(3):
        view = g.without_edges(exclude_indices)
        _ = list(view.connected_components())
    masked_time = (time.perf_counter() - start) / 3

    # Rebuild approach
    exclude_set = set(exclude_indices)
    start = time.perf_counter()
    for _ in range(3):
        rebuilt_edges = [e for i, e in enumerate(edges) if i not in exclude_set]
        g2 = Graph(nodes, rebuilt_edges)
        _ = list(g2.connected_components())
    rebuild_time = (time.perf_counter() - start) / 3

    speedup = rebuild_time / masked_time if masked_time > 0 else float("inf")
    assert speedup > 1.5, f"masked {masked_time:.4f}s vs rebuild {rebuild_time:.4f}s (speedup {speedup:.1f}x)"


# ── Masked bridges vs full rebuild ──


@pytest.mark.parametrize(
    "exponent",
    [4, 5],
    ids=["10K", "100K"],
)
def test_bridges_masked_vs_rebuild(exponent: int) -> None:
    """Masked bridges should be faster than rebuilding."""
    n = 10**exponent
    nodes, edges = _sparse_graph(n)
    g = Graph(nodes, edges)
    exclude_indices = [0]

    start = time.perf_counter()
    for _ in range(3):
        view = g.without_edges(exclude_indices)
        _ = view.bridges()
    masked_time = (time.perf_counter() - start) / 3

    exclude_set = set(exclude_indices)
    start = time.perf_counter()
    for _ in range(3):
        rebuilt_edges = [e for i, e in enumerate(edges) if i not in exclude_set]
        g2 = Graph(nodes, rebuilt_edges)
        _ = g2.bridges()
    rebuild_time = (time.perf_counter() - start) / 3

    speedup = rebuild_time / masked_time if masked_time > 0 else float("inf")
    assert speedup > 1.5, f"masked {masked_time:.4f}s vs rebuild {rebuild_time:.4f}s (speedup {speedup:.1f}x)"


# ── Mask overhead vs no mask ──


@pytest.mark.parametrize(
    ("exponent", "algorithm"),
    [
        (5, "connected_components"),
        (5, "bridges"),
        (5, "articulation_points"),
    ],
    ids=["100K-CC", "100K-bridges", "100K-AP"],
)
def test_mask_overhead_vs_no_mask(exponent: int, algorithm: str) -> None:
    """Mask with no exclusions should not regress more than 15% vs no mask."""
    n = 10**exponent
    nodes, edges = _sparse_graph(n)
    g = Graph(nodes, edges)

    # No mask (direct Graph call)
    fn = getattr(g, algorithm)
    start = time.perf_counter()
    for _ in range(5):
        result = fn()
        if hasattr(result, "__next__"):
            list(result)
    no_mask_time = (time.perf_counter() - start) / 5

    # With empty mask (view, no exclusions)
    view = g.without_edges([])
    fn_view = getattr(view, algorithm)
    start = time.perf_counter()
    for _ in range(5):
        result = fn_view()
        if hasattr(result, "__next__"):
            list(result)
    mask_time = (time.perf_counter() - start) / 5

    overhead = (mask_time - no_mask_time) / no_mask_time if no_mask_time > 0 else 0
    assert overhead < 0.15, (
        f"{algorithm}: mask overhead {overhead:.1%} (no-mask {no_mask_time:.4f}s, mask {mask_time:.4f}s)"
    )


# ── View creation cost ──


@pytest.mark.parametrize(
    ("exponent", "num_exclusions"),
    [(5, 1), (5, 100), (5, 1000), (6, 1)],
    ids=["100K-1excl", "100K-100excl", "100K-1000excl", "1M-1excl"],
)
def test_view_creation_cost(exponent: int, num_exclusions: int) -> None:
    """View creation should be fast — well under 10ms for typical sizes."""
    n = 10**exponent
    nodes, edges = _sparse_graph(n)
    g = Graph(nodes, edges)
    rng = random.Random(99)
    exclude_indices = rng.sample(range(len(edges)), min(num_exclusions, len(edges)))

    iterations = 100
    start = time.perf_counter()
    for _ in range(iterations):
        _ = g.without_edges(exclude_indices)
    elapsed = (time.perf_counter() - start) / iterations

    # View creation should be under 10ms for graphs up to 1M edges
    assert elapsed < 0.01, f"view creation took {elapsed * 1000:.2f}ms"


# ── Edge-addition view creation cost ──


@pytest.mark.parametrize(
    ("exponent", "num_additions"),
    [(4, 1), (4, 100), (5, 1)],
    ids=["10K-1add", "10K-100add", "100K-1add"],
)
def test_with_edges_creation_cost(exponent: int, num_additions: int) -> None:
    """with_edges() rebuilds CSR — should still be reasonable."""
    n = 10**exponent
    nodes, edges = _sparse_graph(n)
    g = Graph(nodes, edges)
    rng = random.Random(99)
    added = [(rng.randint(0, n - 1), rng.randint(0, n - 1)) for _ in range(num_additions)]

    iterations = 10
    start = time.perf_counter()
    for _ in range(iterations):
        _ = g.with_edges(added)
    elapsed = (time.perf_counter() - start) / iterations

    # CSR rebuild at 100K should be under 1 second
    limit = 1.0 if exponent >= 5 else 0.5
    assert elapsed < limit, f"with_edges took {elapsed:.3f}s"
