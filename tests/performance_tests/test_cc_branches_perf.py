"""Performance tests for connected_components_with_branch_ids.

Compares cc vs cc_with_branch_ids at various scales, and measures the
overhead of edge/node exclusions on cc_with_branch_ids.
"""

import random
import time

import pytest
from cgraph import Graph

pytestmark = pytest.mark.performance


def _sparse_graph(
    number_of_nodes: int,
    average_degree: int = 3,
    seed: int = 42,
) -> tuple[list[int], list[tuple[int, int]]]:
    rng = random.Random(seed)
    number_of_edges = (number_of_nodes * average_degree) // 2
    nodes = list(range(number_of_nodes))
    edges: list[tuple[int, int]] = []
    for _ in range(number_of_edges):
        node_a = rng.randint(0, number_of_nodes - 1)
        node_b = rng.randint(0, number_of_nodes - 1)
        if node_a != node_b:
            edges.append((node_a, node_b))
    return nodes, edges


# ── CC vs CC with branch IDs (overhead of tracking branches) ──


@pytest.mark.parametrize(
    "exponent",
    [3, 4, 5, 6],
    ids=["1K", "10K", "100K", "1M"],
)
def test_cc_vs_cc_with_branch_ids(exponent: int) -> None:
    """Measure overhead of tracking branch IDs vs plain CC.

    Both use Graph (parsed context). The branch variant builds additional
    per-component sets for branch IDs, so some overhead is expected.
    """
    number_of_nodes = 10**exponent
    nodes, edges = _sparse_graph(number_of_nodes)
    branch_ids = list(range(len(edges)))
    graph = Graph(nodes, edges)

    runs = 20 if exponent <= 4 else 5

    # Warm up both paths
    list(graph.connected_components())
    list(graph.connected_components_with_branch_ids(branch_ids))

    # Plain CC — best of N
    cc_times = []
    for _ in range(runs):
        start = time.perf_counter()
        list(graph.connected_components())
        cc_times.append(time.perf_counter() - start)
    cc_time = min(cc_times)

    # CC with branch IDs — best of N
    branch_times = []
    for _ in range(runs):
        start = time.perf_counter()
        list(graph.connected_components_with_branch_ids(branch_ids))
        branch_times.append(time.perf_counter() - start)
    cc_branch_time = min(branch_times)

    overhead = (cc_branch_time - cc_time) / cc_time if cc_time > 0 else 0

    print(  # noqa: T201
        f"\n  10^{exponent}:  cc={cc_time:.4f}s"
        f"  | cc+branches={cc_branch_time:.4f}s"
        f"  | overhead={overhead:.0%}",
    )

    # Branch tracking builds additional PySet per component plus iterates all
    # edges to assign branch IDs — 1.5M extra PySet_Add calls at 1M nodes.
    # At small sizes the overhead percentage is noisy (sub-millisecond).
    # At 1M+ the branch set construction (CPython hash-insert floor) dominates.
    # Up to 250% overhead is expected since it does strictly more work.
    assert overhead < 2.5, (
        f"branch overhead {overhead:.0%} (cc {cc_time:.4f}s, cc+branches {cc_branch_time:.4f}s)"
    )


# ── CC with branch IDs: excluded edges ──


@pytest.mark.parametrize(
    ("exponent", "exclusion_fraction"),
    [(4, 0.01), (5, 0.01), (5, 0.10), (5, 0.50)],
    ids=["10K-1%edges", "100K-1%edges", "100K-10%edges", "100K-50%edges"],
)
def test_cc_branch_ids_excluded_edges(exponent: int, exclusion_fraction: float) -> None:
    """CC with branch IDs plus excluded edges vs full rebuild without those edges.

    The masked approach should be faster than filtering edges in Python and
    rebuilding the graph from scratch.
    """
    number_of_nodes = 10**exponent
    nodes, edges = _sparse_graph(number_of_nodes)
    branch_ids = list(range(len(edges)))
    graph = Graph(nodes, edges)
    rng = random.Random(99)

    number_of_exclusions = max(1, int(len(edges) * exclusion_fraction))
    excluded_indices = rng.sample(range(len(edges)), number_of_exclusions)

    runs = 3

    # Masked approach
    start = time.perf_counter()
    for _ in range(runs):
        view = graph.without_edges(excluded_indices)
        list(view.connected_components_with_branch_ids(branch_ids))
    masked_time = (time.perf_counter() - start) / runs

    # Rebuild approach
    excluded_set = set(excluded_indices)
    start = time.perf_counter()
    for _ in range(runs):
        filtered_edges = [edge for edge_index, edge in enumerate(edges) if edge_index not in excluded_set]
        filtered_branch_ids = [
            branch_id for edge_index, branch_id in enumerate(branch_ids) if edge_index not in excluded_set
        ]
        rebuilt_graph = Graph(nodes, filtered_edges)
        list(rebuilt_graph.connected_components_with_branch_ids(filtered_branch_ids))
    rebuild_time = (time.perf_counter() - start) / runs

    speedup = rebuild_time / masked_time if masked_time > 0 else float("inf")

    print(  # noqa: T201
        f"\n  10^{exponent} ({exclusion_fraction:.0%} edges excluded):"
        f"  masked={masked_time:.4f}s"
        f"  | rebuild={rebuild_time:.4f}s"
        f"  | speedup={speedup:.1f}x",
    )

    assert speedup > 1.2, (
        f"masked {masked_time:.4f}s vs rebuild {rebuild_time:.4f}s (speedup {speedup:.1f}x)"
    )


# ── CC with branch IDs: excluded nodes ──


@pytest.mark.parametrize(
    ("exponent", "exclusion_fraction"),
    [(4, 0.01), (5, 0.01), (5, 0.10), (5, 0.50)],
    ids=["10K-1%nodes", "100K-1%nodes", "100K-10%nodes", "100K-50%nodes"],
)
def test_cc_branch_ids_excluded_nodes(exponent: int, exclusion_fraction: float) -> None:
    """CC with branch IDs plus excluded nodes.

    Excluded nodes are filtered from output but edges still connect.
    Compare masked approach vs rebuilding with nodes removed from output
    in Python.
    """
    number_of_nodes = 10**exponent
    nodes, edges = _sparse_graph(number_of_nodes)
    branch_ids = list(range(len(edges)))
    graph = Graph(nodes, edges)
    rng = random.Random(99)

    number_of_exclusions = max(1, int(number_of_nodes * exclusion_fraction))
    excluded_node_ids = rng.sample(nodes, number_of_exclusions)

    runs = 3

    # Masked approach (C-level node filtering)
    start = time.perf_counter()
    for _ in range(runs):
        view = graph.without_nodes(excluded_node_ids)
        list(view.connected_components_with_branch_ids(branch_ids))
    masked_time = (time.perf_counter() - start) / runs

    # Python-level approach: run full CC then filter nodes from output
    excluded_set = set(excluded_node_ids)
    start = time.perf_counter()
    for _ in range(runs):
        components = list(graph.connected_components_with_branch_ids(branch_ids))
        [
            (node_set - excluded_set, branch_set)
            for node_set, branch_set in components
        ]
    python_filter_time = (time.perf_counter() - start) / runs

    overhead = (masked_time - python_filter_time) / python_filter_time if python_filter_time > 0 else 0

    print(  # noqa: T201
        f"\n  10^{exponent} ({exclusion_fraction:.0%} nodes excluded):"
        f"  masked={masked_time:.4f}s"
        f"  | python-filter={python_filter_time:.4f}s"
        f"  | overhead={overhead:+.0%}",
    )

    # C-level filtering should be comparable to Python post-filter.
    # At low exclusion fractions the C version does slightly more work
    # (bytearray check per node), at high fractions it wins by avoiding
    # building sets that would be filtered away.
    assert overhead < 0.75, (
        f"C-level overhead {overhead:.0%} vs Python filter"
        f" (masked {masked_time:.4f}s, python-filter {python_filter_time:.4f}s)"
    )


# ── CC with branch IDs: combined edge + node exclusions ──


@pytest.mark.parametrize(
    "exponent",
    [4, 5],
    ids=["10K", "100K"],
)
def test_cc_branch_ids_combined_exclusions(exponent: int) -> None:
    """CC with branch IDs excluding both edges and nodes simultaneously.

    Measures the combined overhead vs running without any exclusions.
    """
    number_of_nodes = 10**exponent
    nodes, edges = _sparse_graph(number_of_nodes)
    branch_ids = list(range(len(edges)))
    graph = Graph(nodes, edges)
    rng = random.Random(99)

    excluded_edge_count = max(1, len(edges) // 10)
    excluded_node_count = max(1, number_of_nodes // 10)
    excluded_edge_indices = rng.sample(range(len(edges)), excluded_edge_count)
    excluded_node_ids = rng.sample(nodes, excluded_node_count)

    runs = 5

    # No exclusions
    start = time.perf_counter()
    for _ in range(runs):
        list(graph.connected_components_with_branch_ids(branch_ids))
    base_time = (time.perf_counter() - start) / runs

    # Combined exclusions
    start = time.perf_counter()
    for _ in range(runs):
        view = graph.without_edges(excluded_edge_indices).without_nodes(excluded_node_ids)
        list(view.connected_components_with_branch_ids(branch_ids))
    combined_time = (time.perf_counter() - start) / runs

    overhead = (combined_time - base_time) / base_time if base_time > 0 else 0

    print(  # noqa: T201
        f"\n  10^{exponent} (10% edges + 10% nodes excluded):"
        f"  base={base_time:.4f}s"
        f"  | combined={combined_time:.4f}s"
        f"  | overhead={overhead:+.0%}",
    )

    # Exclusions should not regress more than 100% vs no exclusions
    assert overhead < 1.0, (
        f"combined overhead {overhead:.0%}"
        f" (base {base_time:.4f}s, combined {combined_time:.4f}s)"
    )
