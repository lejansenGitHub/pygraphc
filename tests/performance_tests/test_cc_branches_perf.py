"""Performance tests for connected_components_with_branch_ids.

Compares cc vs cc_with_branch_ids at various scales, and measures the
overhead of edge/node exclusions on cc_with_branch_ids.
"""

import random
import time

import pytest

from cgraph import Graph

pytestmark = pytest.mark.performance


class Branch:
    """Domain object simulating a real-world branch/edge with metadata."""

    __slots__ = ("branch_id", "node_a", "node_b")

    def __init__(self, branch_id: int, node_a: int, node_b: int) -> None:
        self.branch_id = branch_id
        self.node_a = node_a
        self.node_b = node_b


def _generate_branches(
    number_of_nodes: int,
    average_degree: int = 3,
    seed: int = 42,
) -> tuple[list[int], list[Branch]]:
    rng = random.Random(seed)
    number_of_edges = (number_of_nodes * average_degree) // 2
    node_ids = list(range(number_of_nodes))
    branches: list[Branch] = []
    for branch_index in range(number_of_edges):
        node_a = rng.randint(0, number_of_nodes - 1)
        node_b = rng.randint(0, number_of_nodes - 1)
        if node_a != node_b:
            branches.append(Branch(branch_index, node_a, node_b))
    return node_ids, branches


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
    graph_plain = Graph(nodes, edges)
    graph_branches = Graph(nodes, edges, branch_ids=branch_ids)

    runs = 20 if exponent <= 4 else 5

    # Warm up both paths
    list(graph_plain.connected_components())
    list(graph_branches.connected_components_with_branch_ids())

    # Plain CC — best of N
    cc_times = []
    for _ in range(runs):
        start = time.perf_counter()
        list(graph_plain.connected_components())
        cc_times.append(time.perf_counter() - start)
    cc_time = min(cc_times)

    # CC with branch IDs — best of N
    branch_times = []
    for _ in range(runs):
        start = time.perf_counter()
        list(graph_branches.connected_components_with_branch_ids())
        branch_times.append(time.perf_counter() - start)
    cc_branch_time = min(branch_times)

    overhead = (cc_branch_time - cc_time) / cc_time if cc_time > 0 else 0

    print(  # noqa: T201
        f"\n  10^{exponent}:  cc={cc_time:.4f}s  | cc+branches={cc_branch_time:.4f}s  | overhead={overhead:.0%}",
    )

    # Branch tracking builds additional PySet per component plus iterates all
    # edges to assign branch IDs — 1.5M extra PySet_Add calls at 1M nodes.
    # At small sizes (<100K) the overhead percentage is noisy (sub-millisecond
    # timings on shared CI runners), so we only assert at 100K+.
    # At 1M+ the branch set construction (CPython hash-insert floor) dominates.
    if exponent >= 5:
        assert overhead < 2.5, f"branch overhead {overhead:.0%} (cc {cc_time:.4f}s, cc+branches {cc_branch_time:.4f}s)"


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
    graph = Graph(nodes, edges, branch_ids=branch_ids)
    rng = random.Random(99)

    number_of_exclusions = max(1, int(len(edges) * exclusion_fraction))
    excluded_branch_ids = rng.sample(branch_ids, number_of_exclusions)

    runs = 3

    # Masked approach
    start = time.perf_counter()
    for _ in range(runs):
        view = graph.without_branches(excluded_branch_ids)
        list(view.connected_components_with_branch_ids())
    masked_time = (time.perf_counter() - start) / runs

    # Rebuild approach
    excluded_set = set(excluded_branch_ids)
    start = time.perf_counter()
    for _ in range(runs):
        filtered_edges = [edge for edge_index, edge in enumerate(edges) if branch_ids[edge_index] not in excluded_set]
        filtered_branch_ids = [branch_id for branch_id in branch_ids if branch_id not in excluded_set]
        rebuilt_graph = Graph(nodes, filtered_edges, branch_ids=filtered_branch_ids)
        list(rebuilt_graph.connected_components_with_branch_ids())
    rebuild_time = (time.perf_counter() - start) / runs

    speedup = rebuild_time / masked_time if masked_time > 0 else float("inf")

    print(  # noqa: T201
        f"\n  10^{exponent} ({exclusion_fraction:.0%} edges excluded):"
        f"  masked={masked_time:.4f}s"
        f"  | rebuild={rebuild_time:.4f}s"
        f"  | speedup={speedup:.1f}x",
    )

    # Only assert at 100K+ with low exclusion fractions. At high fractions
    # (50%) rebuild wins because it creates a smaller graph — the masked
    # approach still iterates all edges.
    if exponent >= 5 and exclusion_fraction <= 0.10:
        assert speedup > 1.2, f"masked {masked_time:.4f}s vs rebuild {rebuild_time:.4f}s (speedup {speedup:.1f}x)"


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
    graph = Graph(nodes, edges, branch_ids=branch_ids)
    rng = random.Random(99)

    number_of_exclusions = max(1, int(number_of_nodes * exclusion_fraction))
    excluded_node_ids = rng.sample(nodes, number_of_exclusions)

    runs = 3

    # Masked approach (C-level node filtering)
    start = time.perf_counter()
    for _ in range(runs):
        view = graph.without_nodes(excluded_node_ids)
        list(view.connected_components_with_branch_ids())
    masked_time = (time.perf_counter() - start) / runs

    # Python-level approach: run full CC then filter nodes from output
    excluded_set = set(excluded_node_ids)
    start = time.perf_counter()
    for _ in range(runs):
        components = list(graph.connected_components_with_branch_ids())
        [(node_set - excluded_set, branch_set) for node_set, branch_set in components]
    python_filter_time = (time.perf_counter() - start) / runs

    overhead = (masked_time - python_filter_time) / python_filter_time if python_filter_time > 0 else 0

    print(  # noqa: T201
        f"\n  10^{exponent} ({exclusion_fraction:.0%} nodes excluded):"
        f"  masked={masked_time:.4f}s"
        f"  | python-filter={python_filter_time:.4f}s"
        f"  | overhead={overhead:+.0%}",
    )

    # C-level filtering should be comparable to Python post-filter.
    # Only assert at 100K+ — at 10K the timings are sub-ms and noisy on CI.
    if exponent >= 5:
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
    graph = Graph(nodes, edges, branch_ids=branch_ids)
    rng = random.Random(99)

    excluded_branch_count = max(1, len(edges) // 10)
    excluded_node_count = max(1, number_of_nodes // 10)
    excluded_branch_ids = rng.sample(branch_ids, excluded_branch_count)
    excluded_node_ids = rng.sample(nodes, excluded_node_count)

    runs = 10

    # Warm-up
    list(graph.connected_components_with_branch_ids())
    view = graph.without_branches(excluded_branch_ids).without_nodes(excluded_node_ids)
    list(view.connected_components_with_branch_ids())

    # No exclusions — best of N
    base_times = []
    for _ in range(runs):
        start = time.perf_counter()
        list(graph.connected_components_with_branch_ids())
        base_times.append(time.perf_counter() - start)
    base_time = min(base_times)

    # Combined exclusions — best of N
    combined_times = []
    for _ in range(runs):
        view = graph.without_branches(excluded_branch_ids).without_nodes(excluded_node_ids)
        start = time.perf_counter()
        list(view.connected_components_with_branch_ids())
        combined_times.append(time.perf_counter() - start)
    combined_time = min(combined_times)

    overhead = (combined_time - base_time) / base_time if base_time > 0 else 0

    print(  # noqa: T201
        f"\n  10^{exponent} (10% edges + 10% nodes excluded):"
        f"  base={base_time:.4f}s"
        f"  | combined={combined_time:.4f}s"
        f"  | overhead={overhead:+.0%}",
    )

    # Exclusions should not regress more than 100% vs no exclusions
    assert overhead < 1.0, f"combined overhead {overhead:.0%} (base {base_time:.4f}s, combined {combined_time:.4f}s)"


# ── End-to-end: from Branch domain objects through gather + algorithm ──


@pytest.mark.parametrize(
    "exponent",
    [4, 5, 6],
    ids=["10K", "100K", "1M"],
)
def test_end_to_end_cc_vs_cc_with_branch_ids(exponent: int) -> None:
    """End-to-end from Branch objects: gather + Graph parse + algorithm.

    Measures the full pipeline cost including extracting edges and branch IDs
    from domain objects, not just the C algorithm.
    """
    number_of_nodes = 10**exponent
    node_ids, branches = _generate_branches(number_of_nodes)
    runs = 10 if exponent <= 5 else 3

    # End-to-end: plain CC (gather edges + Graph + CC)
    def run_cc() -> int:
        edges = [(branch.node_a, branch.node_b) for branch in branches]
        graph = Graph(node_ids, edges)
        return len(list(graph.connected_components()))

    # Warm up
    run_cc()
    cc_times = []
    for _ in range(runs):
        start = time.perf_counter()
        run_cc()
        cc_times.append(time.perf_counter() - start)
    cc_time = min(cc_times)

    # End-to-end: CC with branch IDs (gather edges + branch_ids + Graph + CC)
    def run_cc_branches() -> int:
        edges = [(branch.node_a, branch.node_b) for branch in branches]
        branch_ids = [branch.branch_id for branch in branches]
        graph = Graph(node_ids, edges, branch_ids=branch_ids)
        return len(list(graph.connected_components_with_branch_ids()))

    run_cc_branches()
    branch_times = []
    for _ in range(runs):
        start = time.perf_counter()
        run_cc_branches()
        branch_times.append(time.perf_counter() - start)
    branch_time = min(branch_times)

    overhead = (branch_time - cc_time) / cc_time if cc_time > 0 else 0

    print(  # noqa: T201
        f"\n  10^{exponent} end-to-end:"
        f"  cc={cc_time:.4f}s"
        f"  | cc+branches={branch_time:.4f}s"
        f"  | overhead={overhead:.0%}",
    )

    # End-to-end overhead is lower than algorithm-only because gather + parse
    # costs are shared. At 1M nodes gather is ~35ms, algorithm delta is ~70ms.
    assert overhead < 1.5, f"end-to-end overhead {overhead:.0%} (cc {cc_time:.4f}s, cc+branches {branch_time:.4f}s)"


@pytest.mark.parametrize(
    "exponent",
    [4, 5],
    ids=["10K", "100K"],
)
def test_end_to_end_with_exclusions(exponent: int) -> None:
    """End-to-end from Branch objects with edge and node exclusions.

    Full pipeline: gather from domain objects, build Graph, create view
    with exclusions, run CC with branch IDs.
    """
    number_of_nodes = 10**exponent
    node_ids, branches = _generate_branches(number_of_nodes)
    rng = random.Random(99)

    all_branch_ids = [branch.branch_id for branch in branches]
    excluded_branch_ids = rng.sample(all_branch_ids, len(branches) // 10)
    excluded_node_ids = rng.sample(node_ids, number_of_nodes // 10)

    runs = 10

    # End-to-end: no exclusions
    def run_no_exclusions() -> int:
        edges = [(branch.node_a, branch.node_b) for branch in branches]
        branch_ids = [branch.branch_id for branch in branches]
        graph = Graph(node_ids, edges, branch_ids=branch_ids)
        return len(list(graph.connected_components_with_branch_ids()))

    run_no_exclusions()
    base_times = []
    for _ in range(runs):
        start = time.perf_counter()
        run_no_exclusions()
        base_times.append(time.perf_counter() - start)
    base_time = min(base_times)

    # End-to-end: with exclusions (reuse Graph, create view)
    edges = [(branch.node_a, branch.node_b) for branch in branches]
    branch_ids = [branch.branch_id for branch in branches]
    graph = Graph(node_ids, edges, branch_ids=branch_ids)

    def run_with_exclusions() -> int:
        view = graph.without_branches(excluded_branch_ids).without_nodes(excluded_node_ids)
        return len(list(view.connected_components_with_branch_ids()))

    run_with_exclusions()
    excl_times = []
    for _ in range(runs):
        start = time.perf_counter()
        run_with_exclusions()
        excl_times.append(time.perf_counter() - start)
    excl_time = min(excl_times)

    # Compare: view-based exclusion vs rebuild from scratch
    excluded_branch_set = set(excluded_branch_ids)

    def run_rebuild() -> int:
        filtered_edges = [
            (branch.node_a, branch.node_b) for branch in branches if branch.branch_id not in excluded_branch_set
        ]
        filtered_branch_ids = [branch.branch_id for branch in branches if branch.branch_id not in excluded_branch_set]
        rebuilt_graph = Graph(node_ids, filtered_edges, branch_ids=filtered_branch_ids)
        return len(list(rebuilt_graph.connected_components_with_branch_ids()))

    run_rebuild()
    rebuild_times = []
    for _ in range(runs):
        start = time.perf_counter()
        run_rebuild()
        rebuild_times.append(time.perf_counter() - start)
    rebuild_time = min(rebuild_times)

    speedup_vs_rebuild = rebuild_time / excl_time if excl_time > 0 else float("inf")

    print(  # noqa: T201
        f"\n  10^{exponent} end-to-end with 10% exclusions:"
        f"  no-excl={base_time:.4f}s"
        f"  | view={excl_time:.4f}s"
        f"  | rebuild={rebuild_time:.4f}s"
        f"  | view vs rebuild={speedup_vs_rebuild:.1f}x",
    )

    assert speedup_vs_rebuild > 1.0, (
        f"view {excl_time:.4f}s vs rebuild {rebuild_time:.4f}s (speedup {speedup_vs_rebuild:.1f}x)"
    )
