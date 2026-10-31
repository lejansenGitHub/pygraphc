"""Known superlinear cases, each with its ratchet limit and its cause.

Every case of the complexity tier must scale at most like ``LINEAR_LIMIT``
unless it is listed here. A listed limit is the worst exponent of two runs
when the case was found plus 0.25 (at least 1.6), so the case cannot get worse unnoticed. The MR
that fixes a case deletes its entry, and the case falls back to the linear
limit: the test that found the problem proves the fix.

Garbage-collector-bound cases are listed although their algorithm is
linear: building one tracked Python container per result item makes the
collector walk ever more objects, and callers pay that with the collector on.
Their exponent varies more between runs than an algorithm's, since when the
collector runs depends on the heap, so they share one limit, ``GC_LIMIT``,
above the worst measured 1.64.
"""

GC_LIMIT = 1.9
GC_BOUND = "linear algorithm; one GC-tracked container per result item makes CPython's collector superlinear"

KNOWN_SUPERLINEAR: dict[str, tuple[float, str]] = {
    "bridges_with_branch_ids/star": (
        2.2,
        "quadratic: edge_indices(hub, leaf) per bridge scans the hub's whole adjacency",
    ),
    "reduce_c/two_hubs": (1.9, "quadratic in hub degree; the python and rounds engines stay linear"),
    "reduce_moves/two_hubs": (2.0, "quadratic in hub degree; the python and rounds engines stay linear"),
    "tree_folds/two_hubs": (2.2, "quadratic in time and memory in the width of the parallel tree two hubs produce"),
    "cycle_basis/lattice": (1.9, "inherent: a fundamental cycle basis of a lattice has total length ~n^1.5"),
    "connected_components/singletons": (GC_LIMIT, GC_BOUND),
    "connected_components_with_branch_ids/singletons": (GC_LIMIT, GC_BOUND),
    "masked_connected_components/singletons": (GC_LIMIT, GC_BOUND),
    "biconnected_components/path": (GC_LIMIT, GC_BOUND),
    "biconnected_components/star": (GC_LIMIT, GC_BOUND),
    "two_edge_connected_components/path": (GC_LIMIT, GC_BOUND),
    "two_edge_connected_components/star": (GC_LIMIT, GC_BOUND),
    "two_edge_connected_components/singletons": (GC_LIMIT, GC_BOUND),
    "ports/two_edge_connected_components/wide_edge": (GC_LIMIT, GC_BOUND),
    "ports/two_edge_connected_components/port_chain": (GC_LIMIT, GC_BOUND),
    "cycle_basis/singletons": (GC_LIMIT, GC_BOUND),
    "cycle_basis/sparse_random": (GC_LIMIT, GC_BOUND),
    "nodes_on_simple_paths/path": (GC_LIMIT, GC_BOUND),
    "nodes_on_simple_paths/star": (GC_LIMIT, GC_BOUND),
    "nodes_on_simple_paths/lattice": (GC_LIMIT, GC_BOUND),
    "nodes_on_simple_paths/sparse_random": (GC_LIMIT, GC_BOUND),
    "reduce_python/lattice": (GC_LIMIT, GC_BOUND),
    "directed/strongly_connected_components/directed_acyclic": (GC_LIMIT, GC_BOUND),
    "directed/strongly_connected_components/path": (GC_LIMIT, GC_BOUND),
    "directed/strongly_connected_components/star": (GC_LIMIT, GC_BOUND),
}
