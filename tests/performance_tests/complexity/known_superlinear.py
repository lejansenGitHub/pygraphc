"""Known superlinear cases, each with its ratchet limit and its cause.

Every case of the complexity tier must scale at most like ``LINEAR_LIMIT``
unless it is listed here. A listed limit is the worst exponent of two runs
when the case was found plus 0.25, so the case cannot get worse unnoticed. The MR
that fixes a case deletes its entry, and the case falls back to the linear
limit: the test that found the problem proves the fix.

Garbage-collector-bound cases are listed although their algorithm is
linear: building one tracked Python container per result item makes the
collector walk ever more objects, and callers pay that with the collector on.
"""

GC_BOUND = "linear algorithm; one GC-tracked container per result item makes CPython's collector superlinear"

KNOWN_SUPERLINEAR: dict[str, tuple[float, str]] = {
    "bridges_with_branch_ids/star": (
        2.2,
        "quadratic: edge_indices(hub, leaf) per bridge scans the hub's whole adjacency",
    ),
    "reduce_c/two_hubs": (2.0, "quadratic in hub degree; the python and rounds engines stay linear"),
    "reduce_moves/two_hubs": (2.1, "quadratic in hub degree; the python and rounds engines stay linear"),
    "tree_folds/two_hubs": (2.3, "quadratic in the width of the parallel tree two hubs produce"),
    "cycle_basis/lattice": (1.9, "inherent: a fundamental cycle basis of a lattice has total length ~n^1.5"),
    "connected_components/singletons": (1.9, GC_BOUND),
    "connected_components_with_branch_ids/singletons": (1.9, GC_BOUND),
    "masked_connected_components/singletons": (1.9, GC_BOUND),
    "biconnected_components/path": (1.8, GC_BOUND),
    "biconnected_components/star": (1.8, GC_BOUND),
    "two_edge_connected_components/path": (1.7, GC_BOUND),
    "two_edge_connected_components/star": (1.7, GC_BOUND),
    "two_edge_connected_components/singletons": (1.8, GC_BOUND),
    "cycle_basis/singletons": (1.7, GC_BOUND),
    "cycle_basis/sparse_random": (1.8, GC_BOUND),
    "nodes_on_simple_paths/path": (1.7, GC_BOUND),
    "nodes_on_simple_paths/star": (1.7, GC_BOUND),
    "nodes_on_simple_paths/lattice": (1.6, GC_BOUND),
    "nodes_on_simple_paths/sparse_random": (1.6, GC_BOUND),
    "directed/strongly_connected_components/directed_acyclic": (1.7, GC_BOUND),
    "reduce_python/lattice": (1.6, GC_BOUND),
    "directed/strongly_connected_components/path": (1.9, GC_BOUND),
    "directed/strongly_connected_components/star": (1.9, GC_BOUND),
}
