"""
Structured benchmark: measures the 3 cost phases separately for connected_components.

Phase 1 - Gather: extracting edges from domain objects (Branch instances)
Phase 2 - Parse: cgraph hash map build + edge translation (nid_parse)
Phase 3 - C algo: union-find + result set construction

Tests across interfaces and graph topologies.
"""

import random
import time

import numpy as np

from cgraph import connected_components
from cgraph._core import connected_components as _cc_index


class Branch:
    __slots__ = ('branch_id', 'node_a', 'node_b')

    def __init__(self, branch_id: int, node_a: int, node_b: int):
        self.branch_id = branch_id
        self.node_a = node_a
        self.node_b = node_b


def generate_branches(n, avg_degree=3, seed=42):
    rng = random.Random(seed)
    m = (n * avg_degree) // 2
    branches = []
    for i in range(m):
        a = rng.randint(0, n - 1)
        b = rng.randint(0, n - 1)
        if a != b:
            branches.append(Branch(i, a, b))
    return branches


def median(times):
    return sorted(times)[len(times) // 2]


def bench(func, rounds=7, warmup=3):
    for _ in range(warmup):
        func()
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        func()
        times.append(time.perf_counter() - t0)
    return median(times)


def make_chain_edges(components):
    """Build chain edges for given (start, end) ranges."""
    src, dst = [], []
    for s, e in components:
        for i in range(s, e - 1):
            src.append(i)
            dst.append(i + 1)
    if not src:
        return np.zeros((0, 2), dtype=np.int32)
    return np.column_stack([np.array(src, dtype=np.int32),
                            np.array(dst, dtype=np.int32)])


def run_phase_breakdown(n):
    """3-phase cost breakdown across interfaces."""
    node_ids = list(range(n))
    branches = generate_branches(n)
    m = len(branches)

    edges_tuples = [(b.node_a, b.node_b) for b in branches]
    src_list = [b.node_a for b in branches]
    dst_list = [b.node_b for b in branches]
    src_np = np.array(src_list, dtype=np.int32)
    dst_np = np.array(dst_list, dtype=np.int32)
    edges_np = np.column_stack([src_np, dst_np])

    # Phase 3: pure C algo
    t_algo = bench(lambda: _cc_index(n, edges_np))

    # Total cgraph call per interface
    t_nid_tuples = bench(lambda: list(connected_components(node_ids, edges_tuples)))
    t_split_list = bench(lambda: list(connected_components(node_ids, src_list, dst_list)))
    t_nid_np2d = bench(lambda: list(connected_components(node_ids, edges_np)))
    t_split_np = bench(lambda: list(connected_components(node_ids, src_np, dst_np)))

    p_tuples = t_nid_tuples - t_algo
    p_split_list = t_split_list - t_algo
    p_np2d = t_nid_np2d - t_algo
    p_split_np = t_split_np - t_algo

    g_tuples = bench(lambda: [(b.node_a, b.node_b) for b in branches])
    g_split = bench(lambda: ([b.node_a for b in branches], [b.node_b for b in branches]))

    def gather_np2d():
        s = [b.node_a for b in branches]
        d = [b.node_b for b in branches]
        return np.column_stack([np.array(s, dtype=np.int32), np.array(d, dtype=np.int32)])
    g_np2d = bench(gather_np2d)

    def gather_split_np():
        return (np.array([b.node_a for b in branches], dtype=np.int32),
                np.array([b.node_b for b in branches], dtype=np.int32))
    g_split_np = bench(gather_split_np)

    print(f"\n  n={n:,}  m={m:,}")
    print(f"  {'phase':<25} {'tuples':>9} {'split list':>11} {'np (m,2)':>10} {'split np1d':>11}")
    print(f"  {'-' * 70}")
    print(f"  {'1. gather':<25} {g_tuples * 1000:>8.1f}ms {g_split * 1000:>10.1f}ms"
          f" {g_np2d * 1000:>9.1f}ms {g_split_np * 1000:>10.1f}ms")
    print(f"  {'2. parse':<25} {p_tuples * 1000:>8.1f}ms {p_split_list * 1000:>10.1f}ms"
          f" {p_np2d * 1000:>9.1f}ms {p_split_np * 1000:>10.1f}ms")
    print(f"  {'3. C algo':<25} {t_algo * 1000:>8.1f}ms {t_algo * 1000:>10.1f}ms"
          f" {t_algo * 1000:>9.1f}ms {t_algo * 1000:>10.1f}ms")

    totals = [
        g_tuples + t_nid_tuples,
        g_split + t_split_list,
        g_np2d + t_nid_np2d,
        g_split_np + t_split_np,
    ]
    print(f"  {'TOTAL':<25} {totals[0] * 1000:>8.1f}ms {totals[1] * 1000:>10.1f}ms"
          f" {totals[2] * 1000:>9.1f}ms {totals[3] * 1000:>10.1f}ms")
    print(f"  {'vs tuples':<25} {'base':>9} {totals[0] / totals[1]:>10.2f}x"
          f" {totals[0] / totals[2]:>9.2f}x {totals[0] / totals[3]:>10.2f}x")


def run_topology_scenarios(n):
    """C algo across different graph topologies."""
    print(f"\n  C algo only (index + numpy), n={n:,}")
    print(f"  {'scenario':<50} {'comps':>8} {'time':>8}")
    print(f"  {'-' * 68}")

    scenarios = [
        ("A) 1 component (all connected chain)",
         make_chain_edges([(0, n)])),
        ("B) 1K equal components (1K nodes each)",
         make_chain_edges([(i * 1000, (i + 1) * 1000) for i in range(1000)])),
        ("C) 1 dominant (900K) + 100K isolated",
         make_chain_edges([(0, 900_000)])),
        ("D) 10 components (100K each)",
         make_chain_edges([(i * 100_000, (i + 1) * 100_000) for i in range(10)])),
        ("E) 1M isolated (0 edges)",
         np.zeros((0, 2), dtype=np.int32)),
    ]

    # F) Random sparse
    rng = random.Random(42)
    m = (n * 3) // 2
    s, d = [], []
    for _ in range(m):
        a, b = rng.randint(0, n - 1), rng.randint(0, n - 1)
        if a != b:
            s.append(a)
            d.append(b)
    e_rand = np.column_stack([np.array(s, dtype=np.int32), np.array(d, dtype=np.int32)])
    scenarios.append(("F) Random sparse (avg degree 3)", e_rand))

    for name, edges in scenarios:
        r = _cc_index(n, edges)
        nc = len(r)
        t = bench(lambda: _cc_index(n, edges))
        print(f"  {name:<50} {nc:>8,} {t * 1000:>7.1f}ms")


def main():
    print("=" * 80)
    print("Part 1: 3-phase cost breakdown (gather, parse, C algo)")
    print("=" * 80)

    for exp in [4, 5, 6]:
        run_phase_breakdown(10 ** exp)

    print()
    print("=" * 80)
    print("Part 2: C algo across graph topologies")
    print("=" * 80)

    run_topology_scenarios(1_000_000)


if __name__ == "__main__":
    main()
