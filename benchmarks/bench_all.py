"""
Structured benchmark: measures the 3 cost phases separately for connected_components.

Phase 1 - Gather: extracting edges from domain objects (Branch instances)
Phase 2 - Parse: cgraph hash map build + edge translation (nid_parse)
Phase 3 - C algo: union-find + result construction

Tests across interfaces: tuples, split lists, numpy (m,2), split numpy 1D.
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


def run_benchmark(n):
    node_ids = list(range(n))
    branches = generate_branches(n)
    m = len(branches)

    # Pre-build all edge formats
    edges_tuples = [(b.node_a, b.node_b) for b in branches]
    src_list = [b.node_a for b in branches]
    dst_list = [b.node_b for b in branches]
    src_np = np.array(src_list, dtype=np.int32)
    dst_np = np.array(dst_list, dtype=np.int32)
    edges_np = np.column_stack([src_np, dst_np])

    # Phase 3: pure C algo (index-based + numpy, no nid_parse)
    t_algo = bench(lambda: _cc_index(n, edges_np))

    # Total cgraph call per interface
    t_nid_tuples = bench(lambda: list(connected_components(node_ids, edges_tuples)))
    t_split_list = bench(lambda: list(connected_components(node_ids, src_list, dst_list)))
    t_nid_np2d = bench(lambda: list(connected_components(node_ids, edges_np)))
    t_split_np = bench(lambda: list(connected_components(node_ids, src_np, dst_np)))

    # Phase 2: parse = total cgraph - algo
    p_tuples = t_nid_tuples - t_algo
    p_split_list = t_split_list - t_algo
    p_np2d = t_nid_np2d - t_algo
    p_split_np = t_split_np - t_algo

    # Phase 1: gather from Branch objects
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

    # Print results
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


def main():
    print("=" * 80)
    print("Connected Components: 3-phase cost breakdown")
    print("  1. gather = extracting edges from Branch objects (Python)")
    print("  2. parse  = nid hash map + edge translation (C input handling)")
    print("  3. C algo = union-find + result set construction")
    print("=" * 80)

    for exp in [4, 5, 6]:
        run_benchmark(10 ** exp)


if __name__ == "__main__":
    main()
