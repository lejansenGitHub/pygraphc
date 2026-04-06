"""
Structured benchmark: measures the 3 cost phases separately for connected_components.

Phase 1 - Data gathering: extracting edges from domain objects (Branch instances)
Phase 2 - Input handling: cgraph parsing edges into internal format (nid_parse + parse_edges_mapped)
Phase 3 - C core: the actual algorithm (union-find, CSR build, result construction)

Tests multiple data-gathering strategies:
  A) List of tuples:   [(b.node_a, b.node_b) for b in branches]
  B) Numpy from attrs: np.array([[b.node_a, b.node_b] for b in branches], dtype=np.int32)
  C) Numpy pre-alloc:  pre-allocate array, fill in loop
  D) Numpy from lists:  build two lists, then np.column_stack
"""

import random
import time

import numpy as np

from cgraph import connected_components


class Branch:
    __slots__ = ('branch_id', 'node_a', 'node_b')

    def __init__(self, branch_id: int, node_a: int, node_b: int):
        self.branch_id = branch_id
        self.node_a = node_a
        self.node_b = node_b


def generate_branches(n, avg_degree=3, seed=42):
    """Generate Branch objects simulating a real domain."""
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


def bench_phase(func, rounds=7, warmup=3):
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

    print(f"\n  n={n:,}  m={m:,}")
    print(f"  {'strategy':<28} {'gather':>10} {'cgraph':>10} {'total':>10}  "
          f"{'gather%':>8} {'cgraph%':>8}")
    print(f"  {'-'*82}")

    # --- Strategy A: list of tuples ---
    def gather_tuples():
        return [(b.node_a, b.node_b) for b in branches]

    edges_tuples = gather_tuples()  # pre-build for phase 2+3 measurement

    t_gather_a = bench_phase(gather_tuples)
    t_cgraph_a = bench_phase(lambda: list(connected_components(node_ids, edges_tuples)))
    t_total_a = t_gather_a + t_cgraph_a

    print(f"  {'A) tuples':<28} {t_gather_a*1000:>9.1f}ms {t_cgraph_a*1000:>9.1f}ms "
          f"{t_total_a*1000:>9.1f}ms  {t_gather_a/t_total_a*100:>7.1f}% {t_cgraph_a/t_total_a*100:>7.1f}%")

    # --- Strategy B: numpy from list comprehension ---
    def gather_np_listcomp():
        return np.array([(b.node_a, b.node_b) for b in branches], dtype=np.int32)

    edges_np_b = gather_np_listcomp()

    t_gather_b = bench_phase(gather_np_listcomp)
    t_cgraph_b = bench_phase(lambda: list(connected_components(node_ids, edges_np_b)))
    t_total_b = t_gather_b + t_cgraph_b

    print(f"  {'B) np from listcomp':<28} {t_gather_b*1000:>9.1f}ms {t_cgraph_b*1000:>9.1f}ms "
          f"{t_total_b*1000:>9.1f}ms  {t_gather_b/t_total_b*100:>7.1f}% {t_cgraph_b/t_total_b*100:>7.1f}%")

    # --- Strategy C: numpy pre-allocated ---
    def gather_np_prealloc():
        edges = np.empty((m, 2), dtype=np.int32)
        for i, b in enumerate(branches):
            edges[i, 0] = b.node_a
            edges[i, 1] = b.node_b
        return edges

    edges_np_c = gather_np_prealloc()

    t_gather_c = bench_phase(gather_np_prealloc)
    t_cgraph_c = bench_phase(lambda: list(connected_components(node_ids, edges_np_c)))
    t_total_c = t_gather_c + t_cgraph_c

    print(f"  {'C) np pre-alloc fill':<28} {t_gather_c*1000:>9.1f}ms {t_cgraph_c*1000:>9.1f}ms "
          f"{t_total_c*1000:>9.1f}ms  {t_gather_c/t_total_c*100:>7.1f}% {t_cgraph_c/t_total_c*100:>7.1f}%")

    # --- Strategy D: two lists then column_stack ---
    def gather_np_twolists():
        src = [b.node_a for b in branches]
        dst = [b.node_b for b in branches]
        return np.column_stack([np.array(src, dtype=np.int32),
                                np.array(dst, dtype=np.int32)])

    edges_np_d = gather_np_twolists()

    t_gather_d = bench_phase(gather_np_twolists)
    t_cgraph_d = bench_phase(lambda: list(connected_components(node_ids, edges_np_d)))
    t_total_d = t_gather_d + t_cgraph_d

    print(f"  {'D) np from two lists':<28} {t_gather_d*1000:>9.1f}ms {t_cgraph_d*1000:>9.1f}ms "
          f"{t_total_d*1000:>9.1f}ms  {t_gather_d/t_total_d*100:>7.1f}% {t_cgraph_d/t_total_d*100:>7.1f}%")

    # --- Strategy E: tuples, but pre-build node_ids as range ---
    # (measures: what if node_ids is already list(range(n))?)
    # This is the same as A but highlights that node_ids creation is free

    print()
    return {
        'A_total': t_total_a, 'B_total': t_total_b,
        'C_total': t_total_c, 'D_total': t_total_d,
    }


def main():
    print("=" * 90)
    print("Connected Components: cost breakdown by phase")
    print("  gather = extracting edges from Branch objects (Python)")
    print("  cgraph = connected_components() call (parsing + C algo + result build)")
    print("=" * 90)

    for exp in [4, 5, 6]:
        run_benchmark(10 ** exp)


if __name__ == "__main__":
    main()
