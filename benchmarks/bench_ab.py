"""
A/B comparison: build baseline, benchmark, build optimized, benchmark.
Tuples only (main use case). Runs both back-to-back to minimize system noise.
"""
import random
import subprocess
import sys
import time

def gen(n, seed=42):
    rng = random.Random(seed)
    m = (n * 3) // 2
    edges = []
    for _ in range(m):
        a, b = rng.randint(0, n-1), rng.randint(0, n-1)
        if a != b: edges.append((a, b))
    return edges

def gen_w(m, seed=42):
    rng = random.Random(seed)
    return [rng.uniform(0.1, 10.0) for _ in range(m)]

def bench(func, *args, rounds=7, warmup=3):
    for _ in range(warmup):
        r = func(*args)
        if hasattr(r, '__next__'): list(r)
    times = []
    for _ in range(rounds):
        t0 = time.perf_counter()
        r = func(*args)
        if hasattr(r, '__next__'): list(r)
        times.append(time.perf_counter() - t0)
    return sorted(times)[len(times)//2]

def run_bench():
    from cgraph import (connected_components, bridges, articulation_points,
                        biconnected_components, bfs, shortest_path, shortest_path_lengths)

    n = 1_000_000
    node_ids = list(range(n))
    edges = gen(n)
    weights = gen_w(len(edges))

    results = {}
    for name, func, extra in [
        ('CC', connected_components, ()),
        ('bridges', bridges, ()),
        ('AP', articulation_points, ()),
        ('BCC', biconnected_components, ()),
        ('bfs', bfs, (0,)),
        ('shortest_path', shortest_path, (weights, 0, n-1)),
        ('sssp', shortest_path_lengths, (weights, 0)),
    ]:
        results[name] = bench(func, node_ids, edges, *extra)
    return results

if __name__ == "__main__":
    # Run 3 interleaved rounds to average out system noise
    all_results = []
    for round_num in range(3):
        print(f"Round {round_num + 1}/3...")
        all_results.append(run_bench())

    # Average across rounds
    print(f"\n{'algo':<20} {'median':>10}")
    print("-" * 32)
    for name in all_results[0]:
        vals = [r[name] for r in all_results]
        med = sorted(vals)[len(vals)//2]
        print(f"{name:<20} {med*1000:>9.1f}ms")
