"""
Benchmark: base (0..n-1) vs remapped overhead.

Three scenarios:
1. base:             connected_components(n, edges)
2. remapped_identity: connected_components_remapped(list(range(n)), edges)  — same IDs, measures pure overhead
3. remapped_real:    connected_components_remapped(node_ids, edges)         — realistic non-contiguous IDs

This isolates the cost of:
- Passing the node_ids list to C (PySequence_Fast parsing)
- Looking up nid_items[bucket[j]] instead of PyLong_FromLong(bucket[j])
- Building Python int objects from arbitrary IDs vs sequential ints
"""

import random
import time

from cgraph._core import (
    connected_components as _cc,
    connected_components_remapped as _cc_remapped,
)


def generate_sparse_graph(number_of_nodes: int, average_degree: int = 3, seed: int = 42):
    rng = random.Random(seed)
    number_of_edges = (number_of_nodes * average_degree) // 2
    edges = []
    for _ in range(number_of_edges):
        node_a = rng.randint(0, number_of_nodes - 1)
        node_b = rng.randint(0, number_of_nodes - 1)
        if node_a != node_b:
            edges.append((node_a, node_b))
    return edges


def bench(label, func, *args, warmup=3, rounds=10):
    for _ in range(warmup):
        list(func(*args))

    times = []
    for _ in range(rounds):
        start = time.perf_counter()
        result = list(func(*args))
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    median = sorted(times)[len(times) // 2]
    total_nodes = sum(len(component) for component in result)
    return median, total_nodes


def run_benchmark():
    print(f"{'nodes':>10} | {'base':>10} | {'remap_id':>10} | {'remap_real':>10} | {'overhead_id':>12} | {'overhead_real':>13}")
    print("-" * 85)

    for exponent in [3, 4, 5, 6, 7]:
        number_of_nodes = 10**exponent
        edges = generate_sparse_graph(number_of_nodes)

        # Scenario 1: base
        time_base, nodes_base = bench("base", _cc, number_of_nodes, edges)

        # Scenario 2: remapped with identity (list(range(n))) — pure overhead
        identity_ids = list(range(number_of_nodes))
        time_identity, nodes_identity = bench("remap_id", _cc_remapped, identity_ids, edges)

        # Scenario 3: remapped with realistic non-contiguous IDs (e.g. database IDs)
        real_ids = list(range(1_000_000, 1_000_000 + number_of_nodes))
        time_real, nodes_real = bench("remap_real", _cc_remapped, real_ids, edges)

        assert nodes_base == nodes_identity == nodes_real == number_of_nodes

        overhead_identity = ((time_identity / time_base) - 1) * 100
        overhead_real = ((time_real / time_base) - 1) * 100

        print(
            f"{number_of_nodes:>10,} | {time_base:>10.5f} | {time_identity:>10.5f} | {time_real:>10.5f} | {overhead_identity:>+11.1f}% | {overhead_real:>+12.1f}%"
        )


if __name__ == "__main__":
    run_benchmark()
