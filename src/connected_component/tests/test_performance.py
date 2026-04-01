"""
Performance tests comparing C union-find vs scipy connected_components.

Tests graph sizes from 10^3 to 10^7 nodes with ~3 edges per node on average.
"""

import random
import time

import numpy as np
import pytest

from connected_component import igp_connected_components


def generate_sparse_grid_graph(
    number_of_nodes: int,
    average_degree: int = 3,
    seed: int = 42,
) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    number_of_edges = (number_of_nodes * average_degree) // 2
    edges: list[tuple[int, int]] = []
    for _ in range(number_of_edges):
        node_a = rng.randint(0, number_of_nodes - 1)
        node_b = rng.randint(0, number_of_nodes - 1)
        if node_a != node_b:
            edges.append((node_a, node_b))
    return edges


def _run_scipy(number_of_nodes, edges):
    """Run the original scipy implementation for comparison."""
    from collections import defaultdict

    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components

    edge_array = np.array(edges, dtype=np.int32)
    row = np.concatenate([edge_array[:, 0], edge_array[:, 1]])
    col = np.concatenate([edge_array[:, 1], edge_array[:, 0]])
    data = np.ones(len(row), dtype=np.int8)
    graph = csr_matrix((data, (row, col)), shape=(number_of_nodes, number_of_nodes))
    _n, labels = connected_components(graph, directed=False)
    comps = defaultdict(set)
    for node_id, comp_id in enumerate(labels):
        comps[comp_id].add(node_id)
    return list(comps.values())


@pytest.mark.parametrize(
    ("exponent", "time_limit_seconds"),
    [
        (3, 0.003),
        (4, 0.01),
        (5, 0.2),
        (6, 2),
        (7, 25),
    ],
    ids=["1K", "10K", "100K", "1M", "10M"],
)
def test_connected_components_performance(
    exponent: int,
    time_limit_seconds: float,
) -> None:
    number_of_nodes = 10**exponent
    edges = generate_sparse_grid_graph(number_of_nodes)

    start = time.perf_counter()
    components = list(igp_connected_components(number_of_nodes, edges))
    elapsed = time.perf_counter() - start

    total_nodes = sum(len(c) for c in components)
    assert total_nodes == number_of_nodes
    assert elapsed < time_limit_seconds, (
        f"took {elapsed:.3f}s for 10^{exponent} nodes, limit is {time_limit_seconds}s"
    )


@pytest.mark.parametrize(
    "exponent",
    [3, 4, 5, 6, 7],
    ids=["1K", "10K", "100K", "1M", "10M"],
)
def test_speedup_vs_scipy(exponent: int) -> None:
    """Measure and print speedup vs scipy for all input/output modes."""
    pytest.importorskip("scipy")
    number_of_nodes = 10**exponent
    edges_list = generate_sparse_grid_graph(number_of_nodes)
    edges_np = np.array(edges_list, dtype=np.int32)

    # Warm up
    list(igp_connected_components(number_of_nodes, edges_list))
    _run_scipy(number_of_nodes, edges_list)

    # scipy
    start = time.perf_counter()
    scipy_result = _run_scipy(number_of_nodes, edges_list)
    scipy_time = time.perf_counter() - start

    # C: list-of-tuples -> sets
    start = time.perf_counter()
    c_list_result = list(igp_connected_components(number_of_nodes, edges_list))
    c_list_time = time.perf_counter() - start

    # C: numpy -> sets
    start = time.perf_counter()
    c_np_result = list(igp_connected_components(number_of_nodes, edges_np))
    c_np_time = time.perf_counter() - start

    # Verify
    assert sum(len(c) for c in c_list_result) == number_of_nodes
    assert sum(len(c) for c in c_np_result) == number_of_nodes
    assert len(c_list_result) == len(scipy_result)
    assert len(c_np_result) == len(scipy_result)

    s1 = scipy_time / c_list_time if c_list_time > 0 else float("inf")
    s2 = scipy_time / c_np_time if c_np_time > 0 else float("inf")

    print(  # noqa: T201
        f"\n  10^{exponent}:  scipy={scipy_time:.4f}s"
        f"  | tuples->sets={c_list_time:.4f}s ({s1:.1f}x)"
        f"  | numpy->sets={c_np_time:.4f}s ({s2:.1f}x)"
    )
