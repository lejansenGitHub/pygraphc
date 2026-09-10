"""Performance test for the terminal-preserving reduction on a sparse random graph."""

import random
import time

import pytest

from pygraphc.reduction import MultiGraph, reduce

pytestmark = pytest.mark.performance


def sparse_multigraph(node_count: int, edge_count: int, seed: int) -> MultiGraph[int]:
    rng = random.Random(seed)
    nodes = list(range(node_count))
    endpoints = {edge_id: (rng.randrange(node_count), rng.randrange(node_count)) for edge_id in range(edge_count)}
    return MultiGraph(nodes, endpoints)


def test_reduce_20k_nodes_with_one_percent_terminals() -> None:
    node_count = 20_000
    graph = sparse_multigraph(node_count, edge_count=25_000, seed=42)
    terminals = set(random.Random(7).sample(graph.nodes, node_count // 100))

    start = time.perf_counter()
    reduced = reduce(graph, terminals)
    elapsed = time.perf_counter() - start

    print(  # noqa: T201 — benchmark output is intentional
        f"\n  reduce 20k nodes / 25k edges / 200 terminals: {elapsed:.3f}s "
        f"-> {len(reduced.graph.nodes)} nodes, {len(reduced.graph.endpoints)} edges"
    )
    assert terminals <= set(reduced.graph.nodes)
    assert elapsed < 5.0, f"took {elapsed:.3f}s, limit 5s"
