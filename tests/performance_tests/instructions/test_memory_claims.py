"""Memory claims, asserted on tracemalloc peaks: deterministic like instruction counts, no valgrind needed."""

import random
import tracemalloc
from collections.abc import Callable

import pytest

from pygraphc.reduction import MultiGraph, Partition

pytestmark = pytest.mark.performance


def _sparse_multigraph(node_count: int, edge_count: int, seed: int) -> MultiGraph[int]:
    rng = random.Random(seed)
    endpoints = {edge_id: (rng.randrange(node_count), rng.randrange(node_count)) for edge_id in range(edge_count)}
    return MultiGraph(list(range(node_count)), endpoints)


def _set_based_partition(graph: MultiGraph[int], active: set[int]) -> Partition:
    """The partition path before the label kernel: one Python set per block from ``connected_components``."""
    kernel = graph._kernel
    excluded = [index for index, edge_id in enumerate(kernel.edge_ids) if edge_id not in active]
    block_of: dict[int, int] = {}
    for group in kernel.graph.without_edges(excluded).connected_components():
        representative = min(group)
        for node_id in group:
            block_of[node_id] = representative
    return Partition(block_of)


def _peak_bytes(function: Callable[[], Partition]) -> int:
    tracemalloc.start()
    function()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def test_label_partition_uses_less_memory_than_the_set_partition() -> None:
    """The label kernel returns 4 bytes per node; the set path holds one Python set per block on top of the dict."""
    graph = _sparse_multigraph(100_000, edge_count=125_000, seed=42)
    active = set(graph.endpoints)
    graph._kernel  # noqa: B018 — build the cached C graph outside the measurement

    set_peak = _peak_bytes(lambda: _set_based_partition(graph, active))
    label_peak = _peak_bytes(lambda: Partition.from_components(graph, active))

    assert label_peak < set_peak, f"labels peak {label_peak} not below sets peak {set_peak}"
