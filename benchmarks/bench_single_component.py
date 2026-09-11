"""One component against all of them: ``connected_component`` vs ``connected_components``.

The answer depends on the shape of the graph, not only on its size, so the
sweep covers one giant component, a thousand equal components, all singletons
and a sparse mixture with a giant component plus a long tail. The accessor
builds one set from a scan of the int32 label buffer; the generator builds a
set per component and adds every node to one. Both pay the same label pass, so
the accessor only wins back the set building.

The accessor uses the graph's cached node id to index map when another method
has already built it and scans the node ids when it has not, so the cold column
is what a single lookup on a fresh graph costs and the warm column is what a
lookup costs once the map exists.

Run with the venv python: ``python benchmarks/bench_single_component.py``.
"""

import gc
import random
import time
import tracemalloc
from collections.abc import Callable

from pygraphc import Graph

NODE_COUNTS = (100_000, 1_000_000)
EQUAL_COMPONENT_COUNT = 1_000
MIXTURE_DEGREE = 0.8
REPEATS = 5
SEED = 20260911


def chain_components(node_count: int, component_size: int) -> list[tuple[int, int]]:
    """Paths of ``component_size`` nodes, so every component has the same size."""
    if component_size <= 1:
        return []
    return [(index, index + 1) for index in range(node_count - 1) if (index + 1) % component_size != 0]


def sparse_mixture(node_count: int) -> list[tuple[int, int]]:
    """Random edges at average degree 1.6: one giant component plus a long tail of small ones."""
    rng = random.Random(SEED)
    edge_count = int(MIXTURE_DEGREE * node_count)
    return [(rng.randrange(node_count), rng.randrange(node_count)) for _ in range(edge_count)]


def best_seconds(call: Callable[[], set[int]]) -> float:
    """Wall time of the fastest of ``REPEATS`` calls."""
    best = float("inf")
    for _ in range(REPEATS):
        gc.collect()
        start = time.perf_counter()
        call()
        best = min(best, time.perf_counter() - start)
    return best


def peak_bytes(call: Callable[[], set[int]]) -> int:
    """``tracemalloc`` peak of one call."""
    gc.collect()
    tracemalloc.start()
    call()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak


def pick_probe(graph: Graph, wanted: str) -> tuple[int, int]:
    """A node of the ``"largest"`` or of the ``"smallest"`` component, taken at the median position."""
    size_of: dict[int, int] = {}
    for component in graph.connected_components():
        for member in component:
            size_of[member] = len(component)
    member_count = max(size_of.values()) if wanted == "largest" else min(size_of.values())
    candidates = sorted(node_id for node_id in size_of if size_of[node_id] == member_count)
    return candidates[len(candidates) // 2], member_count


def measure(shape: str, node_count: int, edges: list[tuple[int, int]], wanted: str) -> None:
    """Print one row: the accessor cold and warm against building every component."""
    node_ids = list(range(node_count))
    graph = Graph(node_ids, edges)
    probe, member_count = pick_probe(graph, wanted)

    def one_component() -> set[int]:
        return graph.connected_component(probe)

    def every_component() -> set[int]:
        return next(component for component in graph.connected_components() if probe in component)

    gc.collect()
    cold_graph = Graph(node_ids, edges)
    start = time.perf_counter()
    cold_result = cold_graph.connected_component(probe)
    cold = time.perf_counter() - start
    del cold_graph
    graph.without_nodes([])  # the cheapest public method that builds the shared id map
    warm = best_seconds(one_component)
    every = best_seconds(every_component)
    one_peak = peak_bytes(one_component)
    every_peak = peak_bytes(every_component)
    if cold_result != every_component():
        message = f"{shape}: the two paths disagree"
        raise AssertionError(message)
    print(  # noqa: T201 — a benchmark reports its table on stdout
        f"{shape:<24} {node_count:>9} {member_count:>9} {cold * 1e3:>9.1f} {warm * 1e3:>9.1f} "
        f"{every * 1e3:>9.1f} {every / warm:>8.2f} {one_peak / 2**20:>9.2f} {every_peak / 2**20:>9.2f}"
    )


def main() -> None:
    print(  # noqa: T201 — a benchmark reports its table on stdout
        f"{'shape':<24} {'nodes':>9} {'members':>9} {'cold ms':>9} {'warm ms':>9} "
        f"{'all ms':>9} {'ratio':>8} {'one MiB':>9} {'all MiB':>9}"
    )
    for node_count in NODE_COUNTS:
        measure("one giant component", node_count, chain_components(node_count, node_count), "largest")
        measure(
            "1000 equal components",
            node_count,
            chain_components(node_count, node_count // EQUAL_COMPONENT_COUNT),
            "largest",
        )
        measure("all singletons", node_count, [], "largest")
        mixture = sparse_mixture(node_count)
        measure("sparse mixture, giant", node_count, mixture, "largest")
        measure("sparse mixture, smallest", node_count, mixture, "smallest")


if __name__ == "__main__":
    main()
