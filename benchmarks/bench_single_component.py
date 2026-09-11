"""One component against all of them: ``connected_component`` vs ``connected_components``.

The claim under test is that asking for the component of one node costs about
one component rather than all of them. The graph has many small components, so
the generator has to build a set per component and add every node to one, while
the accessor builds one set from a single scan of the int32 label buffer.

Run with the venv python: ``python benchmarks/bench_single_component.py``.
"""

import time
import tracemalloc
from collections.abc import Callable

from pygraphc import Graph

NODE_COUNTS = (200_000, 1_000_000)
COMPONENT_SIZE = 5


def chain_components(node_count: int) -> Graph:
    """A graph of paths of ``COMPONENT_SIZE`` nodes, so there are many components."""
    edges = [(index, index + 1) for index in range(node_count - 1) if (index + 1) % COMPONENT_SIZE != 0]
    return Graph(list(range(node_count)), edges)


def time_and_peak(call: Callable[[], set[int]]) -> tuple[float, int, int]:
    """Wall time of one call, the tracemalloc peak of a second call, and the result size."""
    start = time.perf_counter()
    result = call()
    elapsed = time.perf_counter() - start
    tracemalloc.start()
    call()
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return elapsed, peak, len(result)


def measure(node_count: int) -> None:
    """Time both paths on one graph size and print the two rows."""
    graph = chain_components(node_count)
    probe = node_count // 2

    def one_component() -> set[int]:
        return graph.connected_component(probe)

    def every_component() -> set[int]:
        return next(component for component in graph.connected_components() if probe in component)

    single, single_peak, single_size = time_and_peak(one_component)
    every, every_peak, every_size = time_and_peak(every_component)
    if single_size != every_size:
        message = f"the two paths disagree: {single_size} members against {every_size}"
        raise AssertionError(message)
    rows = (
        ("connected_component", single, single_peak, f"{every / single:.1f}"),
        ("all, then pick one", every, every_peak, ""),
    )
    for name, seconds, peak, break_even in rows:
        print(  # noqa: T201 — a benchmark reports its table on stdout
            f"{node_count:>10} {name:>22} {seconds:>10.4f} {peak / 2**20:>10.2f} {single_size:>8} {break_even:>10}"
        )


def main() -> None:
    print(  # noqa: T201 — a benchmark reports its table on stdout
        f"{'nodes':>10} {'path':>22} {'seconds':>10} {'peak MiB':>10} {'members':>8} {'break-even':>10}"
    )
    for node_count in NODE_COUNTS:
        measure(node_count)


if __name__ == "__main__":
    main()
