"""Timing test for the bidirectional source-target Dijkstra against networkx.

``nx.shortest_path`` with a weight dispatches to networkx's own bidirectional
Dijkstra; ``nx.dijkstra_path`` is the one-directional algorithm.

Both weight input forms are timed. A float64 buffer is handed to C as is, so
only the edges the search inspects are ever read. A list of floats is converted
and validated element by element first, which is proportional to the whole edge
list and dominates the query on a large graph.

Each assertion targets the reference whose margin cannot flip on a loaded
runner. The float64 row is asserted against the bidirectional
``nx.shortest_path`` because it wins by more than an order of magnitude. The
list row is asserted against the one-directional ``nx.dijkstra_path``, which is
the like-for-like comparison for it — both sides then spend their time on work
proportional to the whole edge list — and the margin there is several hundred
times. The list row against ``nx.shortest_path`` is close, so it is printed for
a human to read and deliberately not asserted. Do not "tighten" that into an
assertion; it is a coin flip, not a gate.
"""

import array
import random
import time
from collections.abc import Callable, Sequence

import networkx as nx
import pytest

from pygraphc import Graph

pytestmark = pytest.mark.performance

NODE_COUNT = 100_000
EXTRA_EDGES_PER_NODE = 2
FAST_REPEATS = 25  # millisecond calls, repeated enough that the printed ratio is stable
SLOW_REPEATS = 5  # nx.dijkstra_path takes half a second per call


def _build_graph(seed: int = 7) -> tuple[Graph, nx.Graph, list[float], int, int]:
    """A sparse weighted graph as both a pygraphc Graph and an nx.Graph.

    A spanning path keeps every node connected, random chords give the graph a
    small diameter, and the source-target pair sits at opposite ends of the
    path so the distance is realistic rather than a couple of hops.
    """
    rng = random.Random(seed)
    node_ids = list(range(NODE_COUNT))
    edges = [(index, index + 1) for index in range(NODE_COUNT - 1)]
    edges += [(rng.randrange(NODE_COUNT), rng.randrange(NODE_COUNT)) for _ in range(NODE_COUNT * EXTRA_EDGES_PER_NODE)]
    weights = [rng.uniform(1.0, 10.0) for _ in edges]

    graph = Graph(node_ids, edges)
    networkx_graph = nx.Graph()
    networkx_graph.add_nodes_from(node_ids)
    for (source, target), weight in zip(edges, weights, strict=True):
        if not networkx_graph.has_edge(source, target) or networkx_graph[source][target]["weight"] > weight:
            networkx_graph.add_edge(source, target, weight=weight)
    return graph, networkx_graph, weights, 5, NODE_COUNT - 5


def _best_of(run: Callable[[], list[int]], repeats: int) -> tuple[float, list[int]]:
    best_seconds = float("inf")
    path: list[int] = []
    for _ in range(repeats):
        start = time.perf_counter()
        path = run()
        best_seconds = min(best_seconds, time.perf_counter() - start)
    return best_seconds, path


def test_bidirectional_dijkstra_beats_networkx() -> None:
    """A single source-target query beats networkx for both weight input forms.

    See the module docstring for why each assertion picks its reference.
    """
    graph, networkx_graph, weights, source, target = _build_graph()
    weight_buffer: Sequence[float] = array.array("d", weights)

    list_seconds, list_path = _best_of(lambda: graph.shortest_path(weights, source, target), FAST_REPEATS)
    buffer_seconds, buffer_path = _best_of(lambda: graph.shortest_path(weight_buffer, source, target), FAST_REPEATS)
    bidirectional_seconds, bidirectional_path = _best_of(
        lambda: nx.shortest_path(networkx_graph, source, target, weight="weight"), FAST_REPEATS
    )
    one_directional_seconds, one_directional_path = _best_of(
        lambda: nx.dijkstra_path(networkx_graph, source, target, weight="weight"), SLOW_REPEATS
    )

    expected_weight = nx.path_weight(networkx_graph, one_directional_path, "weight")
    for label, path in (("list", list_path), ("buffer", buffer_path), ("networkx", bidirectional_path)):
        assert nx.path_weight(networkx_graph, path, "weight") == pytest.approx(expected_weight), label

    print(  # noqa: T201
        f"\n{NODE_COUNT} nodes, {graph.edge_count} edges, path of {len(list_path)} nodes, "
        f"best of {FAST_REPEATS} ({SLOW_REPEATS} for nx.dijkstra_path):"
        f"\n  pygraphc, float64 buffer{buffer_seconds * 1000:8.3f} ms"
        f"  ({bidirectional_seconds / buffer_seconds:7.1f}x vs nx.shortest_path, asserted)"
        f"\n  pygraphc, list weights  {list_seconds * 1000:8.3f} ms"
        f"  ({bidirectional_seconds / list_seconds:7.1f}x vs nx.shortest_path, printed only:"
        f" this row is close, because networkx dispatches to a bidirectional search here"
        f" and a list of weights has to be converted whole before ours starts;"
        f" passing a float64 buffer is what makes the comparison lopsided in our favour)"
        f"\n                          {list_seconds * 1000:8.3f} ms"
        f"  ({one_directional_seconds / list_seconds:7.1f}x vs nx.dijkstra_path, asserted)"
        f"\n  nx.shortest_path        {bidirectional_seconds * 1000:8.3f} ms"
        f"\n  nx.dijkstra_path        {one_directional_seconds * 1000:8.3f} ms"
    )

    # Float64 weights: asserted against the bidirectional reference, won by >10x.
    assert buffer_seconds < bidirectional_seconds, (
        f"float64 buffer {buffer_seconds * 1000:.3f} ms is not faster than "
        f"nx.shortest_path {bidirectional_seconds * 1000:.3f} ms"
    )
    # List weights: asserted against the one-directional reference, the like-for-like
    # comparison, won by >100x. Against nx.shortest_path the margin is ~20% — printed, not gated.
    assert list_seconds < one_directional_seconds, (
        f"list weights {list_seconds * 1000:.3f} ms is not faster than "
        f"nx.dijkstra_path {one_directional_seconds * 1000:.3f} ms"
    )
    # Skipping the per-element weight conversion is worth an order of magnitude.
    assert buffer_seconds < list_seconds
