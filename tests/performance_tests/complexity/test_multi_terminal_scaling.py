"""Operations on edges with terminals must scale linearly along the axes multi-terminal edges add.

One edge with a terminal at every node grows the number of terminals of a single
edge, a chain of three-terminal edges grows the depth, and random edges with two
to four terminals grow the plain size. BFS, Dijkstra and the Tarjan search expand
each multi-terminal edge once, so they may not pay per pair of terminals.
"""

from collections.abc import Callable

import pytest

from performance_tests import shapes
from performance_tests.complexity.scaling import assert_scales_linearly, measure_scaling
from pygraphc import Graph

pytestmark = pytest.mark.performance

Shape = Callable[[int], tuple[list[int], list[tuple[int, ...]]]]
Operation = Callable[[Graph[int, int]], Callable[[], object]]

SHAPES: dict[str, Shape] = {
    "wide_edge": shapes.wide_edge,
    "terminal_chain": shapes.terminal_chain,
    "random_terminals": shapes.random_terminals,
}

OPERATIONS: dict[str, Operation] = {
    "connected_components": lambda graph: lambda: list(graph.connected_components()),
    "connected_components_with_branch_ids": lambda graph: lambda: list(graph.connected_components_with_branch_ids()),
    "component_labels": lambda graph: graph.component_labels,
    "masked_terminals_view": lambda graph: lambda: list(graph.without_terminals([(0, 0)]).connected_components()),
    "bfs": lambda graph: lambda: graph.bfs(0),
    "shortest_path_lengths": lambda graph: _shortest_path_lengths_with_unit_weights(graph),
    "shortest_path": lambda graph: _shortest_path_with_unit_weights(graph),
    "shortest_path_lengths_negative_terminals": lambda graph: _shortest_path_lengths_with_a_negative_terminal(graph),
    "shortest_path_negative_terminals": lambda graph: _shortest_path_with_a_negative_terminal(graph),
    "critical_edges": lambda graph: graph.critical_edges,
    "articulation_points": lambda graph: graph.articulation_points,
    "biconnected_components": lambda graph: lambda: list(graph.biconnected_components()),
    "two_edge_connected_components": lambda graph: lambda: list(graph.two_edge_connected_components()),
    "nodes_on_simple_paths": lambda graph: lambda: graph.nodes_on_simple_paths(0, [graph.node_count - 1]),
    "edge_indices_at": lambda graph: _edge_indices_at_first_edge(graph),
}


def _shortest_path_lengths_with_unit_weights(graph: Graph[int, int]) -> Callable[[], object]:
    """Build one weight per terminal outside the timed call."""
    weights = [1.0] * graph.terminal_count
    return lambda: graph.shortest_path_lengths(weights, 0)


def _shortest_path_with_unit_weights(graph: Graph[int, int]) -> Callable[[], object]:
    """From the first to the last node, with one weight per terminal built outside the timed call."""
    weights = [1.0] * graph.terminal_count
    return lambda: graph.shortest_path(weights, 0, graph.node_count - 1)


def _weights_with_a_negative_terminal(graph: Graph[int, int]) -> list[float]:
    """The first terminal of every edge at -0.5, every other at 1.0: each pass costs at least 0.5."""
    weights = [1.0] * graph.terminal_count
    for edge_index in range(graph.edge_count):
        terminals = graph._terminals_of(edge_index)
        if len(terminals) >= 2:
            weights[terminals.start] = -0.5
    return weights


def _shortest_path_lengths_with_a_negative_terminal(graph: Graph[int, int]) -> Callable[[], object]:
    """The labelled search: a second label per node and per edge state, still linear."""
    weights = _weights_with_a_negative_terminal(graph)
    return lambda: graph.shortest_path_lengths(weights, 0)


def _shortest_path_with_a_negative_terminal(graph: Graph[int, int]) -> Callable[[], object]:
    """The labelled search stopped at the target, path read back through the labels.

    The search runs from the source only, so it stops after the share of the graph
    closer than the target. The target is the farthest node, settled last: a fixed
    node such as the last one lands at a different share for every random size and
    bends the fitted exponent without any change in the work per node.
    """
    weights = _weights_with_a_negative_terminal(graph)
    distances = graph.shortest_path_lengths(weights, 0)
    farthest_node = max(distances, key=distances.__getitem__)
    return lambda: graph.shortest_path(weights, 0, farthest_node)


def _edge_indices_at_first_edge(graph: Graph[int, int]) -> Callable[[], object]:
    """Build the lookup of a fresh graph inside the timed call: one pass over the terminals."""
    node_ids = [0, 1]

    def edge_indices_at() -> object:
        graph._edge_indices_by_terminal_node_indices = None
        return graph.edge_indices_at(node_ids)

    return edge_indices_at


def _prepare(operation: Operation, shape: Shape) -> Callable[[int], Callable[[], object]]:
    def prepare(size: int) -> Callable[[], object]:
        node_ids, edges = shape(size)
        return operation(Graph(node_ids, edges, branch_ids=list(range(len(edges)))))

    return prepare


@pytest.mark.parametrize("shape_name", list(SHAPES))
@pytest.mark.parametrize("operation_name", list(OPERATIONS))
def test_terminal_operation_scales_linearly(operation_name: str, shape_name: str) -> None:
    """Every operation here visits each terminal a bounded number of times."""
    scaling = measure_scaling(_prepare(OPERATIONS[operation_name], SHAPES[shape_name]))
    assert_scales_linearly(f"terminals/{operation_name}/{shape_name}", scaling)


@pytest.mark.parametrize("shape_name", list(SHAPES))
def test_terminal_parsing_scales_linearly(shape_name: str) -> None:
    """Interning node ids and building the terminal incidence is one pass over the terminals."""

    def prepare(size: int) -> Callable[[], object]:
        node_ids, edges = SHAPES[shape_name](size)
        return lambda: Graph(node_ids, edges)

    assert_scales_linearly(f"terminals/parse/{shape_name}", measure_scaling(prepare))
