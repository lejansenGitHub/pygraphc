"""Directed operations must scale linearly (or n log n) along every shape axis.

A long cycle is one strongly connected component as deep as the graph, the
case an iterative Tarjan must handle without quadratic stack work.
"""

from __future__ import annotations

import array
from collections.abc import Callable

import pytest

from performance_tests import shapes
from performance_tests.complexity.scaling import assert_scales_linearly, measure_scaling
from pygraphc import Graph

pytestmark = pytest.mark.performance

Shape = Callable[[int], tuple[list[int], list[tuple[int, int]]]]
Operation = Callable[[Graph[int, int]], Callable[[], object]]


def _long_cycle(node_count: int) -> tuple[list[int], list[tuple[int, int]]]:
    node_ids, edges = shapes.path(node_count)
    return node_ids, [*edges, (node_count - 1, 0)]


ACYCLIC_SHAPES: dict[str, Shape] = {
    "directed_acyclic": shapes.directed_acyclic,
    "path": shapes.path,
    "star": shapes.star,
}
SHAPES: dict[str, Shape] = {**ACYCLIC_SHAPES, "long_cycle": _long_cycle}

ACYCLIC_OPERATIONS: dict[str, Operation] = {
    "topological_sort": lambda graph: graph.topological_sort,
    "dag_longest_path": lambda graph: graph.dag_longest_path,
}
OPERATIONS: dict[str, Operation] = {
    "strongly_connected_components": lambda graph: lambda: list(graph.strongly_connected_components()),
    "weakly_connected_components": lambda graph: lambda: list(graph.weakly_connected_components()),
    "bfs": lambda graph: lambda: graph.bfs(0),
    "shortest_path_lengths": lambda graph: (
        lambda: graph.shortest_path_lengths(array.array("d", [1.0] * graph.edge_count), 0)
    ),
}


def _prepare(operation: Operation, shape: Shape) -> Callable[[int], Callable[[], object]]:
    def prepare(size: int) -> Callable[[], object]:
        node_ids, edges = shape(size)
        return operation(Graph(node_ids, edges, directed=True))

    return prepare


@pytest.mark.parametrize("shape_name", list(SHAPES))
@pytest.mark.parametrize("operation_name", list(OPERATIONS))
def test_operation_scales_linearly(operation_name: str, shape_name: str) -> None:
    """Every operation here visits each node and edge a bounded number of times."""
    scaling = measure_scaling(_prepare(OPERATIONS[operation_name], SHAPES[shape_name]))
    assert_scales_linearly(f"directed/{operation_name}/{shape_name}", scaling)


@pytest.mark.parametrize("shape_name", list(ACYCLIC_SHAPES))
@pytest.mark.parametrize("operation_name", list(ACYCLIC_OPERATIONS))
def test_acyclic_operation_scales_linearly(operation_name: str, shape_name: str) -> None:
    """Kahn's algorithm and the longest-path relaxation each take every edge once."""
    scaling = measure_scaling(_prepare(ACYCLIC_OPERATIONS[operation_name], ACYCLIC_SHAPES[shape_name]))
    assert_scales_linearly(f"directed/{operation_name}/{shape_name}", scaling)
