"""Undirected operations must scale linearly (or n log n) along every shape axis.

Each shape grows one axis a superlinear implementation hides behind: a path
the traversal depth, a star the degree of one node, singletons the number of
components, a lattice the number of cycles, a sparse random graph the plain
size. An operation that is linear on random graphs but quadratic in degree
fails on the star and nowhere else.
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
Operation = Callable[[Graph[int, int], list[int]], Callable[[], object]]

SHAPES: dict[str, Shape] = {
    "path": shapes.path,
    "star": shapes.star,
    "singletons": shapes.singletons,
    "lattice": shapes.lattice,
    "sparse_random": shapes.sparse_random,
}


def _unit_weight_buffer(graph: Graph[int, int]) -> array.array[float]:
    return array.array("d", [1.0] * graph.edge_count)


def _split_hub(graph: Graph[int, int], node_ids: list[int]) -> Callable[[], object]:
    half_of_the_hub_edges = graph.incident_edge_indices(0)[::2]
    return lambda: graph.split_node(0, node_ids[-1] + 1, half_of_the_hub_edges)


OPERATIONS: dict[str, Operation] = {
    "connected_components": lambda graph, _node_ids: lambda: list(graph.connected_components()),
    "connected_component": lambda graph, _node_ids: lambda: graph.connected_component(0),
    "connected_components_with_branch_ids": lambda graph, _node_ids: (
        lambda: list(graph.connected_components_with_branch_ids())
    ),
    "component_labels": lambda graph, _node_ids: graph.component_labels,
    "bridges": lambda graph, _node_ids: graph.bridges,
    "bridges_with_branch_ids": lambda graph, _node_ids: graph.bridges_with_branch_ids,
    "articulation_points": lambda graph, _node_ids: graph.articulation_points,
    "biconnected_components": lambda graph, _node_ids: lambda: list(graph.biconnected_components()),
    "two_edge_connected_components": lambda graph, _node_ids: lambda: list(graph.two_edge_connected_components()),
    "cycle_basis": lambda graph, _node_ids: graph.cycle_basis,
    "bfs": lambda graph, _node_ids: lambda: graph.bfs(0),
    "shortest_path": lambda graph, node_ids: lambda: graph.shortest_path(_unit_weight_buffer(graph), 0, node_ids[-1]),
    "shortest_path_lengths": lambda graph, _node_ids: (
        lambda: graph.shortest_path_lengths(_unit_weight_buffer(graph), 0)
    ),
    "nodes_on_simple_paths": lambda graph, node_ids: lambda: graph.nodes_on_simple_paths(0, [node_ids[-1]]),
    "without_edges": lambda graph, _node_ids: lambda: graph.without_edges(range(0, graph.edge_count, 7)),
    "without_nodes": lambda graph, node_ids: lambda: graph.without_nodes(node_ids[::7]),
    "with_edges": lambda graph, node_ids: (
        lambda: graph.with_edges([(node_ids[0], node_ids[-1])] * 100, list(range(-100, 0)))
    ),
    "split_node": _split_hub,
    "neighbors_of_node_zero": lambda graph, _node_ids: lambda: graph.neighbors(0),
    "masked_connected_components": lambda graph, _node_ids: (
        lambda: list(graph.without_edges([0]).connected_components())
    ),
    "masked_bridges": lambda graph, _node_ids: lambda: graph.without_nodes([0]).bridges(),
}


def _prepare(operation: Operation, shape: Shape) -> Callable[[int], Callable[[], object]]:
    def prepare(size: int) -> Callable[[], object]:
        node_ids, edges = shape(size)
        graph = Graph(node_ids, edges, branch_ids=list(range(len(edges))))
        return operation(graph, node_ids)

    return prepare


@pytest.mark.parametrize("shape_name", list(SHAPES))
@pytest.mark.parametrize("operation_name", list(OPERATIONS))
def test_operation_scales_linearly(operation_name: str, shape_name: str) -> None:
    """Every operation here touches each node and edge a bounded number of times."""
    scaling = measure_scaling(_prepare(OPERATIONS[operation_name], SHAPES[shape_name]))

    assert_scales_linearly(f"{operation_name}/{shape_name}", scaling)


@pytest.mark.parametrize("shape_name", list(SHAPES))
def test_parsing_scales_linearly(shape_name: str) -> None:
    """Interning node ids and building the adjacency is one pass over nodes and edges."""

    def prepare(size: int) -> Callable[[], object]:
        node_ids, edges = SHAPES[shape_name](size)
        branch_ids = list(range(len(edges)))
        return lambda: Graph(node_ids, edges, branch_ids=branch_ids)

    assert_scales_linearly(f"parse/{shape_name}", measure_scaling(prepare))
