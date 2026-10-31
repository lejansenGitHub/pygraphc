"""Operations on edges with ports must scale linearly along the axes multiport edges add.

One edge with a port at every node grows the number of ports of a single
edge, a chain of three-port edges grows the depth, and random edges with two
to four ports grow the plain size. BFS expands each multiport edge once, so it
may not pay per pair of ports.
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
    "port_chain": shapes.port_chain,
    "random_ports": shapes.random_ports,
}

OPERATIONS: dict[str, Operation] = {
    "connected_components": lambda graph: lambda: list(graph.connected_components()),
    "connected_components_with_branch_ids": lambda graph: lambda: list(graph.connected_components_with_branch_ids()),
    "component_labels": lambda graph: graph.component_labels,
    "masked_ports_view": lambda graph: lambda: list(graph.without_ports([(0, 0)]).connected_components()),
    "bfs": lambda graph: lambda: graph.bfs(0),
}


def _prepare(operation: Operation, shape: Shape) -> Callable[[int], Callable[[], object]]:
    def prepare(size: int) -> Callable[[], object]:
        node_ids, edges = shape(size)
        return operation(Graph(node_ids, edges, branch_ids=list(range(len(edges)))))

    return prepare


@pytest.mark.parametrize("shape_name", list(SHAPES))
@pytest.mark.parametrize("operation_name", list(OPERATIONS))
def test_port_operation_scales_linearly(operation_name: str, shape_name: str) -> None:
    """Every operation here visits each port a bounded number of times."""
    scaling = measure_scaling(_prepare(OPERATIONS[operation_name], SHAPES[shape_name]))
    assert_scales_linearly(f"ports/{operation_name}/{shape_name}", scaling)


@pytest.mark.parametrize("shape_name", list(SHAPES))
def test_port_parsing_scales_linearly(shape_name: str) -> None:
    """Interning node ids and building the port incidence is one pass over the ports."""

    def prepare(size: int) -> Callable[[], object]:
        node_ids, edges = SHAPES[shape_name](size)
        return lambda: Graph(node_ids, edges)

    assert_scales_linearly(f"ports/parse/{shape_name}", measure_scaling(prepare))
