"""The reduction and its folds must scale linearly along every shape axis.

Two hubs joined by many two-paths grow the degree of the hubs, the axis on
which a per-move scan of the incidence set was once quadratic. A path between
two preserved nodes grows the depth of one series tree, the axis of every fold.
"""

import random
from collections.abc import Callable

import pytest

from performance_tests import shapes
from performance_tests.complexity.scaling import assert_scales_linearly, measure_scaling
from pygraphc import Graph
from pygraphc.reduction import MultiGraph, Partition, closed, minimal_toggles, paths, reduce, scenario

pytestmark = pytest.mark.performance


def _two_hubs(node_count: int) -> tuple[list[int], list[tuple[int, int]]]:
    """Hubs 0 and 1 joined by (n - 2) two-paths through one middle node each."""
    middles = range(2, node_count)
    return list(range(node_count)), [edge for middle in middles for edge in ((0, middle), (middle, 1))]


SHAPES: dict[str, Callable[[int], tuple[list[int], list[tuple[int, int]]]]] = {
    "tree_with_chords": shapes.tree_with_chords,
    "path": shapes.path,
    "star": shapes.star,
    "two_hubs": _two_hubs,
    "lattice": shapes.lattice,
}


def _preserved_nodes(shape_name: str, node_ids: list[int]) -> set[int]:
    """The two ends for a path and the hubs for two hubs, one node in a hundred otherwise."""
    if shape_name == "path":
        return {node_ids[0], node_ids[-1]}
    if shape_name == "two_hubs":
        return {0, 1}
    return {0, *random.Random(1).sample(node_ids, max(2, len(node_ids) // 100))}


def _multigraph(shape_name: str, size: int) -> tuple[MultiGraph[int], set[int]]:
    node_ids, edges = SHAPES[shape_name](size)
    return MultiGraph(node_ids, dict(enumerate(edges))), _preserved_nodes(shape_name, node_ids)


@pytest.mark.parametrize("shape_name", list(SHAPES))
@pytest.mark.parametrize("engine", ["c", "moves", "rounds"])
def test_reduce_scales_linearly(engine: str, shape_name: str) -> None:
    """Every move removes a node or an edge and touches a bounded neighbourhood."""

    def prepare(size: int) -> Callable[[], object]:
        graph, preserved_nodes = _multigraph(shape_name, size)
        return lambda: reduce(graph, preserved_nodes, engine=engine)

    assert_scales_linearly(f"reduce_{engine}/{shape_name}", measure_scaling(prepare))


@pytest.mark.parametrize("shape_name", list(SHAPES))
def test_python_engine_scales_linearly(shape_name: str) -> None:
    """The reference worklist makes the same moves in Python."""

    def prepare(size: int) -> Callable[[], object]:
        graph, preserved_nodes = _multigraph(shape_name, size)
        return lambda: reduce(graph, preserved_nodes, engine="python")

    assert_scales_linearly(f"reduce_python/{shape_name}", measure_scaling(prepare))


@pytest.mark.parametrize("shape_name", ["path", "two_hubs"])
def test_tree_folds_scale_linearly(shape_name: str) -> None:
    """A path gives one series tree as deep as the graph, two hubs one parallel node as wide."""

    def prepare(size: int) -> Callable[[], object]:
        graph, preserved_nodes = _multigraph(shape_name, size)
        (tree,) = reduce(graph, preserved_nodes).provenance.values()
        all_closed = dict.fromkeys(graph.endpoints, True)

        def fold() -> None:
            closed(tree, all_closed)
            minimal_toggles(tree, all_closed, target_closed=False)
            paths(tree, cutoff=2)

        return fold

    assert_scales_linearly(f"tree_folds/{shape_name}", measure_scaling(prepare))


@pytest.mark.parametrize("shape_name", ["tree_with_chords", "lattice"])
def test_scenario_and_partition_scale_linearly(shape_name: str) -> None:
    """Masking removed edges and labelling components is one pass over the graph."""

    def prepare(size: int) -> Callable[[], object]:
        graph, _preserved_nodes = _multigraph(shape_name, size)
        active = set(graph.endpoints)
        removed = set(range(0, len(active), 50))

        def apply() -> None:
            scenario(graph, active, removed)
            Partition.from_components(graph, active)

        return apply

    assert_scales_linearly(f"scenario_and_partition/{shape_name}", measure_scaling(prepare))


def test_all_edge_paths_scales_linearly_in_depth() -> None:
    """A chain has exactly one path; finding it must not copy the path prefix at every step."""

    def prepare(size: int) -> Callable[[], object]:
        node_ids, edges = shapes.path(size)
        graph = Graph(node_ids, edges)
        return lambda: graph.all_edge_paths(0, node_ids[-1])

    assert_scales_linearly("all_edge_paths/path", measure_scaling(prepare))
