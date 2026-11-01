"""Customer experience: one realistic large input per workflow, end to end, against a generous budget.

These do not track regressions (instruction counts do) or scaling (the
complexity tier does); they catch an operation that should take a second
taking a minute. Each call runs once, cold, from plain lists the way a caller
holds its data, so parsing is inside the time. The budgets are the 1M-node
limits the walltime suite asserted before; they stand in for product
requirements until those are written down.
"""

from __future__ import annotations

import array
import random
import time
from collections.abc import Callable

import pytest

from performance_tests import shapes
from pygraphc import (
    Graph,
    articulation_points,
    bfs,
    biconnected_components,
    bridges,
    connected_components,
    multi_source_shortest_path_lengths,
    shortest_path,
    shortest_path_lengths,
)
from pygraphc.reduction import MultiGraph, reduce

pytestmark = pytest.mark.performance

NODE_COUNT = 1_000_000

Edges = list[tuple[int, int]]


@pytest.fixture(scope="module")
def sparse_graph() -> tuple[list[int], Edges]:
    """One million nodes, a spanning path plus random chords."""
    return shapes.sparse_random(NODE_COUNT)


@pytest.fixture(scope="module")
def directed_acyclic_graph() -> tuple[list[int], Edges]:
    """One million nodes, edges from lower to higher id."""
    return shapes.directed_acyclic(NODE_COUNT)


def _weights(edge_count: int) -> array.array[float]:
    rng = random.Random(7)
    return array.array("d", [rng.uniform(0.1, 10.0) for _ in range(edge_count)])


def _assert_within_budget(call: Callable[[], object], budget_seconds: float) -> None:
    start = time.perf_counter()
    call()
    elapsed = time.perf_counter() - start
    assert elapsed < budget_seconds, f"took {elapsed:.2f}s, budget {budget_seconds:.1f}s"


# ── Undirected, from lists ──


@pytest.mark.parametrize(
    ("operation", "budget_seconds"),
    [
        (lambda node_ids, edges: list(connected_components(node_ids, edges)), 2.0),
        (lambda node_ids, edges: bridges(node_ids, edges), 4.0),
        (lambda node_ids, edges: articulation_points(node_ids, edges), 4.0),
        (lambda node_ids, edges: list(biconnected_components(node_ids, edges)), 4.0),
        (lambda node_ids, edges: bfs(node_ids, edges, 0), 4.0),
        (lambda node_ids, edges: Graph(node_ids, edges).cycle_basis(), 3.0),
    ],
    ids=["connected_components", "bridges", "articulation_points", "biconnected_components", "bfs", "cycle_basis"],
)
def test_undirected_workflow_within_budget(
    sparse_graph: tuple[list[int], Edges],
    operation: Callable[[list[int], Edges], object],
    budget_seconds: float,
) -> None:
    """Parsing a million-node graph and answering one question must feel immediate."""
    node_ids, edges = sparse_graph
    _assert_within_budget(lambda: operation(node_ids, edges), budget_seconds)


@pytest.mark.parametrize(
    ("operation", "budget_seconds"),
    [
        (lambda node_ids, edges, weights: shortest_path(node_ids, edges, weights, 0, NODE_COUNT - 1), 6.0),
        (lambda node_ids, edges, weights: shortest_path_lengths(node_ids, edges, weights, 0), 6.0),
        (
            lambda node_ids, edges, weights: multi_source_shortest_path_lengths(
                node_ids, edges, weights, [0, NODE_COUNT // 2, NODE_COUNT - 1]
            ),
            6.0,
        ),
    ],
    ids=["shortest_path", "shortest_path_lengths", "multi_source_shortest_path_lengths"],
)
def test_weighted_workflow_within_budget(
    sparse_graph: tuple[list[int], Edges],
    operation: Callable[[list[int], Edges, array.array[float]], object],
    budget_seconds: float,
) -> None:
    """Weighted queries on a million-node graph, weights handed over as a float buffer."""
    node_ids, edges = sparse_graph
    weights = _weights(len(edges))
    _assert_within_budget(lambda: operation(node_ids, edges, weights), budget_seconds)


def test_excluding_one_edge_from_a_parsed_graph_is_instant(sparse_graph: tuple[list[int], Edges]) -> None:
    """A loop that removes one edge at a time creates one view per edge; creating it must not cost a parse."""
    graph = Graph(*sparse_graph)
    _assert_within_budget(lambda: graph.without_edges([0]), 0.01)


# ── Directed ──


@pytest.mark.parametrize(
    ("operation", "budget_seconds"),
    [
        (lambda graph: graph.topological_sort(), 5.0),
        (lambda graph: graph.dag_longest_path(), 5.0),
        (lambda graph: list(graph.strongly_connected_components()), 5.0),
        (lambda graph: list(graph.weakly_connected_components()), 3.0),
        (lambda graph: graph.bfs(0), 4.0),
    ],
    ids=["topological_sort", "dag_longest_path", "strongly_connected_components", "weakly_connected_components", "bfs"],
)
def test_directed_workflow_within_budget(
    directed_acyclic_graph: tuple[list[int], Edges],
    operation: Callable[[Graph[int, int]], object],
    budget_seconds: float,
) -> None:
    """Parsing a million-node DAG and ordering or labelling it must feel immediate."""
    node_ids, edges = directed_acyclic_graph
    _assert_within_budget(lambda: operation(Graph(node_ids, edges, directed=True)), budget_seconds)


# ── Reduction ──


def test_reducing_a_million_node_graph_within_budget() -> None:
    """The node-preserving reduction of a tree with chords, one preserved node per hundred nodes."""
    node_ids, edges = shapes.tree_with_chords(NODE_COUNT)
    graph = MultiGraph(node_ids, dict(enumerate(edges)))
    preserved_nodes = set(random.Random(1).sample(node_ids, NODE_COUNT // 100))

    _assert_within_budget(lambda: reduce(graph, preserved_nodes, engine="c"), 30.0)
