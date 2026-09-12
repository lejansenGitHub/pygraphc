"""Performance tests for the terminal-preserving reduction.

A sparse random graph against a straightforward networkx implementation of
the same three moves, and the hub-heavy case that a per-move scan of the
smaller incidence set made quadratic.
"""

import random
import time
import timeit

import networkx
import pytest

from pygraphc.reduction import MultiGraph, leaves, reduce

pytestmark = pytest.mark.performance


def sparse_multigraph(node_count: int, edge_count: int, seed: int) -> MultiGraph[int]:
    rng = random.Random(seed)
    nodes = list(range(node_count))
    endpoints = {edge_id: (rng.randrange(node_count), rng.randrange(node_count)) for edge_id in range(edge_count)}
    return MultiGraph(nodes, endpoints)


def networkx_reduce(graph: MultiGraph[int], terminals: set[int]) -> set[int]:
    """The three moves as a networkx caller writes them: full sweeps to a fixpoint.

    Self-loops take part in no move and are dropped first, terminal-free
    components next. Degree-1 non-terminals are removed, degree-2
    non-terminals with two distinct neighbours are contracted into one of
    them, and parallel edges collapse to one; the sweep repeats until nothing
    changes. Returns the surviving node set.
    """
    nx_graph = networkx.MultiGraph()
    nx_graph.add_nodes_from(graph.nodes)
    for edge_id, (from_node, to_node) in graph.endpoints.items():
        if from_node != to_node:
            nx_graph.add_edge(from_node, to_node, key=edge_id)
    for component in list(networkx.connected_components(nx_graph)):
        if not component & terminals:
            nx_graph.remove_nodes_from(component)
    changed = True
    while changed:
        changed = False
        for node_id in list(nx_graph.nodes):
            if node_id in terminals or node_id not in nx_graph:
                continue
            degree = nx_graph.degree(node_id)
            if degree == 1:
                nx_graph.remove_node(node_id)
                changed = True
            elif degree == 2:
                neighbours = list(nx_graph.neighbors(node_id))
                if len(neighbours) == 2:
                    networkx.contracted_nodes(nx_graph, neighbours[0], node_id, self_loops=False, copy=False)
                    changed = True
        for from_node, to_node in {(min(pair), max(pair)) for pair in nx_graph.edges(keys=False)}:
            keys = list(nx_graph[from_node][to_node])
            for key in keys[1:]:
                nx_graph.remove_edge(from_node, to_node, key=key)
                changed = True
    return set(nx_graph.nodes)


def test_reduce_20k_nodes_is_not_slower_than_the_networkx_moves() -> None:
    node_count = 20_000
    graph = sparse_multigraph(node_count, edge_count=25_000, seed=42)
    terminals = set(random.Random(7).sample(graph.nodes, node_count // 100))

    kernel_elapsed = min(timeit.repeat(lambda: reduce(graph, terminals), number=1, repeat=3))
    networkx_elapsed = min(timeit.repeat(lambda: networkx_reduce(graph, terminals), number=1, repeat=3))
    reduced = reduce(graph, terminals)
    networkx_nodes = networkx_reduce(graph, terminals)

    print(  # noqa: T201 — benchmark output is intentional
        f"\n  reduce 20k nodes / 25k edges / 200 terminals (best of 3): kernel {kernel_elapsed:.3f}s, "
        f"networkx sweeps {networkx_elapsed:.3f}s "
        f"-> {len(reduced.graph.nodes)} nodes, {len(reduced.graph.endpoints)} edges"
    )
    assert terminals <= set(reduced.graph.nodes)
    assert set(reduced.graph.nodes) == networkx_nodes
    assert kernel_elapsed < 1.0, f"took {kernel_elapsed:.3f}s, limit 1s"
    assert kernel_elapsed <= networkx_elapsed, (
        f"kernel {kernel_elapsed:.3f}s slower than networkx {networkx_elapsed:.3f}s"
    )


def test_two_hubs_joined_by_8000_two_paths_reduce_in_linear_time() -> None:
    """Every series move at a hub pair used to scan the smaller hub's incidence
    set for the parallel check, O(d) per move and quadratic overall."""
    path_count = 8_000
    nodes = list(range(path_count + 2))
    endpoints: dict[int, tuple[int, int]] = {}
    for middle in range(2, path_count + 2):
        endpoints[2 * middle] = (0, middle)
        endpoints[2 * middle + 1] = (middle, 1)
    graph = MultiGraph(nodes, endpoints)

    start = time.perf_counter()
    reduced = reduce(graph, terminals={0, 1})
    elapsed = time.perf_counter() - start

    print(f"\n  reduce two hubs with {path_count} two-paths: {elapsed:.3f}s")  # noqa: T201 — benchmark output is intentional
    (tree,) = reduced.provenance.values()
    assert reduced.graph.nodes == [0, 1]
    assert len(leaves(tree)) == 2 * path_count
    assert elapsed < 1.0, f"took {elapsed:.3f}s, limit 1s"
