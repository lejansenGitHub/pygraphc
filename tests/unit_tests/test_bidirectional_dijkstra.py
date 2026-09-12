"""Differential tests for the bidirectional source-target Dijkstra."""

import array
import random

import networkx as nx
import pytest

import pygraphc
from pygraphc import Graph, GraphView

SHAPES = ("sparse", "dense", "parallel", "disconnected")


def _random_case(seed: int) -> dict[str, object]:
    """Build one seeded random weighted graph plus a source-target pair."""
    rng = random.Random(seed)
    shape = SHAPES[seed % len(SHAPES)]
    directed = seed % 2 == 0
    node_count = rng.choice([1, 2, 3, 8, 25, 60])
    node_ids = [node_index * 3 + 1 for node_index in range(node_count)]

    if shape == "sparse":
        edge_count = rng.randint(0, node_count * 2)
    elif shape == "dense":
        edge_count = rng.randint(node_count, node_count * 6)
    elif shape == "parallel":
        edge_count = rng.randint(node_count, node_count * 3)
    else:
        edge_count = rng.randint(0, node_count)

    edges: list[tuple[int, int]] = []
    for _ in range(edge_count):
        if shape == "disconnected":
            half = max(1, node_count // 2)
            block = rng.choice([node_ids[:half], node_ids[half:]]) or node_ids
            edges.append((rng.choice(block), rng.choice(block)))
        elif shape == "parallel" and edges and rng.random() < 0.5:
            edges.append(rng.choice(edges))
        else:
            edges.append((rng.choice(node_ids), rng.choice(node_ids)))

    weights = [rng.choice([0.0, 1.0, rng.uniform(0.0, 10.0)]) for _ in edges]
    excluded_edges = [index for index in range(len(edges)) if rng.random() < 0.15] if seed % 3 == 0 else []
    excluded_nodes = [rng.choice(node_ids)] if seed % 7 == 0 and node_count > 3 else []
    return {
        "node_ids": node_ids,
        "edges": edges,
        "weights": weights,
        "directed": directed,
        "source": rng.choice(node_ids),
        "target": rng.choice(node_ids),
        "excluded_edges": excluded_edges,
        "excluded_nodes": excluded_nodes,
    }


def _min_weight_by_pair(case: dict[str, object]) -> dict[tuple[int, int], float]:
    """Minimum weight per ordered node pair, so parallel edges collapse."""
    excluded_edge_set = set(case["excluded_edges"])
    excluded_node_set = set(case["excluded_nodes"])
    minimum: dict[tuple[int, int], float] = {}
    for edge_index, ((source, target), weight) in enumerate(zip(case["edges"], case["weights"], strict=True)):
        if edge_index in excluded_edge_set:
            continue
        if source in excluded_node_set or target in excluded_node_set:
            continue
        pairs = [(source, target)] if case["directed"] else [(source, target), (target, source)]
        for pair in pairs:
            if pair not in minimum or weight < minimum[pair]:
                minimum[pair] = weight
    return minimum


def _walk_weight(path: list[int], pair_weights: dict[tuple[int, int], float]) -> float:
    """Total weight of a path, asserting every step is an existing edge."""
    total = 0.0
    for step in range(len(path) - 1):
        pair = (path[step], path[step + 1])
        assert pair in pair_weights, f"path step {pair} is not an edge"
        total += pair_weights[pair]
    return total


def _networkx_graph(case: dict[str, object]) -> nx.MultiGraph | nx.MultiDiGraph:
    """The same graph in networkx, with excluded edges and nodes dropped."""
    graph: nx.MultiGraph | nx.MultiDiGraph = nx.MultiDiGraph() if case["directed"] else nx.MultiGraph()
    excluded_node_set = set(case["excluded_nodes"])
    graph.add_nodes_from(node_id for node_id in case["node_ids"] if node_id not in excluded_node_set)
    excluded_edge_set = set(case["excluded_edges"])
    for edge_index, ((source, target), weight) in enumerate(zip(case["edges"], case["weights"], strict=True)):
        if edge_index in excluded_edge_set:
            continue
        if source in excluded_node_set or target in excluded_node_set:
            continue
        graph.add_edge(source, target, weight=weight)
    return graph


def _graph_under_test(case: dict[str, object]) -> Graph | GraphView:
    graph = Graph(case["node_ids"], case["edges"], directed=case["directed"])
    if not case["excluded_nodes"] and not case["excluded_edges"]:
        return graph
    view = graph.without_edges(case["excluded_edges"])
    if case["excluded_nodes"]:
        view = view.without_nodes(case["excluded_nodes"])
    return view


@pytest.mark.parametrize("seed_block", range(7))
def test_bidirectional_matches_one_directional_and_networkx(seed_block: int) -> None:
    """The returned path is a valid walk whose weight matches both references."""
    for seed in range(seed_block * 60, seed_block * 60 + 60):
        case = _random_case(seed)
        source, target = case["source"], case["target"]
        if source in case["excluded_nodes"] or target in case["excluded_nodes"]:
            continue

        graph = _graph_under_test(case)
        path = graph.shortest_path(case["weights"], source, target)
        if source == target:
            assert path == [source], f"seed {seed}"
            continue
        one_directional = graph.shortest_path_lengths(case["weights"], source).get(target)
        try:
            reference = nx.dijkstra_path_length(_networkx_graph(case), source, target, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            reference = None

        if one_directional is None:
            assert path == [], f"seed {seed}: expected no path, got {path}"
            assert reference is None, f"seed {seed}: networkx found a path"
            continue

        assert path, f"seed {seed}: expected a path of weight {one_directional}"
        assert path[0] == source, f"seed {seed}"
        assert path[-1] == target, f"seed {seed}"
        assert len(set(path)) == len(path), f"seed {seed}: {path} visits a node twice"
        assert _walk_weight(path, _min_weight_by_pair(case)) == pytest.approx(one_directional), f"seed {seed}"
        assert reference == pytest.approx(one_directional), f"seed {seed}"


def test_first_node_settled_by_both_searches_is_not_the_answer() -> None:
    """The frontiers meet on node 3, but the shortest path runs through node 2.

    The two searches settle node 3 from both sides first, at 0.8 + 0.8 = 1.6,
    while the shortest path weighs 1.4 and is only ever seen as a relaxed edge.
    """
    edges = [(1, 2), (2, 4), (1, 3), (3, 4)]
    weights = [0.1, 1.3, 0.8, 0.8]
    graph = Graph([1, 2, 3, 4], edges)
    assert graph.shortest_path(weights, 1, 4) == [1, 2, 4]
    assert graph.shortest_path_lengths(weights, 1)[4] == pytest.approx(1.4)


def test_first_node_settled_by_both_searches_is_not_the_answer_directed() -> None:
    """Same shape with one-way edges: the meeting node is still not the answer."""
    edges = [(1, 2), (2, 4), (1, 3), (3, 4)]
    weights = [0.1, 1.3, 0.8, 0.8]
    graph = Graph([1, 2, 3, 4], edges, directed=True)
    assert graph.shortest_path(weights, 1, 4) == [1, 2, 4]


def test_malformed_weight_on_an_uninspected_edge_still_raises() -> None:
    """Every element of a weight sequence is validated, not only the ones read.

    The search from 1 to 3 never looks at the edge between 4 and 5, but a
    sequence of weights is converted whole, so ``shortest_path`` rejects the
    same input that ``shortest_path_lengths`` rejects.
    """
    graph = Graph([1, 2, 3, 4, 5], [(1, 2), (2, 3), (4, 5)])
    weights = [1.0, 1.0, "not a number"]
    with pytest.raises(TypeError):
        graph.shortest_path(weights, 1, 3)
    with pytest.raises(TypeError):
        graph.shortest_path_lengths(weights, 1)


def test_weights_length_is_validated_before_the_search() -> None:
    """A length mismatch is an O(1) check, too short and too long alike."""
    graph = Graph([1, 2, 3], [(1, 2), (2, 3)])
    with pytest.raises(ValueError, match="weights length"):
        graph.shortest_path([1.0], 1, 3)
    with pytest.raises(ValueError, match="weights length"):
        graph.shortest_path([1.0, 1.0, 1.0], 1, 3)
    with pytest.raises(ValueError, match="weights length"):
        graph.shortest_path(array.array("d", [1.0]), 1, 3)


def test_float64_buffer_weights_give_the_same_path_as_a_list() -> None:
    """The zero-copy buffer path and the converted sequence path agree."""
    edges = [(1, 2), (2, 3), (3, 4), (1, 4)]
    weights = [1.0, 1.0, 1.0, 4.0]
    graph = Graph([1, 2, 3, 4], edges)
    assert graph.shortest_path(array.array("d", weights), 1, 4) == graph.shortest_path(weights, 1, 4)


def test_source_equals_target() -> None:
    """A node reaches itself over the empty path, so the path is the node alone."""
    graph = Graph([1, 2, 3], [(1, 2), (2, 3)])
    assert graph.shortest_path([1.0, 1.0], 2, 2) == [2]


def test_unreachable_target() -> None:
    """The two components share no edge, so no path exists."""
    graph = Graph([1, 2, 3, 4], [(1, 2), (3, 4)])
    assert graph.shortest_path([1.0, 1.0], 1, 4) == []


def test_single_node_graph() -> None:
    """The only node is its own source and target."""
    graph = Graph([1], [])
    assert graph.shortest_path([], 1, 1) == [1]


def test_empty_graph() -> None:
    """Neither endpoint exists, so there is nothing to connect."""
    graph = Graph([], [])
    assert graph.shortest_path([], 1, 2) == []


def test_zero_weights_stay_on_the_free_path() -> None:
    """Three free edges beat one edge of weight 5."""
    graph = Graph([1, 2, 3, 4], [(1, 2), (2, 3), (3, 4), (1, 4)])
    assert graph.shortest_path([0.0, 0.0, 0.0, 5.0], 1, 4) == [1, 2, 3, 4]


def test_self_loop_is_never_part_of_a_path() -> None:
    """A loop only adds weight, so the direct edge wins."""
    graph = Graph([1, 2], [(1, 1), (1, 2), (2, 2)])
    assert graph.shortest_path([0.5, 2.0, 0.5], 1, 2) == [1, 2]


def test_parallel_edges_use_the_minimum_weight() -> None:
    """Of the two edges between 1 and 2 the cheaper one sets the distance."""
    graph = Graph([1, 2, 3], [(1, 2), (1, 2), (2, 3)])
    weights = [9.0, 1.0, 1.0]
    assert graph.shortest_path(weights, 1, 3) == [1, 2, 3]
    assert graph.shortest_path_lengths(weights, 1)[3] == pytest.approx(2.0)


def test_directed_path_exists_in_one_direction_only() -> None:
    """The backward search follows incoming edges, so the reverse pair has no path."""
    graph = Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)
    assert graph.shortest_path([1.0, 1.0], 1, 3) == [1, 2, 3]
    assert graph.shortest_path([1.0, 1.0], 3, 1) == []


def test_directed_prefers_the_cheaper_of_two_one_way_routes() -> None:
    """Both routes run source to target, so only the weights decide."""
    graph = Graph([1, 2, 3, 4], [(1, 2), (2, 4), (1, 3), (3, 4)], directed=True)
    assert graph.shortest_path([5.0, 5.0, 1.0, 1.0], 1, 4) == [1, 3, 4]
    assert graph.shortest_path([1.0, 1.0, 5.0, 5.0], 1, 4) == [1, 2, 4]


def test_directed_graph_without_edges() -> None:
    """A directed graph without edges has no reverse adjacency to search."""
    graph = Graph([1, 2], [], directed=True)
    assert graph.shortest_path([], 1, 2) == []


def test_masked_edge_forces_the_detour() -> None:
    """Masking the direct edge leaves only the three-edge route."""
    weights = [1.0, 1.0, 1.0, 1.0]
    graph = Graph([1, 2, 3, 4], [(1, 4), (1, 2), (2, 3), (3, 4)])
    assert graph.shortest_path(weights, 1, 4) == [1, 4]
    assert graph.without_edges([0]).shortest_path(weights, 1, 4) == [1, 2, 3, 4]


def test_masked_node_forces_the_detour() -> None:
    """Masking node 2 drops both of its edges, leaving the route over 3 and 4."""
    weights = [1.0, 1.0, 1.0, 1.0, 1.0]
    graph = Graph([1, 2, 3, 4, 5], [(1, 2), (2, 5), (1, 3), (3, 4), (4, 5)])
    assert graph.shortest_path(weights, 1, 5) == [1, 2, 5]
    assert graph.without_nodes([2]).shortest_path(weights, 1, 5) == [1, 3, 4, 5]


def test_masked_target_has_no_path() -> None:
    """A masked endpoint is not part of the view, so nothing reaches it."""
    graph = Graph([1, 2, 3], [(1, 2), (2, 3)])
    assert graph.without_nodes([3]).shortest_path([1.0, 1.0], 1, 3) == []


def test_free_function_matches_the_graph_method() -> None:
    """The free function and the Graph method run the same search."""
    node_ids = [1, 2, 3, 4]
    edges = [(1, 2), (2, 3), (3, 4), (1, 4)]
    weights = [1.0, 1.0, 1.0, 4.0]
    expected = Graph(node_ids, edges).shortest_path(weights, 1, 4)
    assert pygraphc.shortest_path(node_ids, edges, weights, 1, 4) == expected
