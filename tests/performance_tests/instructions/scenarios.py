"""Instruction-count scenarios: small-to-medium inputs, one call each.

Every scenario is a function that builds its inputs and returns the call to
measure, so construction stays outside the count unless construction is what
the scenario measures. Names are ``<operation>/<shape>_<size>``; a name stays
stable once merged, since the merge-base comparison pairs scenarios by name.
"""

import array
import random
from collections.abc import Callable

import pygraphc
from performance_tests import shapes
from pygraphc import Graph, for_each_edge_excluded
from pygraphc.reduction import MultiGraph, Partition, minimal_toggles, paths, reduce, scenario

Measured = Callable[[], object]
SCENARIOS: dict[str, Callable[[], Measured]] = {}

SIZE = 20_000


def register(name: str) -> Callable[[Callable[[], Measured]], Callable[[], Measured]]:
    """Add a scenario under a name that must be unique."""

    def add(prepare: Callable[[], Measured]) -> Callable[[], Measured]:
        if name in SCENARIOS:
            message = f"scenario {name!r} is registered twice"
            raise ValueError(message)
        SCENARIOS[name] = prepare
        return prepare

    return add


def _graph(shape: Callable[[int], tuple[list[int], list[tuple[int, int]]]], size: int = SIZE) -> Graph[int, int]:
    node_ids, edges = shape(size)
    return Graph(node_ids, edges, branch_ids=list(range(len(edges))))


def _unit_weights(graph: Graph[int, int]) -> list[float]:
    return [1.0] * graph.edge_count


# ── Construction ──


@register("parse/sparse_random_20k")
def _parse_sparse_random() -> Measured:
    node_ids, edges = shapes.sparse_random(SIZE)
    return lambda: Graph(node_ids, edges)


@register("parse_with_branch_ids/sparse_random_20k")
def _parse_with_branch_ids() -> Measured:
    node_ids, edges = shapes.sparse_random(SIZE)
    branch_ids = list(range(len(edges)))
    return lambda: Graph(node_ids, edges, branch_ids=branch_ids)


# ── Connectivity ──


@register("connected_components/sparse_random_20k")
def _connected_components_sparse() -> Measured:
    graph = _graph(shapes.sparse_random)
    return lambda: list(graph.connected_components())


@register("connected_components/singletons_20k")
def _connected_components_singletons() -> Measured:
    graph = _graph(shapes.singletons)
    return lambda: list(graph.connected_components())


@register("connected_components/star_20k")
def _connected_components_star() -> Measured:
    graph = _graph(shapes.star)
    return lambda: list(graph.connected_components())


@register("connected_components_masked/sparse_random_20k")
def _connected_components_masked() -> Measured:
    view = _graph(shapes.sparse_random).without_edges([])
    return lambda: list(view.connected_components())


@register("connected_component/sparse_random_20k")
def _connected_component() -> Measured:
    graph = _graph(shapes.sparse_random)
    return lambda: graph.connected_component(0)


@register("connected_components_with_branch_ids/sparse_random_20k")
def _connected_components_with_branch_ids() -> Measured:
    graph = _graph(shapes.sparse_random)
    return lambda: list(graph.connected_components_with_branch_ids())


@register("component_labels/sparse_random_20k")
def _component_labels() -> Measured:
    graph = _graph(shapes.sparse_random)
    return graph.component_labels


# ── Bridges, articulation points, blocks ──


@register("bridges/sparse_random_20k")
def _bridges() -> Measured:
    graph = _graph(shapes.sparse_random)
    return graph.bridges


@register("bridges_masked/sparse_random_20k")
def _bridges_masked() -> Measured:
    view = _graph(shapes.sparse_random).without_edges([])
    return view.bridges


@register("bridges_with_branch_ids/path_20k")
def _bridges_with_branch_ids() -> Measured:
    graph = _graph(shapes.path)
    return graph.bridges_with_branch_ids


@register("articulation_points/sparse_random_20k")
def _articulation_points() -> Measured:
    graph = _graph(shapes.sparse_random)
    return graph.articulation_points


@register("articulation_points_masked/sparse_random_20k")
def _articulation_points_masked() -> Measured:
    view = _graph(shapes.sparse_random).without_edges([])
    return view.articulation_points


@register("articulation_points/star_20k")
def _articulation_points_star() -> Measured:
    graph = _graph(shapes.star)
    return graph.articulation_points


@register("biconnected_components/lattice_20k")
def _biconnected_components() -> Measured:
    graph = _graph(shapes.lattice)
    return lambda: list(graph.biconnected_components())


@register("bcc_edge_labels/sparse_random_20k")
def _bcc_edge_labels() -> Measured:
    graph = _graph(shapes.sparse_random)
    return graph.bcc_edge_labels


@register("two_edge_connected_components/sparse_random_20k")
def _two_edge_connected_components() -> Measured:
    graph = _graph(shapes.sparse_random)
    return lambda: list(graph.two_edge_connected_components())


@register("nodes_on_simple_paths/sparse_random_20k")
def _nodes_on_simple_paths() -> Measured:
    graph = _graph(shapes.sparse_random)
    return lambda: graph.nodes_on_simple_paths(0, [SIZE - 1])


# ── Traversal and paths ──


@register("bfs/sparse_random_20k")
def _bfs() -> Measured:
    graph = _graph(shapes.sparse_random)
    return lambda: graph.bfs(0)


@register("shortest_path/sparse_random_20k")
def _shortest_path() -> Measured:
    graph = _graph(shapes.sparse_random)
    weights = _unit_weights(graph)
    return lambda: graph.shortest_path(weights, 0, SIZE - 1)


@register("shortest_path_lengths/sparse_random_20k")
def _shortest_path_lengths() -> Measured:
    graph = _graph(shapes.sparse_random)
    weights = _unit_weights(graph)
    return lambda: graph.shortest_path_lengths(weights, 0)


@register("multi_source_shortest_path_lengths/meshed_feeders_20k")
def _multi_source_shortest_path_lengths() -> Measured:
    graph = _graph(shapes.meshed_feeders)
    weights = _unit_weights(graph)
    sources = list(range(0, SIZE, 1_000))
    return lambda: graph.multi_source_shortest_path_lengths(weights, sources)


@register("all_edge_paths/lattice_400_cutoff_10")
def _all_edge_paths() -> Measured:
    graph = _graph(shapes.lattice, 400)
    four_right_four_down = 4 * 20 + 4
    return lambda: list(graph.all_edge_paths(0, [four_right_four_down], cutoff=10))


@register("cycle_basis/lattice_20k")
def _cycle_basis() -> Measured:
    graph = _graph(shapes.lattice)
    return graph.cycle_basis


# ── Views ──


@register("without_edges/sparse_random_20k")
def _without_edges() -> Measured:
    graph = _graph(shapes.sparse_random)
    excluded = list(range(0, graph.edge_count, 7))
    return lambda: graph.without_edges(excluded)


@register("without_nodes/sparse_random_20k")
def _without_nodes() -> Measured:
    graph = _graph(shapes.sparse_random)
    excluded = list(range(0, SIZE, 7))
    return lambda: graph.without_nodes(excluded)


@register("with_edges/sparse_random_20k")
def _with_edges() -> Measured:
    graph = _graph(shapes.sparse_random)
    added = [(node_id, node_id + 2) for node_id in range(0, SIZE - 2, 10)]
    return lambda: graph.with_edges(added, list(range(10**6, 10**6 + len(added))))


@register("for_each_edge_excluded/path_2k")
def _for_each_edge_excluded() -> Measured:
    graph = _graph(shapes.path, 2_000)
    return lambda: list(for_each_edge_excluded(graph, "connected_components"))


# ── Directed ──


def _directed_graph() -> Graph[int, int]:
    node_ids, edges = shapes.directed_acyclic(SIZE)
    return Graph(node_ids, edges, directed=True)


@register("topological_sort/directed_acyclic_20k")
def _topological_sort() -> Measured:
    graph = _directed_graph()
    return graph.topological_sort


@register("dag_longest_path/directed_acyclic_20k")
def _dag_longest_path() -> Measured:
    graph = _directed_graph()
    return graph.dag_longest_path


@register("strongly_connected_components/directed_acyclic_20k")
def _strongly_connected_components() -> Measured:
    graph = _directed_graph()
    return lambda: list(graph.strongly_connected_components())


@register("weakly_connected_components/directed_acyclic_20k")
def _weakly_connected_components() -> Measured:
    graph = _directed_graph()
    return lambda: list(graph.weakly_connected_components())


# ── Reduction ──


def _feeder_multigraph(size: int) -> tuple[MultiGraph[int], set[int]]:
    node_ids, edges = shapes.meshed_feeders(size)
    terminals = set(random.Random(1).sample(node_ids, size // 100))
    return MultiGraph(node_ids, dict(enumerate(edges))), terminals


@register("reduce_c/meshed_feeders_20k")
def _reduce_c() -> Measured:
    graph, terminals = _feeder_multigraph(SIZE)
    return lambda: reduce(graph, terminals, engine="c")


@register("reduce_python/meshed_feeders_5k")
def _reduce_python() -> Measured:
    graph, terminals = _feeder_multigraph(5_000)
    return lambda: reduce(graph, terminals, engine="python")


@register("reduce_moves/meshed_feeders_5k")
def _reduce_moves() -> Measured:
    graph, terminals = _feeder_multigraph(5_000)
    return lambda: reduce(graph, terminals, engine="moves")


@register("reduce_rounds/meshed_feeders_5k")
def _reduce_rounds() -> Measured:
    graph, terminals = _feeder_multigraph(5_000)
    return lambda: reduce(graph, terminals, engine="rounds")


@register("scenario/meshed_feeders_20k")
def _scenario() -> Measured:
    graph, _terminals = _feeder_multigraph(SIZE)
    active = set(graph.endpoints)
    removed = set(range(0, len(active), 50))
    return lambda: scenario(graph, active, removed)


@register("partition_from_components/meshed_feeders_20k")
def _partition_from_components() -> Measured:
    graph, _terminals = _feeder_multigraph(SIZE)
    active = set(graph.endpoints)
    return lambda: Partition.from_components(graph, active)


@register("tree_folds/meshed_feeders_20k")
def _tree_folds() -> Measured:
    graph, terminals = _feeder_multigraph(SIZE)
    reduced = reduce(graph, terminals)
    trees = list(reduced.provenance.values())
    all_closed = dict.fromkeys(graph.endpoints, True)

    def fold_every_tree() -> None:
        for tree in trees:
            minimal_toggles(tree, all_closed, target_closed=False)
            paths(tree, cutoff=3)

    return fold_every_tree


# ── Pairs behind the design claims in test_claims.py ──


def _rebuilt_without_edges(node_ids: list[int], edges: list[tuple[int, int]], excluded: set[int]) -> Graph[int, int]:
    return Graph(node_ids, [edge for index, edge in enumerate(edges) if index not in excluded])


@register("connected_components_node_masked/sparse_random_20k")
def _connected_components_node_masked() -> Measured:
    view = _graph(shapes.sparse_random).without_nodes([])
    return lambda: list(view.connected_components())


@register("bridges_node_masked/sparse_random_20k")
def _bridges_node_masked() -> Measured:
    view = _graph(shapes.sparse_random).without_nodes([])
    return view.bridges


@register("articulation_points_node_masked/sparse_random_20k")
def _articulation_points_node_masked() -> Measured:
    view = _graph(shapes.sparse_random).without_nodes([])
    return view.articulation_points


@register("connected_components_without_one_edge_by_mask/sparse_random_20k")
def _connected_components_without_one_edge_by_mask() -> Measured:
    graph = _graph(shapes.sparse_random)
    return lambda: list(graph.without_edges([0]).connected_components())


@register("connected_components_without_one_edge_by_rebuild/sparse_random_20k")
def _connected_components_without_one_edge_by_rebuild() -> Measured:
    node_ids, edges = shapes.sparse_random(SIZE)
    return lambda: list(_rebuilt_without_edges(node_ids, edges, {0}).connected_components())


@register("bridges_without_one_edge_by_mask/sparse_random_20k")
def _bridges_without_one_edge_by_mask() -> Measured:
    graph = _graph(shapes.sparse_random)
    return lambda: graph.without_edges([0]).bridges()


@register("bridges_without_one_edge_by_rebuild/sparse_random_20k")
def _bridges_without_one_edge_by_rebuild() -> Measured:
    node_ids, edges = shapes.sparse_random(SIZE)
    return lambda: _rebuilt_without_edges(node_ids, edges, {0}).bridges()


@register("connected_components_without_ten_nodes_by_mask/sparse_random_20k")
def _connected_components_without_ten_nodes_by_mask() -> Measured:
    graph = _graph(shapes.sparse_random)
    excluded = list(range(0, SIZE, SIZE // 10))
    return lambda: list(graph.without_nodes(excluded).connected_components())


@register("connected_components_without_ten_nodes_by_rebuild/sparse_random_20k")
def _connected_components_without_ten_nodes_by_rebuild() -> Measured:
    node_ids, edges = shapes.sparse_random(SIZE)
    excluded = set(range(0, SIZE, SIZE // 10))

    def rebuild() -> object:
        kept_node_ids = [node_id for node_id in node_ids if node_id not in excluded]
        kept_edges = [edge for edge in edges if edge[0] not in excluded and edge[1] not in excluded]
        return list(Graph(kept_node_ids, kept_edges).connected_components())

    return rebuild


@register("parse_directed/sparse_random_20k")
def _parse_directed() -> Measured:
    node_ids, edges = shapes.sparse_random(SIZE)
    return lambda: Graph(node_ids, edges, directed=True)


@register("reduce_c/meshed_feeders_5k")
def _reduce_c_small() -> Measured:
    graph, terminals = _feeder_multigraph(5_000)
    return lambda: reduce(graph, terminals, engine="c")


@register("shortest_path_float_buffer/sparse_random_20k")
def _shortest_path_float_buffer() -> Measured:
    graph = _graph(shapes.sparse_random)
    weights = array.array("d", _unit_weights(graph))
    return lambda: graph.shortest_path(weights, 0, SIZE - 1)


class _Branch:
    """A domain object as callers hold them: the graph is gathered from these on every request."""

    __slots__ = ("branch_id", "from_node_id", "to_node_id")

    def __init__(self, branch_id: int, from_node_id: int, to_node_id: int) -> None:
        self.branch_id = branch_id
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id


def _branches() -> tuple[list[int], list[_Branch]]:
    node_ids, edges = shapes.sparse_random(SIZE)
    return node_ids, [_Branch(index, first, second) for index, (first, second) in enumerate(edges)]


@register("workflow_components/branches_20k")
def _workflow_components() -> Measured:
    node_ids, branches = _branches()

    def gather_parse_and_label() -> object:
        edges = [(branch.from_node_id, branch.to_node_id) for branch in branches]
        return list(Graph(node_ids, edges).connected_components())

    return gather_parse_and_label


@register("workflow_components_with_branch_ids/branches_20k")
def _workflow_components_with_branch_ids() -> Measured:
    node_ids, branches = _branches()

    def gather_parse_and_label() -> object:
        edges = [(branch.from_node_id, branch.to_node_id) for branch in branches]
        branch_ids = [branch.branch_id for branch in branches]
        return list(Graph(node_ids, edges, branch_ids=branch_ids).connected_components_with_branch_ids())

    return gather_parse_and_label


# ── Edges with ports ──


def _port_graph(size: int = SIZE) -> Graph[int, int]:
    node_ids, edges = shapes.random_ports(size)
    return Graph(node_ids, edges, branch_ids=list(range(len(edges))))


@register("parse_ports/random_ports_20k")
def _parse_ports() -> Measured:
    node_ids, edges = shapes.random_ports(SIZE)
    return lambda: Graph(node_ids, edges)


@register("port_connected_components/random_ports_20k")
def _port_connected_components() -> Measured:
    graph = _port_graph()
    return lambda: list(graph.connected_components())


@register("port_connected_components_with_branch_ids/random_ports_20k")
def _port_connected_components_with_branch_ids() -> Measured:
    graph = _port_graph()
    return lambda: list(graph.connected_components_with_branch_ids())


@register("port_component_labels/random_ports_20k")
def _port_component_labels() -> Measured:
    graph = _port_graph()
    return graph.component_labels


@register("port_connected_components_masked_ports/random_ports_20k")
def _port_connected_components_masked_ports() -> Measured:
    graph = _port_graph()
    masked = [(edge_index, 0) for edge_index in range(SIZE - 1, graph.edge_count, 5)]
    return lambda: list(graph.without_ports(masked).connected_components())


@register("port_bfs/random_ports_20k")
def _port_bfs() -> Measured:
    graph = _port_graph()
    return lambda: graph.bfs(0)


@register("port_shortest_path_lengths/random_ports_20k")
def _port_shortest_path_lengths() -> Measured:
    graph = _port_graph()
    rng = random.Random(0)
    weights = [rng.uniform(0.1, 10.0) for _ in range(graph.port_count)]
    return lambda: graph.shortest_path_lengths(weights, 0)


# ── Structure learning ──


@register("hill_climb_k2/10_variables_500_samples")
def _hill_climb_k2() -> Measured:
    rng = random.Random(0)
    data = [[rng.randrange(2) for _ in range(10)] for _ in range(500)]
    return lambda: pygraphc.hill_climb_k2(data, [2] * 10, max_indegree=2)
