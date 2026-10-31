"""Edges with ports: storage, connectivity, masks, membership, traversal and weighted paths.

Each test names the decision of ``docs/multiport-decisions.md`` or the masking
case of the README table it checks. Throughout, a three-port edge joins nodes
2, 3 and 4 between two two-port edges, unless a test builds its own graph.
"""

import pytest
from pygraphc._core import port_graph_shape  # noqa: PLC2701 — D1 checks what the C core stores

from pygraphc import Graph, PortGraph


def _three_port_edge_between_two_port_edges() -> PortGraph[int, int]:
    """1 - 2, the three-port edge (2, 3, 4), 4 - 5; branch ids 10, 11, 12."""
    graph = Graph([1, 2, 3, 4, 5], [(1, 2), (2, 3, 4), (4, 5)], branch_ids=[10, 11, 12])
    assert isinstance(graph, PortGraph)
    return graph


def _components(view: object) -> list[list[int]]:
    return sorted(sorted(component) for component in view.connected_components())  # type: ignore[attr-defined]


def _memberships(view: object) -> dict[int, list[int]]:
    """Branch id → sorted nodes of the component it belongs to."""
    return {
        branch_id: sorted(node_ids)
        for node_ids, branch_ids in view.connected_components_with_branch_ids()  # type: ignore[attr-defined]
        for branch_id in branch_ids
    }


# ── D1: no auxiliary nodes or edges ──


def test_multiport_graph_holds_exactly_the_given_nodes_and_edges() -> None:
    """D1: the C core stores an edge as its ports; a star expansion would add a node and replace the edge."""
    graph = _three_port_edge_between_two_port_edges()

    nodes_in_c, edges_in_c, ports_in_c = port_graph_shape(graph._graph_capsule)

    # --- Assert ---
    assert (graph.node_count, graph.edge_count) == (5, 3)
    assert (nodes_in_c, edges_in_c, ports_in_c) == (5, 3, 7)


# ── D2: tuples of node ids, port index is the position ──


def test_port_index_is_the_position_in_the_tuple() -> None:
    """D2: port 1 of (2, 3, 4) is at node 3, so masking it detaches node 3 only."""
    view = _three_port_edge_between_two_port_edges().without_ports([(1, 1)])

    assert _components(view) == [[1, 2, 4, 5], [3]]


def test_two_ports_at_one_node_stay_distinguishable() -> None:
    """D2 and masking case 7: with one of two ports at node 2 masked, node 2 stays attached by the other."""
    graph = Graph([2, 3], [(2, 2, 3)])

    view = graph.without_ports([(0, 0)])

    assert _components(view) == [[2, 3]]


# ── D3: graphs of two-port edges are unchanged ──


def test_a_graph_of_two_port_edges_stays_a_plain_graph() -> None:
    """D3: pairs take the unchanged two-port parser; only a non-pair edge makes a PortGraph."""
    assert type(Graph([1, 2, 3], [(1, 2), (2, 3)])) is Graph


def test_two_port_edge_errors_are_reported_as_before() -> None:
    """D3: an unknown endpoint in a graph of two-port edges raises the two-port parser's own error."""
    with pytest.raises(ValueError, match="edge references unknown node ID"):
        Graph([1, 2], [(1, 9)])


def test_a_port_with_an_unknown_node_is_rejected() -> None:
    """O1 as recommended: every port needs a known node, as every two-port edge endpoint does."""
    with pytest.raises(ValueError, match="edge references unknown node ID"):
        Graph([1, 2], [(1, 2, 9)])


# ── D4: directed edges ──


def test_directed_pair_keeps_its_meaning() -> None:
    """D4: (u, v) in a directed graph with a multiport edge still gives port 0 at u and port 1 at v."""
    graph = Graph([1, 2, 3, 4], [(1, 2), ((2,), (3, 4))], directed=True)

    assert port_graph_shape(graph._graph_capsule) == (4, 2, 5)


def test_directed_multiport_needs_an_in_and_an_out_port() -> None:
    """D4: an edge with no in-nodes or no out-nodes cannot carry anything."""
    with pytest.raises(ValueError, match="at least one port"):
        Graph([1, 2, 3], [((), (2, 3))], directed=True)


# ── D9: masking cases 1 to 8 (connection and membership) ──


def test_masking_case_1_masked_edge_joins_nothing() -> None:
    """Case 1: the masked edge joins nothing and belongs to no component; its nodes keep their two-port edges."""
    view = _three_port_edge_between_two_port_edges().without_edges([1])

    # --- Assert ---
    assert _components(view) == [[1, 2], [3], [4, 5]]
    assert 11 not in _memberships(view)


def test_masking_case_2_remaining_ports_stay_joined() -> None:
    """Case 2: with port 0 (node 2) masked, ports at 3 and 4 still join, and the edge belongs there."""
    view = _three_port_edge_between_two_port_edges().without_ports([(1, 0)])

    # --- Assert ---
    assert _components(view) == [[1, 2], [3, 4, 5]]
    assert _memberships(view)[11] == [3, 4, 5]


def test_masking_case_3_one_active_port_joins_nothing() -> None:
    """Case 3: one active port (node 4) joins nothing; the edge belongs to that port's component."""
    view = _three_port_edge_between_two_port_edges().without_ports([(1, 0), (1, 1)])

    # --- Assert ---
    assert _components(view) == [[1, 2], [3], [4, 5]]
    assert _memberships(view)[11] == [4, 5]


def test_masking_case_4_no_active_port_belongs_nowhere() -> None:
    """Case 4: every port masked is the masked edge of case 1."""
    view = _three_port_edge_between_two_port_edges().without_ports([(1, 0), (1, 1), (1, 2)])

    # --- Assert ---
    assert _components(view) == [[1, 2], [3], [4, 5]]
    assert 11 not in _memberships(view)


def test_masking_case_5_two_port_edge_with_one_port_masked() -> None:
    """Case 5: a two-port edge with port 1 masked joins nothing and belongs to its active end."""
    view = _three_port_edge_between_two_port_edges().without_ports([(0, 1)])

    # --- Assert ---
    assert _components(view) == [[1], [2, 3, 4, 5]]
    assert _memberships(view)[10] == [1]


def test_masking_case_6_masked_node_leaves_and_the_rest_stays_joined() -> None:
    """Case 6: node 3 leaves every result; the edge's ports at 2 and 4 stay joined."""
    view = _three_port_edge_between_two_port_edges().without_nodes([3])

    # --- Assert ---
    assert _components(view) == [[1, 2, 4, 5]]
    assert view.connected_component(2) == {1, 2, 4, 5}
    with pytest.raises(ValueError, match="excluded"):
        view.connected_component(3)


def test_masking_case_7_masked_duplicate_port_keeps_the_node_attached() -> None:
    """Case 7: ports 0 and 1 both sit at node 2; masking one leaves node 2 joined to 3."""
    graph = Graph([1, 2, 3], [(1, 2), (2, 2, 3)], branch_ids=[10, 11])

    view = graph.without_ports([(1, 1)])

    # --- Assert ---
    assert _components(view) == [[1, 2, 3]]
    assert _memberships(view)[11] == [1, 2, 3]


def test_masking_case_8_all_ports_at_one_node_join_nothing() -> None:
    """Case 8: an edge whose ports all sit at node 2 joins nothing and belongs to node 2's component."""
    graph = Graph([1, 2, 3], [(1, 2), (2, 2, 2)], branch_ids=[10, 11])

    # --- Assert ---
    assert _components(graph) == [[1, 2], [3]]
    assert _memberships(graph)[11] == [1, 2]


# ── D10: masks combine as a union ──


def test_masks_combine_as_a_union_in_any_order() -> None:
    """D10: an edge, a port and a node mask give the same view in any order; the two-port edge 4-5 stays."""
    graph = _three_port_edge_between_two_port_edges()

    first = graph.without_edges([0]).without_ports([(1, 2)]).without_nodes([3])
    second = graph.without_nodes([3]).without_edges([0]).without_ports([(1, 2)])

    # --- Assert ---
    assert _components(first) == _components(second) == [[1], [2], [4, 5]]
    assert list(first.component_labels()) == list(second.component_labels())


def test_a_port_beyond_the_edge_is_rejected() -> None:
    """Port indices are per edge, so a three-port edge has no port 3."""
    with pytest.raises(IndexError, match="edge 1 has no port 3"):
        _three_port_edge_between_two_port_edges().without_ports([(1, 3)])


# ── D12: single-port edges ──


def test_single_port_edge_joins_nothing_and_belongs_to_its_node() -> None:
    """D12: an undirected edge with one port, such as a load, joins nothing and belongs to its node's component."""
    graph = Graph([1, 2], [(1,), (1, 2)], branch_ids=[10, 11])

    # --- Assert ---
    assert _components(graph) == [[1, 2]]
    assert _memberships(graph) == {10: [1, 2], 11: [1, 2]}


# ── Labels and later stages ──


def test_component_labels_name_the_smallest_node_index() -> None:
    """As on a graph of two-port edges: the smallest node index per component, -1 for a masked node."""
    view = _three_port_edge_between_two_port_edges().without_edges([1]).without_nodes([5])

    assert list(view.component_labels()) == [0, 0, 2, 3, -1]


def test_methods_of_later_stages_say_so() -> None:
    """Until their stage lands, methods not yet on ports raise instead of answering on the two-port structure alone."""
    with pytest.raises(NotImplementedError, match="critical-elements stage"):
        _three_port_edge_between_two_port_edges().articulation_points()


# ── Traversal ──


def test_bfs_crosses_a_multiport_edge_to_all_of_its_ports() -> None:
    """D7: reaching one port of the three-port edge reaches the nodes of all of its ports, in port order."""
    graph = _three_port_edge_between_two_port_edges()

    # --- Assert ---
    assert graph.bfs(1) == [1, 2, 3, 4, 5]
    assert graph.bfs(3) == [3, 2, 4, 1, 5]


def test_one_active_in_port_reaches_every_out_port() -> None:
    """D5: with the in-port at node 1 masked, the in-port at node 2 alone still feeds both out-ports."""
    graph = Graph([1, 2, 3, 4], [((1, 2), (3, 4))], directed=True)

    view = graph.without_ports([(0, 0)])

    # --- Assert ---
    assert view.bfs(2) == [2, 3, 4]
    assert view.bfs(1) == [1]
    assert view.shortest_path([1.0] * 4, 2, 4) == [2, 4]
    assert view.shortest_path([1.0] * 4, 1, 4) == []


def test_masking_case_9_directed_edge_without_an_active_in_port_transmits_nothing() -> None:
    """Case 9: with both in-ports masked, nothing reaches the out-port."""
    view = Graph([1, 2, 3], [((1, 2), (3,))], directed=True).without_ports([(0, 0), (0, 1)])

    assert view.bfs(1) == [1]


# ── Weighted paths ──


def test_path_through_a_multiport_edge_costs_entry_plus_exit() -> None:
    """D11: one weight per port; the three-port edge costs its entry port's weight plus its exit port's weight."""
    graph = _three_port_edge_between_two_port_edges()
    weights = [1.0, 1.0, 2.0, 5.0, 3.0, 1.0, 1.0]

    # --- Assert ---
    assert graph.shortest_path_lengths(weights, 1) == {1: 0.0, 2: 2.0, 3: 9.0, 4: 7.0, 5: 9.0}
    assert graph.shortest_path_lengths(weights, 3) == {3: 0.0, 2: 7.0, 4: 8.0, 1: 9.0, 5: 10.0}
    assert graph.shortest_path(weights, 1, 5) == [1, 2, 4, 5]
    assert graph.shortest_path(weights, 5, 3) == [5, 4, 3]


def test_masking_case_10_paths_only_through_active_ports() -> None:
    """Case 10: with the port at node 4 masked, no path enters or leaves the three-port edge there."""
    view = _three_port_edge_between_two_port_edges().without_ports([(1, 2)])
    weights = [1.0, 1.0, 2.0, 5.0, 3.0, 1.0, 1.0]

    # --- Assert ---
    assert view.shortest_path_lengths(weights, 1) == {1: 0.0, 2: 2.0, 3: 9.0}
    assert view.shortest_path_lengths(weights, 4) == {4: 0.0, 5: 2.0}
    assert view.shortest_path(weights, 1, 3) == [1, 2, 3]
    assert view.shortest_path(weights, 1, 5) == []


def test_bfs_under_an_edge_mask() -> None:
    """Case 1: a masked three-port edge carries nothing, so node 2 is as far as 1 gets."""
    view = _three_port_edge_between_two_port_edges().without_edges([1])

    assert view.bfs(1) == [1, 2]


def test_bfs_under_a_port_mask() -> None:
    """Cases 2 and 5: a masked port detaches only its node; a two-port edge with a masked port carries nothing."""
    graph = _three_port_edge_between_two_port_edges()

    three_port_edge_port_masked = graph.without_ports([(1, 1)])
    two_port_edge_port_masked = graph.without_ports([(0, 1)])

    # --- Assert ---
    assert three_port_edge_port_masked.bfs(1) == [1, 2, 4, 5]
    assert three_port_edge_port_masked.bfs(3) == [3]
    assert two_port_edge_port_masked.bfs(1) == [1]
    assert two_port_edge_port_masked.bfs(2) == [2, 3, 4, 5]


def test_bfs_under_a_node_mask() -> None:
    """Case 6: a masked node is never reached while the other ports of its edge stay joined."""
    view = _three_port_edge_between_two_port_edges().without_nodes([3])

    assert view.bfs(1) == [1, 2, 4, 5]


def test_bfs_enters_through_the_active_one_of_two_ports_at_a_node() -> None:
    """Case 7: with port 0 at node 2 masked, the edge (2, 2, 3) is entered through port 1."""
    view = Graph([2, 3], [(2, 2, 3)]).without_ports([(0, 0)])

    assert view.bfs(2) == [2, 3]


def test_bfs_never_steps_through_a_single_port_edge() -> None:
    """D12: a single-port edge joins nothing, so it adds no step."""
    graph = Graph([1, 2, 3], [(1,), (1, 2), (3,)])

    # --- Assert ---
    assert graph.bfs(2) == [2, 1]
    assert graph.bfs(3) == [3]


def test_directed_bfs_follows_in_ports_to_out_ports() -> None:
    """D4 and D5: from 1 the search runs through the in-port at 2 to both out-ports, and nothing runs back."""
    graph = Graph([1, 2, 3, 4, 5, 6], [(1, 2), ((2,), (3, 4)), ((4, 5), (6,))], directed=True)

    # --- Assert ---
    assert graph.bfs(1) == [1, 2, 3, 4, 6]
    assert graph.bfs(5) == [5, 6]
    assert graph.bfs(6) == [6]


def test_bfs_over_a_wide_edge_reaches_every_port_once() -> None:
    """One edge with a port at each of 50 nodes is expanded once, in port order."""
    node_ids = list(range(50))

    assert Graph(node_ids, [tuple(node_ids)]).bfs(7) == [7, *range(7), *range(8, 50)]


def test_bfs_contracts_match_graphs_of_two_port_edges() -> None:
    """Unknown and masked sources raise as on a graph of two-port edges."""
    two_port_view = Graph([1, 2, 3], [(1, 2), (2, 3)]).without_nodes([1])
    multiport_view = Graph([1, 2, 3], [(1, 2, 3)]).without_nodes([1])

    # --- Assert ---
    for view in (two_port_view, multiport_view):
        with pytest.raises(ValueError, match="source not in node_ids"):
            view.bfs(9)
        with pytest.raises(ValueError, match="source node is excluded"):
            view.bfs(1)
        assert view.bfs(2) == [2, 3]


# ── Differential check against a brute-force reading of the rules ──

REFERENCE_EDGES = [(0, 1), (1, 2, 3), (3, 4), (2, 5, 5, 6), (6,), (4, 6), (6, 7, 0)]
REFERENCE_DIRECTED_EDGES = [(0, 1), ((1,), (2, 3)), (3, 4), ((2, 5), (5, 6)), ((4,), (6,)), ((6, 7), (0,))]
REFERENCE_MASKS = [
    {},
    {"ports": [(3, 1)]},
    {"edges": [2]},
    {"nodes": [3]},
    {"ports": [(1, 0), (3, 3)], "edges": [5], "nodes": [7]},
]


def _ports_of(edge: tuple[object, ...], *, directed: bool) -> list[tuple[int, bool, bool]]:
    """``(node, can enter, can leave)`` per port, in port order."""
    if not directed:
        return [(node, True, True) for node in edge]  # type: ignore[misc]
    in_nodes, out_nodes = ((edge[0],), (edge[1],)) if isinstance(edge[0], int) else edge
    return [(node, True, False) for node in in_nodes] + [(node, False, True) for node in out_nodes]  # type: ignore[union-attr]


def _reference_steps(
    edges: list[tuple[object, ...]], masks: dict[str, list[object]], *, directed: bool
) -> set[tuple[int, int]]:
    """Every (from node, to node) step over an (entry port, exit port) pair of one edge: O(ports²) per edge."""
    steps: set[tuple[int, int]] = set()
    for edge_index, edge in enumerate(edges):
        active = [
            (position, node, can_enter, can_leave)
            for position, (node, can_enter, can_leave) in enumerate(_ports_of(edge, directed=directed))
            if edge_index not in masks.get("edges", [])
            and (edge_index, position) not in masks.get("ports", [])
            and node not in masks.get("nodes", [])
        ]
        steps |= {
            (entry_node, exit_node)
            for entry, entry_node, can_enter, _ in active
            for exit_port, exit_node, _, can_leave in active
            if entry != exit_port and can_enter and can_leave
        }
    return steps


def _reference_reach(steps: set[tuple[int, int]], source: int) -> set[int]:
    """Closure of the steps from source, by repeated sweeps rather than a queue."""
    reached = {source}
    for _round in range(len(steps) + 1):
        reached |= {to_node for from_node, to_node in steps if from_node in reached}
    return reached


@pytest.mark.parametrize(
    ("direction", "masks"),
    [(direction, masks) for direction in ("undirected", "directed") for masks in REFERENCE_MASKS],
)
def test_bfs_agrees_with_a_brute_force_reference(direction: str, masks: dict[str, list[object]]) -> None:
    """From every unmasked node, BFS reaches exactly the nodes of the pairwise reading of D5 to D9."""
    directed = direction == "directed"
    edges: list[tuple[object, ...]] = REFERENCE_DIRECTED_EDGES if directed else REFERENCE_EDGES  # type: ignore[assignment]
    view = Graph(list(range(8)), edges, directed=directed).without_edges(masks.get("edges", []))
    view = view.without_ports(masks.get("ports", [])).without_nodes(masks.get("nodes", []))
    steps = _reference_steps(edges, masks, directed=directed)

    # --- Assert ---
    for source in (node for node in range(8) if node not in masks.get("nodes", [])):
        order = view.bfs(source)
        assert len(order) == len(set(order))
        assert set(order) == _reference_reach(steps, source)
