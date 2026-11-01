"""Edges with terminals: storage, connectivity, masks, membership, traversal and weighted paths.

Each test names the decision of ``docs/multi_terminal-decisions.md`` or the masking
case of the README table it checks. Throughout, a three-terminal edge joins nodes
2, 3 and 4 between two two-terminal edges, unless a test builds its own graph.
"""

import array

import pytest

from pygraphc import Graph
from pygraphc._core import graph_shape, has_multi_terminal_edges  # noqa: PLC2701 — D1 checks what the C core stores


def _three_terminal_edge_between_two_terminal_edges() -> Graph[int, int]:
    """1 - 2, the three-terminal edge (2, 3, 4), 4 - 5; branch ids 10, 11, 12."""
    return Graph([1, 2, 3, 4, 5], [(1, 2), (2, 3, 4), (4, 5)], branch_ids=[10, 11, 12])


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


def test_multi_terminal_graph_holds_exactly_the_given_nodes_and_edges() -> None:
    """D1: the C core stores an edge as its terminals; a star expansion would add a node and replace the edge."""
    graph = _three_terminal_edge_between_two_terminal_edges()

    nodes_in_c, edges_in_c, terminals_in_c = graph_shape(graph._graph_capsule)

    # --- Assert ---
    assert (graph.node_count, graph.edge_count) == (5, 3)
    assert (nodes_in_c, edges_in_c, terminals_in_c) == (5, 3, 7)


# ── D2: tuples of node ids, terminal index is the position ──


def test_terminal_index_is_the_position_in_the_tuple() -> None:
    """D2: terminal 1 of (2, 3, 4) is at node 3, so masking it detaches node 3 only."""
    view = _three_terminal_edge_between_two_terminal_edges().without_terminals([(1, 1)])

    assert _components(view) == [[1, 2, 4, 5], [3]]


def test_two_terminals_at_one_node_stay_distinguishable() -> None:
    """D2 and masking case 7: with one of two terminals at node 2 masked, node 2 stays attached by the other."""
    graph = Graph([2, 3], [(2, 2, 3)])

    view = graph.without_terminals([(0, 0)])

    assert _components(view) == [[2, 3]]


# ── D3: graphs of two-terminal edges are unchanged ──


def test_a_graph_of_two_terminal_edges_keeps_no_terminal_list() -> None:
    """D3: pairs take the unchanged two-terminal parser; only a non-pair edge adds a terminal list to the C core."""
    graph = Graph([1, 2, 3], [(1, 2), (2, 3)])

    # --- Assert ---
    assert graph.terminal_count == 4
    assert graph_shape(graph._graph_capsule) == (3, 2, 4)
    assert not has_multi_terminal_edges(graph._graph_capsule)


def _edge_buffer(pairs: list[tuple[int, int]]) -> memoryview:
    """Edges as an int32 buffer of shape (edge_count, 2), the layout a numpy array has."""
    flat = array.array("i", [node_id for pair in pairs for node_id in pair])
    return memoryview(flat).cast("B").cast("i", shape=[len(pairs), 2])


def test_an_edge_buffer_builds_the_same_graph_as_pairs() -> None:
    """The (edge_count, 2) buffer input reads two-terminal edges directly."""
    graph = Graph([1, 2, 3, 4], _edge_buffer([(1, 2), (2, 3)]))

    # --- Assert ---
    assert graph.terminal_count == 4
    assert sorted(sorted(component) for component in graph.connected_components()) == [[1, 2, 3], [4]]


def test_an_edge_buffer_with_an_unknown_node_raises() -> None:
    """The buffer path cleans up after an unknown node instead of leaving a half-built edge list."""
    with pytest.raises(ValueError, match="edge references unknown node ID"):
        Graph([1, 2], _edge_buffer([(1, 9)]))


def test_two_terminal_edge_errors_are_reported_as_before() -> None:
    """D3: an unknown endpoint in a graph of two-terminal edges raises the two-terminal parser's own error."""
    with pytest.raises(ValueError, match="edge references unknown node ID"):
        Graph([1, 2], [(1, 9)])


@pytest.mark.multi_terminal_edge_case
def test_a_terminal_with_an_unknown_node_is_rejected() -> None:
    """O1 as recommended: every terminal needs a known node, as every two-terminal edge endpoint does."""
    with pytest.raises(ValueError, match="edge references unknown node ID"):
        Graph([1, 2], [(1, 2, 9)])


# ── D4: directed edges ──


def test_directed_pair_keeps_its_meaning() -> None:
    """D4: (u, v) in a directed graph with a multi-terminal edge still gives terminal 0 at u and terminal 1 at v."""
    graph = Graph([1, 2, 3, 4], [(1, 2), ((2,), (3, 4))], directed=True)

    assert graph_shape(graph._graph_capsule) == (4, 2, 5)


@pytest.mark.multi_terminal_edge_case
def test_directed_multi_terminal_needs_an_in_and_an_out_terminal() -> None:
    """D4: an edge with no in-nodes or no out-nodes cannot carry anything."""
    with pytest.raises(ValueError, match="at least one terminal"):
        Graph([1, 2, 3], [((), (2, 3))], directed=True)


# ── D9: masking cases 1 to 8 (connection and membership) ──


def test_masking_case_1_masked_edge_joins_nothing() -> None:
    """Case 1: the masked edge joins nothing and belongs to no component; its nodes keep their two-terminal edges."""
    view = _three_terminal_edge_between_two_terminal_edges().without_edges([1])

    # --- Assert ---
    assert _components(view) == [[1, 2], [3], [4, 5]]
    assert 11 not in _memberships(view)


def test_masking_case_2_remaining_terminals_stay_joined() -> None:
    """Case 2: with terminal 0 (node 2) masked, terminals at 3 and 4 still join, and the edge belongs there."""
    view = _three_terminal_edge_between_two_terminal_edges().without_terminals([(1, 0)])

    # --- Assert ---
    assert _components(view) == [[1, 2], [3, 4, 5]]
    assert _memberships(view)[11] == [3, 4, 5]


@pytest.mark.multi_terminal_edge_case
def test_masking_case_3_one_active_terminal_joins_nothing() -> None:
    """Case 3: one active terminal (node 4) joins nothing; the edge belongs to that terminal's component."""
    view = _three_terminal_edge_between_two_terminal_edges().without_terminals([(1, 0), (1, 1)])

    # --- Assert ---
    assert _components(view) == [[1, 2], [3], [4, 5]]
    assert _memberships(view)[11] == [4, 5]


def test_masking_case_4_no_active_terminal_belongs_nowhere() -> None:
    """Case 4: every terminal masked is the masked edge of case 1."""
    view = _three_terminal_edge_between_two_terminal_edges().without_terminals([(1, 0), (1, 1), (1, 2)])

    # --- Assert ---
    assert _components(view) == [[1, 2], [3], [4, 5]]
    assert 11 not in _memberships(view)


@pytest.mark.multi_terminal_edge_case
def test_masking_case_5_two_terminal_edge_with_one_terminal_masked() -> None:
    """Case 5: a two-terminal edge with terminal 1 masked joins nothing and belongs to its active end."""
    view = _three_terminal_edge_between_two_terminal_edges().without_terminals([(0, 1)])

    # --- Assert ---
    assert _components(view) == [[1], [2, 3, 4, 5]]
    assert _memberships(view)[10] == [1]


def test_masking_case_6_masked_node_leaves_and_the_rest_stays_joined() -> None:
    """Case 6: node 3 leaves every result; the edge's terminals at 2 and 4 stay joined."""
    view = _three_terminal_edge_between_two_terminal_edges().without_nodes([3])

    # --- Assert ---
    assert _components(view) == [[1, 2, 4, 5]]
    assert view.connected_component(2) == {1, 2, 4, 5}
    with pytest.raises(ValueError, match="excluded"):
        view.connected_component(3)


@pytest.mark.multi_terminal_edge_case
def test_masking_case_7_masked_duplicate_terminal_keeps_the_node_attached() -> None:
    """Case 7: terminals 0 and 1 both sit at node 2; masking one leaves node 2 joined to 3."""
    graph = Graph([1, 2, 3], [(1, 2), (2, 2, 3)], branch_ids=[10, 11])

    view = graph.without_terminals([(1, 1)])

    # --- Assert ---
    assert _components(view) == [[1, 2, 3]]
    assert _memberships(view)[11] == [1, 2, 3]


@pytest.mark.multi_terminal_edge_case
def test_masking_case_8_all_terminals_at_one_node_join_nothing() -> None:
    """Case 8: an edge whose terminals all sit at node 2 joins nothing and belongs to node 2's component."""
    graph = Graph([1, 2, 3], [(1, 2), (2, 2, 2)], branch_ids=[10, 11])

    # --- Assert ---
    assert _components(graph) == [[1, 2], [3]]
    assert _memberships(graph)[11] == [1, 2]


# ── D10: masks combine as a union ──


def test_masks_combine_as_a_union_in_any_order() -> None:
    """D10: an edge, a terminal and a node mask give the same view in any order; the two-terminal edge 4-5 stays."""
    graph = _three_terminal_edge_between_two_terminal_edges()

    first = graph.without_edges([0]).without_terminals([(1, 2)]).without_nodes([3])
    second = graph.without_nodes([3]).without_edges([0]).without_terminals([(1, 2)])

    # --- Assert ---
    assert _components(first) == _components(second) == [[1], [2], [4, 5]]
    assert list(first.component_labels()) == list(second.component_labels())


def test_a_terminal_beyond_the_edge_is_rejected() -> None:
    """Terminal indices are per edge, so a three-terminal edge has no terminal 3."""
    with pytest.raises(IndexError, match="edge 1 has no terminal 3"):
        _three_terminal_edge_between_two_terminal_edges().without_terminals([(1, 3)])


# ── D12: single-terminal edges ──


@pytest.mark.multi_terminal_edge_case
def test_single_terminal_edge_joins_nothing_and_belongs_to_its_node() -> None:
    """D12: an undirected edge with one terminal, such as a load, joins nothing and belongs to its node's component."""
    graph = Graph([1, 2], [(1,), (1, 2)], branch_ids=[10, 11])

    # --- Assert ---
    assert _components(graph) == [[1, 2]]
    assert _memberships(graph) == {10: [1, 2], 11: [1, 2]}


# ── Labels and later stages ──


def test_component_labels_name_the_smallest_node_index() -> None:
    """As on a graph of two-terminal edges: the smallest node index per component, -1 for a masked node."""
    view = _three_terminal_edge_between_two_terminal_edges().without_edges([1]).without_nodes([5])

    assert list(view.component_labels()) == [0, 0, 2, 3, -1]


@pytest.mark.multi_terminal_edge_case
def test_methods_of_later_stages_say_so() -> None:
    """Until their stage lands, methods not yet on terminals raise instead of answering on the two-terminal structure
    alone."""
    graph = _three_terminal_edge_between_two_terminal_edges()

    # --- Assert ---
    with pytest.raises(NotImplementedError, match="multi-terminal edges"):
        graph.all_edge_paths(1, 5)
    with pytest.raises(NotImplementedError, match="multi-terminal edges"):
        graph.without_edges([0]).quotient_edges(memoryview(bytes(20)).cast("i"))


@pytest.mark.multi_terminal_edge_case
def test_methods_of_later_stages_refuse_a_terminal_mask_on_two_terminal_edges() -> None:
    """A terminal mask on a graph of two-terminal edges is not silently dropped by a method that cannot read it yet."""
    view = Graph([1, 2, 3], [(1, 2), (2, 3)]).without_terminals([(0, 1)])

    with pytest.raises(NotImplementedError, match="terminal masks"):
        view.all_edge_paths(1, 3)


_METHODS_WITHOUT_TERMINAL_MASKS = {
    "all_edge_paths": lambda view: view.all_edge_paths(1, 3),
    "quotient_edges": lambda view: view.quotient_edges(memoryview(bytes(12)).cast("i")),
    "series_parallel_reduce": lambda view: view.series_parallel_reduce(bytes(3), bytes(3)),
}


@pytest.mark.multi_terminal_edge_case
@pytest.mark.parametrize("method_name", list(_METHODS_WITHOUT_TERMINAL_MASKS))
def test_methods_of_later_stages_refuse_every_terminal_mask(method_name: str) -> None:
    """D19: each method whose kernel does not read terminal masks yet raises on a view with one."""
    view = Graph([1, 2, 3], [(1, 2), (2, 3)], branch_ids=[10, 11]).without_terminals([(0, 1)])

    with pytest.raises(NotImplementedError, match=f"{method_name} does not read terminal masks"):
        _METHODS_WITHOUT_TERMINAL_MASKS[method_name](view)


# ── Critical edges, critical nodes and blocks ──


def _sorted_blocks(view: object) -> list[list[int]]:
    return sorted(sorted(block) for block in view.biconnected_components())  # type: ignore[attr-defined]


@pytest.mark.multi_terminal_edge_case
def test_tree_of_edges_has_every_edge_critical() -> None:
    """D14 and case 12: in a tree, every edge is critical, its inner terminal nodes are critical, each edge is one
    block."""
    graph = _three_terminal_edge_between_two_terminal_edges()

    # --- Assert ---
    assert sorted(graph.critical_edges()) == [0, 1, 2]
    assert graph.articulation_points() == {2, 4}
    assert _sorted_blocks(graph) == [[1, 2], [2, 3, 4], [4, 5]]


def test_multi_terminal_edge_on_a_cycle_is_not_critical() -> None:
    """A three-terminal edge whose terminals are also joined by two-terminal edges: one block, nothing critical."""
    graph = Graph([1, 2, 3], [(1, 2), (2, 3), (1, 2, 3)])

    # --- Assert ---
    assert graph.critical_edges() == []
    assert graph.articulation_points() == set()
    assert _sorted_blocks(graph) == [[1, 2, 3]]


@pytest.mark.multi_terminal_edge_case
def test_blocks_merge_at_a_multi_terminal_edge() -> None:
    """D14: the edge (1, 2, 4) lies on the cycle 1-2-3 and holds node 4, so all four nodes form one block."""
    graph = Graph([1, 2, 3, 4], [(1, 2), (2, 3), (3, 1), (1, 2, 4)])

    # --- Assert ---
    assert _sorted_blocks(graph) == [[1, 2, 3, 4]]
    assert graph.articulation_points() == set()
    assert graph.critical_edges() == [3]
    assert sorted(sorted(component) for component in graph.two_edge_connected_components()) == [[1, 2, 3], [4]]


@pytest.mark.multi_terminal_edge_case
def test_masking_case_3_an_edge_with_one_active_terminal_is_never_critical() -> None:
    """Case 3: the three-terminal edge keeps only its terminal at node 2, so neither it nor node 2 is critical through
    it."""
    view = Graph([1, 2, 3, 4], [(1, 2), (2, 3, 4)]).without_terminals([(1, 1), (1, 2)])

    # --- Assert ---
    assert view.critical_edges() == [0]
    assert view.articulation_points() == set()
    assert _sorted_blocks(view) == [[1, 2]]


def test_masking_case_11_critical_elements_on_active_terminals_only() -> None:
    """Case 11: masking the terminal at node 3 breaks the cycle, so the remaining edges become critical."""
    graph = Graph([1, 2, 3], [(1, 2), (2, 3), (1, 3, 2)])
    view = graph.without_terminals([(2, 1)])

    # --- Assert ---
    assert graph.critical_edges() == []
    assert sorted(view.critical_edges()) == [1]
    assert view.articulation_points() == {2}
    assert _sorted_blocks(view) == [[1, 2], [2, 3]]


@pytest.mark.multi_terminal_edge_case
def test_pair_methods_raise_on_multi_terminal_graphs() -> None:
    """D13: a node pair cannot name an edge with more than two terminals, so the pair methods name their replacement."""
    graph = _three_terminal_edge_between_two_terminal_edges()

    # --- Assert ---
    with pytest.raises(NotImplementedError, match=r"critical_edges\(\)"):
        graph.bridges()
    with pytest.raises(NotImplementedError, match=r"critical_edges\(\)"):
        graph.bridges_with_branch_ids()
    with pytest.raises(NotImplementedError, match=r"edge_indices_at\(\)"):
        graph.edge_indices(2, 3)


@pytest.mark.multi_terminal_edge_case
def test_edge_indices_at_finds_edges_by_their_terminal_nodes() -> None:
    """D13: edge_indices_at replaces the pair lookup and also finds two-terminal edges."""
    graph = _three_terminal_edge_between_two_terminal_edges()

    # --- Assert ---
    assert graph.edge_indices_at([4, 3, 2]) == [1]
    assert graph.edge_indices_at([1, 2]) == [0]
    assert graph.edge_indices_at([2, 3]) == []


def test_terminal_mask_on_two_terminal_edges_in_critical_elements() -> None:
    """Case 5 in the Tarjan family: a parallel edge with a masked terminal no longer protects the other one."""
    graph = Graph([1, 2, 3], [(1, 2), (1, 2), (2, 3)], branch_ids=[10, 11, 12])
    view = graph.without_terminals([(1, 0)])

    # --- Assert ---
    assert graph.bridges() == [(2, 3)]
    assert sorted(view.bridges()) == [(1, 2), (2, 3)]
    assert sorted(view.bridges_with_branch_ids()) == [(1, 2, 10), (2, 3, 12)]
    assert sorted(view.critical_edges()) == [0, 2]


# ── Graphs of two-terminal edges follow the same rules ──


@pytest.mark.multi_terminal_edge_case
def test_masking_case_5_on_a_graph_of_two_terminal_edges() -> None:
    """Case 5 without any multi-terminal edge: the edge with a masked terminal joins nothing and belongs to its active
    end."""
    graph = Graph([1, 2, 3], [(1, 2), (2, 3)], branch_ids=[10, 11])
    view = graph.without_terminals([(0, 1)])

    # --- Assert ---
    assert _components(view) == [[1], [2, 3]]
    assert sorted(view.connected_components_with_branch_ids(), key=sorted) == [({1}, {10}), ({2, 3}, {11})]
    assert view.bfs(2) == [2, 3]
    assert view.shortest_path_lengths([1.0, 1.0], 1) == {1: 0.0}
    assert view.shortest_path([1.0, 1.0], 1, 3) == []


@pytest.mark.multi_terminal_edge_case
def test_two_terminal_edge_with_both_ends_masked_belongs_to_no_component() -> None:
    """D8 on a graph of two-terminal edges: with both endpoints masked, the edge has no active terminal and no
    component."""
    view = Graph([1, 2, 3], [(1, 2), (2, 3)], branch_ids=[10, 11]).without_nodes([1, 2])

    assert list(view.connected_components_with_branch_ids()) == [({3}, {11})]


# ── Traversal ──


def test_bfs_crosses_a_multi_terminal_edge_to_all_of_its_terminals() -> None:
    """D7: reaching one terminal of the three-terminal edge reaches the nodes of all of its terminals, in terminal
    order."""
    graph = _three_terminal_edge_between_two_terminal_edges()

    # --- Assert ---
    assert graph.bfs(1) == [1, 2, 3, 4, 5]
    assert graph.bfs(3) == [3, 2, 4, 1, 5]


@pytest.mark.multi_terminal_edge_case
def test_one_active_in_terminal_reaches_every_out_terminal() -> None:
    """D5: with the in-terminal at node 1 masked, the in-terminal at node 2 alone still feeds both out-terminals."""
    graph = Graph([1, 2, 3, 4], [((1, 2), (3, 4))], directed=True)

    view = graph.without_terminals([(0, 0)])

    # --- Assert ---
    assert view.bfs(2) == [2, 3, 4]
    assert view.bfs(1) == [1]
    assert view.shortest_path([1.0] * 4, 2, 4) == [2, 4]
    assert view.shortest_path([1.0] * 4, 1, 4) == []


@pytest.mark.multi_terminal_edge_case
def test_masking_case_9_directed_edge_without_an_active_in_terminal_transmits_nothing() -> None:
    """Case 9: with both in-terminals masked, nothing reaches the out-terminal."""
    view = Graph([1, 2, 3], [((1, 2), (3,))], directed=True).without_terminals([(0, 0), (0, 1)])

    assert view.bfs(1) == [1]


# ── Weighted paths ──


@pytest.mark.multi_terminal_edge_case
def test_path_through_a_multi_terminal_edge_costs_entry_plus_exit() -> None:
    """D11: one weight per terminal; the three-terminal edge costs its entry terminal's weight plus its exit terminal's
    weight."""
    graph = _three_terminal_edge_between_two_terminal_edges()
    weights = [1.0, 1.0, 2.0, 5.0, 3.0, 1.0, 1.0]

    # --- Assert ---
    assert graph.shortest_path_lengths(weights, 1) == {1: 0.0, 2: 2.0, 3: 9.0, 4: 7.0, 5: 9.0}
    assert graph.shortest_path_lengths(weights, 3) == {3: 0.0, 2: 7.0, 4: 8.0, 1: 9.0, 5: 10.0}
    assert graph.shortest_path(weights, 1, 5) == [1, 2, 4, 5]
    assert graph.shortest_path(weights, 5, 3) == [5, 4, 3]


@pytest.mark.multi_terminal_edge_case
def test_path_halves_never_join_into_a_second_pass_right_at_the_negative_terminal() -> None:
    """D21: the halves meet at node 3, one leaving and the other entering the three-terminal edge there; the cheapest
    path joins a best label to a second one, taking the loop over node 6 between the two passes."""
    graph = Graph([1, 2, 3, 4, 5, 6], [(1, 2), (2, 3, 4), (4, 5), (3, 6)])
    weights = [0.5, 0.5, 3.0, -1.0, 2.0, 0.5, 0.5, 0.25, 0.25]

    # --- Assert ---
    assert graph.shortest_path(weights, 1, 5) == [1, 2, 3, 6, 3, 4, 5]
    assert graph.shortest_path_lengths(weights, 1)[5] == 6.0


@pytest.mark.multi_terminal_edge_case
def test_path_halves_never_join_in_a_u_turn_through_an_edge() -> None:
    """D21, edge case 20: both halves reach the three-terminal edge through its terminal of weight 0 at node 2;
    joining there would be a U-turn and return the path [1, 2, 2]."""
    graph = Graph([1, 2], [(1, 2, 2), (1, 2)])
    weights = [3.0, 1.0, 0.0, -1.0, 1.0]

    # --- Assert ---
    assert graph.shortest_path(weights, 1, 2) == [1, 2]
    assert graph.shortest_path_lengths(weights, 1)[2] == 0.0


@pytest.mark.multi_terminal_edge_case
def test_path_halves_meeting_inside_an_edge_count_its_weights_once() -> None:
    """D11, D21: each half entering the three-terminal edge adds its lowest terminal weight as a shift; a meeting
    inside the edge takes both off again, or the detour 1 - 2 - 3 (costing 3) would look as cheap as the direct pass
    (costing 1)."""
    graph = Graph([1, 2, 3], [(1, 2, 3)])
    weights = [-1.0, 1.0, 2.0]

    # --- Assert ---
    assert graph.shortest_path(weights, 1, 3) == [1, 3]
    assert graph.shortest_path_lengths(weights, 1)[3] == 1.0


@pytest.mark.multi_terminal_edge_case
def test_path_halves_join_on_a_pass_leaving_through_the_negative_terminal() -> None:
    """D21: the forward half enters the three-terminal edge at node 2 and leaves through its negative terminal at node
    3; the backward half represents the same pass as a direct one, so they meet on the pass, before the dearer edge
    (1, 3) ends the search."""
    graph = Graph([1, 2, 3, 6], [(1, 2), (2, 3, 6), (1, 3)])
    weights = [0.5, 0.5, 2.0, -1.0, 5.0, 1.25, 1.25]

    # --- Assert ---
    assert graph.shortest_path(weights, 1, 3) == [1, 2, 3]
    assert graph.shortest_path_lengths(weights, 1)[3] == 2.0


def test_masking_case_10_paths_only_through_active_terminals() -> None:
    """Case 10: with the terminal at node 4 masked, no path enters or leaves the three-terminal edge there."""
    view = _three_terminal_edge_between_two_terminal_edges().without_terminals([(1, 2)])
    weights = [1.0, 1.0, 2.0, 5.0, 3.0, 1.0, 1.0]

    # --- Assert ---
    assert view.shortest_path_lengths(weights, 1) == {1: 0.0, 2: 2.0, 3: 9.0}
    assert view.shortest_path_lengths(weights, 4) == {4: 0.0, 5: 2.0}
    assert view.shortest_path(weights, 1, 3) == [1, 2, 3]
    assert view.shortest_path(weights, 1, 5) == []


def test_bfs_under_an_edge_mask() -> None:
    """Case 1: a masked three-terminal edge carries nothing, so node 2 is as far as 1 gets."""
    view = _three_terminal_edge_between_two_terminal_edges().without_edges([1])

    assert view.bfs(1) == [1, 2]


def test_bfs_under_a_terminal_mask() -> None:
    """Cases 2 and 5: a masked terminal detaches only its node; a two-terminal edge with a masked terminal carries
    nothing."""
    graph = _three_terminal_edge_between_two_terminal_edges()

    three_terminal_edge_terminal_masked = graph.without_terminals([(1, 1)])
    two_terminal_edge_terminal_masked = graph.without_terminals([(0, 1)])

    # --- Assert ---
    assert three_terminal_edge_terminal_masked.bfs(1) == [1, 2, 4, 5]
    assert three_terminal_edge_terminal_masked.bfs(3) == [3]
    assert two_terminal_edge_terminal_masked.bfs(1) == [1]
    assert two_terminal_edge_terminal_masked.bfs(2) == [2, 3, 4, 5]


def test_bfs_under_a_node_mask() -> None:
    """Case 6: a masked node is never reached while the other terminals of its edge stay joined."""
    view = _three_terminal_edge_between_two_terminal_edges().without_nodes([3])

    assert view.bfs(1) == [1, 2, 4, 5]


def test_bfs_enters_through_the_active_one_of_two_terminals_at_a_node() -> None:
    """Case 7: with terminal 0 at node 2 masked, the edge (2, 2, 3) is entered through terminal 1."""
    view = Graph([2, 3], [(2, 2, 3)]).without_terminals([(0, 0)])

    assert view.bfs(2) == [2, 3]


def test_bfs_never_steps_through_a_single_terminal_edge() -> None:
    """D12: a single-terminal edge joins nothing, so it adds no step."""
    graph = Graph([1, 2, 3], [(1,), (1, 2), (3,)])

    # --- Assert ---
    assert graph.bfs(2) == [2, 1]
    assert graph.bfs(3) == [3]


def test_directed_bfs_follows_in_terminals_to_out_terminals() -> None:
    """D4 and D5: from 1 the search runs through the in-terminal at 2 to both out-terminals, and nothing runs back."""
    graph = Graph([1, 2, 3, 4, 5, 6], [(1, 2), ((2,), (3, 4)), ((4, 5), (6,))], directed=True)

    # --- Assert ---
    assert graph.bfs(1) == [1, 2, 3, 4, 6]
    assert graph.bfs(5) == [5, 6]
    assert graph.bfs(6) == [6]


def test_bfs_over_a_wide_edge_reaches_every_terminal_once() -> None:
    """One edge with a terminal at each of 50 nodes is expanded once, in terminal order."""
    node_ids = list(range(50))

    assert Graph(node_ids, [tuple(node_ids)]).bfs(7) == [7, *range(7), *range(8, 50)]


def test_bfs_contracts_match_graphs_of_two_terminal_edges() -> None:
    """Unknown and masked sources raise as on a graph of two-terminal edges."""
    two_terminal_view = Graph([1, 2, 3], [(1, 2), (2, 3)]).without_nodes([1])
    multi_terminal_view = Graph([1, 2, 3], [(1, 2, 3)]).without_nodes([1])

    # --- Assert ---
    for view in (two_terminal_view, multi_terminal_view):
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
    {"terminals": [(3, 1)]},
    {"edges": [2]},
    {"nodes": [3]},
    {"terminals": [(1, 0), (3, 3)], "edges": [5], "nodes": [7]},
]


def _terminals_of(edge: tuple[object, ...], *, directed: bool) -> list[tuple[int, bool, bool]]:
    """``(node, can enter, can leave)`` per terminal, in terminal order."""
    if not directed:
        return [(node, True, True) for node in edge]  # type: ignore[misc]
    in_nodes, out_nodes = ((edge[0],), (edge[1],)) if isinstance(edge[0], int) else edge
    return [(node, True, False) for node in in_nodes] + [(node, False, True) for node in out_nodes]  # type: ignore[union-attr]


def _reference_steps(
    edges: list[tuple[object, ...]], masks: dict[str, list[object]], *, directed: bool
) -> set[tuple[int, int]]:
    """Every (from node, to node) step over an (entry terminal, exit terminal) pair of one edge: O(terminals²) per
    edge."""
    steps: set[tuple[int, int]] = set()
    for edge_index, edge in enumerate(edges):
        active = [
            (position, node, can_enter, can_leave)
            for position, (node, can_enter, can_leave) in enumerate(_terminals_of(edge, directed=directed))
            if edge_index not in masks.get("edges", [])
            and (edge_index, position) not in masks.get("terminals", [])
            and node not in masks.get("nodes", [])
        ]
        steps |= {
            (entry_node, exit_node)
            for entry, entry_node, can_enter, _ in active
            for exit_terminal, exit_node, _, can_leave in active
            if entry != exit_terminal and can_enter and can_leave
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
    view = view.without_terminals(masks.get("terminals", [])).without_nodes(masks.get("nodes", []))
    steps = _reference_steps(edges, masks, directed=directed)

    # --- Assert ---
    for source in (node for node in range(8) if node not in masks.get("nodes", [])):
        order = view.bfs(source)
        assert len(order) == len(set(order))
        assert set(order) == _reference_reach(steps, source)


# ── Step C: directed algorithms on multi-terminal edges (D5, D23) ──


def _strong_components(view: object) -> list[list[int]]:
    return sorted(sorted(component) for component in view.strongly_connected_components())  # type: ignore[attr-defined]


def test_strongly_connected_components_pass_through_a_multi_terminal_edge() -> None:
    """D5: 1 reaches 3 through the edge and 3 reaches 1, so they share a component; 2 only reaches them."""
    graph = Graph([1, 2, 3], [((1, 2), (3,)), (3, 1)], directed=True)

    # --- Assert ---
    assert _strong_components(graph) == [[1, 3], [2]]


def test_a_cycle_through_two_out_terminals_is_one_component() -> None:
    """Every active out-terminal is reached: 1 feeds 2 and 3, both lead back to 1."""
    graph = Graph([1, 2, 3, 4], [((1,), (2, 3)), (2, 1), (3, 1), (3, 4)], directed=True)

    # --- Assert ---
    assert _strong_components(graph) == [[1, 2, 3], [4]]


def test_masking_an_in_terminal_breaks_the_strong_component() -> None:
    """Case 9 under a terminal mask: with the in-terminal at 1 masked, 1 no longer reaches 3."""
    view = Graph([1, 2, 3], [((1, 2), (3,)), (3, 1)], directed=True).without_terminals([(0, 0)])

    # --- Assert ---
    assert _strong_components(view) == [[1], [2], [3]]


def test_topological_sort_puts_every_in_terminal_before_every_out_terminal() -> None:
    """((1, 2), (3, 4)): 1 and 2 before 3 and 4."""
    order = Graph([1, 2, 3, 4], [((1, 2), (3, 4))], directed=True).topological_sort()

    # --- Assert ---
    assert set(order[:2]) == {1, 2}
    assert set(order[2:]) == {3, 4}


def test_topological_sort_raises_on_a_cycle_through_a_multi_terminal_edge() -> None:
    """1 feeds 3 through the edge and 3 leads back to 1."""
    graph = Graph([1, 2, 3], [((1, 2), (3,)), (3, 1)], directed=True)

    # --- Assert ---
    with pytest.raises(ValueError, match="cycle"):
        graph.topological_sort()


def test_an_in_and_an_out_terminal_at_one_node_is_a_cycle() -> None:
    """Like a self-loop: the edge leads from node 1 back to node 1."""
    graph = Graph([1, 2], [((1,), (1, 2))], directed=True)

    # --- Assert ---
    with pytest.raises(ValueError, match="cycle"):
        graph.topological_sort()


def test_an_edge_without_an_active_in_terminal_orders_nothing() -> None:
    """Case 9: with its only in-terminal masked, the edge no longer constrains its out-terminals."""
    view = Graph([1, 2, 3], [((1,), (2, 3)), (3, 1)], directed=True).without_terminals([(0, 0)])

    # --- Assert ---
    assert view.topological_sort() == [2, 3, 1]


def test_dag_longest_path_counts_a_pass_through_an_edge_as_one() -> None:
    """Unweighted, 1 -> (edge) -> 4 -> 5 is two passes, longer than any single pass."""
    graph = Graph([1, 2, 3, 4, 5], [((1, 2), (3, 4)), (4, 5)], directed=True)

    # --- Assert ---
    assert graph.dag_longest_path() == [1, 4, 5]


def test_dag_longest_path_weighs_a_pass_entry_plus_exit() -> None:
    """D11: one weight per terminal; entering at 2 (5) and leaving at 4 (7) beats every other pass."""
    graph = Graph([1, 2, 3, 4], [((1, 2), (3, 4))], directed=True)

    # --- Assert ---
    assert graph.dag_longest_path([1.0, 5.0, 2.0, 7.0]) == [2, 4]


# ── Step D: local queries on multi-terminal edges (D24) ──


def test_neighbors_through_a_multi_terminal_edge_are_its_other_terminal_nodes() -> None:
    """D24: every other active terminal of an edge at the node is a neighbor, as two nodes in one hyperedge are."""
    graph = _three_terminal_edge_between_two_terminal_edges()

    # --- Assert ---
    assert graph.neighbors(2) == {1, 3, 4}
    assert graph.neighbors(3) == {2, 4}
    assert graph.incident_edge_indices(2) == [0, 1]


def test_degree_counts_the_terminals_at_the_node() -> None:
    """D24: one per joined terminal at the node; degrees agrees with degree for every node."""
    graph = Graph([1, 2, 3, 4], [(1, 2), (2, 3, 4), (3, 3, 4)])

    # --- Assert ---
    assert [graph.degree(node_id) for node_id in [1, 2, 3, 4]] == [1, 2, 3, 2]
    assert list(graph.degrees()) == [1, 2, 3, 2]
    assert graph.incident_edge_indices(3) == [1, 2]


def test_a_masked_terminal_leaves_the_queries_of_its_node() -> None:
    """D24 with case 2: the other terminals stay joined; the masked terminal's node loses the edge."""
    view = _three_terminal_edge_between_two_terminal_edges().without_terminals([(1, 1)])

    # --- Assert ---
    assert view.neighbors(2) == {1, 4}
    assert view.neighbors(3) == set()
    assert [view.degree(node_id) for node_id in [1, 2, 3, 4, 5]] == [1, 2, 0, 2, 1]
    assert list(view.degrees()) == [1, 2, 0, 2, 1]
    assert view.incident_edge_indices(3) == []


def test_an_edge_with_one_active_terminal_is_incident_to_nothing() -> None:
    """D24 with case 3: a single active terminal joins nothing, so it counts toward no degree."""
    view = _three_terminal_edge_between_two_terminal_edges().without_terminals([(1, 1), (1, 2)])

    # --- Assert ---
    assert view.neighbors(2) == {1}
    assert view.degree(2) == 1
    assert view.incident_edge_indices(2) == [0]
    assert list(view.degrees()) == [1, 1, 0, 1, 1]


def test_a_masked_node_leaves_the_neighbors_of_the_others() -> None:
    """D24 with case 6: its terminal is inactive, the remaining terminals stay joined."""
    view = _three_terminal_edge_between_two_terminal_edges().without_nodes([4])

    # --- Assert ---
    assert view.neighbors(2) == {1, 3}
    assert view.degree(4) == 0
    assert list(view.degrees()) == [1, 2, 1, 0, 0]


def test_directed_queries_follow_in_and_out_terminals() -> None:
    """D24: successors are the out-terminal nodes of edges at the node's in-terminals; predecessors the reverse."""
    graph = Graph([1, 2, 3, 4, 5], [((1, 2), (3, 4)), (4, 5)], directed=True)

    # --- Assert ---
    assert graph.successors(1) == {3, 4}
    assert graph.predecessors(4) == {1, 2}
    assert graph.successors(4) == {5}
    assert (graph.out_degree(1), graph.in_degree(4), graph.in_degree(1)) == (1, 1, 0)
    assert graph.outgoing_edge_indices(1) == [0]
    assert graph.incoming_edge_indices(4) == [0]
    assert list(graph.degrees()) == [1, 1, 0, 1, 0]


def test_a_directed_edge_without_an_active_out_terminal_has_no_successors() -> None:
    """D24 with case 12: an edge left without an active out-terminal transmits nothing."""
    view = Graph([1, 2, 3, 4], [((1, 2), (3, 4))], directed=True).without_terminals([(0, 2), (0, 3)])

    # --- Assert ---
    assert view.successors(1) == set()
    assert view.out_degree(1) == 0
    assert view.outgoing_edge_indices(1) == []
    assert view.predecessors(3) == set()
    assert list(view.degrees()) == [0, 0, 0, 0]


def test_cycle_basis_passes_through_a_multi_terminal_edge() -> None:
    """D25: a Berge cycle enters an edge at one terminal and leaves at another; the tree around it has none."""
    ring = Graph([1, 2, 3, 4], [(1, 2, 3), (3, 4), (4, 1)])

    # --- Assert ---
    assert [sorted(cycle) for cycle in ring.cycle_basis()] == [[1, 3, 4]]
    assert _three_terminal_edge_between_two_terminal_edges().cycle_basis() == []


def test_cycle_basis_counts_the_rank_of_the_incidence_graph() -> None:
    """D25: terminals - nodes - edges + components; one wide edge alone closes no cycle."""
    graph = Graph([1, 2, 3, 4, 5], [(1, 2, 3, 4, 5), (1, 2), (3, 4, 5)])

    # --- Assert ---
    assert Graph([1, 2, 3, 4], [(1, 2, 3, 4)]).cycle_basis() == []
    assert len(graph.cycle_basis()) == (5 + 2 + 3) - 5 - 3 + 1


def test_a_masked_terminal_opens_the_cycle_through_its_edge() -> None:
    """D25 with case 2: the edge still joins its other terminals, so a cycle needing the masked one is gone."""
    ring = Graph([1, 2, 3, 4], [(1, 2, 3), (3, 4), (4, 1)])

    # --- Assert ---
    assert ring.without_terminals([(0, 0)]).cycle_basis() == []
    assert [sorted(cycle) for cycle in ring.without_terminals([(0, 1)]).cycle_basis()] == [[1, 3, 4]]
    assert ring.without_nodes([2]).cycle_basis() == ring.cycle_basis()


# ── Step D: parallel edges, rebuilt views and edge labels (D26, D27) ──


def test_is_multigraph_finds_two_edges_at_the_same_terminal_nodes() -> None:
    """D26: parallel edges have the same terminal nodes in any order, XGI's duplicate edges."""
    # --- Assert ---
    assert Graph([1, 2, 3], [(1, 2, 3), (3, 2, 1)]).is_multigraph
    assert not Graph([1, 2, 3, 4], [(1, 2, 3), (1, 2, 4)]).is_multigraph
    assert not _three_terminal_edge_between_two_terminal_edges().is_multigraph


def test_is_multigraph_compares_a_directed_edge_per_side() -> None:
    """D26: in-terminals with in-terminals, out-terminals with out-terminals."""
    # --- Assert ---
    assert Graph([1, 2, 3], [((1, 2), (3,)), ((2, 1), (3,))], directed=True).is_multigraph
    assert not Graph([1, 2, 3], [((1, 2), (3,)), ((3,), (1, 2))], directed=True).is_multigraph
    assert Graph([1, 2], [(1, 2), ((1,), (2,))], directed=True).is_multigraph


def test_is_multigraph_reads_a_graph_built_from_source_and_target_lists() -> None:
    """D26: the C terminal list holds every edge however it was given; before, src/dst graphs always said False."""
    # --- Assert ---
    assert Graph([1, 2], [1, 2], [2, 1]).is_multigraph
    assert not Graph([1, 2, 3], [1, 2], [2, 3]).is_multigraph


def test_with_edges_adds_a_multi_terminal_edge() -> None:
    """D26: an added edge is written as for Graph() and its new terminal nodes join the view."""
    view = _three_terminal_edge_between_two_terminal_edges().with_edges([(5, 6, 1)], [13])

    # --- Assert ---
    assert view.neighbors(6) == {1, 5}
    assert view.cycle_basis() != []
    assert _components(view.without_branches([13])) == [[1, 2, 3, 4, 5], [6]]


def test_with_edges_keeps_the_terminal_mask_and_adds_active_terminals() -> None:
    """D26: base terminals keep their indices, so the mask still names them; added terminals start active."""
    view = _three_terminal_edge_between_two_terminal_edges().without_terminals([(1, 1)]).with_edges([(3, 5)], [13])

    # --- Assert ---
    assert view.neighbors(3) == {5}
    assert view.neighbors(2) == {1, 4}


def test_split_node_moves_every_terminal_of_an_edge_at_the_node() -> None:
    """D26: an edge index moves all its terminals at the node, here both of (2, 2, 3)."""
    graph = Graph([1, 2, 3], [(1, 2), (2, 2, 3)])
    view = graph.split_node(2, 9, [1])

    # --- Assert ---
    assert view.neighbors(9) == {3}
    assert view.degree(9) == 2
    assert view.neighbors(2) == {1}


def test_split_node_moves_one_terminal() -> None:
    """D26: a terminal in terminals_to_new_node moves alone, e.g. to split a busbar at a coupler."""
    graph = Graph([1, 2, 3], [(1, 2), (2, 2, 3)])
    view = graph.split_node(2, 9, terminals_to_new_node=[(1, 1)])

    # --- Assert ---
    assert view.neighbors(9) == {2, 3}
    assert view.neighbors(2) == {1, 3, 9}


def test_split_node_reroutes_a_directed_edge_in_its_own_form() -> None:
    """D26: (in_nodes, out_nodes) stays in that form; the moved terminal keeps its side."""
    graph = Graph([1, 2, 3], [((1, 2), (3,))], directed=True)
    view = graph.split_node(2, 9, [0])

    # --- Assert ---
    assert view.successors(9) == {3}
    assert view.predecessors(3) == {1, 9}


def test_split_node_carries_a_masked_terminal_with_its_edge() -> None:
    """D26: the rerouted edge keeps the masks of the edge it replaces."""
    view = _three_terminal_edge_between_two_terminal_edges().without_terminals([(1, 1)]).split_node(2, 9, [1])

    # --- Assert ---
    assert view.neighbors(9) == {4}
    assert view.neighbors(3) == set()


def test_split_node_refuses_a_terminal_elsewhere() -> None:
    """D26: a terminal not at the node, or not of the edge, raises instead of moving something else."""
    graph = _three_terminal_edge_between_two_terminal_edges()

    # --- Assert ---
    with pytest.raises(ValueError, match="terminal 1 of edge 1 is not incident to node 2"):
        graph.split_node(2, 9, terminals_to_new_node=[(1, 1)])
    with pytest.raises(IndexError, match="edge 1 has no terminal 3"):
        graph.split_node(2, 9, terminals_to_new_node=[(1, 3)])


def test_bcc_edge_labels_put_a_multi_terminal_edge_in_one_block() -> None:
    """D27 on D14: a tree-shaped wide edge is one block; the two-terminal edges around it are bridges."""
    labels = list(_three_terminal_edge_between_two_terminal_edges().bcc_edge_labels())

    # --- Assert ---
    assert len(set(labels)) == 3
    assert min(labels) == 0


def test_bcc_edge_labels_merge_the_edges_of_a_cycle_through_a_wide_edge() -> None:
    """D27: the ring through (1, 2, 3) is one block, the edge (3, 5) another."""
    labels = list(Graph([1, 2, 3, 4, 5], [(1, 2, 3), (3, 4), (4, 1), (3, 5)]).bcc_edge_labels())

    # --- Assert ---
    assert labels[0] == labels[1] == labels[2]
    assert labels[3] not in {labels[0], -1}


def test_bcc_edge_labels_under_a_terminal_mask() -> None:
    """D27 with case 2: masking the ring's terminal at 1 opens it; an edge left at one node gets -1."""
    ring = Graph([1, 2, 3, 4], [(1, 2, 3), (3, 4), (4, 1)])
    labels = list(ring.without_terminals([(0, 0)]).bcc_edge_labels())

    # --- Assert ---
    assert len(set(labels)) == 3
    assert list(ring.without_terminals([(0, 0), (0, 1)]).bcc_edge_labels())[0] == -1
