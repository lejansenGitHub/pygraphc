"""Multi-terminal edge case behavior up for discussion: one test per entry of the README list.

Each test pins the current answer to one entry of the README section
*Multi-terminal edge case behavior*. When a decision changes, change its test here
first, then the code. ``pytest -m multi_terminal_edge_case`` selects these tests and
every other test that pins the same behavior.
"""

import pytest

from pygraphc import Graph

pytestmark = pytest.mark.multi_terminal_edge_case


def _memberships(view: object) -> dict[int, list[int]]:
    """Branch id → sorted nodes of the component it belongs to."""
    return {
        branch_id: sorted(node_ids)
        for node_ids, branch_ids in view.connected_components_with_branch_ids()  # type: ignore[attr-defined]
        for branch_id in branch_ids
    }


def _components(view: object) -> list[list[int]]:
    return sorted(sorted(component) for component in view.connected_components())  # type: ignore[attr-defined]


def test_edge_case_1_two_terminal_edge_with_both_ends_masked_belongs_nowhere() -> None:
    """EC1: no active terminal, no component; it used to form a component without nodes."""
    view = Graph([1, 2, 3], [(1, 2), (2, 3)], branch_ids=[10, 11]).without_nodes([1, 2])

    assert _memberships(view) == {11: [3]}


def test_edge_case_2_one_active_terminal_belongs_to_its_node_and_is_never_critical() -> None:
    """EC2: joins nothing, belongs to the component of its one active terminal, makes nothing critical."""
    graph = Graph([1, 2, 3, 4], [(1, 2), (2, 3, 4)], branch_ids=[10, 11])
    view = graph.without_terminals([(1, 1), (1, 2)])

    assert _components(view) == [[1, 2], [3], [4]]
    assert _memberships(view) == {10: [1, 2], 11: [1, 2]}
    assert view.critical_edges() == [0]
    assert view.articulation_points() == set()


def test_edge_case_3_two_terminal_edge_with_one_terminal_masked_joins_nothing() -> None:
    """EC3: like a masked edge for connection, but it stays a member of its active end's component."""
    view = Graph([1, 2], [(1, 2)], branch_ids=[10]).without_terminals([(0, 1)])

    assert _components(view) == [[1], [2]]
    assert _memberships(view) == {10: [1]}


def test_edge_case_4_single_terminal_edge() -> None:
    """EC4: an undirected edge may have one terminal; a directed edge needs an in-terminal and an out-terminal."""
    graph = Graph([1, 2], [(1,), (1, 2)], branch_ids=[10, 11])

    assert _memberships(graph) == {10: [1, 2], 11: [1, 2]}
    assert graph.critical_edges() == [1]
    with pytest.raises(ValueError, match="at least one terminal"):
        Graph([1, 2], [((1,), ())], directed=True)


def test_edge_case_5_all_terminals_at_one_node() -> None:
    """EC5: joins nothing, belongs to that node's component, forms no block and is never critical."""
    graph = Graph([1, 2], [(1, 1, 1), (2, 2)], branch_ids=[10, 11])

    assert _memberships(graph) == {10: [1], 11: [2]}
    assert list(graph.biconnected_components()) == []
    assert graph.critical_edges() == []


def test_edge_case_6_two_terminals_of_one_edge_at_one_node() -> None:
    """EC6: the edge is still critical, and masking one of the two terminals keeps the node attached."""
    graph = Graph([1, 2], [(1, 1, 2)])

    assert graph.critical_edges() == [0]
    assert _components(graph.without_terminals([(0, 0)])) == [[1, 2]]


def test_edge_case_7_a_terminal_at_an_unknown_node_is_rejected() -> None:
    """EC7: as for two-terminal edges; "not connected" is a terminal mask, not a missing node."""
    with pytest.raises(ValueError, match="unknown"):
        Graph([1, 2], [(1, 2, 9)])


def test_edge_case_8_one_active_in_terminal_reaches_every_out_terminal() -> None:
    """EC8: OR semantics; AND (all in-terminals needed) is not offered."""
    graph = Graph([1, 2, 3, 4], [((1, 2), (3, 4))], directed=True)

    assert sorted(graph.bfs(1)) == [1, 3, 4]


def test_edge_case_9_directed_edge_without_an_active_out_terminal() -> None:
    """EC9: transmits nothing, yet its active in-terminals stay joined for weak components."""
    view = Graph([1, 2, 3], [((1, 2), (3,))], directed=True).without_terminals([(0, 2)])

    assert view.bfs(1) == [1]
    assert sorted(sorted(component) for component in view.weakly_connected_components()) == [[1, 2], [3]]


def test_edge_case_10_weights_per_terminal_once_any_edge_has_more_than_two_terminals() -> None:
    """EC10: a two-terminal edge there costs its two terminal weights; a graph of two-terminal edges takes one per
    edge."""
    graph = Graph([1, 2, 3, 4], [(1, 2), (2, 3, 4)])

    assert graph.shortest_path_lengths([1, 2, 10, 20, 30], 1) == {1: 0.0, 2: 3.0, 3: 33.0, 4: 43.0}
    with pytest.raises(ValueError, match="terminal count 5"):
        graph.shortest_path_lengths([1, 1], 1)
    with pytest.raises(ValueError, match="edge count 2"):
        Graph([1, 2, 3], [(1, 2), (2, 3)]).shortest_path_lengths([1, 1, 1, 1], 1)


def test_edge_case_11_passing_through_an_edge_costs_entry_plus_exit() -> None:
    """EC11: the entry terminal's weight plus the exit terminal's weight, whichever terminals the path uses."""
    graph = Graph([1, 2, 3], [(1, 2, 3)])

    assert graph.shortest_path_lengths([1, 10, 100], 2) == {2: 0.0, 1: 11.0, 3: 110.0}


def test_edge_case_12_critical_multi_terminal_edge_on_a_cycle() -> None:
    """EC12: critical when removing the whole edge separates its terminal nodes, even inside a block with a cycle."""
    graph = Graph([1, 2, 3, 4], [(1, 2), (2, 3), (1, 2, 3, 4)])

    assert graph.critical_edges() == [2]
    assert graph.articulation_points() == set()


def test_edge_case_13_blocks_never_split_at_a_multi_terminal_edge() -> None:
    """EC13: a tree-shaped edge is one block; a terminal node hanging on an edge of a cycle joins the cycle's block."""
    tree_shaped = Graph([1, 2, 3], [(1, 2, 3)])
    on_a_cycle = Graph([1, 2, 3, 4], [(1, 2), (2, 3), (1, 2, 3, 4)])

    assert [sorted(block) for block in tree_shaped.biconnected_components()] == [[1, 2, 3]]
    assert [sorted(block) for block in on_a_cycle.biconnected_components()] == [[1, 2, 3, 4]]


def test_edge_case_14_two_edge_connected_components_are_nodes_no_single_edge_separates() -> None:
    """EC14: a critical multi-terminal edge on a cycle still joins the nodes its cycle passes through."""
    tree_shaped = Graph([1, 2, 3], [(1, 2, 3)])
    on_a_cycle = Graph([1, 2, 3, 4], [(1, 2), (2, 3), (1, 2, 3, 4)])
    two_critical_edges_on_one_cycle = Graph([1, 2, 3, 4, 5], [(1, 2, 4), (2, 3, 5), (1, 3)])
    joined_only_through_one_edge = Graph([1, 2, 3, 4], [(1, 2, 3, 4), (1, 2), (3, 4)])

    def components(graph: Graph[int, int]) -> list[list[int]]:
        return sorted(sorted(component) for component in graph.two_edge_connected_components())

    assert components(tree_shaped) == [[1], [2], [3]]
    assert components(on_a_cycle) == [[1, 2, 3], [4]]
    assert components(two_critical_edges_on_one_cycle) == [[1, 2, 3], [4], [5]]
    assert components(joined_only_through_one_edge) == [[1, 2], [3, 4]]


def test_edge_case_15_simple_paths_use_each_multi_terminal_edge_once() -> None:
    """EC15: a simple path enters and leaves an edge through one terminal each and never uses the edge twice."""
    tree_shaped = Graph([1, 2, 3], [(1, 2, 3)])
    through_the_edge_once = Graph([1, 2, 3, 4], [(1, 2, 3), (2, 4), (4, 3)])

    assert tree_shaped.nodes_on_simple_paths(1, [2]) == {1, 2}
    assert through_the_edge_once.nodes_on_simple_paths(1, [2]) == {1, 2, 3, 4}


def test_edge_case_16_edge_indices_at_matches_the_set_of_terminal_nodes() -> None:
    """EC16: exact set match; order, repeats and direction do not matter, an unknown node matches nothing."""
    graph = Graph([1, 2, 3], [(1, 2, 3), (3, 2, 1), (1, 1, 2, 3), (1, 2)])
    directed = Graph([1, 2], [(1, 2), (2, 1)], directed=True)

    assert graph.edge_indices_at([3, 1, 2, 2]) == [0, 1, 2]
    assert graph.edge_indices_at([1, 2]) == [3]
    assert graph.edge_indices_at([1, 9]) == []
    assert directed.edge_indices_at([1, 2]) == [0, 1]


def test_edge_case_17_pair_methods_refuse_the_whole_graph() -> None:
    """EC17: on a graph with one multi-terminal edge, the pair methods raise even when asked about a two-terminal
    edge."""
    graph = Graph([1, 2, 3, 4], [(1, 2), (2, 3, 4)])

    with pytest.raises(NotImplementedError, match=r"critical_edges\(\)"):
        graph.bridges()
    with pytest.raises(NotImplementedError, match=r"edge_indices_at\(\)"):
        graph.edge_indices(1, 2)


def test_edge_case_18_methods_not_yet_on_terminals_refuse_the_whole_graph() -> None:
    """EC18: a method not yet on terminals raises on a graph with a multi-terminal edge or a view with a terminal
    mask."""
    graph = Graph([1, 2, 3, 4], [(1, 2), (2, 3, 4)])
    two_terminal_view = Graph([1, 2, 3], [(1, 2), (2, 3)]).without_terminals([(0, 1)])

    with pytest.raises(NotImplementedError, match="multi-terminal edges"):
        graph.quotient_edges(memoryview(bytes(16)).cast("i"))
    with pytest.raises(NotImplementedError, match="terminal masks"):
        two_terminal_view.quotient_edges(memoryview(bytes(12)).cast("i"))


def test_edge_case_19_negative_terminal_weights_need_every_pass_to_cost_at_least_zero() -> None:
    """EC19: a star equivalent may have one negative terminal weight; a pass that would cost less than 0 is rejected."""
    star = Graph([1, 2, 3], [(1, 2, 3)])
    directed = Graph([1, 2, 3], [((1, 2), (3,))], directed=True)

    assert star.shortest_path_lengths([3, -1, 2], 1) == {1: 0.0, 2: 2.0, 3: 5.0}
    assert directed.shortest_path_lengths([-1, 4, 2], 1) == {1: 0.0, 3: 1.0}
    with pytest.raises(ValueError, match="pass costing -2"):
        star.shortest_path_lengths([-3, 1, 2], 1)
    with pytest.raises(ValueError, match="every in-terminal and out-terminal"):
        directed.shortest_path_lengths([-2, 1, 1], 1)


def test_edge_case_20_a_path_never_leaves_an_edge_through_the_terminal_it_entered_by() -> None:
    """EC20: no U-turn; with a negative terminal weight it would undercut the node itself."""
    star = Graph([1, 2, 3], [(1, 2, 3)])
    star_with_a_tail = Graph([1, 2, 3, 4], [(1, 2, 3), (2, 4)])

    assert star.shortest_path_lengths([3, -1, 2], 2) == {1: 2.0, 2: 0.0, 3: 1.0}
    assert star_with_a_tail.shortest_path_lengths([3, -1, 2, 0.5, 0.5], 4) == {1: 3.0, 2: 1.0, 3: 2.0, 4: 0.0}


def test_edge_case_21_no_second_pass_right_at_the_negative_terminal() -> None:
    """EC21: after leaving an edge through its negative terminal, a path does not pass that edge again from there."""
    star = Graph([1, 2, 3], [(1, 2, 3)])

    assert star.shortest_path_lengths([3, -1, 2], 1)[3] == 5.0
    assert star.shortest_path([3, -1, 2], 1, 3) == [1, 3]


def test_edge_case_22_a_path_may_come_back_to_the_negative_terminal_another_way() -> None:
    """EC22: only the immediate second pass is banned; a detour back to the node allows it, so a node may repeat."""
    star_with_a_loop = Graph([1, 2, 3, 4], [(1, 2, 3), (2, 4), (4, 2)])
    weights = [3, -1, 2, 0.25, 0.25, 0.25, 0.25]

    assert star_with_a_loop.shortest_path_lengths(weights, 1)[3] == 4.0
    assert star_with_a_loop.shortest_path(weights, 1, 3) == [1, 2, 4, 2, 3]


def test_edge_case_23_an_in_and_an_out_terminal_at_one_node_is_a_cycle() -> None:
    """EC23: like a self-loop; topological sort raises and the node stays its own strong component."""
    graph = Graph([1, 2], [((1,), (1, 2))], directed=True)

    with pytest.raises(ValueError, match="cycle"):
        graph.topological_sort()
    assert sorted(sorted(component) for component in graph.strongly_connected_components()) == [[1], [2]]


def test_edge_case_24_an_unweighted_pass_through_an_edge_counts_one() -> None:
    """EC24: 1 -> (edge) -> 4 -> 5 is two passes."""
    graph = Graph([1, 2, 3, 4, 5], [((1, 2), (3, 4)), (4, 5)], directed=True)

    assert graph.dag_longest_path() == [1, 4, 5]


def test_edge_case_25_an_edge_without_an_active_in_terminal_constrains_no_order() -> None:
    """EC25: its out-terminal nodes do not wait for it."""
    view = Graph([1, 2, 3], [((1,), (2, 3)), (3, 1)], directed=True).without_terminals([(0, 0)])

    assert view.topological_sort() == [2, 3, 1]


def test_edge_case_26_degree_counts_terminals_not_edges() -> None:
    """EC26: an edge with two terminals at a node counts twice, like a self-loop; XGI and HyperNetX count edges."""
    graph = Graph([1, 2], [(1, 1, 2)])

    assert graph.degree(1) == 2
    assert graph.incident_edge_indices(1) == [0]


def test_edge_case_27_a_node_is_never_its_own_neighbor() -> None:
    """EC27: neither through a self-loop nor through two terminals of one edge; NetworkX lists the node itself."""
    assert Graph([1, 2], [(1, 1, 2)]).neighbors(1) == {2}
    assert Graph([1], [(1, 1)]).neighbors(1) == set()


def test_edge_case_28_two_terminals_of_one_edge_at_one_node_close_a_cycle() -> None:
    """EC28: like a self-loop, [node]; a strict Berge cycle needs distinct nodes and would not count it."""
    assert Graph([1, 2], [(1, 1, 2)]).cycle_basis() == [[1]]


def test_edge_case_29_two_edges_sharing_two_nodes_close_a_cycle() -> None:
    """EC29: a Berge cycle of length 2, like two parallel two-terminal edges; some authors start at length 3."""
    assert [sorted(cycle) for cycle in Graph([1, 2, 3, 4], [(1, 2, 3), (2, 3, 4)]).cycle_basis()] == [[2, 3]]


def test_edge_case_30_cycles_are_those_of_the_incidence_graph() -> None:
    """EC30: the cycle space is the incidence graph's; a hypercycle notion (reaction networks) is not used."""
    graph = Graph([1, 2, 3, 4], [(1, 2, 3), (2, 3, 4), (1, 4)])

    assert len(graph.cycle_basis()) == (3 + 3 + 2) - 4 - 3 + 1


def test_edge_case_31_parallel_edges_compare_their_terminal_nodes_as_multisets() -> None:
    """EC31: (1, 1, 2) and (1, 2) are not parallel; XGI's edges are node sets and cannot tell them apart."""
    assert not Graph([1, 2], [(1, 1, 2), (1, 2)]).is_multigraph
    assert Graph([1, 2], [(1, 1, 2), (2, 1, 1)]).is_multigraph
    assert Graph([1], [(1,), (1,)]).is_multigraph


def test_edge_case_32_split_node_moves_every_terminal_of_an_edge_at_the_node() -> None:
    """EC32: a self-loop moves whole to the new node; before, only its first end moved, joining the two halves."""
    graph = Graph([1, 2], [(1, 1), (1, 2)])

    assert graph.split_node(1, 9, [0]).neighbors(9) == set()
    assert graph.split_node(1, 9, terminals_to_new_node=[(0, 0)]).neighbors(9) == {1}


def test_edge_case_33_an_edge_joining_no_two_nodes_has_no_biconnected_component() -> None:
    """EC33: like a self-loop, an edge whose active terminals sit at one node gets -1, not a block of its own."""
    labels = list(Graph([1, 2], [(1, 1, 1), (1, 2), (2,)]).bcc_edge_labels())

    assert labels == [-1, 0, -1]


def test_edge_case_34_edge_paths_are_edge_sequences_not_node_paths() -> None:
    """EC34: two node paths through (1, 2, 3) then (2, 3, 4) share one edge sequence and are reported once."""
    assert Graph([1, 2, 3, 4], [(1, 2, 3), (2, 3, 4)]).all_edge_paths(1, 4) == [[0, 1]]
