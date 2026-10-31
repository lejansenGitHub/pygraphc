"""Multiport edge case behavior up for discussion: one test per entry of the README list.

Each test pins the current answer to one entry of the README section
*Multiport edge case behavior*. When a decision changes, change its test here
first, then the code. ``pytest -m multiport_edge_case`` selects these tests and
every other test that pins the same behavior.
"""

import pytest

from pygraphc import Graph

pytestmark = pytest.mark.multiport_edge_case


def _memberships(view: object) -> dict[int, list[int]]:
    """Branch id → sorted nodes of the component it belongs to."""
    return {
        branch_id: sorted(node_ids)
        for node_ids, branch_ids in view.connected_components_with_branch_ids()  # type: ignore[attr-defined]
        for branch_id in branch_ids
    }


def _components(view: object) -> list[list[int]]:
    return sorted(sorted(component) for component in view.connected_components())  # type: ignore[attr-defined]


def test_edge_case_1_two_port_edge_with_both_ends_masked_belongs_nowhere() -> None:
    """EC1: no active port, no component; it used to form a component without nodes."""
    view = Graph([1, 2, 3], [(1, 2), (2, 3)], branch_ids=[10, 11]).without_nodes([1, 2])

    assert _memberships(view) == {11: [3]}


def test_edge_case_2_one_active_port_belongs_to_its_node_and_is_never_critical() -> None:
    """EC2: joins nothing, belongs to the component of its one active port, makes nothing critical."""
    graph = Graph([1, 2, 3, 4], [(1, 2), (2, 3, 4)], branch_ids=[10, 11])
    view = graph.without_ports([(1, 1), (1, 2)])

    assert _components(view) == [[1, 2], [3], [4]]
    assert _memberships(view) == {10: [1, 2], 11: [1, 2]}
    assert view.critical_edges() == [0]
    assert view.articulation_points() == set()


def test_edge_case_3_two_port_edge_with_one_port_masked_joins_nothing() -> None:
    """EC3: like a masked edge for connection, but it stays a member of its active end's component."""
    view = Graph([1, 2], [(1, 2)], branch_ids=[10]).without_ports([(0, 1)])

    assert _components(view) == [[1], [2]]
    assert _memberships(view) == {10: [1]}


def test_edge_case_4_single_port_edge() -> None:
    """EC4: an undirected edge may have one port; a directed edge needs an in-port and an out-port."""
    graph = Graph([1, 2], [(1,), (1, 2)], branch_ids=[10, 11])

    assert _memberships(graph) == {10: [1, 2], 11: [1, 2]}
    assert graph.critical_edges() == [1]
    with pytest.raises(ValueError, match="at least one port"):
        Graph([1, 2], [((1,), ())], directed=True)


def test_edge_case_5_all_ports_at_one_node() -> None:
    """EC5: joins nothing, belongs to that node's component, forms no block and is never critical."""
    graph = Graph([1, 2], [(1, 1, 1), (2, 2)], branch_ids=[10, 11])

    assert _memberships(graph) == {10: [1], 11: [2]}
    assert list(graph.biconnected_components()) == []
    assert graph.critical_edges() == []


def test_edge_case_6_two_ports_of_one_edge_at_one_node() -> None:
    """EC6: the edge is still critical, and masking one of the two ports keeps the node attached."""
    graph = Graph([1, 2], [(1, 1, 2)])

    assert graph.critical_edges() == [0]
    assert _components(graph.without_ports([(0, 0)])) == [[1, 2]]


def test_edge_case_7_a_port_at_an_unknown_node_is_rejected() -> None:
    """EC7: as for two-port edges; "not connected" is a port mask, not a missing node."""
    with pytest.raises(ValueError, match="unknown"):
        Graph([1, 2], [(1, 2, 9)])


def test_edge_case_8_one_active_in_port_reaches_every_out_port() -> None:
    """EC8: OR semantics; AND (all in-ports needed) is not offered."""
    graph = Graph([1, 2, 3, 4], [((1, 2), (3, 4))], directed=True)

    assert sorted(graph.bfs(1)) == [1, 3, 4]


def test_edge_case_9_directed_edge_without_an_active_out_port() -> None:
    """EC9: transmits nothing, yet its active in-ports stay joined for weak components."""
    view = Graph([1, 2, 3], [((1, 2), (3,))], directed=True).without_ports([(0, 2)])

    assert view.bfs(1) == [1]
    assert sorted(sorted(component) for component in view.weakly_connected_components()) == [[1, 2], [3]]


def test_edge_case_10_weights_per_port_once_any_edge_has_more_than_two_ports() -> None:
    """EC10: a two-port edge there costs its two port weights; a graph of two-port edges takes one per edge."""
    graph = Graph([1, 2, 3, 4], [(1, 2), (2, 3, 4)])

    assert graph.shortest_path_lengths([1, 2, 10, 20, 30], 1) == {1: 0.0, 2: 3.0, 3: 33.0, 4: 43.0}
    with pytest.raises(ValueError, match="port count 5"):
        graph.shortest_path_lengths([1, 1], 1)
    with pytest.raises(ValueError, match="edge count 2"):
        Graph([1, 2, 3], [(1, 2), (2, 3)]).shortest_path_lengths([1, 1, 1, 1], 1)


def test_edge_case_11_passing_through_an_edge_costs_entry_plus_exit() -> None:
    """EC11: the entry port's weight plus the exit port's weight, whichever ports the path uses."""
    graph = Graph([1, 2, 3], [(1, 2, 3)])

    assert graph.shortest_path_lengths([1, 10, 100], 2) == {2: 0.0, 1: 11.0, 3: 110.0}


def test_edge_case_12_critical_multiport_edge_on_a_cycle() -> None:
    """EC12: critical when removing the whole edge separates its port nodes, even inside a block with a cycle."""
    graph = Graph([1, 2, 3, 4], [(1, 2), (2, 3), (1, 2, 3, 4)])

    assert graph.critical_edges() == [2]
    assert graph.articulation_points() == set()


def test_edge_case_13_blocks_never_split_at_a_multiport_edge() -> None:
    """EC13: a tree-shaped edge is one block; a port node hanging on an edge of a cycle joins the cycle's block."""
    tree_shaped = Graph([1, 2, 3], [(1, 2, 3)])
    on_a_cycle = Graph([1, 2, 3, 4], [(1, 2), (2, 3), (1, 2, 3, 4)])

    assert [sorted(block) for block in tree_shaped.biconnected_components()] == [[1, 2, 3]]
    assert [sorted(block) for block in on_a_cycle.biconnected_components()] == [[1, 2, 3, 4]]


def test_edge_case_14_two_edge_connected_components_are_nodes_no_single_edge_separates() -> None:
    """EC14: a critical multiport edge on a cycle still joins the nodes its cycle passes through."""
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


def test_edge_case_15_simple_paths_use_each_multiport_edge_once() -> None:
    """EC15: a simple path enters and leaves an edge through one port each and never uses the edge twice."""
    tree_shaped = Graph([1, 2, 3], [(1, 2, 3)])
    through_the_edge_once = Graph([1, 2, 3, 4], [(1, 2, 3), (2, 4), (4, 3)])

    assert tree_shaped.nodes_on_simple_paths(1, [2]) == {1, 2}
    assert through_the_edge_once.nodes_on_simple_paths(1, [2]) == {1, 2, 3, 4}


def test_edge_case_16_edge_indices_at_matches_the_set_of_port_nodes() -> None:
    """EC16: exact set match; order, repeats and direction do not matter, an unknown node matches nothing."""
    graph = Graph([1, 2, 3], [(1, 2, 3), (3, 2, 1), (1, 1, 2, 3), (1, 2)])
    directed = Graph([1, 2], [(1, 2), (2, 1)], directed=True)

    assert graph.edge_indices_at([3, 1, 2, 2]) == [0, 1, 2]
    assert graph.edge_indices_at([1, 2]) == [3]
    assert graph.edge_indices_at([1, 9]) == []
    assert directed.edge_indices_at([1, 2]) == [0, 1]


def test_edge_case_17_pair_methods_refuse_the_whole_graph() -> None:
    """EC17: on a graph with one multiport edge, the pair methods raise even when asked about a two-port edge."""
    graph = Graph([1, 2, 3, 4], [(1, 2), (2, 3, 4)])

    with pytest.raises(NotImplementedError, match=r"critical_edges\(\)"):
        graph.bridges()
    with pytest.raises(NotImplementedError, match=r"edge_indices_at\(\)"):
        graph.edge_indices(1, 2)


def test_edge_case_18_methods_not_yet_on_ports_refuse_the_whole_graph() -> None:
    """EC18: a method not yet on ports raises on a graph with a multiport edge or a view with a port mask."""
    graph = Graph([1, 2, 3, 4], [(1, 2), (2, 3, 4)])
    two_port_view = Graph([1, 2, 3], [(1, 2), (2, 3)]).without_ports([(0, 1)])

    with pytest.raises(NotImplementedError, match="multiport edges"):
        graph.cycle_basis()
    with pytest.raises(NotImplementedError, match="port masks"):
        two_port_view.cycle_basis()


def test_edge_case_19_negative_port_weights_need_every_pass_to_cost_at_least_zero() -> None:
    """EC19: a star equivalent may have one negative port weight; a pass that would cost less than 0 is rejected."""
    star = Graph([1, 2, 3], [(1, 2, 3)])
    directed = Graph([1, 2, 3], [((1, 2), (3,))], directed=True)

    assert star.shortest_path_lengths([3, -1, 2], 1) == {1: 0.0, 2: 2.0, 3: 5.0}
    assert directed.shortest_path_lengths([-1, 4, 2], 1) == {1: 0.0, 3: 1.0}
    with pytest.raises(ValueError, match="pass costing -2"):
        star.shortest_path_lengths([-3, 1, 2], 1)
    with pytest.raises(ValueError, match="every in-port and out-port"):
        directed.shortest_path_lengths([-2, 1, 1], 1)


def test_edge_case_20_a_path_never_leaves_an_edge_through_the_port_it_entered_by() -> None:
    """EC20: no U-turn; with a negative port weight it would undercut the node itself."""
    star = Graph([1, 2, 3], [(1, 2, 3)])
    star_with_a_tail = Graph([1, 2, 3, 4], [(1, 2, 3), (2, 4)])

    assert star.shortest_path_lengths([3, -1, 2], 2) == {1: 2.0, 2: 0.0, 3: 1.0}
    assert star_with_a_tail.shortest_path_lengths([3, -1, 2, 0.5, 0.5], 4) == {1: 3.0, 2: 1.0, 3: 2.0, 4: 0.0}


def test_edge_case_21_no_second_pass_right_at_the_negative_port() -> None:
    """EC21: after leaving an edge through its negative port, a path does not pass that edge again from there."""
    star = Graph([1, 2, 3], [(1, 2, 3)])

    assert star.shortest_path_lengths([3, -1, 2], 1)[3] == 5.0
    assert star.shortest_path([3, -1, 2], 1, 3) == [1, 3]


def test_edge_case_22_a_path_may_come_back_to_the_negative_port_another_way() -> None:
    """EC22: only the immediate second pass is banned; a detour back to the node allows it, so a node may repeat."""
    star_with_a_loop = Graph([1, 2, 3, 4], [(1, 2, 3), (2, 4), (4, 2)])
    weights = [3, -1, 2, 0.25, 0.25, 0.25, 0.25]

    assert star_with_a_loop.shortest_path_lengths(weights, 1)[3] == 4.0
    assert star_with_a_loop.shortest_path(weights, 1, 3) == [1, 2, 4, 2, 3]
