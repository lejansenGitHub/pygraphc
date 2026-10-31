"""Edges with ports, stage 1: storage, connectivity, masks and membership.

Each test names the decision of ``docs/multiport-decisions.md`` or the masking
case of the README table it checks. Throughout, a three-port edge joins nodes
2, 3 and 4 between two lines, unless a test builds its own graph.
"""

import pytest
from pygraphc._ports import port_graph_shape  # noqa: PLC2701 — D1 checks what the C core stores

from pygraphc import Graph, PortGraph


def _transformer_between_lines() -> PortGraph[int, int]:
    """1 -line- 2, the three-port edge (2, 3, 4), 4 -line- 5; branch ids 10, 11, 12."""
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
    graph = _transformer_between_lines()

    nodes_in_c, edges_in_c, ports_in_c = port_graph_shape(graph._graph_capsule)

    # --- Assert ---
    assert (graph.node_count, graph.edge_count) == (5, 3)
    assert (nodes_in_c, edges_in_c, ports_in_c) == (5, 3, 7)


# ── D2: tuples of node ids, port index is the position ──


def test_port_index_is_the_position_in_the_tuple() -> None:
    """D2: port 1 of (2, 3, 4) is at node 3, so masking it detaches node 3 only."""
    view = _transformer_between_lines().without_ports([(1, 1)])

    assert _components(view) == [[1, 2, 4, 5], [3]]


def test_two_ports_at_one_node_stay_distinguishable() -> None:
    """D2 and masking case 7: with one of two ports at node 2 masked, node 2 stays attached by the other."""
    graph = Graph([2, 3], [(2, 2, 3)])

    view = graph.without_ports([(0, 0)])

    assert _components(view) == [[2, 3]]


# ── D3: graphs of lines are unchanged ──


def test_a_graph_of_lines_stays_a_plain_graph() -> None:
    """D3: pairs take the unchanged line path; only a non-pair edge makes a PortGraph."""
    assert type(Graph([1, 2, 3], [(1, 2), (2, 3)])) is Graph


def test_line_errors_are_reported_as_before() -> None:
    """D3: an unknown endpoint in a graph of lines raises the line parser's own error."""
    with pytest.raises(ValueError, match="edge references unknown node ID"):
        Graph([1, 2], [(1, 9)])


def test_a_port_with_an_unknown_node_is_rejected() -> None:
    """O1 as recommended: every port needs a known node, as every line endpoint does."""
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
    """Case 1: the masked edge joins nothing and belongs to no component; its nodes keep their lines."""
    view = _transformer_between_lines().without_edges([1])

    # --- Assert ---
    assert _components(view) == [[1, 2], [3], [4, 5]]
    assert 11 not in _memberships(view)


def test_masking_case_2_remaining_ports_stay_joined() -> None:
    """Case 2: with port 0 (node 2) masked, ports at 3 and 4 still join, and the edge belongs there."""
    view = _transformer_between_lines().without_ports([(1, 0)])

    # --- Assert ---
    assert _components(view) == [[1, 2], [3, 4, 5]]
    assert _memberships(view)[11] == [3, 4, 5]


def test_masking_case_3_one_active_port_joins_nothing() -> None:
    """Case 3: one active port (node 4) joins nothing; the edge belongs to that port's component."""
    view = _transformer_between_lines().without_ports([(1, 0), (1, 1)])

    # --- Assert ---
    assert _components(view) == [[1, 2], [3], [4, 5]]
    assert _memberships(view)[11] == [4, 5]


def test_masking_case_4_no_active_port_belongs_nowhere() -> None:
    """Case 4: every port masked is the masked edge of case 1."""
    view = _transformer_between_lines().without_ports([(1, 0), (1, 1), (1, 2)])

    # --- Assert ---
    assert _components(view) == [[1, 2], [3], [4, 5]]
    assert 11 not in _memberships(view)


def test_masking_case_5_line_with_one_port_masked() -> None:
    """Case 5: a line with port 1 masked joins nothing and belongs to its active end."""
    view = _transformer_between_lines().without_ports([(0, 1)])

    # --- Assert ---
    assert _components(view) == [[1], [2, 3, 4, 5]]
    assert _memberships(view)[10] == [1]


def test_masking_case_6_masked_node_leaves_and_the_rest_stays_joined() -> None:
    """Case 6: node 3 leaves every result; the edge's ports at 2 and 4 stay joined."""
    view = _transformer_between_lines().without_nodes([3])

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
    """D10: an edge, a port and a node mask give the same view in any order; the line 4-5 stays."""
    graph = _transformer_between_lines()

    first = graph.without_edges([0]).without_ports([(1, 2)]).without_nodes([3])
    second = graph.without_nodes([3]).without_edges([0]).without_ports([(1, 2)])

    # --- Assert ---
    assert _components(first) == _components(second) == [[1], [2], [4, 5]]
    assert list(first.component_labels()) == list(second.component_labels())


def test_a_port_beyond_the_edge_is_rejected() -> None:
    """Port indices are per edge, so a three-port edge has no port 3."""
    with pytest.raises(IndexError, match="edge 1 has no port 3"):
        _transformer_between_lines().without_ports([(1, 3)])


# ── D12: single-port edges ──


def test_single_port_edge_joins_nothing_and_belongs_to_its_node() -> None:
    """D12: an undirected edge with one port, such as a load, joins nothing and belongs to its node's component."""
    graph = Graph([1, 2], [(1,), (1, 2)], branch_ids=[10, 11])

    # --- Assert ---
    assert _components(graph) == [[1, 2]]
    assert _memberships(graph) == {10: [1, 2], 11: [1, 2]}


# ── Labels and later stages ──


def test_component_labels_name_the_smallest_node_index() -> None:
    """Labels follow the line graph's contract: smallest node index per component, -1 for a masked node."""
    view = _transformer_between_lines().without_edges([1]).without_nodes([5])

    assert list(view.component_labels()) == [0, 0, 2, 3, -1]


def test_methods_of_later_stages_say_so() -> None:
    """Until their stage lands, methods not yet on ports raise instead of answering on an empty line structure."""
    with pytest.raises(NotImplementedError, match="traversal stage"):
        _transformer_between_lines().bfs(1)
