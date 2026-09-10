"""Regression tests for the defects of the 2026-09-10 code review (H1-H6, M1, M2, M4).

These tests were first added as pins that passed while the defects existed.
The wrappers are gone; every test now asserts the fixed behaviour directly.
"""

import pytest

from pygraphc import Graph

# ── H1: weights must match the edge count ──


def test_h1_weights_shorter_than_edge_count_are_rejected():
    """Every edge needs a weight. A shorter list has no defined meaning, so the
    kernel raises instead of reading past the buffer."""
    # --- Input ---
    # 1 -- 2 -- 3 with only ONE weight for TWO edges
    graph = Graph([1, 2, 3], [(1, 2), (2, 3)])

    # --- Assert ---
    with pytest.raises(ValueError, match="weights length 1 does not match edge count 2"):
        graph.shortest_path_lengths([1.0], source=1)


def test_h1_empty_weights_are_rejected():
    """The empty list used to segfault. It is a length mismatch like any other."""
    # --- Input ---
    graph = Graph([1, 2, 3], [(1, 2), (2, 3)])

    # --- Assert ---
    with pytest.raises(ValueError, match="weights length 0 does not match edge count 2"):
        graph.shortest_path([], source=1, target=3)


def test_h1_weights_longer_than_edge_count_are_rejected():
    """A longer list hides a caller bug just as well as a shorter one."""
    # --- Input ---
    graph = Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)

    # --- Assert ---
    with pytest.raises(ValueError, match="weights length 3 does not match edge count 2"):
        graph.dag_longest_path(weights=[1.0, 1.0, 1.0])


# ── H2: with_edges keeps edge and node masks on rebuilt views ──


def test_h2_with_edges_keeps_node_mask():
    """A view is the base graph minus its exclusions. Adding edges to a view must
    not resurrect an excluded node."""
    # --- Input ---
    # 0 -- 1 -- 2, node 1 excluded, then edge (2,3) added
    graph = Graph([0, 1, 2, 3], [(0, 1), (1, 2)])
    view = graph.without_nodes([1]).with_edges([(2, 3)])

    # --- Assert ---
    # Node 1 stays excluded: {0} and {2, 3}
    assert sorted(view.connected_components(), key=min) == [{0}, {2, 3}]


def test_h2_with_edges_keeps_edge_exclusion_after_rebuild():
    """An exclusion applied after one rebuild survives the next rebuild, so the
    result does not depend on the order in which views were chained."""
    # --- Input ---
    # 0 -- 1 -- 2, rebuilt once, then edge 0 excluded, then rebuilt again
    graph = Graph([0, 1, 2, 3, 4], [(0, 1), (1, 2)])
    view = graph.with_edges([(2, 3)]).without_edges([0]).with_edges([(3, 4)])

    # --- Assert ---
    # Edge (0,1) stays excluded: {0} and {1, 2, 3, 4}
    assert sorted(view.connected_components(), key=min) == [{0}, {1, 2, 3, 4}]


# ── H3: split_node on a rebuilt view reroutes instead of duplicating ──


def test_h3_split_node_on_rebuilt_view_reroutes_instead_of_duplicating():
    """Splitting moves an edge to the new node. A rebuilt view behaves like the
    base graph, as the README promises."""
    # --- Input ---
    # 0 -- 1 -- 2, rebuilt with (2,3), then node 1 split so that edge 1 (1,2) moves to 99
    graph = Graph([0, 1, 2], [(0, 1), (1, 2)])
    view = graph.with_edges([(2, 3)]).split_node(node_id=1, new_node_id=99, edge_indices_to_new_node=[1])

    # --- Assert ---
    # Edges (0,1), (99,2), (2,3): components {0, 1} and {2, 3, 99}
    assert sorted(view.connected_components(), key=min) == [{0, 1}, {2, 3, 99}]


# ── H4: with_edges on a split-list graph refuses instead of dropping edges ──


def test_h4_with_edges_on_split_list_graph_refuses():
    """Split-list construction keeps no Python edge list, so a rebuild cannot
    know the base edges. Refusing loudly matches split_node; silently
    dropping every base edge is what happened before."""
    # --- Input ---
    graph = Graph([0, 1, 2, 3], [0, 1], [1, 2])

    # --- Assert ---
    with pytest.raises(ValueError, match="edge-pair construction"):
        graph.with_edges([(2, 3)])


# ── H5: a self-loop is one edge in path enumeration ──


def test_h5_self_loop_paths_are_emitted_once():
    """There are exactly two edge-disjoint paths from 1 to 2: through the loop,
    and direct. The undirected CSR stores the loop in two slots and the DFS
    must visit it through one of them only."""
    # --- Input ---
    # self-loop 0:(1,1) and edge 1:(1,2)
    graph = Graph([1, 2], [(1, 1), (1, 2)])

    # --- Assert ---
    assert sorted(graph.all_edge_paths(source=1, targets=2)) == [[0, 1], [1]]


def test_h5_interior_self_loop_paths_are_emitted_once():
    """A loop on an interior node doubles the walk into the whole subtree if the
    second slot is not skipped."""
    # --- Input ---
    # 0:(1,2), 1:(2,2), 2:(2,3)
    graph = Graph([1, 2, 3], [(1, 2), (2, 2), (2, 3)])

    # --- Assert ---
    assert sorted(graph.all_edge_paths(source=1, targets=3)) == [[0, 1, 2], [0, 2]]


# ── H6: rebuilds keep edge identity and branch ids ──


def test_h6_edge_indices_are_stable_across_rebuild():
    """A rebuild keeps every base edge at its original index and appends added
    edges. Excluded edges stay masked instead of being compacted away, so
    node 3 is still incident to edge index 2."""
    # --- Input ---
    # 0 -- 1 -- 2 -- 3, edge 0 excluded, then an empty rebuild
    graph = Graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
    view = graph.without_edges([0]).with_edges([])

    # --- Assert ---
    assert view.incident_edge_indices(3) == [2]
    assert view.incident_edge_indices(1) == [1]  # edge 0 is masked, not renumbered


def test_h6_added_edges_are_appended_after_base_edges():
    """Added edges receive the indices after the last base edge."""
    # --- Input ---
    graph = Graph([0, 1, 2], [(0, 1)])
    view = graph.with_edges([(1, 2)])

    # --- Assert ---
    assert view.incident_edge_indices(2) == [1]
    assert view.incident_edge_indices(1) == [0, 1]


def test_h6_branch_ids_survive_rebuild():
    """Branch ids belong to the base graph and remain usable on every derived
    view. Excluding branch 12 on the rebuilt view removes edge (2,3)."""
    # --- Input ---
    # 0 -- 1 -- 2 -- 3 with branch ids, edge 0 excluded, then an empty rebuild
    graph = Graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)], branch_ids=[10, 11, 12])
    view = graph.without_edges([0]).with_edges([])

    # --- Assert ---
    assert sorted(view.without_branches([12]).connected_components(), key=min) == [{0}, {1, 2}, {3}]


def test_h6_added_edges_need_branch_ids_when_base_has_them():
    """Without an id the added edge would be unreachable by branch id, so the
    rebuild demands one id per added edge."""
    # --- Input ---
    graph = Graph([0, 1, 2], [(0, 1)], branch_ids=[10])

    # --- Assert ---
    with pytest.raises(ValueError, match="one per added edge"):
        graph.with_edges([(1, 2)])
    view = graph.with_edges([(1, 2)], added_branch_ids=[11])
    assert sorted(view.without_branches([11]).connected_components(), key=min) == [{0, 1}, {2}]


def test_h6_rerouted_edge_keeps_its_branch_id():
    """split_node moves a physical branch to a new node, so the rerouted edge
    carries the branch id of the edge it replaces."""
    # --- Input ---
    # 0 -- 1 -- 2 with branch ids, edge 1 (1,2) rerouted to (99,2)
    graph = Graph([0, 1, 2], [(0, 1), (1, 2)], branch_ids=[10, 11])
    view = graph.split_node(node_id=1, new_node_id=99, edge_indices_to_new_node=[1])

    # --- Assert ---
    # Excluding branch 11 removes the rerouted edge as well
    assert sorted(view.without_branches([11]).connected_components(), key=min) == [{0, 1}, {2}, {99}]


# ── M1: edge_indices(u, u) reports a self-loop once ──


def test_m1_self_loop_edge_index_is_reported_once():
    """One self-loop is one edge, as incident_edge_indices already reports it."""
    # --- Input ---
    graph = Graph([1], [(1, 1)])

    # --- Assert ---
    assert graph.edge_indices(1, 1) == [0]


# ── M2: branch_ids are validated and may be many-to-one ──


def test_m2_branch_ids_length_mismatch_is_rejected():
    """branch_ids is a side table indexed by edge. A list of the wrong length
    leaves edges without an id or ids without an edge, so it is rejected at
    construction."""
    # --- Assert ---
    with pytest.raises(ValueError, match="branch_ids length 1 does not match edge count 2"):
        Graph([1, 2, 3], [(1, 2), (2, 3)], branch_ids=[100])


def test_m2_duplicate_branch_ids_exclude_every_edge():
    """Two edges may share a branch id, the set-typed branch APIs already allow
    it. Excluding that id excludes both edges, so three isolated nodes remain.
    Before the fix the id-to-index dict kept only the last edge."""
    # --- Input ---
    graph = Graph([1, 2, 3], [(1, 2), (2, 3)], branch_ids=[7, 7])

    # --- Assert ---
    assert sorted(graph.without_branches([7]).connected_components(), key=min) == [{1}, {2}, {3}]


# ── M4: duplicate node ids are rejected ──


def test_m4_duplicate_node_ids_are_rejected():
    """Node ids are keys. Listing one twice used to create a phantom isolated
    node carrying the same id."""
    # --- Assert ---
    with pytest.raises(ValueError, match="duplicate node id 1"):
        Graph([1, 1, 2], [(1, 2)])


def test_m4_duplicate_node_ids_are_rejected_for_split_lists():
    """The split-list parser shares the check."""
    # --- Assert ---
    with pytest.raises(ValueError, match="duplicate node id 1"):
        Graph([1, 1, 2], [1], [2])
