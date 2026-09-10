"""Pins for known defects (code review of 2026-09-10, items H1-H6, M1, M2, M4).

Every test asserts the CORRECT behaviour inside ``pytest.raises(AssertionError)``.
The test therefore passes while the defect exists and fails the moment the defect
is fixed. Fixing a defect means deleting its ``pytest.raises`` wrapper (and, where
present, the ``_raises`` helper call), so the inner assertion becomes the
regression test.
"""

from collections.abc import Callable

import pytest

from pygraphc import Graph


def _raises(exception_type: type[BaseException], call: Callable[[], object]) -> bool:
    """Return True when ``call()`` raises ``exception_type``."""
    try:
        call()
    except exception_type:
        return True
    return False


# ── H1: weights shorter than the edge list are read out of bounds ──


def test_h1_weights_shorter_than_edge_count_must_be_rejected():
    """Every edge needs a weight. A shorter list has no defined meaning, so the
    only correct answer is ValueError; today the C kernel reads past the buffer
    and returns distances built from garbage memory."""
    # --- Input ---
    # 1 -- 2 -- 3 with only ONE weight for TWO edges
    graph = Graph([1, 2, 3], [(1, 2), (2, 3)])

    # --- Expected once fixed ---
    with pytest.raises(AssertionError):  # H1: remove once weights length is validated
        assert _raises(ValueError, lambda: graph.shortest_path_lengths([1.0], source=1))


# ── H2: with_edges drops edge and node masks on rebuilt views ──


def test_h2_with_edges_keeps_node_mask():
    """A view is the base graph minus its exclusions. Adding edges to a view must
    not resurrect an excluded node, otherwise chaining without_nodes and
    with_edges silently changes connectivity."""
    # --- Input ---
    # 0 -- 1 -- 2, node 1 excluded, then edge (2,3) added
    graph = Graph([0, 1, 2, 3], [(0, 1), (1, 2)])
    view = graph.without_nodes([1]).with_edges([(2, 3)])

    # --- Expected once fixed ---
    # Node 1 stays excluded: {0} and {2, 3}
    components = sorted(view.connected_components(), key=min)
    with pytest.raises(AssertionError):  # H2: remove once node masks survive with_edges
        assert components == [{0}, {2, 3}]


def test_h2_with_edges_keeps_edge_exclusion_after_rebuild():
    """An exclusion applied after one rebuild must survive the next rebuild.
    Otherwise the result depends on the order in which views were chained."""
    # --- Input ---
    # 0 -- 1 -- 2, rebuilt once, then edge 0 excluded, then rebuilt again
    graph = Graph([0, 1, 2, 3, 4], [(0, 1), (1, 2)])
    view = graph.with_edges([(2, 3)]).without_edges([0]).with_edges([(3, 4)])

    # --- Expected once fixed ---
    # Edge (0,1) stays excluded: {0} and {1, 2, 3, 4}
    components = sorted(view.connected_components(), key=min)
    with pytest.raises(AssertionError):  # H2: remove once exclusions survive with_edges
        assert components == [{0}, {1, 2, 3, 4}]


# ── H3: split_node on a rebuilt view keeps the original edge ──


def test_h3_split_node_on_rebuilt_view_reroutes_instead_of_duplicating():
    """Splitting moves an edge to the new node; the edge count must not change.
    The README promises split_node works on views, so a rebuilt view must
    behave like the base graph."""
    # --- Input ---
    # 0 -- 1 -- 2, rebuilt with (2,3), then node 1 split so that edge 1 (1,2) moves to 99
    graph = Graph([0, 1, 2], [(0, 1), (1, 2)])
    view = graph.with_edges([(2, 3)]).split_node(node_id=1, new_node_id=99, edge_indices_to_new_node=[1])

    # --- Expected once fixed ---
    # Edges (0,1), (99,2), (2,3): components {0, 1} and {2, 3, 99}
    components = sorted(view.connected_components(), key=min)
    with pytest.raises(AssertionError):  # H3: remove once split_node works on rebuilt views
        assert components == [{0, 1}, {2, 3, 99}]


# ── H4: with_edges on a split-list graph drops every base edge ──


def test_h4_with_edges_on_split_list_graph_keeps_base_edges():
    """Both constructors describe the same graph, so with_edges must give the
    same result for both. Today the split-list variant loses all base edges
    because the Python-side edge list is None."""
    # --- Input ---
    # 0 -- 1 -- 2 built from split lists, then edge (2,3) added
    graph = Graph([0, 1, 2, 3], [0, 1], [1, 2])
    view = graph.with_edges([(2, 3)])

    # --- Expected once fixed ---
    # One component {0, 1, 2, 3}
    components = sorted(view.connected_components(), key=min)
    with pytest.raises(AssertionError):  # H4: remove once split-list graphs support with_edges
        assert components == [{0, 1, 2, 3}]


# ── H5: paths through a self-loop are emitted twice ──


def test_h5_self_loop_paths_are_emitted_once():
    """A self-loop is one edge. There are exactly two edge-disjoint paths from
    1 to 2 here: through the loop, and direct. The undirected CSR stores the
    loop in two slots and the DFS re-enters through the second one."""
    # --- Input ---
    # self-loop 0:(1,1) and edge 1:(1,2)
    graph = Graph([1, 2], [(1, 1), (1, 2)])

    # --- Expected once fixed ---
    paths = sorted(graph.all_edge_paths(source=1, targets=2))
    with pytest.raises(AssertionError):  # H5: remove once the duplicate CSR slot is skipped
        assert paths == [[0, 1], [1]]


# ── H6: rebuilds renumber edge indices and drop branch ids ──


def test_h6_edge_indices_are_stable_across_rebuild():
    """Edge indices are the public identity of edges. A rebuild that compacts
    excluded edges away renumbers them, so an index held by the caller
    silently points at a different edge."""
    # --- Input ---
    # 0 -- 1 -- 2 -- 3, edge 0 excluded, then an empty rebuild
    graph = Graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])
    view = graph.without_edges([0]).with_edges([])

    # --- Expected once fixed ---
    # Node 3 is still incident to edge index 2
    with pytest.raises(AssertionError):  # H6: remove once rebuilds preserve edge identity
        assert view.incident_edge_indices(3) == [2]


def test_h6_branch_ids_survive_rebuild():
    """Branch ids are a property of the base graph and must remain usable on
    every derived view. Today the rebuilt Graph is created without them and
    without_branches raises ValueError."""
    # --- Input ---
    # 0 -- 1 -- 2 -- 3 with branch ids, edge 0 excluded, then an empty rebuild
    graph = Graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)], branch_ids=[10, 11, 12])
    view = graph.without_edges([0]).with_edges([])

    # --- Expected once fixed ---
    with pytest.raises(AssertionError):  # H6: remove once rebuilds propagate branch_ids
        assert not _raises(ValueError, lambda: view.without_branches([12]))


# ── M1: edge_indices(u, u) returns the self-loop twice ──


def test_m1_self_loop_edge_index_is_reported_once():
    """One self-loop is one edge and must be reported once, as
    incident_edge_indices already does. edge_indices matches both CSR slots."""
    # --- Input ---
    graph = Graph([1], [(1, 1)])

    # --- Expected once fixed ---
    with pytest.raises(AssertionError):  # M1: remove once consecutive equal eids are deduped
        assert graph.edge_indices(1, 1) == [0]


# ── M2: branch_ids are never validated against the edge list ──


def test_m2_branch_ids_length_mismatch_is_rejected():
    """branch_ids is a side table indexed by edge. A shorter list leaves edges
    without an id and is silently truncated today; rejecting it at construction
    is the only behaviour that keeps the table consistent with the graph."""
    # --- Expected once fixed ---
    with pytest.raises(AssertionError):  # M2: remove once branch_ids length is validated
        assert _raises(ValueError, lambda: Graph([1, 2, 3], [(1, 2), (2, 3)], branch_ids=[100]))


def test_m2_duplicate_branch_ids_are_rejected():
    """without_branches resolves a branch id to one edge index. Two edges with
    the same id make that mapping ambiguous, and today the second silently
    wins, so exclusion by id removes only one of the two edges."""
    # --- Expected once fixed ---
    with pytest.raises(AssertionError):  # M2: remove once branch_ids uniqueness is validated
        assert _raises(ValueError, lambda: Graph([1, 2, 3], [(1, 2), (2, 3)], branch_ids=[7, 7]))


# ── M4: duplicate node ids create a phantom isolated node ──


def test_m4_duplicate_node_ids_are_rejected():
    """Node ids are keys. Listing one twice creates a second index that no
    edge can ever reference, so it shows up as a phantom isolated component
    carrying the same id. Rejecting the input is the only consistent answer."""
    # --- Expected once fixed ---
    with pytest.raises(AssertionError):  # M4: remove once duplicate node ids are detected
        assert _raises(ValueError, lambda: Graph([1, 1, 2], [(1, 2)]))
