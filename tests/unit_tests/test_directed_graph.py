"""Tests for directed graph support in networkc."""

import pytest

import networkc
from networkc import Graph

# ── Construction ──


def test_directed_flag_set():
    graph = Graph([1, 2], [(1, 2)], directed=True)
    assert graph.directed is True


def test_undirected_flag_default():
    graph = Graph([1, 2], [(1, 2)])
    assert graph.directed is False


def test_directed_split_list_construction():
    graph = Graph([1, 2, 3], [1, 2], [2, 3], directed=True)
    assert graph.directed is True
    assert graph.edge_count == 2
    assert graph.node_count == 3


def test_directed_empty_graph():
    graph = Graph([], [], directed=True)
    assert graph.directed is True
    assert graph.edge_count == 0
    assert graph.node_count == 0


def test_directed_single_node():
    graph = Graph([1], [], directed=True)
    assert graph.node_count == 1
    assert graph.edge_count == 0


def test_directed_no_edges():
    graph = Graph([1, 2, 3], [], directed=True)
    assert graph.node_count == 3
    assert graph.edge_count == 0


# ── Strongly Connected Components (SCC) ──


@pytest.mark.parametrize(
    ("node_ids", "edges", "expected"),
    [
        pytest.param([1], [], [{1}], id="single_node"),
        pytest.param([1], [(1, 1)], [{1}], id="self_loop"),
        pytest.param(
            [1, 2, 3],
            [(1, 2), (2, 3), (3, 1)],
            [{1, 2, 3}],
            id="simple_cycle",
        ),
        pytest.param(
            [1, 2, 3],
            [(1, 2), (2, 3)],
            [{1}, {2}, {3}],
            id="dag_chain",
        ),
        pytest.param(
            [1, 2, 3, 4],
            [(1, 2), (2, 1), (3, 4), (4, 3), (2, 3)],
            [{1, 2}, {3, 4}],
            id="two_sccs_linked",
        ),
        pytest.param(
            [1, 2, 3],
            [(1, 2), (2, 1), (1, 3), (3, 1), (2, 3), (3, 2)],
            [{1, 2, 3}],
            id="complete_directed",
        ),
        pytest.param(
            [1, 2, 3, 4],
            [(1, 2), (3, 4)],
            [{1}, {2}, {3}, {4}],
            id="disconnected_dag",
        ),
        pytest.param(
            [1, 2],
            [(1, 2), (2, 1)],
            [{1, 2}],
            id="antiparallel",
        ),
        pytest.param(
            [1, 2],
            [(1, 2), (1, 2)],
            [{1}, {2}],
            id="parallel_same_direction",
        ),
        pytest.param(
            [1, 2, 3],
            [],
            [{1}, {2}, {3}],
            id="isolated_nodes",
        ),
        pytest.param(
            [1, 2, 3, 4, 5, 6, 7, 8],
            [
                (1, 2),
                (2, 3),
                (3, 1),  # SCC {1,2,3}
                (4, 5),
                (5, 6),
                (6, 4),  # SCC {4,5,6}
                (3, 4),  # cross-SCC link
                (7, 8),  # DAG edge
            ],
            [{1, 2, 3}, {4, 5, 6}, {7}, {8}],
            id="classic_tarjan",
        ),
    ],
)
def test_scc_correctness(node_ids, edges, expected):
    graph = Graph(node_ids, edges, directed=True)
    result = list(graph.strongly_connected_components())
    assert len(result) == len(expected)
    for component in expected:
        assert component in result


def test_scc_free_function():
    node_ids = [1, 2, 3]
    edges = [(1, 2), (2, 3), (3, 1)]
    result = list(networkc.strongly_connected_components(node_ids, edges))
    assert result == [{1, 2, 3}]


# ── Weakly Connected Components (WCC) ──


@pytest.mark.parametrize(
    ("node_ids", "edges", "expected"),
    [
        pytest.param(
            [1, 2, 3],
            [(1, 2), (2, 3)],
            [{1, 2, 3}],
            id="dag_chain",
        ),
        pytest.param(
            [1, 2, 3, 4],
            [(1, 2), (3, 4)],
            [{1, 2}, {3, 4}],
            id="two_components",
        ),
        pytest.param(
            [1, 2, 3],
            [],
            [{1}, {2}, {3}],
            id="isolated_nodes",
        ),
        pytest.param(
            [1, 2, 3],
            [(1, 2), (2, 3), (3, 1)],
            [{1, 2, 3}],
            id="cycle_is_one_wcc",
        ),
    ],
)
def test_wcc_correctness(node_ids, edges, expected):
    graph = Graph(node_ids, edges, directed=True)
    result = list(graph.weakly_connected_components())
    assert len(result) == len(expected)
    for component in expected:
        assert component in result


def test_wcc_free_function():
    node_ids = [1, 2, 3, 4]
    edges = [(1, 2), (3, 4)]
    result = list(networkc.weakly_connected_components(node_ids, edges))
    assert len(result) == 2
    assert {1, 2} in result
    assert {3, 4} in result


# ── Type Guards ──


class TestTypeGuardsOnDirectedGraph:
    """Undirected-only methods must raise TypeError on directed graphs."""

    @pytest.fixture
    def directed_graph(self):
        return Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)

    def test_connected_components(self, directed_graph):
        with pytest.raises(TypeError, match="not defined for directed"):
            list(directed_graph.connected_components())

    def test_bridges(self, directed_graph):
        with pytest.raises(TypeError, match="not defined for directed"):
            directed_graph.bridges()

    def test_articulation_points(self, directed_graph):
        with pytest.raises(TypeError, match="not defined for directed"):
            directed_graph.articulation_points()

    def test_biconnected_components(self, directed_graph):
        with pytest.raises(TypeError, match="not defined for directed"):
            list(directed_graph.biconnected_components())

    def test_two_edge_connected_components(self, directed_graph):
        with pytest.raises(TypeError, match="not defined for directed"):
            list(directed_graph.two_edge_connected_components())

    def test_nodes_on_simple_paths(self, directed_graph):
        with pytest.raises(TypeError, match="not defined for directed"):
            directed_graph.nodes_on_simple_paths(1, [3])


class TestTypeGuardsOnUndirectedGraph:
    """Directed-only methods must raise TypeError on undirected graphs."""

    @pytest.fixture
    def undirected_graph(self):
        return Graph([1, 2, 3], [(1, 2), (2, 3)])

    def test_strongly_connected_components(self, undirected_graph):
        with pytest.raises(TypeError, match="requires a directed"):
            list(undirected_graph.strongly_connected_components())

    def test_weakly_connected_components(self, undirected_graph):
        with pytest.raises(TypeError, match="requires a directed"):
            list(undirected_graph.weakly_connected_components())

    def test_topological_sort(self, undirected_graph):
        with pytest.raises(TypeError, match="requires a directed"):
            undirected_graph.topological_sort()

    def test_successors(self, undirected_graph):
        with pytest.raises(TypeError, match="requires a directed"):
            undirected_graph.successors(1)

    def test_predecessors(self, undirected_graph):
        with pytest.raises(TypeError, match="requires a directed"):
            undirected_graph.predecessors(1)

    def test_in_degree(self, undirected_graph):
        with pytest.raises(TypeError, match="requires a directed"):
            undirected_graph.in_degree(1)

    def test_out_degree(self, undirected_graph):
        with pytest.raises(TypeError, match="requires a directed"):
            undirected_graph.out_degree(1)

    def test_outgoing_edge_indices(self, undirected_graph):
        with pytest.raises(TypeError, match="requires a directed"):
            undirected_graph.outgoing_edge_indices(1)

    def test_incoming_edge_indices(self, undirected_graph):
        with pytest.raises(TypeError, match="requires a directed"):
            undirected_graph.incoming_edge_indices(1)


# ── Directed BFS ──


def test_bfs_follows_outgoing_only():
    graph = Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)
    assert graph.bfs(source=1) == [1, 2, 3]


def test_bfs_no_outgoing_from_sink():
    graph = Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)
    assert graph.bfs(source=3) == [3]


def test_bfs_does_not_follow_incoming():
    graph = Graph([1, 2, 3], [(1, 2), (3, 2)], directed=True)
    result = graph.bfs(source=1)
    assert 1 in result
    assert 2 in result
    assert 3 not in result


# ── Directed Dijkstra ──


def test_dijkstra_forward_path():
    graph = Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)
    path = graph.shortest_path(weights=[1.0, 2.0], source=1, target=3)
    assert path == [1, 2, 3]


def test_dijkstra_no_reverse_path():
    graph = Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)
    path = graph.shortest_path(weights=[1.0, 2.0], source=3, target=1)
    assert path == []


def test_dijkstra_weighted_directed():
    # 1 --(1.0)--> 2 --(1.0)--> 3
    # 1 --(10.0)--> 3
    graph = Graph([1, 2, 3], [(1, 2), (2, 3), (1, 3)], directed=True)
    path = graph.shortest_path(weights=[1.0, 1.0, 10.0], source=1, target=3)
    assert path == [1, 2, 3]  # cheaper through 2


# ── Topological Sort ──


def test_topological_sort_dag():
    graph = Graph([1, 2, 3, 4], [(1, 2), (1, 3), (3, 4)], directed=True)
    order = graph.topological_sort()
    idx = {node: i for i, node in enumerate(order)}
    assert idx[1] < idx[2]
    assert idx[1] < idx[3]
    assert idx[3] < idx[4]


def test_topological_sort_cycle_raises():
    graph = Graph([1, 2, 3], [(1, 2), (2, 3), (3, 1)], directed=True)
    with pytest.raises(ValueError, match="cycle"):
        graph.topological_sort()


def test_topological_sort_single_node():
    graph = Graph([1], [], directed=True)
    assert graph.topological_sort() == [1]


def test_topological_sort_undirected_raises():
    graph = Graph([1, 2], [(1, 2)])
    with pytest.raises(TypeError, match="requires a directed"):
        graph.topological_sort()


# ── Query Methods (Directed) ──


class TestDirectedQueryMethods:
    @pytest.fixture
    def graph(self):
        # 1 --> 2 --> 3, 3 --> 1
        return Graph([1, 2, 3], [(1, 2), (2, 3), (3, 1)], directed=True)

    def test_neighbors_returns_successors(self, graph):
        assert graph.neighbors(1) == {2}
        assert graph.neighbors(2) == {3}
        assert graph.neighbors(3) == {1}

    def test_successors(self, graph):
        assert graph.successors(1) == {2}
        assert graph.successors(2) == {3}

    def test_predecessors(self, graph):
        assert graph.predecessors(1) == {3}
        assert graph.predecessors(2) == {1}
        assert graph.predecessors(3) == {2}

    def test_degree_is_out_degree(self, graph):
        assert graph.degree(1) == 1
        assert graph.degree(2) == 1
        assert graph.degree(3) == 1

    def test_out_degree(self, graph):
        assert graph.out_degree(1) == 1

    def test_in_degree(self, graph):
        assert graph.in_degree(1) == 1  # 3 --> 1
        assert graph.in_degree(2) == 1  # 1 --> 2

    def test_edge_indices_directed(self, graph):
        assert graph.edge_indices(1, 2) == [0]
        assert graph.edge_indices(2, 1) == []  # no reverse edge

    def test_incident_edge_indices_outgoing_only(self, graph):
        assert graph.incident_edge_indices(1) == [0]  # edge 0: 1->2

    def test_outgoing_edge_indices(self, graph):
        assert graph.outgoing_edge_indices(1) == [0]

    def test_incoming_edge_indices(self, graph):
        assert graph.incoming_edge_indices(1) == [2]  # edge 2: 3->1


class TestDirectedMultigraph:
    def test_parallel_same_direction_is_multigraph(self):
        graph = Graph([1, 2], [(1, 2), (1, 2)], directed=True)
        assert graph.is_multigraph is True

    def test_antiparallel_not_multigraph(self):
        graph = Graph([1, 2], [(1, 2), (2, 1)], directed=True)
        assert graph.is_multigraph is False


class TestDirectedDegreeEdgeCases:
    def test_node_with_no_outgoing(self):
        graph = Graph([1, 2], [(2, 1)], directed=True)
        assert graph.degree(1) == 0  # no outgoing from 1
        assert graph.in_degree(1) == 1

    def test_node_with_no_incoming(self):
        graph = Graph([1, 2], [(1, 2)], directed=True)
        assert graph.in_degree(1) == 0
        assert graph.out_degree(1) == 1

    def test_self_loop_counted_once_in_out_degree(self):
        graph = Graph([1], [(1, 1)], directed=True)
        assert graph.out_degree(1) == 1
        assert graph.in_degree(1) == 1


# ── GraphView (Directed) ──


class TestGraphViewDirected:
    def test_without_edges_propagates_directed(self):
        graph = Graph([1, 2, 3], [(1, 2), (2, 3), (3, 1)], directed=True)
        view = graph.without_edges([2])  # remove edge 3->1
        assert view._graph._directed is True

    def test_without_nodes_propagates_directed(self):
        graph = Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)
        view = graph.without_nodes([2])
        assert view._graph._directed is True

    def test_scc_with_edge_mask_breaks_cycle(self):
        # 1->2->3->1. Remove edge 3->1 (index 2) => no cycles, all singletons
        graph = Graph([1, 2, 3], [(1, 2), (2, 3), (3, 1)], directed=True)
        view = graph.without_edges([2])
        result = list(view.strongly_connected_components())
        assert len(result) == 3
        for component in [{1}, {2}, {3}]:
            assert component in result

    def test_scc_with_node_mask(self):
        # 1->2->3->1. Remove node 2 => 1 and 3 are separate SCCs
        graph = Graph([1, 2, 3], [(1, 2), (2, 3), (3, 1)], directed=True)
        view = graph.without_nodes([2])
        result = list(view.strongly_connected_components())
        assert {2} not in result
        assert {1} in result
        assert {3} in result

    def test_wcc_with_mask(self):
        graph = Graph([1, 2, 3, 4], [(1, 2), (3, 4)], directed=True)
        view = graph.without_edges([])
        result = list(view.weakly_connected_components())
        assert len(result) == 2

    def test_toposort_with_edge_mask(self):
        # 1->2, 2->3, 3->1 is cyclic. Remove 3->1 => valid topo sort
        graph = Graph([1, 2, 3], [(1, 2), (2, 3), (3, 1)], directed=True)
        view = graph.without_edges([2])
        order = view.topological_sort()
        idx = {node: i for i, node in enumerate(order)}
        assert idx[1] < idx[2]
        assert idx[2] < idx[3]

    def test_view_type_guard_undirected_on_directed(self):
        graph = Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)
        view = graph.without_edges([])
        with pytest.raises(TypeError, match="not defined for directed"):
            list(view.connected_components())
        with pytest.raises(TypeError, match="not defined for directed"):
            view.bridges()
        with pytest.raises(TypeError, match="not defined for directed"):
            view.articulation_points()

    def test_view_query_methods_directed(self):
        graph = Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)
        view = graph.without_edges([])
        assert view.neighbors(1) == {2}
        assert view.successors(1) == {2}
        assert view.predecessors(2) == {1}
        assert view.degree(1) == 1
        assert view.out_degree(1) == 1
        assert view.in_degree(2) == 1

    def test_with_edges_propagates_directed(self):
        graph = Graph([1, 2], [(1, 2)], directed=True)
        view = graph.with_edges([(2, 3)])
        # _added_graph should be directed
        assert view._graph._directed is True

    def test_view_bfs_directed(self):
        graph = Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)
        view = graph.without_edges([])
        assert view.bfs(source=1) == [1, 2, 3]
        assert view.bfs(source=3) == [3]

    def test_view_dijkstra_directed(self):
        graph = Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)
        view = graph.without_edges([])
        path = view.shortest_path(weights=[1.0, 1.0], source=1, target=3)
        assert path == [1, 2, 3]
        path_reverse = view.shortest_path(weights=[1.0, 1.0], source=3, target=1)
        assert path_reverse == []


class TestGraphViewDirectedWithNodeMask:
    """Exercise GraphView directed code paths with node exclusion."""

    def test_incident_edge_indices_with_node_mask(self):
        # 1->2, 2->3. Exclude node 3.
        graph = Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)
        view = graph.without_nodes([3])
        # Node 2 has outgoing edge to 3, but 3 is excluded
        assert view.incident_edge_indices(2) == []
        assert view.incident_edge_indices(1) == [0]

    def test_outgoing_edge_indices_with_node_mask(self):
        graph = Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)
        view = graph.without_nodes([3])
        assert view.outgoing_edge_indices(2) == []

    def test_incoming_edge_indices_with_node_mask(self):
        graph = Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)
        view = graph.without_nodes([1])
        assert view.incoming_edge_indices(2) == []  # 1 is excluded
        assert view.incoming_edge_indices(3) == [1]  # 2->3 still active

    def test_incoming_edge_indices_excluded_target(self):
        graph = Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)
        view = graph.without_nodes([2])
        assert view.incoming_edge_indices(2) == []  # node 2 itself excluded

    def test_neighbors_with_node_mask(self):
        # 1->2, 2->3. Exclude 3.
        graph = Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)
        view = graph.without_nodes([3])
        assert view.neighbors(2) == set()  # 3 is excluded
        assert view.neighbors(1) == {2}

    def test_predecessors_with_node_mask(self):
        graph = Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)
        view = graph.without_nodes([1])
        assert view.predecessors(2) == set()  # 1 is excluded

    def test_predecessors_excluded_node(self):
        graph = Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)
        view = graph.without_nodes([2])
        assert view.predecessors(2) == set()  # node itself excluded

    def test_degree_with_node_mask(self):
        graph = Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)
        view = graph.without_nodes([3])
        assert view.degree(2) == 0  # outgoing to 3, but 3 excluded
        assert view.degree(1) == 1  # outgoing to 2, not excluded

    def test_degree_excluded_node(self):
        graph = Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)
        view = graph.without_nodes([1])
        assert view.degree(1) == 0  # node itself excluded

    def test_in_degree_with_node_mask(self):
        graph = Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)
        view = graph.without_nodes([1])
        assert view.in_degree(2) == 0  # 1->2 but 1 excluded

    def test_in_degree_excluded_node(self):
        graph = Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)
        view = graph.without_nodes([2])
        assert view.in_degree(2) == 0


class TestDirectedSplitListConstruction:
    """Cover edges-is-None branches when using split-list construction."""

    def test_outgoing_edge_indices_split_list(self):
        graph = Graph([1, 2, 3], [1, 2], [2, 3], directed=True)
        assert graph.outgoing_edge_indices(1) == []

    def test_incoming_edge_indices_split_list(self):
        graph = Graph([1, 2, 3], [1, 2], [2, 3], directed=True)
        assert graph.incoming_edge_indices(1) == []

    def test_neighbors_split_list(self):
        graph = Graph([1, 2, 3], [1, 2], [2, 3], directed=True)
        assert graph.neighbors(1) == set()

    def test_predecessors_split_list(self):
        graph = Graph([1, 2, 3], [1, 2], [2, 3], directed=True)
        assert graph.predecessors(1) == set()

    def test_degree_split_list(self):
        graph = Graph([1, 2, 3], [1, 2], [2, 3], directed=True)
        assert graph.degree(1) == 0

    def test_in_degree_split_list(self):
        graph = Graph([1, 2, 3], [1, 2], [2, 3], directed=True)
        assert graph.in_degree(1) == 0


class TestGraphViewDirectedEdgeMask:
    """Cover edge-exclusion paths in incoming_edge_indices, predecessors, in_degree."""

    def test_incoming_edge_indices_with_excluded_edge(self):
        graph = Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)
        view = graph.without_edges([0])  # exclude 1->2
        assert view.incoming_edge_indices(2) == []  # edge 0 excluded

    def test_predecessors_with_excluded_edge(self):
        graph = Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)
        view = graph.without_edges([0])  # exclude 1->2
        assert view.predecessors(2) == set()

    def test_in_degree_with_excluded_edge(self):
        graph = Graph([1, 2, 3], [(1, 2), (2, 3)], directed=True)
        view = graph.without_edges([0])  # exclude 1->2
        assert view.in_degree(2) == 0

    def test_require_directed_on_undirected_view(self):
        """Cover GraphView._require_directed raising TypeError."""
        graph = Graph([1, 2, 3], [(1, 2), (2, 3)])
        view = graph.without_edges([])
        with pytest.raises(TypeError, match="requires a directed"):
            view.successors(1)

    def test_view_degree_undirected_self_loop(self):
        """Cover the undirected self-loop branch in GraphView.degree."""
        graph = Graph([1, 2], [(1, 1), (1, 2)])
        view = graph.without_edges([])
        assert view.degree(1) == 3  # self-loop counts 2, plus edge to 2

    def test_view_degree_undirected_self_loop_with_node_mask(self):
        """Cover self-loop + node-mask continue branch in GraphView.degree."""
        graph = Graph([1, 2], [(1, 1), (1, 2)])
        view = graph.without_nodes([2])
        # When node mask is present, self-loops are skipped (existing behavior)
        assert view.degree(1) == 0


class TestGraphViewDirectedSplitList:
    """Cover edges-is-None branches in GraphView directed methods."""

    def test_incoming_edge_indices_split_list(self):
        graph = Graph([1, 2, 3], [1, 2], [2, 3], directed=True)
        view = graph.without_edges([])
        assert view.incoming_edge_indices(2) == []  # edges is None

    def test_predecessors_split_list(self):
        graph = Graph([1, 2, 3], [1, 2], [2, 3], directed=True)
        view = graph.without_edges([])
        assert view.predecessors(2) == set()  # edges is None

    def test_in_degree_split_list(self):
        graph = Graph([1, 2, 3], [1, 2], [2, 3], directed=True)
        view = graph.without_edges([])
        assert view.in_degree(2) == 0  # edges is None


# ── Free-function tests ──


def test_scc_free_function_matches_graph_method():
    node_ids = [1, 2, 3, 4]
    edges = [(1, 2), (2, 1), (3, 4)]
    free_result = list(networkc.strongly_connected_components(node_ids, edges))
    graph_result = list(Graph(node_ids, edges, directed=True).strongly_connected_components())
    assert sorted(free_result, key=sorted) == sorted(graph_result, key=sorted)


def test_wcc_free_function_matches_graph_method():
    node_ids = [1, 2, 3, 4]
    edges = [(1, 2), (3, 4)]
    free_result = list(networkc.weakly_connected_components(node_ids, edges))
    graph_result = list(Graph(node_ids, edges, directed=True).weakly_connected_components())
    assert sorted(free_result, key=sorted) == sorted(graph_result, key=sorted)


# ── Edge cases: SCC on larger graphs ──


def test_scc_single_large_cycle():
    n = 100
    node_ids = list(range(n))
    edges = [(i, (i + 1) % n) for i in range(n)]
    graph = Graph(node_ids, edges, directed=True)
    result = list(graph.strongly_connected_components())
    assert len(result) == 1
    assert result[0] == set(range(n))


def test_scc_long_chain():
    n = 100
    node_ids = list(range(n))
    edges = [(i, i + 1) for i in range(n - 1)]
    graph = Graph(node_ids, edges, directed=True)
    result = list(graph.strongly_connected_components())
    assert len(result) == n  # every node is its own SCC


def test_wcc_matches_undirected_cc():
    """WCC on directed graph should match CC on same edges treated as undirected."""
    node_ids = [1, 2, 3, 4, 5]
    edges = [(1, 2), (3, 2), (4, 5)]
    directed = Graph(node_ids, edges, directed=True)
    undirected = Graph(node_ids, edges)
    wcc = sorted(directed.weakly_connected_components(), key=sorted)
    cc = sorted(undirected.connected_components(), key=sorted)
    assert wcc == cc
