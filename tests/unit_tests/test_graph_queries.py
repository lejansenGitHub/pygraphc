"""Tests for neighbors(), degree(), incident_edge_indices(), bridges_with_branch_ids()."""

import pytest

from networkc import Graph

# ── Fixtures ──


@pytest.fixture
def triangle():
    """Triangle: 0-1-2-0. Edges: 0:(0,1), 1:(1,2), 2:(2,0)."""
    return Graph([0, 1, 2], [(0, 1), (1, 2), (2, 0)])


@pytest.fixture
def chain():
    """Chain: 0-1-2-3. Edges: 0:(0,1), 1:(1,2), 2:(2,3)."""
    return Graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])


@pytest.fixture
def star():
    """Star: center=0, leaves=1,2,3. Edges: 0:(0,1), 1:(0,2), 2:(0,3)."""
    return Graph([0, 1, 2, 3], [(0, 1), (0, 2), (0, 3)])


# ── neighbors() on Graph ──


def test_neighbors_triangle(triangle):
    assert triangle.neighbors(0) == {1, 2}
    assert triangle.neighbors(1) == {0, 2}
    assert triangle.neighbors(2) == {0, 1}


def test_neighbors_chain_endpoints(chain):
    assert chain.neighbors(0) == {1}
    assert chain.neighbors(3) == {2}


def test_neighbors_chain_middle(chain):
    assert chain.neighbors(1) == {0, 2}
    assert chain.neighbors(2) == {1, 3}


def test_neighbors_star_center(star):
    assert star.neighbors(0) == {1, 2, 3}


def test_neighbors_star_leaf(star):
    assert star.neighbors(1) == {0}


def test_neighbors_isolated_node():
    graph = Graph([0, 1, 2], [(0, 1)])
    assert graph.neighbors(2) == set()


def test_neighbors_self_loop():
    """Self-loop does not add the node as its own neighbor."""
    graph = Graph([0, 1], [(0, 0), (0, 1)])
    assert graph.neighbors(0) == {1}


def test_neighbors_parallel_edges():
    """Parallel edges: neighbor appears only once."""
    graph = Graph([0, 1], [(0, 1), (0, 1)])
    assert graph.neighbors(0) == {1}


def test_neighbors_empty_graph():
    graph = Graph([], [])
    # No node to query — but the method shouldn't crash
    assert graph.neighbors(99) == set()


# ── neighbors() on GraphView ──


def test_neighbors_view_excluded_edge(triangle):
    view = triangle.without_edges([0])  # exclude (0,1)
    assert view.neighbors(0) == {2}
    assert view.neighbors(1) == {2}


def test_neighbors_view_excluded_node(chain):
    view = chain.without_nodes([1])
    assert view.neighbors(0) == set()  # edge (0,1) is excluded because node 1 is
    assert view.neighbors(2) == {3}


def test_neighbors_view_excluded_node_is_queried(chain):
    """Querying neighbors of an excluded node returns empty set."""
    view = chain.without_nodes([1])
    assert view.neighbors(1) == set()


def test_neighbors_view_all_edges_excluded(triangle):
    view = triangle.without_edges([0, 1, 2])
    assert view.neighbors(0) == set()
    assert view.neighbors(1) == set()


# ── degree() on Graph ──


def test_degree_triangle(triangle):
    assert triangle.degree(0) == 2
    assert triangle.degree(1) == 2
    assert triangle.degree(2) == 2


def test_degree_chain_endpoints(chain):
    assert chain.degree(0) == 1
    assert chain.degree(3) == 1


def test_degree_chain_middle(chain):
    assert chain.degree(1) == 2
    assert chain.degree(2) == 2


def test_degree_star_center(star):
    assert star.degree(0) == 3


def test_degree_star_leaf(star):
    assert star.degree(1) == 1


def test_degree_isolated_node():
    graph = Graph([0, 1, 2], [(0, 1)])
    assert graph.degree(2) == 0


def test_degree_self_loop():
    """Self-loop counts as 2."""
    graph = Graph([0, 1], [(0, 0), (0, 1)])
    assert graph.degree(0) == 3  # 2 for self-loop + 1 for (0,1)


def test_degree_parallel_edges():
    """Two parallel edges count as degree 2."""
    graph = Graph([0, 1], [(0, 1), (0, 1)])
    assert graph.degree(0) == 2


# ── degree() on GraphView ──


def test_degree_view_excluded_edge(triangle):
    view = triangle.without_edges([0])  # exclude (0,1)
    assert view.degree(0) == 1  # only (2,0) remains
    assert view.degree(1) == 1  # only (1,2) remains


def test_degree_view_excluded_node(chain):
    view = chain.without_nodes([1])
    assert view.degree(0) == 0
    assert view.degree(1) == 0  # node itself excluded
    assert view.degree(2) == 1  # only (2,3) remains


def test_degree_view_all_excluded(triangle):
    view = triangle.without_edges([0, 1, 2])
    assert view.degree(0) == 0


# ── incident_edge_indices() on Graph ──


def test_incident_edge_indices_triangle(triangle):
    assert sorted(triangle.incident_edge_indices(0)) == [0, 2]
    assert sorted(triangle.incident_edge_indices(1)) == [0, 1]
    assert sorted(triangle.incident_edge_indices(2)) == [1, 2]


def test_incident_edge_indices_chain_endpoint(chain):
    assert chain.incident_edge_indices(0) == [0]
    assert chain.incident_edge_indices(3) == [2]


def test_incident_edge_indices_isolated_node():
    graph = Graph([0, 1, 2], [(0, 1)])
    assert graph.incident_edge_indices(2) == []


def test_incident_edge_indices_self_loop():
    """Self-loop appears once in the list (it's one edge)."""
    graph = Graph([0, 1], [(0, 0), (0, 1)])
    assert sorted(graph.incident_edge_indices(0)) == [0, 1]


def test_incident_edge_indices_parallel_edges():
    """Both parallel edges are returned."""
    graph = Graph([0, 1], [(0, 1), (0, 1)])
    assert sorted(graph.incident_edge_indices(0)) == [0, 1]


# ── incident_edge_indices() on GraphView ──


def test_incident_edge_indices_view_excluded_edge(triangle):
    view = triangle.without_edges([0])  # exclude (0,1)
    assert view.incident_edge_indices(0) == [2]  # only (2,0)
    assert view.incident_edge_indices(1) == [1]  # only (1,2)


def test_incident_edge_indices_view_excluded_node(chain):
    view = chain.without_nodes([1])
    assert view.incident_edge_indices(0) == []  # edge (0,1) excluded via node 1
    assert view.incident_edge_indices(1) == []  # node itself excluded
    assert view.incident_edge_indices(2) == [2]  # only (2,3)


def test_incident_edge_indices_view_all_excluded(triangle):
    view = triangle.without_edges([0, 1, 2])
    assert view.incident_edge_indices(0) == []


# ── bridges_with_branch_ids() on Graph ──


def test_bridges_with_branch_ids_chain():
    """Chain: every edge is a bridge."""
    graph = Graph([0, 1, 2], [(0, 1), (1, 2)], branch_ids=[100, 200])
    result = graph.bridges_with_branch_ids()
    branch_ids_found = {branch_id for _, _, branch_id in result}
    assert branch_ids_found == {100, 200}


def test_bridges_with_branch_ids_triangle():
    """Triangle: no bridges."""
    graph = Graph([0, 1, 2], [(0, 1), (1, 2), (2, 0)], branch_ids=[100, 200, 300])
    assert graph.bridges_with_branch_ids() == []


def test_bridges_with_branch_ids_bridge_graph():
    """Two triangles connected by a bridge."""
    # (0-1-2-0) --bridge(2,3)-- (3-4-5-3)
    nodes = [0, 1, 2, 3, 4, 5]
    edges = [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)]
    branch_ids = [10, 20, 30, 40, 50, 60, 70]
    graph = Graph(nodes, edges, branch_ids=branch_ids)
    result = graph.bridges_with_branch_ids()
    assert len(result) == 1
    _, _, bridge_branch_id = result[0]
    assert bridge_branch_id == 40  # edge (2,3)


def test_bridges_with_branch_ids_raises_without_branch_ids():
    graph = Graph([0, 1], [(0, 1)])
    with pytest.raises(ValueError, match="no branch_ids"):
        graph.bridges_with_branch_ids()


# ── bridges_with_branch_ids() on GraphView ──


def test_bridges_with_branch_ids_view():
    """Removing a triangle edge makes the other edges bridges."""
    # Triangle + tail: 0-1-2-0, 2-3
    nodes = [0, 1, 2, 3]
    edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
    branch_ids = [10, 20, 30, 40]
    graph = Graph(nodes, edges, branch_ids=branch_ids)

    # Without mask: only edge (2,3) is a bridge
    assert len(graph.bridges_with_branch_ids()) == 1

    # Exclude edge (2,0) — now (0,1), (1,2), (2,3) are all bridges
    view = graph.without_edges([2])
    result = view.bridges_with_branch_ids()
    branch_ids_found = {branch_id for _, _, branch_id in result}
    assert branch_ids_found == {10, 20, 40}


def test_bridges_with_branch_ids_view_raises_without_branch_ids():
    graph = Graph([0, 1], [(0, 1)])
    view = graph.without_edges([])
    with pytest.raises(ValueError, match="no branch_ids"):
        view.bridges_with_branch_ids()


# ── Large node IDs ──


def test_neighbors_large_node_ids():
    graph = Graph([1000, 2000, 3000], [(1000, 2000), (2000, 3000)])
    assert graph.neighbors(2000) == {1000, 3000}


def test_degree_large_node_ids():
    graph = Graph([1000, 2000, 3000], [(1000, 2000), (2000, 3000)])
    assert graph.degree(2000) == 2


def test_incident_edge_indices_large_node_ids():
    graph = Graph([1000, 2000, 3000], [(1000, 2000), (2000, 3000)])
    assert sorted(graph.incident_edge_indices(2000)) == [0, 1]


# ── Combined view operations ──


def test_neighbors_after_edge_and_node_exclusion():
    """Chain with both edge and node masks."""
    graph = Graph([0, 1, 2, 3, 4], [(0, 1), (1, 2), (2, 3), (3, 4)])
    view = graph.without_edges([0]).without_nodes([3])
    # Edge (0,1) excluded, node 3 excluded (which also excludes edges (2,3) and (3,4))
    assert view.neighbors(0) == set()
    assert view.neighbors(1) == {2}
    assert view.neighbors(2) == {1}
    assert view.neighbors(4) == set()


def test_degree_after_edge_and_node_exclusion():
    graph = Graph([0, 1, 2, 3, 4], [(0, 1), (1, 2), (2, 3), (3, 4)])
    view = graph.without_edges([0]).without_nodes([3])
    assert view.degree(0) == 0
    assert view.degree(1) == 1
    assert view.degree(2) == 1
    assert view.degree(4) == 0
