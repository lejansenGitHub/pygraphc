"""Correctness tests for the C union-find connected components."""

from connected_component import igp_connected_components, igp_connected_components_with_branch_ids


def test_no_edges():
    components = list(igp_connected_components(5, []))
    assert len(components) == 5
    assert all(len(c) == 1 for c in components)
    assert {frozenset(c) for c in components} == {frozenset({i}) for i in range(5)}


def test_single_component():
    edges = [(0, 1), (1, 2), (2, 3)]
    components = list(igp_connected_components(4, edges))
    assert len(components) == 1
    assert components[0] == {0, 1, 2, 3}


def test_two_components():
    edges = [(0, 1), (2, 3)]
    components = list(igp_connected_components(4, edges))
    assert len(components) == 2
    component_sets = {frozenset(c) for c in components}
    assert component_sets == {frozenset({0, 1}), frozenset({2, 3})}


def test_isolated_nodes():
    edges = [(0, 1)]
    components = list(igp_connected_components(4, edges))
    assert len(components) == 3
    component_sets = {frozenset(c) for c in components}
    assert component_sets == {frozenset({0, 1}), frozenset({2}), frozenset({3})}


def test_with_branch_ids_no_edges():
    results = list(igp_connected_components_with_branch_ids(3, [], []))
    assert len(results) == 3
    for nodes, branches in results:
        assert len(nodes) == 1
        assert branches == set()


def test_with_branch_ids():
    edges = [(0, 1), (2, 3)]
    branch_ids = [100, 200]
    results = list(igp_connected_components_with_branch_ids(4, edges, branch_ids))
    assert len(results) == 2
    result_map = {frozenset(nodes): branches for nodes, branches in results}
    assert result_map[frozenset({0, 1})] == {100}
    assert result_map[frozenset({2, 3})] == {200}


def test_with_branch_ids_merged():
    edges = [(0, 1), (1, 2)]
    branch_ids = [10, 20]
    results = list(igp_connected_components_with_branch_ids(3, edges, branch_ids))
    assert len(results) == 1
    nodes, branches = results[0]
    assert nodes == {0, 1, 2}
    assert branches == {10, 20}


def test_node_coverage():
    """Every node must appear in exactly one component."""
    n = 1000
    edges = [(i, i + 1) for i in range(0, n - 1, 2)]
    components = list(igp_connected_components(n, edges))
    all_nodes = set()
    for c in components:
        assert not all_nodes & c, "overlap"
        all_nodes |= c
    assert all_nodes == set(range(n))
