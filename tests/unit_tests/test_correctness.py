"""Correctness tests for all cgraph algorithms."""

import pytest

from cgraph import (
    articulation_points,
    bfs,
    biconnected_components,
    bridges,
    connected_components,
    connected_components_with_branch_ids,
    eccentricity,
    multi_source_shortest_path_lengths,
    nodes_on_simple_paths,
    shortest_path,
    shortest_path_lengths,
    two_edge_connected_components,
)

# ── Connected Components ──


@pytest.mark.parametrize(
    ("node_ids", "edges", "expected"),
    [
        pytest.param([], [], [], id="no_nodes"),
        pytest.param([0], [], [{0}], id="single_node_no_edges"),
        pytest.param([0, 1, 2, 3], [], [{0}, {1}, {2}, {3}], id="all_nodes_isolated"),
        pytest.param(
            [0, 1, 2, 3],
            [(0, 1), (1, 2), (2, 3)],
            [{0, 1, 2, 3}],
            id="single_connected_component",
        ),
        pytest.param(
            [0, 1, 2],
            [(0, 1), (1, 2), (2, 0)],
            [{0, 1, 2}],
            id="cycle_graph",
        ),
        pytest.param(
            [0, 1, 2, 3, 4],
            [(0, 1), (0, 2), (0, 3), (0, 4)],
            [{0, 1, 2, 3, 4}],
            id="star_topology",
        ),
        pytest.param(
            [0, 1, 2, 3],
            [(0, 1), (2, 3)],
            [{0, 1}, {2, 3}],
            id="two_disconnected_components",
        ),
        pytest.param(
            [0, 1, 2, 3],
            [(0, 1), (0, 1), (2, 3)],
            [{0, 1}, {2, 3}],
            id="parallel_edges",
        ),
        pytest.param(
            [0, 1, 2],
            [(0, 0), (1, 2)],
            [{0}, {1, 2}],
            id="self_loop",
        ),
        pytest.param(
            [0, 1, 2, 3, 4],
            [(1, 3)],
            [{0}, {1, 3}, {2}, {4}],
            id="mixed_isolated_and_connected",
        ),
        pytest.param(
            [0, 1, 2, 3, 4],
            [(i, j) for i in range(5) for j in range(i + 1, 5)],
            [{0, 1, 2, 3, 4}],
            id="complete_graph",
        ),
    ],
)
def test_connected_components(
    node_ids: list[int],
    edges: list[tuple[int, int]],
    expected: list[set[int]],
) -> None:
    result = list(connected_components(node_ids, edges))
    assert len(result) == len(expected)
    for component in expected:
        assert component in result


@pytest.mark.parametrize(
    ("node_ids", "edges", "branch_ids", "expected"),
    [
        pytest.param([], [], [], [], id="no_nodes"),
        pytest.param([0], [], [], [({0}, set())], id="single_node_no_edges"),
        pytest.param(
            [0, 1, 2, 3],
            [],
            [],
            [({0}, set()), ({1}, set()), ({2}, set()), ({3}, set())],
            id="all_nodes_isolated",
        ),
        pytest.param(
            [0, 1, 2, 3],
            [(0, 1), (1, 2), (2, 3)],
            [10, 11, 12],
            [({0, 1, 2, 3}, {10, 11, 12})],
            id="single_connected_component",
        ),
        pytest.param(
            [0, 1, 2],
            [(0, 1), (1, 2), (2, 0)],
            [20, 21, 22],
            [({0, 1, 2}, {20, 21, 22})],
            id="cycle_graph",
        ),
        pytest.param(
            [0, 1, 2, 3, 4],
            [(0, 1), (0, 2), (0, 3), (0, 4)],
            [30, 31, 32, 33],
            [({0, 1, 2, 3, 4}, {30, 31, 32, 33})],
            id="star_topology",
        ),
        pytest.param(
            [0, 1, 2, 3],
            [(0, 1), (2, 3)],
            [40, 41],
            [({0, 1}, {40}), ({2, 3}, {41})],
            id="two_disconnected_components",
        ),
        pytest.param(
            [0, 1, 2, 3],
            [(0, 1), (0, 1), (2, 3)],
            [50, 51, 52],
            [({0, 1}, {50, 51}), ({2, 3}, {52})],
            id="parallel_edges",
        ),
        pytest.param(
            [0, 1, 2],
            [(0, 0), (1, 2)],
            [60, 61],
            [({0}, {60}), ({1, 2}, {61})],
            id="self_loop",
        ),
        pytest.param(
            [0, 1, 2, 3, 4],
            [(1, 3)],
            [70],
            [({0}, set()), ({1, 3}, {70}), ({2}, set()), ({4}, set())],
            id="mixed_isolated_and_connected",
        ),
        pytest.param(
            [0, 1, 2, 3, 4],
            [(i, j) for i in range(5) for j in range(i + 1, 5)],
            list(range(100, 110)),
            [({0, 1, 2, 3, 4}, set(range(100, 110)))],
            id="complete_graph",
        ),
    ],
)
def test_connected_components_with_branch_ids(
    node_ids: list[int],
    edges: list[tuple[int, int]],
    branch_ids: list[int],
    expected: list[tuple[set[int], set[int]]],
) -> None:
    result = list(connected_components_with_branch_ids(node_ids, edges, branch_ids))
    assert len(result) == len(expected)
    for expected_nodes, expected_branches in expected:
        assert (expected_nodes, expected_branches) in result


def test_node_coverage() -> None:
    node_ids = list(range(1000))
    edges = [(i, i + 1) for i in range(0, 999, 2)]
    components = list(connected_components(node_ids, edges))
    all_nodes: set[int] = set()
    for component in components:
        assert not all_nodes & component, "overlap"
        all_nodes |= component
    assert all_nodes == set(range(1000))


# ── Remapped variants (non-contiguous node IDs) ──


def test_remapped_basic() -> None:
    node_ids = [100, 200, 300, 400]
    edges = [(0, 1), (2, 3)]
    result = list(connected_components(node_ids, edges))
    assert len(result) == 2
    assert {100, 200} in result
    assert {300, 400} in result


def test_remapped_no_edges() -> None:
    node_ids = [10, 20, 30]
    result = list(connected_components(node_ids, []))
    assert len(result) == 3
    assert {10} in result
    assert {20} in result
    assert {30} in result


def test_remapped_single_component() -> None:
    node_ids = [5, 10, 15]
    edges = [(0, 1), (1, 2)]
    result = list(connected_components(node_ids, edges))
    assert len(result) == 1
    assert result[0] == {5, 10, 15}


def test_remapped_empty() -> None:
    result = list(connected_components([], []))
    assert result == []


def test_remapped_with_branches_basic() -> None:
    node_ids = [100, 200, 300, 400]
    edges = [(0, 1), (2, 3)]
    branch_ids = [901, 902]
    result = list(connected_components_with_branch_ids(node_ids, edges, branch_ids))
    assert len(result) == 2
    result_map = {frozenset(nodes): branches for nodes, branches in result}
    assert result_map[frozenset({100, 200})] == {901}
    assert result_map[frozenset({300, 400})] == {902}


def test_remapped_with_branches_merged() -> None:
    node_ids = [10, 20, 30]
    edges = [(0, 1), (1, 2)]
    branch_ids = [500, 600]
    result = list(connected_components_with_branch_ids(node_ids, edges, branch_ids))
    assert len(result) == 1
    nodes, br = result[0]
    assert nodes == {10, 20, 30}
    assert br == {500, 600}


def test_remapped_with_branches_isolated() -> None:
    node_ids = [7, 8, 9]
    result = list(connected_components_with_branch_ids(node_ids, [], []))
    assert len(result) == 3
    for nodes, br in result:
        assert len(nodes) == 1
        assert br == set()


# ── Bridges ──


@pytest.mark.parametrize(
    ("node_ids", "edges", "expected"),
    [
        pytest.param([], [], [], id="empty"),
        pytest.param([0, 1], [], [], id="no_edges"),
        pytest.param(
            [0, 1],
            [(0, 1)],
            [(0, 1)],
            id="single_edge_bridge",
        ),
        pytest.param(
            [0, 1, 2],
            [(0, 1), (1, 2), (2, 0)],
            [],
            id="triangle_no_bridges",
        ),
        pytest.param(
            [0, 1, 2, 3],
            [(0, 1), (1, 2), (2, 0), (2, 3)],
            [(2, 3)],
            id="triangle_with_pendant_bridge",
        ),
        pytest.param(
            [0, 1, 2, 3, 4],
            [(0, 1), (1, 2), (2, 3), (3, 4)],
            [(0, 1), (1, 2), (2, 3), (3, 4)],
            id="path_all_bridges",
        ),
        pytest.param(
            [0, 1, 2, 3],
            [(0, 1), (0, 1), (2, 3)],
            [(2, 3)],
            id="parallel_edges_not_bridge",
        ),
    ],
)
def test_bridges(
    node_ids: list[int],
    edges: list[tuple[int, int]],
    expected: list[tuple[int, int]],
) -> None:
    result = bridges(node_ids, edges)
    expected_set = {(min(u, v), max(u, v)) for u, v in expected}
    result_set = {(min(u, v), max(u, v)) for u, v in result}
    assert result_set == expected_set


# ── Articulation Points ──


@pytest.mark.parametrize(
    ("node_ids", "edges", "expected"),
    [
        pytest.param([], [], set(), id="empty"),
        pytest.param([0, 1], [(0, 1)], set(), id="single_edge_no_ap"),
        pytest.param(
            [0, 1, 2],
            [(0, 1), (1, 2)],
            {1},
            id="path_middle_is_ap",
        ),
        pytest.param(
            [0, 1, 2],
            [(0, 1), (1, 2), (2, 0)],
            set(),
            id="triangle_no_ap",
        ),
        pytest.param(
            [0, 1, 2, 3],
            [(0, 1), (1, 2), (2, 0), (2, 3)],
            {2},
            id="triangle_with_pendant",
        ),
        pytest.param(
            [0, 1, 2, 3, 4],
            [(0, 1), (1, 2), (2, 3), (3, 4)],
            {1, 2, 3},
            id="path_interior_are_aps",
        ),
        pytest.param(
            [0, 1, 2, 3, 4],
            [(0, 1), (0, 2), (0, 3), (0, 4)],
            {0},
            id="star_center_is_ap",
        ),
    ],
)
def test_articulation_points(
    node_ids: list[int],
    edges: list[tuple[int, int]],
    expected: set[int],
) -> None:
    result = articulation_points(node_ids, edges)
    assert result == expected


# ── Biconnected Components ──


@pytest.mark.parametrize(
    ("node_ids", "edges", "expected"),
    [
        pytest.param([], [], [], id="empty"),
        pytest.param([0, 1, 2], [], [], id="isolated_nodes"),
        pytest.param(
            [0, 1],
            [(0, 1)],
            [{0, 1}],
            id="single_edge",
        ),
        pytest.param(
            [0, 1, 2],
            [(0, 1), (1, 2), (2, 0)],
            [{0, 1, 2}],
            id="triangle",
        ),
        pytest.param(
            [0, 1, 2, 3],
            [(0, 1), (1, 2), (2, 0), (2, 3)],
            [{0, 1, 2}, {2, 3}],
            id="triangle_with_pendant",
        ),
        pytest.param(
            [0, 1, 2, 3, 4],
            [(0, 1), (1, 2), (2, 3), (3, 4)],
            [{0, 1}, {1, 2}, {2, 3}, {3, 4}],
            id="path_each_edge_is_bcc",
        ),
    ],
)
def test_biconnected_components(
    node_ids: list[int],
    edges: list[tuple[int, int]],
    expected: list[set[int]],
) -> None:
    result = list(biconnected_components(node_ids, edges))
    assert len(result) == len(expected)
    for comp in expected:
        assert comp in result


# ── BFS ──


@pytest.mark.parametrize(
    ("node_ids", "edges", "source", "expected_set"),
    [
        pytest.param(
            [0, 1, 2],
            [(0, 1), (1, 2)],
            0,
            {0, 1, 2},
            id="path_from_start",
        ),
        pytest.param(
            [0, 1, 2, 3],
            [(0, 1), (2, 3)],
            0,
            {0, 1},
            id="disconnected_from_start",
        ),
        pytest.param(
            [10, 20, 30],
            [(0, 1), (1, 2)],
            10,
            {10, 20, 30},
            id="remapped_ids",
        ),
    ],
)
def test_bfs(
    node_ids: list[int],
    edges: list[tuple[int, int]],
    source: int,
    expected_set: set[int],
) -> None:
    result = bfs(node_ids, edges, source)
    assert result[0] == source
    assert set(result) == expected_set


def test_bfs_order() -> None:
    """BFS should visit neighbors before their neighbors."""
    # Star graph: 0 connected to 1,2,3,4
    node_ids = list(range(5))
    edges = [(0, 1), (0, 2), (0, 3), (0, 4)]
    result = bfs(node_ids, edges, 0)
    assert result[0] == 0
    assert set(result[1:]) == {1, 2, 3, 4}


# ── Shortest Path (Dijkstra) ──


def test_shortest_path_simple() -> None:
    node_ids = [0, 1, 2, 3]
    edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
    weights = [1.0, 1.0, 1.0, 10.0]
    result = shortest_path(node_ids, edges, weights, 0, 3)
    assert result == [0, 1, 2, 3]


def test_shortest_path_direct() -> None:
    node_ids = [0, 1, 2, 3]
    edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
    weights = [1.0, 1.0, 1.0, 2.0]
    result = shortest_path(node_ids, edges, weights, 0, 3)
    assert result == [0, 3]


def test_shortest_path_same_node() -> None:
    node_ids = [0, 1]
    edges = [(0, 1)]
    weights = [1.0]
    result = shortest_path(node_ids, edges, weights, 0, 0)
    assert result == [0]


def test_shortest_path_unreachable() -> None:
    node_ids = [0, 1, 2, 3]
    edges = [(0, 1), (2, 3)]
    weights = [1.0, 1.0]
    result = shortest_path(node_ids, edges, weights, 0, 3)
    assert result == []


def test_shortest_path_remapped() -> None:
    node_ids = [100, 200, 300]
    edges = [(0, 1), (1, 2)]
    weights = [3.0, 4.0]
    result = shortest_path(node_ids, edges, weights, 100, 300)
    assert result == [100, 200, 300]


# ── SSSP Lengths ──


def test_sssp_lengths_basic() -> None:
    node_ids = [0, 1, 2, 3]
    edges = [(0, 1), (1, 2), (2, 3)]
    weights = [1.0, 2.0, 3.0]
    result = shortest_path_lengths(node_ids, edges, weights, 0)
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(1.0)
    assert result[2] == pytest.approx(3.0)
    assert result[3] == pytest.approx(6.0)


def test_sssp_lengths_with_cutoff() -> None:
    node_ids = [0, 1, 2, 3]
    edges = [(0, 1), (1, 2), (2, 3)]
    weights = [1.0, 2.0, 3.0]
    result = shortest_path_lengths(node_ids, edges, weights, 0, cutoff=2.5)
    assert 0 in result
    assert 1 in result
    assert 2 not in result  # distance 3.0 > cutoff 2.5
    assert 3 not in result


def test_sssp_lengths_disconnected() -> None:
    node_ids = [0, 1, 2, 3]
    edges = [(0, 1), (2, 3)]
    weights = [1.0, 1.0]
    result = shortest_path_lengths(node_ids, edges, weights, 0)
    assert 0 in result
    assert 1 in result
    assert 2 not in result
    assert 3 not in result


# ── Multi-Source Shortest Path Lengths ──


def test_multi_source_basic() -> None:
    node_ids = [0, 1, 2, 3, 4]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    weights = [1.0, 1.0, 1.0, 1.0]
    result = multi_source_shortest_path_lengths(
        node_ids, edges, weights, [0, 4],
    )
    assert result[0] == pytest.approx(0.0)
    assert result[4] == pytest.approx(0.0)
    assert result[2] == pytest.approx(2.0)
    assert result[1] == pytest.approx(1.0)
    assert result[3] == pytest.approx(1.0)


def test_multi_source_with_cutoff() -> None:
    node_ids = [0, 1, 2, 3, 4]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    weights = [1.0, 1.0, 1.0, 1.0]
    result = multi_source_shortest_path_lengths(
        node_ids, edges, weights, [0], cutoff=2.0,
    )
    assert 0 in result
    assert 1 in result
    assert 2 in result
    assert 3 not in result
    assert 4 not in result


# ── Eccentricity ──


def test_eccentricity_path() -> None:
    node_ids = [0, 1, 2, 3, 4]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    weights = [1.0, 1.0, 1.0, 1.0]
    assert eccentricity(node_ids, edges, weights, 0) == pytest.approx(4.0)
    assert eccentricity(node_ids, edges, weights, 2) == pytest.approx(2.0)


def test_eccentricity_single_node() -> None:
    node_ids = [0]
    assert eccentricity(node_ids, [], [], 0) == pytest.approx(0.0)


# ── Two-Edge-Connected Components ──


@pytest.mark.parametrize(
    ("node_ids", "edges", "expected"),
    [
        pytest.param([], [], [], id="empty"),
        pytest.param(
            [0, 1, 2],
            [(0, 1), (1, 2), (2, 0)],
            [{0, 1, 2}],
            id="triangle",
        ),
        pytest.param(
            [0, 1, 2, 3],
            [(0, 1), (1, 2), (2, 0), (2, 3)],
            [{0, 1, 2}, {3}],
            id="triangle_with_pendant",
        ),
        pytest.param(
            [0, 1, 2, 3, 4],
            [(0, 1), (1, 2), (2, 3), (3, 4)],
            [{0}, {1}, {2}, {3}, {4}],
            id="path_all_singletons",
        ),
    ],
)
def test_two_edge_connected_components(
    node_ids: list[int],
    edges: list[tuple[int, int]],
    expected: list[set[int]],
) -> None:
    result = list(two_edge_connected_components(node_ids, edges))
    assert len(result) == len(expected)
    for comp in expected:
        assert comp in result


# ── Nodes on Simple Paths ──


def test_nodes_on_simple_paths_path_graph() -> None:
    """All nodes on the unique path from 0 to 4."""
    node_ids = [0, 1, 2, 3, 4]
    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    result = nodes_on_simple_paths(node_ids, edges, 0, [4])
    assert result == {0, 1, 2, 3, 4}


def test_nodes_on_simple_paths_excludes_dead_end() -> None:
    """Dead-end node 5 should NOT be on any simple path from 0 to 3."""
    # 0-1-2-3, plus 1-5 (dead end)
    node_ids = [0, 1, 2, 3, 5]
    edges = [(0, 1), (1, 2), (2, 3), (1, 4)]  # 4 is index of node 5
    result = nodes_on_simple_paths(node_ids, edges, 0, [3])
    assert result == {0, 1, 2, 3}


def test_nodes_on_simple_paths_triangle() -> None:
    """In a triangle, all nodes are on some simple path between any two."""
    node_ids = [0, 1, 2]
    edges = [(0, 1), (1, 2), (2, 0)]
    result = nodes_on_simple_paths(node_ids, edges, 0, [2])
    assert result == {0, 1, 2}


def test_nodes_on_simple_paths_source_is_target() -> None:
    node_ids = [0, 1, 2]
    edges = [(0, 1), (1, 2)]
    result = nodes_on_simple_paths(node_ids, edges, 0, [0])
    assert result == {0}


def test_nodes_on_simple_paths_multiple_targets() -> None:
    # 0-1-2, 0-3-4
    node_ids = [0, 1, 2, 3, 4]
    edges = [(0, 1), (1, 2), (0, 3), (3, 4)]
    result = nodes_on_simple_paths(node_ids, edges, 0, [2, 4])
    assert result == {0, 1, 2, 3, 4}
