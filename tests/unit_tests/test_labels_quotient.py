"""Tests for the int32 label kernels: ``component_labels``, ``quotient_edges``,
``degrees`` and ``bcc_edge_labels`` on ``Graph`` and ``GraphView``.

Every kernel is checked on masks, self-loops, parallel edges, excluded nodes
and the empty graph, and the two label kernels are cross-checked against the
set-returning algorithms on random multigraphs.
"""

import random
from array import array

import pytest

from pygraphc import Graph

SEED = 20260910
PROPERTY_TRIALS = 300


def labels_of(view) -> list[int]:
    return view.component_labels().tolist()


def random_graph(rng: random.Random, node_count: int, edge_count: int) -> tuple[list[int], list[tuple[int, int]]]:
    """Node ids in random order and edges with self-loops and parallel edges."""
    nodes = rng.sample(range(1000), node_count)
    edges: list[tuple[int, int]] = []
    for _ in range(edge_count):
        from_node = rng.choice(nodes)
        roll = rng.random()
        if roll < 0.08:
            to_node = from_node
        elif roll < 0.25 and edges:
            from_node, to_node = edges[rng.randrange(len(edges))]
        else:
            to_node = rng.choice(nodes)
        edges.append((from_node, to_node))
    return nodes, edges


def random_view(rng: random.Random, nodes: list[int], edges: list[tuple[int, int]]):
    """The parsed graph, or a view of it with random excluded edges and nodes."""
    graph = Graph(nodes, edges)
    excluded_edges = [index for index in range(len(edges)) if rng.random() < 0.2]
    excluded_nodes = [node_id for node_id in nodes if rng.random() < 0.1]
    if not excluded_edges and not excluded_nodes:
        return graph
    return graph.without_edges(excluded_edges).without_nodes(excluded_nodes)


# ── component_labels ──


def test_component_labels_are_the_smallest_node_index_of_each_component():
    """The label must identify a component by its smallest node index, so a
    caller with sorted node ids reads the numerically smallest member off it."""
    # --- Input ---
    #  0:10 -- 1:20 -- 2:30    3:40 -- 4:50    5:60
    graph = Graph([10, 20, 30, 40, 50, 60], [(10, 20), (20, 30), (40, 50)])

    # --- Assert ---
    assert labels_of(graph) == [0, 0, 0, 3, 3, 5]


def test_component_labels_return_an_int32_view_with_one_entry_per_node():
    """Callers index the result by node position and hand it to array or numpy,
    so it must be an int32 view of exactly node_count entries."""
    # --- Input ---
    graph = Graph([1, 2, 3], [(1, 2)])

    # --- Execute ---
    labels = graph.component_labels()

    # --- Assert ---
    assert labels.format == "i"
    assert labels.itemsize == 4
    assert len(labels) == 3


def test_component_labels_ignore_excluded_edges():
    """An excluded edge must not connect, so masking the middle edge of a path
    splits the labels without rebuilding the graph."""
    # --- Input ---
    graph = Graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3)])

    # --- Execute ---
    view = graph.without_edges([1])

    # --- Assert ---
    assert labels_of(graph) == [0, 0, 0, 0]
    assert labels_of(view) == [0, 0, 2, 2]


def test_component_labels_mark_excluded_nodes_with_minus_one_and_do_not_connect_through_them():
    """An excluded node belongs to no block, so it must be labelled -1 and its
    edges must not join its neighbours."""
    # --- Input ---
    graph = Graph([0, 1, 2], [(0, 1), (1, 2)])

    # --- Execute ---
    view = graph.without_nodes([1])

    # --- Assert ---
    assert labels_of(view) == [0, -1, 2]


def test_component_labels_with_self_loops_and_parallel_edges():
    """A self-loop and a parallel edge add no connectivity, so the labels equal
    those of the simple graph underneath."""
    # --- Input ---
    graph = Graph([0, 1, 2], [(0, 0), (0, 1), (1, 0), (2, 2)])

    # --- Assert ---
    assert labels_of(graph) == [0, 0, 2]


def test_component_labels_of_the_empty_graph_are_empty():
    """The empty graph has no nodes to label; the kernel must not touch a zero-length buffer."""
    # --- Input ---
    graph = Graph([], [])

    # --- Assert ---
    assert labels_of(graph) == []
    assert labels_of(Graph([7], [])) == [0]


def test_component_labels_agree_with_connected_components_on_random_multigraphs():
    """The label kernel reuses the masked union-find of connected_components,
    so grouping node ids by label must give exactly the component sets."""
    rng = random.Random(SEED)
    for _ in range(PROPERTY_TRIALS):
        # --- Input ---
        nodes, edges = random_graph(rng, rng.randrange(1, 12), rng.randrange(0, 20))
        view = random_view(rng, nodes, edges)

        # --- Execute ---
        indices_by_label: dict[int, list[int]] = {}
        for index, label in enumerate(view.component_labels()):
            if label != -1:
                indices_by_label.setdefault(label, []).append(index)
        node_sets = [{nodes[index] for index in indices} for indices in indices_by_label.values()]

        # --- Assert ---
        assert all(label == min(indices) for label, indices in indices_by_label.items())
        assert sorted(map(sorted, node_sets)) == sorted(map(sorted, view.connected_components()))


# ── quotient_edges ──


def test_quotient_edges_split_crossing_and_internal_edges_in_index_order():
    """Crossing edges must come back as three parallel arrays in edge index order
    with the edge index kept, and internal edges as their indices, so a caller
    maps every edge back to its own id and never keys anything by endpoint pair."""
    # --- Input ---
    #  labels: node 0,1 -> 0   node 2,3 -> 2
    graph = Graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    labels = memoryview(array("i", [0, 0, 2, 2]))

    # --- Execute ---
    src_labels, dst_labels, edge_indices, internal_edge_indices = graph.quotient_edges(labels)

    # --- Assert ---
    assert src_labels.tolist() == [0, 2, 0]
    assert dst_labels.tolist() == [2, 0, 2]
    assert edge_indices.tolist() == [1, 3, 4]
    assert internal_edge_indices.tolist() == [0, 2]


def test_quotient_edges_keep_parallel_crossing_edges_as_separate_indices():
    """Two parallel edges between the same blocks are two meta edges; collapsing
    them would lose the edge identity the reduction is built on."""
    # --- Input ---
    graph = Graph([0, 1], [(0, 1), (1, 0), (0, 1)])
    labels = memoryview(array("i", [0, 1]))

    # --- Execute ---
    src_labels, dst_labels, edge_indices, internal_edge_indices = graph.quotient_edges(labels)

    # --- Assert ---
    assert list(zip(src_labels, dst_labels, edge_indices, strict=True)) == [(0, 1, 0), (1, 0, 1), (0, 1, 2)]
    assert internal_edge_indices.tolist() == []


def test_quotient_edges_report_self_loops_as_internal():
    """A self-loop has equal endpoint labels by construction, so it is internal
    to its block and never a meta edge."""
    # --- Input ---
    graph = Graph([0, 1], [(0, 0), (0, 1)])
    labels = memoryview(array("i", [0, 1]))

    # --- Execute ---
    _src, _dst, edge_indices, internal_edge_indices = graph.quotient_edges(labels)

    # --- Assert ---
    assert edge_indices.tolist() == [1]
    assert internal_edge_indices.tolist() == [0]


def test_quotient_edges_skip_excluded_edges():
    """The crossing set of a quotient is a mask on the same cached graph, so an
    excluded edge must appear in neither output array."""
    # --- Input ---
    graph = Graph([0, 1, 2], [(0, 1), (1, 2), (0, 0)])
    labels = memoryview(array("i", [0, 1, 1]))

    # --- Execute ---
    _src, _dst, edge_indices, internal_edge_indices = graph.without_edges([0, 2]).quotient_edges(labels)

    # --- Assert ---
    assert edge_indices.tolist() == []
    assert internal_edge_indices.tolist() == [1]


def test_quotient_edges_skip_edges_at_a_node_labelled_minus_one():
    """-1 is the label of an excluded node; an edge at such a node belongs to no
    block and must be neither crossing nor internal."""
    # --- Input ---
    graph = Graph([0, 1, 2], [(0, 1), (1, 2), (0, 2)])
    labels = memoryview(array("i", [0, -1, 2]))

    # --- Execute ---
    src_labels, dst_labels, edge_indices, internal_edge_indices = graph.quotient_edges(labels)

    # --- Assert ---
    assert graph.without_nodes([1]).component_labels().tolist() == [0, -1, 0]
    assert (src_labels.tolist(), dst_labels.tolist(), edge_indices.tolist()) == ([0], [2], [2])
    assert internal_edge_indices.tolist() == []


def test_quotient_edges_accept_the_component_labels_view_and_raw_bytes():
    """The labels usually come straight from component_labels; raw bytes of the
    right length are the same buffer without the cast and must work too."""
    # --- Input ---
    graph = Graph([0, 1, 2], [(0, 1), (1, 2)])
    labels = graph.without_edges([1]).component_labels()

    # --- Execute ---
    from_view = graph.quotient_edges(labels)
    from_bytes = graph.quotient_edges(memoryview(bytes(labels)))

    # --- Assert ---
    assert [part.tolist() for part in from_view] == [[0], [2], [1], [0]]
    assert [part.tolist() for part in from_bytes] == [part.tolist() for part in from_view]


def test_quotient_edges_reject_a_labels_buffer_of_the_wrong_length():
    """The kernel indexes the labels by node index, so a short or long buffer
    would read out of bounds; it must be rejected before the pass."""
    # --- Input ---
    graph = Graph([0, 1, 2], [(0, 1)])

    # --- Assert ---
    with pytest.raises(ValueError, match="one int32 per node"):
        graph.quotient_edges(memoryview(array("i", [0, 0])))
    with pytest.raises(ValueError, match="one int32 per node"):
        graph.quotient_edges(memoryview(array("i", [0, 0, 0, 0])))


def test_quotient_edges_reject_a_labels_buffer_that_is_not_int32():
    """A float64 buffer of the right byte length would be read as garbage labels,
    so the item format is checked, not only the length."""
    # --- Input ---
    graph = Graph([0, 1], [(0, 1)])

    # --- Assert ---
    with pytest.raises(TypeError, match="int32"):
        graph.quotient_edges(memoryview(array("d", [0.0])))


def test_quotient_edges_of_the_empty_graph_are_four_empty_arrays():
    """No nodes and no edges give four empty int32 views, not an error."""
    # --- Input ---
    graph = Graph([], [])

    # --- Execute ---
    parts = graph.quotient_edges(memoryview(array("i", [])))

    # --- Assert ---
    assert [part.tolist() for part in parts] == [[], [], [], []]
    assert all(part.format == "i" for part in parts)


# ── degrees ──


def test_degrees_count_incidences_with_self_loops_twice():
    """A self-loop is incident to its node at both ends, so it contributes two
    like it does in the degree method and in networkx."""
    # --- Input ---
    graph = Graph([0, 1, 2], [(0, 1), (1, 1), (1, 2), (1, 2)])

    # --- Assert ---
    assert graph.degrees().tolist() == [1, 5, 2]
    assert graph.degrees().tolist() == [graph.degree(node_id) for node_id in [0, 1, 2]]


def test_degrees_drop_excluded_edges_and_zero_excluded_nodes():
    """A masked edge counts at neither end, and an excluded node has no
    incidences at all, its neighbours losing the shared edge."""
    # --- Input ---
    graph = Graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 3), (3, 3)])

    # --- Execute ---
    edge_view = graph.without_edges([1])
    node_view = graph.without_nodes([2])

    # --- Assert ---
    assert edge_view.degrees().tolist() == [1, 1, 1, 3]
    assert node_view.degrees().tolist() == [1, 1, 0, 2]
    assert node_view.degrees().tolist() == [node_view.degree(node_id) for node_id in [0, 1, 2, 3]]


def test_degrees_of_a_directed_graph_are_out_degrees():
    """The degree method of a directed graph is the out-degree, and the array
    version must agree with it entry by entry."""
    # --- Input ---
    graph = Graph([0, 1, 2], [(0, 1), (0, 2), (2, 1)], directed=True)

    # --- Assert ---
    assert graph.degrees().tolist() == [2, 0, 1]


def test_degrees_of_the_empty_graph_and_of_isolated_nodes():
    """No edges means every degree is zero and the empty graph gives an empty view."""
    # --- Input ---
    graph = Graph([4, 5], [])

    # --- Assert ---
    assert graph.degrees().tolist() == [0, 0]
    assert Graph([], []).degrees().tolist() == []


# ── bcc_edge_labels ──


def test_bcc_edge_labels_make_bridges_singleton_components():
    """A bridge lies in no cycle, so it forms its own biconnected component and
    the two triangles it joins keep their own ids."""
    # --- Input ---
    #  triangle 0-1-2, bridge 2-3, triangle 3-4-5
    graph = Graph([0, 1, 2, 3, 4, 5], [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)])

    # --- Execute ---
    labels = graph.bcc_edge_labels().tolist()

    # --- Assert ---
    assert len(labels) == 7
    assert labels[0] == labels[1] == labels[2]
    assert labels[4] == labels[5] == labels[6]
    assert len({labels[0], labels[3], labels[4]}) == 3


def test_bcc_edge_labels_put_parallel_edges_in_one_component_and_self_loops_in_none():
    """Two parallel edges form a cycle, so neither is a bridge; a self-loop lies
    in no two-vertex cycle and gets -1."""
    # --- Input ---
    graph = Graph([0, 1, 2], [(0, 1), (1, 0), (1, 2), (2, 2)])

    # --- Execute ---
    labels = graph.bcc_edge_labels().tolist()

    # --- Assert ---
    assert labels[0] == labels[1]
    assert labels[2] not in {labels[0], -1}
    assert labels[3] == -1


def test_bcc_edge_labels_mark_excluded_edges_and_edges_at_excluded_nodes_with_minus_one():
    """Masked material takes part in no component; removing one triangle edge
    also turns the other two into bridges."""
    # --- Input ---
    graph = Graph([0, 1, 2, 3], [(0, 1), (1, 2), (2, 0), (2, 3)])

    # --- Execute ---
    edge_labels = graph.without_edges([2]).bcc_edge_labels().tolist()
    node_labels = graph.without_nodes([3]).bcc_edge_labels().tolist()

    # --- Assert ---
    assert edge_labels[2] == -1
    assert len({edge_labels[0], edge_labels[1], edge_labels[3]}) == 3
    assert node_labels[3] == -1
    assert node_labels[0] == node_labels[1] == node_labels[2] != -1


def test_bcc_edge_labels_of_the_empty_graph_and_without_edges():
    """No edges means nothing to label, and a directed graph has no biconnected components."""
    # --- Input ---
    directed = Graph([0, 1], [(0, 1)], directed=True)

    # --- Assert ---
    assert Graph([], []).bcc_edge_labels().tolist() == []
    assert Graph([0, 1], []).bcc_edge_labels().tolist() == []
    with pytest.raises(TypeError, match="directed"):
        directed.bcc_edge_labels()


def test_bcc_edge_labels_agree_with_biconnected_components_on_random_multigraphs():
    """The edge labels come from the same Tarjan pass as biconnected_components,
    so the endpoint sets of the edges of every label must be exactly the node
    sets that biconnected_components yields."""
    rng = random.Random(SEED)
    for _ in range(PROPERTY_TRIALS):
        # --- Input ---
        nodes, edges = random_graph(rng, rng.randrange(1, 12), rng.randrange(0, 20))
        view = random_view(rng, nodes, edges)

        # --- Execute ---
        node_sets: dict[int, set[int]] = {}
        for edge_index, label in enumerate(view.bcc_edge_labels()):
            if label != -1:
                node_sets.setdefault(label, set()).update(edges[edge_index])

        # --- Assert ---
        assert sorted(map(sorted, node_sets.values())) == sorted(map(sorted, view.biconnected_components()))
