"""The C reduction engine against the Python worklist, move for move.

The C engine returns a flat operation log and the Python tier folds it into
the very trees the worklist builds, so the two engines must agree on every
part of ``Reduced``: the residual multigraph, the leaf set and the canonical
record log of every provenance tree, the folded node material and the dropped
pendant trees. The differential test compares all of that on seeded random
multigraphs with self-loops, parallel edges, terminals and protected nodes.
"""

import random
from typing import Literal

import pytest

import pygraphc
from pygraphc.reduction import EdgeId, MultiGraph, Reduced, VirtualEdgeId, leaves, reduce, tree_records


def random_multigraph(seed: int) -> tuple[MultiGraph[int], set[int], set[int], bool]:
    """A multigraph with self-loops and parallel edges plus its terminals, protected nodes and fold flag."""
    rng = random.Random(seed)
    node_count = rng.randint(4, 40)
    nodes = list(range(node_count))
    edge_count = rng.randint(0, 3 * node_count)
    endpoints: dict[int, tuple[int, int]] = {}
    for edge_id in range(edge_count):
        from_node = rng.randrange(node_count)
        if rng.random() < 0.1:
            endpoints[edge_id] = (from_node, from_node)
        elif rng.random() < 0.2 and endpoints:
            endpoints[edge_id] = endpoints[rng.choice(list(endpoints))]
        else:
            endpoints[edge_id] = (from_node, rng.randrange(node_count))
    terminals = set(rng.sample(nodes, rng.randint(0, 4)))
    protected = set(rng.sample(nodes, rng.randint(0, 2)))
    return MultiGraph(nodes, endpoints), terminals, protected, rng.random() < 0.5


def canonical(reduced: Reduced[EdgeId]) -> object:
    """Everything ``Reduced`` carries, in a form two runs can be compared by equality.

    Trees compare by their canonical record log, in which a virtual edge id
    appears only as the key of its own residual edge, so the comparison pins
    the generated ids as well as the structure.
    """
    return (
        sorted(reduced.graph.nodes),
        sorted((_edge_key(edge_id), pair) for edge_id, pair in reduced.graph.endpoints.items()),
        sorted(
            (_edge_key(edge_id), sorted(leaves(tree)), tree_records(tree))
            for edge_id, tree in reduced.provenance.items()
        ),
        {node_id: sorted(material) for node_id, material in reduced.folded_nodes.items()},
        {node_id: sorted(material) for node_id, material in reduced.folded_interior.items()},
        [(neighbour, tree_records(tree)) for neighbour, tree in reduced.dropped],
    )


def _edge_key(edge_id: object) -> tuple[int, int]:
    """Order over input and generated edge ids that never compares an int with a dataclass."""
    return (1, edge_id.index) if isinstance(edge_id, VirtualEdgeId) else (0, int(str(edge_id)))


@pytest.mark.parametrize("seed", range(500))
def test_the_two_engines_agree_on_random_multigraphs(seed: int) -> None:
    """The Python worklist is the reference semantics, so any difference is a defect of the C loop or the fold."""
    graph, terminals, protected, fold_leaves = random_multigraph(seed)
    from_c = reduce(graph, terminals, protected, fold_leaves=fold_leaves, engine="c")
    from_python = reduce(graph, terminals, protected, fold_leaves=fold_leaves, engine="python")
    assert canonical(from_c) == canonical(from_python)


def test_the_engines_agree_with_both_fold_settings_on_the_same_graph() -> None:
    """Only the fold flag differs, so a difference cannot come from the graph."""
    for seed in range(50):
        graph, terminals, protected, _ = random_multigraph(seed)
        for fold_leaves in (True, False):
            from_c = reduce(graph, terminals, protected, fold_leaves=fold_leaves, engine="c")
            from_python = reduce(graph, terminals, protected, fold_leaves=fold_leaves, engine="python")
            assert canonical(from_c) == canonical(from_python), (seed, fold_leaves)


def test_the_empty_graph_reduces_to_itself() -> None:
    """No node carries a terminal and none can move, so the only correct residual is the empty graph."""
    graph: MultiGraph[int] = MultiGraph([], {})
    reduced = reduce(graph, terminals=set())
    assert reduced.graph.nodes == []
    assert reduced.graph.endpoints == {}
    assert canonical(reduced) == canonical(reduce(graph, terminals=set(), engine="python"))


def test_a_single_node_survives_only_as_a_terminal() -> None:
    """Its component holds no terminal without it, and a component without a terminal goes whole, which
    also drops its entry from the folded node material because no node absorbs it."""
    graph: MultiGraph[int] = MultiGraph([7], {})
    assert reduce(graph, terminals={7}).graph.nodes == [7]
    assert reduce(graph, terminals=set()).graph.nodes == []
    assert reduce(graph, terminals=set()).folded_nodes == {}


def test_all_nodes_terminal_leaves_the_graph_untouched() -> None:
    """Every move needs a non-terminal to eliminate, so a six-cycle of terminals is already a fixpoint."""
    nodes = list(range(6))
    endpoints = {index: (index, (index + 1) % 6) for index in range(6)}
    graph = MultiGraph(nodes, endpoints)
    reduced = reduce(graph, terminals=set(nodes))
    assert reduced.graph.nodes == nodes
    assert reduced.graph.endpoints == endpoints
    assert canonical(reduced) == canonical(reduce(graph, terminals=set(nodes), engine="python"))


def test_no_terminal_at_all_removes_every_component() -> None:
    """Terminal-free components go before any move, so nothing is left to absorb their nodes or to report."""
    graph = MultiGraph([0, 1, 2, 3], {0: (0, 1), 1: (2, 3), 2: (3, 3)})
    reduced = reduce(graph, terminals=set())
    assert reduced.graph.nodes == []
    assert reduced.graph.endpoints == {}
    assert reduced.folded_nodes == {}
    assert reduced.dropped == []
    assert canonical(reduced) == canonical(reduce(graph, terminals=set(), engine="python"))


def test_a_chain_of_3000_nodes_becomes_one_edge() -> None:
    """The C loop never recurses and the fold carries interior nodes without walking the tree."""
    node_count = 3_000
    nodes = list(range(node_count))
    endpoints = {index: (index, index + 1) for index in range(node_count - 1)}
    graph = MultiGraph(nodes, endpoints)
    reduced = reduce(graph, terminals={0, node_count - 1})

    assert reduced.graph.nodes == [0, node_count - 1]
    ((edge_id, tree),) = reduced.provenance.items()
    assert len(leaves(tree)) == node_count - 1
    assert reduced.graph.endpoints[edge_id] in {(0, node_count - 1), (node_count - 1, 0)}
    assert sorted(reduced.folded_interior) == list(range(1, node_count - 1))


def test_two_hubs_joined_by_many_two_paths_collapse_to_one_parallel_tree() -> None:
    """Each two-path is one series move and the pair it lands on is merged at once, so one edge carrying
    every input edge as a leaf is the only possible residual; the pair lookup must not become quadratic."""
    path_count = 2_000
    nodes = list(range(path_count + 2))
    endpoints: dict[int, tuple[int, int]] = {}
    for middle in range(2, path_count + 2):
        endpoints[2 * middle] = (0, middle)
        endpoints[2 * middle + 1] = (middle, 1)
    graph = MultiGraph(nodes, endpoints)

    reduced = reduce(graph, terminals={0, 1})

    # --- Assert ---
    ((_edge_id, tree),) = reduced.provenance.items()
    assert reduced.graph.nodes == [0, 1]
    assert len(leaves(tree)) == 2 * path_count
    assert canonical(reduced) == canonical(reduce(graph, terminals={0, 1}, engine="python"))


def test_a_graph_of_only_self_loops_keeps_its_terminals_and_their_loops() -> None:
    """A self-loop takes part in no move and leaves with its node, so a terminal keeps all of its loops
    unmerged and the non-terminal node goes with its own."""
    nodes = [0, 1, 2]
    endpoints = {0: (0, 0), 1: (0, 0), 2: (1, 1), 3: (2, 2)}
    graph = MultiGraph(nodes, endpoints)

    reduced = reduce(graph, terminals={0, 2})
    assert reduced.graph.nodes == [0, 2]
    assert reduced.graph.endpoints == {0: (0, 0), 1: (0, 0), 3: (2, 2)}
    assert reduced.dropped == []
    assert canonical(reduced) == canonical(reduce(graph, terminals={0, 2}, engine="python"))


def test_a_caller_supplied_order_selects_the_python_engine() -> None:
    """The C work queue is fixed to increasing node index, so ``order`` picks the worklist."""
    graph = MultiGraph([0, 1, 2, 3, 4], {0: (0, 1), 1: (1, 2), 2: (2, 3), 3: (3, 4)})
    ordered = reduce(graph, terminals={0, 4}, order=[3, 2, 1])
    assert canonical(ordered) == canonical(reduce(graph, terminals={0, 4}, order=[3, 2, 1], engine="python"))


def test_unknown_terminals_are_rejected_by_both_engines() -> None:
    """The check belongs to ``reduce``, above the engine choice, so neither engine can see an unknown node."""
    graph = MultiGraph([0, 1], {0: (0, 1)})
    engines: tuple[Literal["c", "python"], ...] = ("c", "python")
    for engine in engines:
        with pytest.raises(ValueError, match="unknown: \\[9\\]"):
            reduce(graph, terminals={9}, engine=engine)


def test_a_view_reduces_under_its_own_masks() -> None:
    """Two triangles joined by a bridge, one terminal in each. With the bridge the two terminals keep
    one edge between them; masking it leaves each triangle collapsing onto its own terminal, so the
    residual has no edge at all. Masking a triangle node takes that node out of every move."""
    graph = pygraphc.Graph([0, 1, 2, 3, 4, 5], [(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 5), (5, 3)])
    terminal_mask = bytes([1, 0, 0, 0, 1, 0])

    whole = graph.series_parallel_reduce(terminal_mask, bytes(6))
    without_the_bridge = graph.without_edges([3]).series_parallel_reduce(terminal_mask, bytes(6))
    without_a_triangle_node = graph.without_nodes([1]).series_parallel_reduce(terminal_mask, bytes(6))

    # --- Assert ---
    assert whole.surviving_nodes.tolist() == [0, 4]
    assert (whole.residual_u.tolist(), whole.residual_v.tolist()) == ([0], [4])
    assert without_the_bridge.surviving_nodes.tolist() == [0, 4]
    assert without_the_bridge.residual_op.tolist() == []
    assert without_a_triangle_node.surviving_nodes.tolist() == [0, 4]
    assert 1 not in without_a_triangle_node.interior_node.tolist()
    assert len(without_a_triangle_node.op_kind) < len(whole.op_kind)


def test_the_loop_refuses_a_directed_graph() -> None:
    """The moves are defined on incidences, which a directed graph does not have, so there is no
    meaning to fall back on and the call must say so instead of reducing the underlying graph."""
    graph = pygraphc.Graph([0, 1], [(0, 1)], directed=True)
    with pytest.raises(TypeError, match="series_parallel_reduce"):
        graph.series_parallel_reduce(bytes(2), bytes(2))


def test_the_loop_refuses_a_missing_terminal_mask() -> None:
    """With no terminal mask every component is terminal-free and the whole graph goes, which is
    what no caller means by leaving the argument out. An all-zero mask still says it deliberately."""
    graph = pygraphc.Graph([0, 1, 2], [(0, 1), (1, 2)])

    with pytest.raises(TypeError, match="requires a terminal mask"):
        graph.series_parallel_reduce(None, bytes(3))  # type: ignore[arg-type] — the runtime guard is the subject
    with pytest.raises(TypeError, match="requires a terminal mask"):
        graph.without_nodes([2]).series_parallel_reduce(None, bytes(3))  # type: ignore[arg-type] — same guard
    assert graph.series_parallel_reduce(bytes(3), bytes(3)).surviving_nodes.tolist() == []


def test_a_node_mask_is_a_buffer_and_not_a_list_of_node_ids() -> None:
    """Both masks are read as buffers of one byte per node index, so the list that the earlier
    ``Collection[int]`` annotation invited is a TypeError rather than a mask of two terminals."""
    graph = pygraphc.Graph([0, 1, 2], [(0, 1), (1, 2)])

    with pytest.raises(TypeError):
        graph.series_parallel_reduce([0, 2], bytes(3))  # type: ignore[arg-type] — the runtime guard is the subject
