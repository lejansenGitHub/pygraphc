"""The Python-orchestrated reduction engines against the monolithic C loop.

``"moves"`` drives the fixpoint from Python over one C primitive per move and
applies the moves in the very order the monolith does, so it must reproduce
every part of ``Reduced``, provenance tree shape included. ``"rounds"`` applies
a whole batch of independent moves per call, which changes the order the merges
happen in, so only the canonical properties can be compared: the residual node
set and endpoint multiset, the leaves of every residual edge, the folded
payload of every surviving node, the material that left the graph, and
``paths``, ``closed`` and ``minimal_toggles`` over the residual trees.

The one canonical property that genuinely differs is the ``dropped`` list, and
``test_the_batched_engine_splits_a_series_then_pendant...`` pins the minimal
case rather than leaving it to the differential test.
"""

import random
from typing import Literal

import pytest

import pygraphc
from pygraphc import ReductionState
from pygraphc.reduction import (
    EdgeId,
    MultiGraph,
    Reduced,
    VirtualEdgeId,
    closed,
    leaves,
    minimal_toggles,
    paths,
    reduce,
    tree_records,
)

ENGINES: tuple[Literal["moves", "python", "rounds"], ...] = ("python", "moves", "rounds")


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


def _edge_key(edge_id: object) -> tuple[int, int]:
    """Order over input and generated edge ids that never compares an int with a dataclass."""
    return (1, edge_id.index) if isinstance(edge_id, VirtualEdgeId) else (0, int(str(edge_id)))


def strict(reduced: Reduced[EdgeId]) -> object:
    """Everything ``Reduced`` carries, generated edge ids and tree shapes included."""
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


def _fold_samples(tree: object, edge_set: list[int], label: object) -> list[object]:
    """``closed`` and both directions of ``minimal_toggles`` under four leaf states drawn from ``label``."""
    samples: list[object] = []
    for trial in range(4):
        draw = random.Random(f"{label}|{trial}")
        states = {leaf: draw.random() < 0.5 for leaf in edge_set}
        samples.append((
            tuple(states[leaf] for leaf in edge_set),
            closed(tree, states),  # type: ignore[arg-type]
            frozenset(minimal_toggles(tree, states, target_closed=True)),  # type: ignore[arg-type]
            frozenset(minimal_toggles(tree, states, target_closed=False)),  # type: ignore[arg-type]
        ))
    return samples


def canonical(reduced: Reduced[int]) -> object:
    """The properties a change of merge order must leave alone.

    Every tree is addressed by its sorted endpoint pair and leaf set rather
    than by its generated edge id, because the engines number virtual edges in
    their own creation order and may orient one either way. ``dropped`` appears
    only as the union of its leaf sets, the material that left the graph: which
    pendant move carried which part of it away does depend on the order, which
    ``test_the_batched_engine_splits_a_series_then_pendant...`` pins.
    """
    endpoints = reduced.graph.endpoints
    records = []
    for edge_id, tree in reduced.provenance.items():
        edge_set = sorted(leaves(tree))
        label = (tuple(sorted(endpoints[edge_id])), tuple(edge_set))
        cutoffs = tuple((cutoff, frozenset(paths(tree, cutoff))) for cutoff in (None, 1, 2, 3, 5))
        records.append((label, cutoffs, tuple(_fold_samples(tree, edge_set, label))))
    dropped_leaves: frozenset[int] = frozenset()
    for _neighbour, tree in reduced.dropped:
        dropped_leaves |= leaves(tree)
    return (
        sorted(reduced.graph.nodes),
        sorted(tuple(sorted(pair)) for pair in endpoints.values()),
        frozenset(records),
        {node_id: sorted(material) for node_id, material in reduced.folded_nodes.items()},
        {node_id: sorted(material) for node_id, material in reduced.folded_interior.items()},
        dropped_leaves,
    )


@pytest.mark.parametrize("seed", range(500))
def test_every_engine_agrees_on_the_canonical_properties(seed: int) -> None:
    """The monolith is the reference; a difference is a defect of a primitive, of the batch
    independence rule, or of the Python loop that drives them."""
    graph, terminals, protected, fold_leaves = random_multigraph(seed)
    reference = canonical(reduce(graph, terminals, protected, fold_leaves=fold_leaves, engine="c"))
    for engine in ENGINES:
        reduced = reduce(graph, terminals, protected, fold_leaves=fold_leaves, engine=engine)
        assert canonical(reduced) == reference, engine


@pytest.mark.parametrize("seed", range(500))
def test_the_per_move_engine_reproduces_the_monolith_exactly(seed: int) -> None:
    """It asks for the same move the monolith would make next and applies it, so the operation log
    is the same log and the provenance trees must be identical in shape, not only in leaves."""
    graph, terminals, protected, fold_leaves = random_multigraph(seed)
    from_c = reduce(graph, terminals, protected, fold_leaves=fold_leaves, engine="c")
    from_moves = reduce(graph, terminals, protected, fold_leaves=fold_leaves, engine="moves")
    assert strict(from_moves) == strict(from_c)


def test_the_batched_engine_splits_a_series_then_pendant_into_two_pendants() -> None:
    """The minimal divergence, on the path 0-1-2-3 with the terminal at node 1.

    The monolith reaches node 2 while it still has degree two and merges it
    into an edge from 1 to 3, which the pendant move at 3 then drops whole.
    The batched engine deletes the pendants 0 and 3 first, which leaves node 2
    a pendant of its own. The material that left node 1 is the same either
    way, and so is every residual tree, because the residual is a bare
    terminal; only the split of ``dropped`` differs.
    """
    graph = MultiGraph([0, 1, 2, 3], {0: (0, 1), 1: (1, 2), 2: (2, 3)})

    from_c = reduce(graph, terminals={1}, engine="c")
    from_rounds = reduce(graph, terminals={1}, engine="rounds")

    # --- Assert ---
    assert [(node, sorted(leaves(tree))) for node, tree in from_c.dropped] == [(1, [0]), (1, [1, 2])]
    assert [(node, sorted(leaves(tree))) for node, tree in from_rounds.dropped] == [
        (1, [0]),
        (2, [2]),
        (1, [1]),
    ]
    assert {node: sorted(material) for node, material in from_c.folded_nodes.items()} == {1: [0, 2, 3]}
    assert {node: sorted(material) for node, material in from_rounds.folded_nodes.items()} == {1: [0, 2, 3]}
    assert canonical(from_c) == canonical(from_rounds)


def test_the_series_batch_admits_no_two_adjacent_nodes() -> None:
    """Two adjacent degree-two nodes conflict, so a path is thinned by alternate nodes per round
    and the round count is logarithmic in its length rather than linear."""
    node_count = 130
    graph = MultiGraph(list(range(node_count)), {index: (index, index + 1) for index in range(node_count - 1)})
    kernel = graph._kernel
    terminal_mask = bytearray(node_count)
    terminal_mask[0] = terminal_mask[node_count - 1] = 1

    rounds = 0
    with kernel.graph.series_parallel_state(terminal_mask, bytes(node_count)) as state:
        first = state.batch_moves(ReductionState.SERIES)
        assert first is not None
        admitted = memoryview(first).cast("i")[::5].tolist()
        progressed = True
        while progressed:
            progressed = False
            rounds += 1
            for kind in (ReductionState.PENDANT, ReductionState.SERIES, ReductionState.PARALLEL):
                batch = state.batch_moves(kind)
                if batch is not None:
                    state.apply_batch(kind, batch)
                    progressed = True

    # --- Assert ---
    assert admitted == list(range(1, node_count - 1, 2))
    assert rounds <= 10, f"{node_count} nodes took {rounds} rounds"
    assert reduce(graph, terminals={0, node_count - 1}, engine="rounds").graph.nodes == [0, node_count - 1]


def test_the_handle_carries_the_graph_it_was_built_from() -> None:
    """Every primitive is a method of the state, so a state can never be paired with another graph."""
    graph = pygraphc.Graph([0, 1, 2], [(0, 1), (1, 2)])
    with graph.series_parallel_state(bytes([1, 0, 1]), bytes(3)) as state:
        assert state.graph is graph


def test_a_freed_handle_refuses_every_primitive() -> None:
    """The capsule outlives the arrays it pointed at, so the handle goes inert instead of dangling."""
    graph = pygraphc.Graph([0, 1, 2], [(0, 1), (1, 2)])
    state = graph.series_parallel_state(bytes([1, 0, 1]), bytes(3))
    state.free()
    for call in (state.next_move, state.log, state.free):
        with pytest.raises(ValueError, match="already freed"):
            call()
    with pytest.raises(ValueError, match="already freed"):
        state.batch_moves(ReductionState.SERIES)


def test_an_unknown_move_kind_is_refused() -> None:
    """A kind outside the three moves is a caller defect, not an empty batch."""
    graph = pygraphc.Graph([0, 1, 2], [(0, 1), (1, 2)])
    with graph.series_parallel_state(bytes([1, 0, 1]), bytes(3)) as state:
        with pytest.raises(ValueError, match="pendant, series or parallel"):
            state.batch_moves(7)
        with pytest.raises(ValueError, match="pendant, series or parallel"):
            state.apply_batch(7, bytes(20))
        with pytest.raises(ValueError, match="pendant or a series"):
            state.apply_move((ReductionState.PARALLEL, 1, 0, 1, 0, 2))


def test_a_move_naming_an_index_the_state_does_not_have_is_refused() -> None:
    """The primitives write into raw arrays, so an index a caller made up must raise rather than
    corrupt the incidence structure. An edge must run between exactly the two nodes named."""
    graph = pygraphc.Graph([0, 1, 2], [(0, 1), (1, 2)])
    wrong_moves = [
        (ReductionState.SERIES, 900_000, 0, 1, 0, 2),  # no such node
        (ReductionState.SERIES, 1, 77, 1, 0, 2),  # no such edge
        (ReductionState.SERIES, 1, 0, 0, 0, 2),  # edge 0 does not run from 1 to 2
        (ReductionState.PENDANT, 1, 0, -1, 2, -1),  # edge 0 does not run from 1 to 2
    ]
    # --- Assert ---
    with graph.series_parallel_state(bytes([1, 0, 1]), bytes(3)) as state:
        for move in wrong_moves:
            with pytest.raises(ValueError, match="live edge between two live nodes"):
                state.apply_move(move)
        with pytest.raises(ValueError, match="two endpoints, a count and that many edges"):
            state.apply_batch(ReductionState.PARALLEL, bytes(12))
        with pytest.raises(ValueError, match="five int32 per move"):
            state.apply_batch(ReductionState.SERIES, bytes(13))
        assert state.apply_move(state.next_move()) == (2, 0)


def test_a_fresh_state_has_applied_no_move_but_dropped_the_terminal_free_components() -> None:
    """Construction is not a move: it links the incidences, emits one leaf per edge and removes the
    components that carry no terminal, which must happen before any move can apply."""
    graph = pygraphc.Graph([0, 1, 2, 3], [(0, 1), (2, 3)])
    with graph.series_parallel_state(bytes([1, 0, 0, 0]), bytes(4)) as state:
        log = state.log()
        assert log.op_kind.tolist() == [0, 0]
        assert log.surviving_nodes.tolist() == [0, 1]
        assert (log.residual_u.tolist(), log.residual_v.tolist()) == ([0], [1])


def test_a_parallel_merge_needs_at_least_two_edges() -> None:
    """``pair_edges`` reports None below two, so passing one through is a caller defect."""
    graph = pygraphc.Graph([0, 1, 2], [(0, 1), (0, 1), (1, 2)])
    with graph.series_parallel_state(bytes([1, 0, 1]), bytes(3)) as state:
        assert state.pair_edges(1, 2) is None
        members = state.pair_edges(0, 1)
        assert members is not None
        assert len(members) == 8
        with pytest.raises(ValueError, match="at least two edges"):
            state.apply_parallel(0, 1, members[:4])


def test_the_state_refuses_a_directed_graph() -> None:
    """The moves are defined on incidences, which a directed graph does not have."""
    graph = pygraphc.Graph([0, 1], [(0, 1)], directed=True)
    with pytest.raises(TypeError, match="series_parallel_state is not defined for directed graphs"):
        graph.series_parallel_state(bytes(2), bytes(2))
