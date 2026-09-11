"""The per-node pendant policy and the series-ineligible set, in both engines.

``PendantPolicy`` splits the old ``fold_leaves`` boolean into a per-node
decision with three values and ``series_ineligible`` splits the series half out
of ``protected``. Both are per-node questions the C loop answers from one byte
mask each, so the two engines must stay differentially equal under random
policies, which is what the sweep below checks.

The targeted tests pin the four notions apart: a terminal survives every move,
a protected node is still deleted as a pendant, a ``keep`` node is still merged
away in series, and a series-ineligible node still has its parallel edges
merged. The last two tests are about confluence: a ``keep`` action makes the
residual order dependent, a series-ineligible node does not.
"""

import random
from typing import Literal

import pytest
from test_reduction_c_engine import canonical, random_multigraph

from pygraphc.reduction import (
    MultiGraph,
    Parallel,
    PendantAction,
    PendantPolicy,
    Reduced,
    leaves,
    reduce,
)

ENGINES: tuple[Literal["c", "python"], ...] = ("c", "python")


def random_policies(seed: int, nodes: list[int]) -> tuple[PendantPolicy, frozenset[int]]:
    """A pendant policy and a series-ineligible set for the graph of ``random_multigraph(seed)``.

    Drawn from a stream of its own so that the graphs stay the ones the
    engine sweep in ``test_reduction_c_engine`` already covers.
    """
    rng = random.Random(10_000 + seed)
    actions: tuple[PendantAction, ...] = ("absorb", "discard", "keep")
    default = rng.choice(("absorb", "absorb", "discard", "keep"))
    exceptions = {node_id: rng.choice(actions) for node_id in rng.sample(nodes, rng.randint(0, min(6, len(nodes))))}
    series_ineligible = frozenset(rng.sample(nodes, rng.randint(0, min(3, len(nodes)))))
    return PendantPolicy(default, exceptions), series_ineligible


def reduced_with(graph: MultiGraph[int], terminals: set[int], **arguments: object) -> Reduced[int]:
    """One reduction per engine, asserted equal, so every targeted test covers both."""
    from_c = reduce(graph, terminals, engine="c", **arguments)
    from_python = reduce(graph, terminals, engine="python", **arguments)
    assert canonical(from_c) == canonical(from_python)
    return from_c


# ── Both engines under random policies ──


@pytest.mark.parametrize("seed", range(400))
def test_the_two_engines_agree_under_random_policies(seed: int) -> None:
    """The Python worklist is the reference semantics, so any difference is a defect of the C masks or the fold."""
    graph, terminals, protected, _fold_leaves = random_multigraph(seed)
    pendant, series_ineligible = random_policies(seed, graph.nodes)
    from_c = reduce(graph, terminals, protected, pendant=pendant, series_ineligible=series_ineligible, engine="c")
    from_python = reduce(
        graph, terminals, protected, pendant=pendant, series_ineligible=series_ineligible, engine="python"
    )
    assert canonical(from_c) == canonical(from_python)


def test_the_two_engines_agree_on_every_uniform_policy() -> None:
    """Only the default action differs, so a difference cannot come from the graph or from the exceptions."""
    actions: tuple[PendantAction, ...] = ("absorb", "discard", "keep")
    for seed in range(40):
        graph, terminals, protected, _fold_leaves = random_multigraph(seed)
        for action in actions:
            from_c = reduce(graph, terminals, protected, pendant=PendantPolicy(action), engine="c")
            from_python = reduce(graph, terminals, protected, pendant=PendantPolicy(action), engine="python")
            assert canonical(from_c) == canonical(from_python)


# ── The three pendant actions ──


def test_absorb_hands_the_pendant_material_to_the_neighbour() -> None:
    """The default action, and what ``fold_leaves=True`` meant."""
    graph = MultiGraph([0, 1], {101: (0, 1)})
    reduced = reduced_with(graph, {0}, pendant=PendantPolicy("absorb"))
    assert reduced.graph.nodes == [0]
    assert reduced.folded_nodes == {0: [1]}
    assert [(neighbour, leaves(tree)) for neighbour, tree in reduced.dropped] == [(0, frozenset({101}))]


def test_discard_drops_the_material_and_still_reports_the_dropped_tree() -> None:
    """The material is gone, the provenance of the removed edge is not: ``dropped`` does not depend on the action."""
    graph = MultiGraph([0, 1], {101: (0, 1)})
    reduced = reduced_with(graph, {0}, pendant=PendantPolicy("discard"))
    assert reduced.graph.nodes == [0]
    assert reduced.folded_nodes == {0: []}
    assert [(neighbour, leaves(tree)) for neighbour, tree in reduced.dropped] == [(0, frozenset({101}))]


def test_keep_leaves_the_pendant_node_and_its_edge_in_the_residual() -> None:
    """No pendant move applies at the node, so nothing is dropped either."""
    graph = MultiGraph([0, 1], {101: (0, 1)})
    reduced = reduced_with(graph, {0}, pendant=PendantPolicy("absorb", {1: "keep"}))
    assert reduced.graph.nodes == [0, 1]
    assert reduced.graph.endpoints == {101: (0, 1)}
    assert reduced.folded_nodes == {0: [], 1: []}
    assert reduced.dropped == []


def test_a_pendant_chain_whose_middle_node_keeps_and_whose_outer_node_absorbs() -> None:
    """The outer pendant folds into the middle one, which then keeps and carries the material into the residual.

    Node ids put the outer node before the middle one in the processing
    order, so the middle node is a degree-1 node by the time it is reached
    and its ``keep`` action is what saves it; reached at degree 2 it would
    have been merged away in series, see the order-dependence test below.
    """
    graph = MultiGraph([0, 1, 2], {101: (0, 2), 102: (2, 1)})
    reduced = reduced_with(graph, {0}, pendant=PendantPolicy("absorb", {2: "keep"}))
    assert reduced.graph.nodes == [0, 2]
    assert reduced.graph.endpoints == {101: (0, 2)}
    assert reduced.folded_nodes == {0: [], 2: [1]}
    assert [(neighbour, leaves(tree)) for neighbour, tree in reduced.dropped] == [(2, frozenset({102}))]


def test_a_discarding_node_drops_the_material_absorbed_into_it_earlier() -> None:
    """The action of a pendant move is the action of the node it removes, so absorbing into a discarder loses it."""
    graph = MultiGraph([0, 1, 2], {101: (0, 2), 102: (2, 1)})
    policy = PendantPolicy("absorb", {2: "discard"})
    reduced = reduced_with(graph, {0}, pendant=policy)
    assert reduced.graph.nodes == [0]
    assert reduced.folded_nodes == {0: []}


def test_a_keep_node_is_not_a_terminal_and_is_still_merged_away_in_series() -> None:
    """``keep`` answers the pendant question only; a node that must survive every move is a terminal."""
    graph = MultiGraph([0, 1, 2], {101: (0, 1), 102: (1, 2)})
    reduced = reduced_with(graph, {0, 2}, pendant=PendantPolicy("absorb", {1: "keep"}))
    assert reduced.graph.nodes == [0, 2]
    assert reduced.folded_interior == {1: []}


def test_a_keep_policy_does_not_save_a_terminal_free_component() -> None:
    """Components without a terminal go whole before any move, so no pendant action is ever consulted in them."""
    graph = MultiGraph([0, 1, 2, 3], {101: (0, 1), 102: (2, 3)})
    reduced = reduced_with(graph, {0}, pendant=PendantPolicy("keep"))
    assert reduced.graph.nodes == [0, 1]
    assert reduced.folded_nodes == {0: [], 1: []}


# ── Series eligibility against protection ──


def test_a_series_ineligible_node_keeps_its_two_edges_apart() -> None:
    """The series move is the only one blocked, so the node stays with exactly its two incidences."""
    graph = MultiGraph([0, 1, 2], {101: (0, 1), 102: (1, 2)})
    reduced = reduced_with(graph, {0, 2}, series_ineligible=frozenset({1}))
    assert reduced.graph.nodes == [0, 1, 2]
    assert reduced.graph.endpoints == {101: (0, 1), 102: (1, 2)}


def test_a_series_ineligible_node_still_has_its_parallel_edges_merged() -> None:
    """This is the whole difference to ``protected``: the parallel pair at the node merges, the series move does not."""
    graph = MultiGraph([0, 1, 2], {101: (0, 1), 102: (0, 1), 103: (1, 2)})
    reduced = reduced_with(graph, {0, 2}, series_ineligible=frozenset({1}))
    assert reduced.graph.nodes == [0, 1, 2]
    merged = [tree for tree in reduced.provenance.values() if isinstance(tree, Parallel)]
    assert [leaves(tree) for tree in merged] == [frozenset({101, 102})]
    assert sorted(reduced.graph.endpoints.values()) == [(0, 1), (1, 2)]


def test_protecting_the_same_node_leaves_the_parallel_pair_alone() -> None:
    """Same graph, the coarser flag: ``protected`` blocks the parallel merge as well, so all three edges survive."""
    graph = MultiGraph([0, 1, 2], {101: (0, 1), 102: (0, 1), 103: (1, 2)})
    reduced = reduced_with(graph, {0, 2}, protected=frozenset({1}))
    assert sorted(reduced.graph.endpoints) == [101, 102, 103]


def test_protection_and_series_ineligibility_compose_by_union_on_the_series_question() -> None:
    """Both block the series move, so marking one node twice adds nothing and marking a second one adds that node."""
    graph = MultiGraph([0, 1, 2, 3], {101: (0, 1), 102: (1, 2), 103: (2, 3)})
    protected_only = reduced_with(graph, {0, 3}, protected=frozenset({1}))
    assert protected_only.graph.nodes == [0, 1, 3]

    same_node_twice = reduced_with(graph, {0, 3}, protected=frozenset({1}), series_ineligible=frozenset({1}))
    assert canonical(same_node_twice) == canonical(protected_only)

    both_nodes = reduced_with(graph, {0, 3}, protected=frozenset({1}), series_ineligible=frozenset({2}))
    assert both_nodes.graph.nodes == [0, 1, 2, 3]


def test_a_protected_node_is_still_deleted_as_a_pendant() -> None:
    """``protected`` answers the merge questions only; a node that must stay is a terminal or a ``keep`` node."""
    graph = MultiGraph([0, 1], {101: (0, 1)})
    reduced = reduced_with(graph, {0}, protected=frozenset({1}))
    assert reduced.graph.nodes == [0]
    assert reduced.folded_nodes == {0: [1]}


# ── The deprecated boolean ──


@pytest.mark.parametrize("engine", ENGINES)
def test_the_deprecated_boolean_maps_to_a_uniform_policy(engine: Literal["c", "python"]) -> None:
    """``True`` is a uniform ``absorb``, ``False`` a uniform ``discard``, on every graph of the sweep."""
    for seed in range(60):
        graph, terminals, protected, _fold_leaves = random_multigraph(seed)
        for fold_leaves, action in ((True, "absorb"), (False, "discard")):
            from_boolean = reduce(graph, terminals, protected, fold_leaves=fold_leaves, engine=engine)
            from_policy = reduce(graph, terminals, protected, pendant=PendantPolicy(action), engine=engine)
            assert canonical(from_boolean) == canonical(from_policy)


@pytest.mark.parametrize("engine", ENGINES)
def test_the_default_is_still_the_old_default(engine: Literal["c", "python"]) -> None:
    """No pendant argument at all keeps the material, as ``fold_leaves=True`` did."""
    graph, terminals, protected, _fold_leaves = random_multigraph(3)
    plain = reduce(graph, terminals, protected, engine=engine)
    explicit = reduce(graph, terminals, protected, fold_leaves=True, engine=engine)
    assert canonical(plain) == canonical(explicit)


def test_the_policy_and_the_deprecated_boolean_cannot_both_be_given() -> None:
    """The two say the same thing in two ways, so a caller giving both has no answer to which one wins."""
    graph = MultiGraph([0, 1], {101: (0, 1)})
    with pytest.raises(ValueError, match="not both"):
        reduce(graph, {0}, pendant=PendantPolicy("discard"), fold_leaves=True)


# ── Rejected input ──


def test_an_unknown_pendant_action_is_rejected() -> None:
    """The three actions are the whole vocabulary; a fourth word would be silently ignored at move time."""
    with pytest.raises(ValueError, match="pendant action must be one of"):
        PendantPolicy("fold")
    with pytest.raises(ValueError, match="pendant action must be one of"):
        PendantPolicy("absorb", {1: "hold"})


def test_policy_exceptions_and_series_ineligible_nodes_must_be_nodes_of_the_graph() -> None:
    """The C masks are indexed by node position, so an unknown id would land on the wrong node instead of nowhere."""
    graph = MultiGraph([0, 1], {101: (0, 1)})
    with pytest.raises(ValueError, match=r"unknown: \[9\]"):
        reduce(graph, {0}, pendant=PendantPolicy("absorb", {9: "keep"}))
    with pytest.raises(ValueError, match=r"unknown: \[9\]"):
        reduce(graph, {0}, series_ineligible=frozenset({9}))


def test_a_policy_exception_for_a_node_that_never_becomes_a_pendant_is_harmless() -> None:
    """The action is read only by the pendant move, so naming a node that is merged in series changes nothing."""
    graph = MultiGraph([0, 1, 2], {101: (0, 1), 102: (1, 2)})
    marked = reduced_with(graph, {0, 2}, pendant=PendantPolicy("absorb", {1: "keep"}))
    plain = reduced_with(graph, {0, 2}, pendant=PendantPolicy("absorb"))
    assert canonical(marked) == canonical(plain)


# ── Confluence ──


def test_a_keep_action_makes_the_residual_order_dependent() -> None:
    """A triangle with one terminal in which the ``keep`` node's fate is decided by which move reaches it first.

    In id order node 1 is merged away in series first, which leaves node 2 a
    pendant whose ``keep`` action saves it. Processing node 2 first merges it
    away in series instead, because ``keep`` blocks no series move, and the
    residual is the terminal alone. The module answers in id order, so the
    kept node survives; both engines agree because the C work queue is the
    same increasing node index.
    """
    # --- Input ---
    graph = MultiGraph([0, 1, 2], {101: (0, 1), 102: (1, 2), 103: (2, 0)})
    policy = PendantPolicy("absorb", {2: "keep"})

    # --- Assert ---
    in_id_order = reduced_with(graph, {0}, pendant=policy)
    assert in_id_order.graph.nodes == [0, 2]
    assert [leaves(tree) for tree in in_id_order.provenance.values()] == [frozenset({101, 102, 103})]
    assert in_id_order.folded_nodes == {0: [], 2: []}

    keep_node_first = reduce(graph, {0}, pendant=policy, order=[2, 1], engine="python")
    assert keep_node_first.graph.nodes == [0]
    assert keep_node_first.folded_nodes == {0: [1, 2]}


def test_a_series_ineligible_node_alone_leaves_the_residual_order_independent() -> None:
    """Blocking only the series move removes a move without creating a competing pair, so confluence survives.

    A node is a pendant candidate at degree one and a series candidate at
    degree two, never both, and degrees only fall. Blocking the series move
    leaves a degree-two node inert, so nothing about it depends on when it is
    reached; a ``keep`` action instead blocks the move that fires at the lower
    degree while leaving the one that removes the node, which is what the test
    above exploits. ``protected`` breaks confluence through its parallel block,
    which can leave one of two competing nodes inert; series ineligibility
    alone cannot. Random orders over random graphs found no counterexample.
    """
    # --- Assert ---
    rng = random.Random(1234)
    for seed in range(120):
        graph, terminals, _protected, _fold_leaves = random_multigraph(seed)
        series_ineligible = frozenset(rng.sample(graph.nodes, rng.randint(1, min(4, len(graph.nodes)))))
        answers = set()
        for _attempt in range(8):
            order = sorted(graph.nodes) if not answers else rng.sample(graph.nodes, len(graph.nodes))
            reduced = reduce(graph, terminals, series_ineligible=series_ineligible, order=order, engine="python")
            answers.add(order_independent_part(reduced))
        assert len(answers) == 1


def order_independent_part(reduced: Reduced[int]) -> str:
    """The residual graph and the folded material, the part the confluence claim is about.

    Not the shape of the provenance trees, the generated edge ids or the order
    of ``dropped`` and of the folded lists: a chain of series merges nests its
    tree in the order it was applied, and that is order dependent with no
    protected node and no policy at all.
    """
    return repr((
        sorted(reduced.graph.nodes),
        sorted(tuple(sorted(pair)) for pair in reduced.graph.endpoints.values()),
        sorted(repr(sorted(map(repr, leaves(tree)))) for tree in reduced.provenance.values()),
        sorted((node_id, sorted(material)) for node_id, material in reduced.folded_nodes.items()),
        sorted((node_id, sorted(material)) for node_id, material in reduced.folded_interior.items()),
    ))
