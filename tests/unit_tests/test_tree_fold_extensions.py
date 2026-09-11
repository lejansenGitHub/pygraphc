"""Tests for the two provenance-tree extensions in ``pygraphc.reduction``.

``minimal_toggles(..., togglable_leaves=...)`` restricts the leaves the fold
may flip and answers ``None`` when the target state is out of reach, and
``series_chain`` exposes the ordered alternating walk of a series chain.
"""

import random

import pytest

from pygraphc.reduction import (
    Leaf,
    Parallel,
    Series,
    SeriesStep,
    SPTree,
    closed,
    leaves,
    minimal_toggles,
    series_chain,
)

SEED = 20260911
PROPERTY_TRIALS = 300
BRUTE_FORCE_LEAF_LIMIT = 10


def series_of_parallels() -> Series[str]:
    """``0 -(a || b)- 1 -(c || d)- 2``: two parallel pairs in series."""
    left = Parallel(frozenset({Leaf("a"), Leaf("b")}))
    right = Parallel(frozenset({Leaf("c"), Leaf("d")}))
    return Series((left, right), (1,))


ALL_OPEN = {"a": False, "b": False, "c": False, "d": False}
ALL_CLOSED = {"a": True, "b": True, "c": True, "d": True}
LEFT_OPEN = {"a": False, "b": False, "c": True, "d": True}
RIGHT_OPEN = {"a": True, "b": True, "c": False, "d": False}
ONE_OPEN_PER_PAIR = {"a": False, "b": True, "c": True, "d": False}


# --- Part one: fixed leaves in the minimal-toggle fold ---


@pytest.mark.parametrize(
    ("edge_closed", "target_closed", "togglable_leaves", "expected"),
    [
        # Already in the target state: the empty set, whatever may be toggled.
        (ALL_CLOSED, True, None, frozenset()),
        (ALL_CLOSED, True, frozenset(), frozenset()),
        (ALL_CLOSED, True, frozenset({"a"}), frozenset()),
        (ALL_OPEN, False, None, frozenset()),
        (ALL_OPEN, False, frozenset(), frozenset()),
        # Reachable: one leaf of the open parallel pair closes the chain.
        (LEFT_OPEN, True, None, frozenset({"a"})),
        (LEFT_OPEN, True, frozenset({"a", "b"}), frozenset({"a"})),
        (LEFT_OPEN, True, frozenset({"b"}), frozenset({"b"})),
        (LEFT_OPEN, True, frozenset({"b", "c", "d"}), frozenset({"b"})),
        # Unreachable through a fixed leaf: the open pair cannot be closed.
        (LEFT_OPEN, True, frozenset({"c", "d"}), None),
        (LEFT_OPEN, True, frozenset({"c"}), None),
        # All leaves fixed, and the empty permitted set, away from the target.
        (LEFT_OPEN, True, frozenset(), None),
        (ALL_CLOSED, False, frozenset(), None),
        (ALL_OPEN, True, frozenset(), None),
        # Opening a closed chain needs every leaf of one parallel pair.
        (ALL_CLOSED, False, None, frozenset({"a", "b"})),
        (ALL_CLOSED, False, frozenset({"a", "b", "c"}), frozenset({"a", "b"})),
        (ALL_CLOSED, False, frozenset({"c", "d"}), frozenset({"c", "d"})),
        (ALL_CLOSED, False, frozenset({"a", "c", "d"}), frozenset({"c", "d"})),
        (ALL_CLOSED, False, frozenset({"a", "c"}), None),
        # Closing an open chain needs one leaf of each parallel pair.
        (ALL_OPEN, True, None, frozenset({"a", "c"})),
        (ALL_OPEN, True, frozenset({"b", "d"}), frozenset({"b", "d"})),
        (ALL_OPEN, True, frozenset({"a", "b", "c", "d"}), frozenset({"a", "c"})),
        (ALL_OPEN, True, frozenset({"a", "b"}), None),
    ],
)
def test_minimal_toggles_respects_the_togglable_leaves(edge_closed, target_closed, togglable_leaves, expected):
    """Table over the series-of-parallels tree, in both target directions."""
    tree = series_of_parallels()
    toggles = minimal_toggles(tree, edge_closed, target_closed=target_closed, togglable_leaves=togglable_leaves)
    assert toggles == expected


@pytest.mark.parametrize("edge_closed", [ALL_OPEN, ALL_CLOSED, LEFT_OPEN, RIGHT_OPEN, ONE_OPEN_PER_PAIR])
@pytest.mark.parametrize("target_closed", [True, False])
def test_the_default_is_every_leaf_togglable(edge_closed, target_closed):
    """Omitting the parameter and passing every leaf give the same answer."""
    tree = series_of_parallels()
    assert minimal_toggles(tree, edge_closed, target_closed=target_closed) == minimal_toggles(
        tree,
        edge_closed,
        target_closed=target_closed,
        togglable_leaves=leaves(tree),
    )


def test_an_empty_set_and_none_are_different_answers():
    """The empty set means already in the target state, ``None`` means out of reach."""
    tree = series_of_parallels()
    assert minimal_toggles(tree, ALL_CLOSED, target_closed=True, togglable_leaves=frozenset()) == frozenset()
    assert minimal_toggles(tree, ALL_CLOSED, target_closed=False, togglable_leaves=frozenset()) is None


def test_ties_among_equally_small_candidates_stay_deterministic():
    """Both leaves of the open pair cost one; the smaller edge id wins, as without the parameter."""
    tree = Series((Parallel(frozenset({Leaf("b"), Leaf("a")})), Leaf("z")), (1,))
    edge_closed = {"a": False, "b": False, "z": True}
    assert minimal_toggles(tree, edge_closed, target_closed=True) == frozenset({"a"})
    assert minimal_toggles(tree, edge_closed, target_closed=True, togglable_leaves=frozenset({"a", "b"})) == frozenset({
        "a"
    })


# --- Part one: property test over random trees ---


def random_tree(rng: random.Random, edge_ids: list[str], depth: int) -> SPTree[str]:
    """A random series-parallel tree over the given edge ids, each used once."""
    if len(edge_ids) == 1 or depth == 0:
        if len(edge_ids) == 1:
            return Leaf(edge_ids[0])
        return Parallel(frozenset(Leaf(edge_id) for edge_id in edge_ids))
    split = rng.randrange(1, len(edge_ids))
    left = random_tree(rng, edge_ids[:split], depth - 1)
    right = random_tree(rng, edge_ids[split:], depth - 1)
    if rng.random() < 0.5:
        return Series((left, right), (1000 + rng.randrange(1000),))
    return Parallel(frozenset({left, right}))


def apply_toggles(edge_closed: dict[str, bool], toggles: frozenset[str]) -> dict[str, bool]:
    return {edge_id: state != (edge_id in toggles) for edge_id, state in edge_closed.items()}


def smallest_reaching_subset_size(
    tree: SPTree[str],
    edge_closed: dict[str, bool],
    togglable_leaves: frozenset[str],
    *,
    target_closed: bool,
) -> int | None:
    """Brute force over every subset of the togglable leaves; ``None`` if none reaches the target."""
    candidates = sorted(togglable_leaves)
    for size in range(len(candidates) + 1):
        for mask in range(1 << len(candidates)):
            subset = frozenset(candidates[index] for index in range(len(candidates)) if mask >> index & 1)
            if len(subset) == size and closed(tree, apply_toggles(edge_closed, subset)) == target_closed:
                return size
    return None


def test_minimal_toggles_property_over_random_trees():
    """Subset of the permitted leaves, reaches the target, and no smaller permitted subset does."""
    rng = random.Random(SEED)
    for _ in range(PROPERTY_TRIALS):
        edge_ids = [f"e{index}" for index in range(rng.randrange(1, 8))]
        tree = random_tree(rng, edge_ids, depth=4)
        edge_closed = {edge_id: rng.random() < 0.5 for edge_id in edge_ids}
        togglable_leaves = frozenset(edge_id for edge_id in edge_ids if rng.random() < 0.6)
        target_closed = rng.random() < 0.5

        # --- Act ---
        toggles = minimal_toggles(
            tree,
            edge_closed,
            target_closed=target_closed,
            togglable_leaves=togglable_leaves,
        )

        # --- Assert ---
        brute_force_size = None
        if len(edge_ids) <= BRUTE_FORCE_LEAF_LIMIT:
            brute_force_size = smallest_reaching_subset_size(
                tree,
                edge_closed,
                togglable_leaves,
                target_closed=target_closed,
            )
        if toggles is None:
            assert brute_force_size is None, "the fold reported unreachable but a permitted subset reaches the target"
            continue
        assert toggles <= togglable_leaves
        assert closed(tree, apply_toggles(edge_closed, toggles)) == target_closed
        assert brute_force_size == len(toggles)


# --- Part two: the ordered walk of a series chain ---


def nested_chain() -> Series[str]:
    """``0 -x- 4 -y- 5 -z- 9`` with ``y`` and ``z`` in a series node nested in the outer one."""
    return Series((Leaf("x"), Series((Leaf("y"), Leaf("z")), (5,))), (4,))


def test_walking_from_either_endpoint_gives_mutually_reversed_sequences():
    """The chain is one sequence of positions, so reading it backwards means reading every step backwards."""
    tree = series_of_parallels()
    forward = series_chain(tree, (0, 2), 0)
    backward = series_chain(tree, (0, 2), 2)
    assert [SeriesStep(step.to_node, step.subtree, step.from_node) for step in reversed(forward)] == backward


def test_the_steps_chain_end_to_start():
    """Consecutive steps share a node, so the walk runs from the start node to the other endpoint unbroken."""
    tree = nested_chain()
    for start_node, first_node, last_node in [(0, 0, 9), (9, 9, 0)]:
        steps = series_chain(tree, (0, 9), start_node)
        assert steps[0].from_node == first_node
        assert steps[-1].to_node == last_node
        assert all(earlier.to_node == later.from_node for earlier, later in zip(steps, steps[1:], strict=False))


def test_a_nested_chain_is_flattened_in_order():
    """Positions along the whole chain are addressable by index, the sub-chain spliced in place."""
    steps = series_chain(nested_chain(), (0, 9), 0)
    assert [step.subtree for step in steps] == [Leaf("x"), Leaf("y"), Leaf("z")]
    assert [(step.from_node, step.to_node) for step in steps] == [(0, 4), (4, 5), (5, 9)]
    assert [step.subtree for step in series_chain(nested_chain(), (0, 9), 9)] == [Leaf("z"), Leaf("y"), Leaf("x")]


def test_a_parallel_child_is_a_single_position():
    """A parallel subtree has no order, so the whole subtree sits at one position."""
    parallel = Parallel(frozenset({Leaf("c"), Leaf("d")}))
    steps = series_chain(Series((Leaf("x"), parallel), (4,)), (0, 9), 0)
    assert [step.subtree for step in steps] == [Leaf("x"), parallel]


def test_a_leaf_is_a_chain_of_one_step():
    """A leaf spans its two endpoints and nothing between them, so the chain has exactly one position."""
    assert series_chain(Leaf("q"), (3, 7), 3) == [SeriesStep(3, Leaf("q"), 7)]
    assert series_chain(Leaf("q"), (3, 7), 7) == [SeriesStep(7, Leaf("q"), 3)]


def test_a_parallel_root_has_no_chain_order():
    """The children of a parallel node are alternatives between the same endpoints, so no order exists to return."""
    tree = Parallel(frozenset({Leaf("a"), Leaf("b")}))
    with pytest.raises(ValueError, match="parallel tree has no chain order"):
        series_chain(tree, (0, 1), 0)


def test_a_start_node_that_is_not_an_endpoint_is_rejected():
    """A walk has to begin at an end of the chain, so an unrelated node is an error and not an empty walk."""
    with pytest.raises(ValueError, match="start node 7 is not an endpoint of the tree"):
        series_chain(series_of_parallels(), (0, 2), 7)


def test_an_interior_node_is_not_a_start_node():
    """The node between the children is on the chain but not an end of it."""
    with pytest.raises(ValueError, match="endpoints are 0 and 2"):
        series_chain(series_of_parallels(), (0, 2), 1)


def test_a_series_node_with_the_wrong_interior_node_count_is_rejected():
    """Two children have exactly one node between them, so a second interior node leaves the chain ambiguous."""
    with pytest.raises(ValueError, match="needs 1 interior nodes, got 2"):
        series_chain(Series((Leaf("x"), Leaf("y")), (4, 5)), (0, 9), 0)
