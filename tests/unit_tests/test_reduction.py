"""Tests for the terminal-preserving graph reduction kernel in ``pygraphc.reduction``.

The four reference scenarios and the two property checks mirror the paper's
``check_kernel.py``: parallel edges survive as one parallel node, a long
chain keeps every edge, removing a parallel pair splits its block, a two-level
quotient keeps edge identity, and the provenance tree round-trips against a
brute force path enumeration on random multigraphs.
"""

import dataclasses
import random
import subprocess
import sys
from collections import defaultdict

import pytest

import pygraphc
from pygraphc.reduction import (
    Leaf,
    MultiGraph,
    Parallel,
    Partition,
    Series,
    VirtualEdgeId,
    closed,
    leaves,
    lift,
    minimal_toggles,
    paths,
    quotient,
    reduce,
    scenario,
    tree_from_records,
    tree_records,
)

SEED = 20260910
PROPERTY_TRIALS = 300


def random_multigraph(rng: random.Random, node_count: int, edge_count: int) -> MultiGraph[int]:
    """Random multigraph with an 8 percent self-loop and a 17 percent parallel rate."""
    nodes = list(range(node_count))
    endpoints: dict[int, tuple[int, int]] = {}
    for edge_id in range(edge_count):
        from_node = rng.randrange(node_count)
        roll = rng.random()
        if roll < 0.08:
            to_node = from_node
        elif roll < 0.25 and endpoints:
            from_node, to_node = endpoints[rng.randrange(len(endpoints))]
        else:
            to_node = rng.randrange(node_count)
        endpoints[edge_id] = (from_node, to_node)
    return MultiGraph(nodes, endpoints)


def brute_force_paths(
    graph: MultiGraph[int],
    allowed_edges: frozenset[int],
    source: int,
    target: int,
    forbidden_interior: set[int],
) -> set[frozenset[int]]:
    """Node-simple paths from source to target over the allowed edges that never
    pass through a forbidden interior node (the surviving nodes)."""
    incidence: dict[int, list[int]] = defaultdict(list)
    for edge_id in allowed_edges:
        from_node, to_node = graph.endpoints[edge_id]
        if from_node == to_node:
            continue
        incidence[from_node].append(edge_id)
        incidence[to_node].append(edge_id)
    found: set[frozenset[int]] = set()

    def walk(node: int, used: list[int], visited: set[int]) -> None:
        for edge_id in incidence[node]:
            if edge_id in used:
                continue
            from_node, to_node = graph.endpoints[edge_id]
            next_node = to_node if from_node == node else from_node
            if next_node == target:
                found.add(frozenset([*used, edge_id]))
                continue
            if next_node in visited or next_node in forbidden_interior:
                continue
            walk(next_node, [*used, edge_id], visited | {next_node})

    walk(source, [], {source})
    return found


def canonical(reduced) -> tuple:
    """Residual nodes, every residual edge as (sorted pair, sorted leaves), and
    the folded material per surviving and per interior node as sorted lists."""
    edges = sorted(
        (tuple(sorted(pair)), tuple(sorted(leaves(reduced.provenance[edge_id]))))
        for edge_id, pair in reduced.graph.endpoints.items()
    )
    folded = {node_id: sorted(members) for node_id, members in reduced.folded_nodes.items()}
    folded_interior = {node_id: sorted(members) for node_id, members in reduced.folded_interior.items()}
    return (tuple(reduced.graph.nodes), tuple(edges), folded, folded_interior)


def tree_interior_nodes(tree) -> list[int]:
    """Every node recorded in a series node of the tree, in increasing id."""
    return sorted(node_id for _index, kind, _children, interior, _leaf in tree_records(tree) for node_id in interior)


def interior_nodes(reduced) -> list[int]:
    """Every node recorded in a series node of any residual tree."""
    return [node_id for tree in reduced.provenance.values() for node_id in tree_interior_nodes(tree)]


def same_tree(actual, expected) -> bool:
    """Structural equality; ``==`` on series and parallel nodes is identity."""
    return tree_records(actual) == tree_records(expected)


# ── MultiGraph ──


def test_multigraph_rejects_duplicate_nodes():
    """The C tier interns node ids; a duplicate would create a phantom node."""
    with pytest.raises(ValueError, match=r"unique, duplicates: \[1\]"):
        MultiGraph([1, 1, 2], {})


def test_multigraph_rejects_non_int_node_ids():
    """The C tier interns ints only; a string would die deep inside it with a type error."""
    with pytest.raises(TypeError, match="node ids must be ints, got 'a'"):
        MultiGraph(["a", "b"], {})


def test_multigraph_rejects_bool_node_ids():
    """``True == 1`` would silently alias a node; bools are ints to Python but not node ids."""
    with pytest.raises(TypeError, match="node ids must be ints, got True"):
        MultiGraph([True, 2], {})


def test_multigraph_rejects_negative_node_ids():
    """The C tier treats negative ids as missing and reports an unrelated unknown-node error."""
    with pytest.raises(ValueError, match="non-negative, got -1"):
        MultiGraph([-1, 2], {})


def test_multigraph_stores_any_node_sequence_as_a_list():
    """A ``range`` and the equivalent list describe the same graph and compare equal."""
    assert MultiGraph(range(3), {}) == MultiGraph([0, 1, 2], {})
    assert MultiGraph(range(3), {}).nodes == [0, 1, 2]


def test_from_groups_rejects_non_int_node_ids():
    """Explicit groups feed the same int-only block ids as the C tier does."""
    with pytest.raises(TypeError, match="node ids must be ints, got 'a'"):
        Partition.from_groups([["a", "b"]])


def test_multigraph_rejects_edge_with_unknown_endpoint():
    """An endpoint outside the node list has no block and no incidence slot."""
    with pytest.raises(ValueError, match=r"'e' has an endpoint that is not a node: \(1, 3\)"):
        MultiGraph([1, 2], {"e": (1, 3)})


# ── Partition ──


def test_from_components_uses_minimum_member_as_block_id():
    """Block ids are content addressed: the minimum member id, so two
    partitions of the same components compare equal regardless of input order."""
    # --- Input ---
    graph = MultiGraph([5, 3, 9, 7], {"a": (9, 3), "b": (5, 7)})

    # --- Execute ---
    partition = Partition.from_components(graph, {"a", "b"})

    # --- Assert ---
    assert partition.blocks() == {3: [3, 9], 5: [5, 7]}
    assert partition.block_of[9] == 3


def test_from_components_masks_inactive_edges_without_rebuilding(mocker):
    """Inactive edges are a byte mask on the cached C graph: a second call on
    the same multigraph reuses the parsed graph, so the C graph is built once."""
    # --- Input ---
    graph = MultiGraph([1, 2, 3, 4], {"e12": (1, 2), "e23": (2, 3), "e34": (3, 4), "loop": (2, 2)})
    graph_constructor = mocker.patch("pygraphc.Graph", wraps=pygraphc.Graph)

    # --- Execute ---
    all_active = Partition.from_components(graph, set(graph.endpoints))
    masked = Partition.from_components(graph, {"e12", "e34"})

    # --- Assert ---
    assert all_active.blocks() == {1: [1, 2, 3, 4]}
    assert masked.blocks() == {1: [1, 2], 3: [3, 4]}
    assert graph_constructor.call_count == 1


def test_from_groups_gives_content_addressed_block_ids():
    """Callers that already know the blocks get the same content-addressed ids."""
    # --- Input ---
    partition = Partition.from_groups([[4, 2], [7]])

    # --- Assert ---
    assert partition.block_of == {4: 2, 2: 2, 7: 7}


def test_blocks_are_sorted_regardless_of_input_order():
    """Outputs must be serializable and cacheable, so member and block order
    cannot depend on the order the caller listed the groups in."""
    # --- Input ---
    forward = Partition.from_groups([[1, 9], [3, 5]])
    backward = Partition.from_groups([[5, 3], [9, 1]])

    # --- Assert ---
    assert list(forward.blocks()) == [1, 3]
    assert forward.blocks() == backward.blocks() == {1: [1, 9], 3: [3, 5]}


def test_compose_equals_the_direct_partition_of_the_union_mask():
    """A second quotient level composed with the first must describe the same
    blocks as partitioning the original graph by both edge classes at once."""
    # --- Input ---
    # 1 -c- 2 -x- 3 -c- 4    5 -c- 6
    graph = MultiGraph(range(1, 7), {"c12": (1, 2), "x23": (2, 3), "c34": (3, 4), "c56": (5, 6)})
    base_mask = {"c12", "c34", "c56"}
    crossing = {"x23"}

    # --- Execute ---
    level_one = Partition.from_components(graph, base_mask)
    meta, _internal = quotient(level_one, graph, crossing)
    level_two = Partition.from_components(meta, set(meta.endpoints)).compose(level_one)

    # --- Assert ---
    assert level_two == Partition.from_components(graph, base_mask | crossing)
    assert level_two.blocks() == {1: [1, 2, 3, 4], 5: [5, 6]}


def test_refines_holds_for_the_finer_partition_only():
    """Refinement is the direction that lets a scenario partition be checked
    against its base: every fine block lies inside one coarse block, not vice versa."""
    # --- Input ---
    finer = Partition.from_groups([[1, 2], [3], [4]])
    coarser = Partition.from_groups([[1, 2, 3], [4]])

    # --- Assert ---
    assert finer.refines(coarser)
    assert not coarser.refines(finer)


# ── Quotient and lift ──


def test_quotient_keeps_parallel_crossing_edges_and_reports_internal_ones():
    """Two crossing edges between the same blocks stay two meta edges with their
    own ids; a crossing edge inside a block is returned as internal, not dropped."""
    # --- Input ---
    graph = MultiGraph([1, 2, 3, 4], {"x": (2, 3), "y": (3, 2), "z": (1, 2), "c12": (1, 2), "c34": (3, 4)})
    partition = Partition.from_components(graph, {"c12", "c34"})

    # --- Execute ---
    meta, internal = quotient(partition, graph, {"x", "y", "z"})

    # --- Assert ---
    assert meta.nodes == [1, 3]
    assert meta.endpoints == {"x": (1, 3), "y": (3, 1)}
    assert internal == {1: ["z"]}


def test_quotient_rejects_a_crossing_edge_id_that_is_not_an_edge_of_the_graph():
    """The crossing set is an edge mask on the cached C graph, under which an
    unknown id simply matches nothing. A typo would be dropped without a word,
    so the ids are checked against the graph first."""
    # --- Input ---
    graph = MultiGraph([1, 2, 3], {"a": (1, 2), "b": (2, 3)})
    partition = Partition.from_groups([[1, 2], [3]])

    # --- Assert ---
    with pytest.raises(KeyError, match="not edges of the graph"):
        quotient(partition, graph, {"a", "typo"})


def test_quotient_requires_a_block_for_every_node_of_the_graph():
    """The C kernel takes one label per node, so a partition that names blocks
    only for the endpoints of the crossing edges is no longer enough."""
    # --- Input ---
    graph = MultiGraph([1, 2, 3], {"a": (1, 2), "b": (2, 3)})
    partition = Partition.from_groups([[1, 2]])

    # --- Assert ---
    with pytest.raises(KeyError):
        quotient(partition, graph, {"a"})


def test_lift_combines_in_increasing_node_order():
    """The combination order is fixed so a non-commutative operation still
    gives reproducible results."""
    # --- Input ---
    partition = Partition.from_groups([[3, 1, 2], [4]])
    attribute = {3: "c", 1: "a", 2: "b", 4: "d"}

    # --- Assert ---
    assert lift(partition, attribute, lambda left, right: left + right) == {1: "abc", 4: "d"}


# ── Tree folds ──


def test_leaves_collects_every_input_edge():
    """A leaf set is the edge material of the residual edge, whatever the nesting."""
    # --- Input ---
    tree = Parallel(frozenset({Series((Leaf("a"), Leaf("b")), (2,)), Leaf("c")}))

    # --- Assert ---
    assert leaves(tree) == {"a", "b", "c"}


def test_paths_series_is_the_product_and_parallel_the_union():
    """Series children are traversed together, parallel children are alternatives."""
    # --- Input ---
    tree = Parallel(frozenset({Series((Leaf("a"), Leaf("b")), (2,)), Leaf("c")}))

    # --- Assert ---
    assert paths(tree) == {frozenset({"a", "b"}), frozenset({"c"})}


def test_paths_cutoff_prunes_inside_the_product():
    """The cutoff bounds the number of edges per path; a longer alternative
    disappears while the short one through the same series node survives."""
    # --- Input ---
    detour = Series((Leaf("b"), Leaf("c")), (1,))
    tree = Series((Parallel(frozenset({Leaf("a"), detour})), Leaf("d")), (2,))

    # --- Assert ---
    assert paths(tree, cutoff=2) == {frozenset({"a", "d"})}
    assert paths(tree, cutoff=3) == {frozenset({"a", "d"}), frozenset({"b", "c", "d"})}


def test_closed_series_is_and_parallel_is_or():
    """A series node is closed only if every child is; a parallel node if any child is."""
    # --- Input ---
    series = Series((Leaf("a"), Leaf("b")), (1,))
    parallel = Parallel(frozenset({Leaf("a"), Leaf("b")}))
    one_open = {"a": True, "b": False}

    # --- Assert ---
    assert not closed(series, one_open)
    assert closed(parallel, one_open)
    assert closed(series, {"a": True, "b": True})
    assert not closed(parallel, {"a": False, "b": False})


def test_minimal_toggles_series_to_closed_needs_every_child():
    """Closing a series node needs every leaf; nothing less changes the state."""
    # --- Input ---
    tree = Series((Leaf("a"), Leaf("b")), (1,))

    # --- Assert ---
    assert minimal_toggles(tree, {"a": False, "b": False}, target_closed=True) == {"a", "b"}


def test_minimal_toggles_parallel_to_open_needs_every_child():
    """Opening a parallel node needs every child toggled, the case a flat leaf
    list without the tree structure gets wrong."""
    # --- Input ---
    tree = Parallel(frozenset({Leaf("a"), Leaf("b")}))

    # --- Assert ---
    assert minimal_toggles(tree, {"a": True, "b": True}, target_closed=False) == {"a", "b"}


def test_minimal_toggles_takes_the_cheapest_child_and_breaks_ties_by_id():
    """Where one child suffices the smallest toggle set wins; equal sizes are
    decided by edge id order so the answer never depends on hash order."""
    # --- Input ---
    short = Series((Leaf("a"), Leaf("b")), (1,))
    long = Series((Leaf("c"), Series((Leaf("d"), Leaf("e")), (3,))), (2,))
    by_size = Parallel(frozenset({short, long}))
    by_id = Parallel(frozenset({Leaf("b"), Leaf("a")}))
    all_open = dict.fromkeys("abcde", False)

    # --- Assert ---
    assert minimal_toggles(by_size, all_open, target_closed=True) == {"a", "b"}
    assert minimal_toggles(by_id, all_open, target_closed=True) == {"a"}


def test_minimal_toggles_is_empty_when_already_in_the_target_state():
    """Nothing needs toggling when the tree already has the requested state."""
    # --- Input ---
    tree = Series((Leaf("a"), Leaf("b")), (1,))

    # --- Assert ---
    assert minimal_toggles(tree, {"a": True, "b": True}, target_closed=True) == frozenset()


def test_folds_survive_a_chain_deeper_than_the_recursion_limit():
    """Long chains are the common input shape, so a nested series tree wrapped
    in a parallel node must hash and fold without recursion."""
    # --- Input ---
    length = 3000
    endpoints = {f"w{index}": (index, index + 1) for index in range(length)}
    endpoints["direct"] = (0, length)
    graph = MultiGraph(range(length + 1), endpoints)

    # --- Execute ---
    reduced = reduce(graph, terminals={0, length})
    (tree,) = reduced.provenance.values()

    # --- Assert ---
    assert isinstance(tree, Parallel)
    assert len(leaves(tree)) == length + 1
    assert paths(tree) == {frozenset({"direct"}), frozenset(endpoints) - {"direct"}}
    assert closed(tree, dict.fromkeys(endpoints, True))
    assert minimal_toggles(tree, dict.fromkeys(endpoints, True), target_closed=False) == {"direct", "w0"}


# ── Reduce: single moves ──


def test_pendant_node_is_folded_into_its_neighbour():
    """Pendant material carries no path between terminals but carries payload,
    so the node moves to its neighbour instead of vanishing."""
    # --- Input ---
    graph = MultiGraph([1, 2, 3], {"e": (1, 2), "f": (1, 3)})

    # --- Execute ---
    reduced = reduce(graph, terminals={1, 2})

    # --- Assert ---
    assert reduced.graph.nodes == [1, 2]
    assert reduced.graph.endpoints == {"e": (1, 2)}
    assert reduced.folded_nodes == {1: [3], 2: []}
    assert reduced.folded_interior == {}


def test_pendant_node_is_dropped_without_folding():
    """Callers without node payloads ask for the drop and get no bookkeeping."""
    # --- Input ---
    graph = MultiGraph([1, 2, 3], {"e": (1, 2), "f": (1, 3)})

    # --- Execute ---
    reduced = reduce(graph, terminals={1, 2}, fold_leaves=False)

    # --- Assert ---
    assert reduced.folded_nodes == {1: [], 2: []}
    assert [(neighbour, leaves(tree)) for neighbour, tree in reduced.dropped] == [(1, {"f"})]


def test_series_merge_is_oriented_from_first_to_second_edge():
    """The series node is ordered: children[0] touches endpoints[0], and the
    eliminated node is recorded so its payload can be folded."""
    # --- Input ---
    graph = MultiGraph([1, 2, 3], {"a": (2, 1), "b": (2, 3)})

    # --- Execute ---
    reduced = reduce(graph, terminals={1, 3})
    ((edge_id, pair),) = reduced.graph.endpoints.items()

    # --- Assert ---
    assert edge_id == VirtualEdgeId(1)
    assert pair == (1, 3)
    assert same_tree(reduced.provenance[edge_id], Series((Leaf("a"), Leaf("b")), (2,)))


def test_parallel_merge_yields_one_parallel_node_with_sorted_endpoints():
    """All edges between one pair merge in one move into one unordered node."""
    # --- Input ---
    graph = MultiGraph([2, 1], {"a": (2, 1), "b": (1, 2), "c": (2, 1)})

    # --- Execute ---
    reduced = reduce(graph, terminals={1, 2})

    # --- Assert ---
    assert reduced.graph.endpoints == {VirtualEdgeId(1): (1, 2)}
    assert same_tree(reduced.provenance[VirtualEdgeId(1)], Parallel(frozenset({Leaf("a"), Leaf("b"), Leaf("c")})))


def test_series_then_parallel_nests_the_trees():
    """A series edge that becomes parallel to an existing edge is merged with
    it, so the residual tree is a parallel node over a series node and a leaf."""
    # --- Input ---
    graph = MultiGraph([1, 2, 3], {"direct": (1, 3), "a": (1, 2), "b": (2, 3)})

    # --- Execute ---
    reduced = reduce(graph, terminals={1, 3})
    (tree,) = reduced.provenance.values()

    # --- Assert ---
    assert same_tree(tree, Parallel(frozenset({Leaf("direct"), Series((Leaf("a"), Leaf("b")), (2,))})))
    assert list(reduced.graph.endpoints) == [VirtualEdgeId(2)]


def test_terminals_survive_even_as_pendants():
    """Terminals carry the question and must never be removed."""
    # --- Input ---
    graph = MultiGraph([1, 2, 3], {"a": (1, 2), "b": (2, 3)})

    # --- Execute ---
    reduced = reduce(graph, terminals={1, 3})

    # --- Assert ---
    assert reduced.graph.nodes == [1, 3]
    assert reduced.graph.endpoints == {VirtualEdgeId(1): (1, 3)}


def test_protected_node_blocks_series_and_parallel_but_not_pendant():
    """A protected node will be split by a later scenario, so its incident
    edges must stay separate; pendant deletion of a protected node is still fine."""
    # --- Input ---
    # 1 -a- 2 -b- 3, 2 protected and not a terminal; 4 protected pendant on 1; parallels c, d at 3
    graph = MultiGraph([1, 2, 3, 4], {"a": (1, 2), "b": (2, 3), "c": (1, 3), "d": (1, 3), "p": (1, 4)})

    # --- Execute ---
    reduced = reduce(graph, terminals={1}, protected={2, 3, 4})

    # --- Assert ---
    assert reduced.graph.nodes == [1, 2, 3]
    assert reduced.graph.endpoints == {"a": (1, 2), "b": (2, 3), "c": (1, 3), "d": (1, 3)}
    assert reduced.folded_nodes == {1: [4], 2: [], 3: []}


def test_self_loop_never_moves_and_leaves_with_its_node():
    """Degree counts incidences: a node with a loop and one further edge is a
    pendant, and the loop is dropped with it. A loop on a survivor stays."""
    # --- Input ---
    graph = MultiGraph([1, 2, 3], {"a": (1, 2), "b": (2, 3), "loop3": (3, 3), "loop1": (1, 1)})

    # --- Execute ---
    reduced = reduce(graph, terminals={1})

    # --- Assert ---
    assert reduced.graph.nodes == [1]
    assert reduced.graph.endpoints == {"loop1": (1, 1)}
    assert reduced.provenance == {"loop1": Leaf("loop1")}
    assert reduced.folded_nodes == {1: [3, 2]}
    assert reduced.folded_interior == {}


def test_node_with_two_edges_to_one_protected_neighbour_is_not_a_series_candidate():
    """Series requires two distinct neighbours; with a protected neighbour the
    two edges cannot merge either, so the node survives with both edges."""
    # --- Input ---
    graph = MultiGraph([1, 2], {"a": (1, 2), "b": (1, 2)})

    # --- Execute ---
    reduced = reduce(graph, terminals={1}, protected={1})

    # --- Assert ---
    assert reduced.graph.nodes == [1, 2]
    assert reduced.graph.endpoints == {"a": (1, 2), "b": (1, 2)}


def test_path_without_terminals_reduces_to_the_empty_graph():
    """A component without a terminal carries no question and is removed whole."""
    # --- Input ---
    graph = MultiGraph([0, 1, 2], {"a": (0, 1), "b": (1, 2)})

    # --- Execute ---
    reduced = reduce(graph, terminals=set())

    # --- Assert ---
    assert reduced.graph == MultiGraph([], {})
    assert reduced.provenance == {}
    assert reduced.folded_nodes == {}
    assert reduced.dropped == []


def test_terminal_free_components_are_removed_before_the_moves():
    """Removing terminal-free components first makes the residual independent of
    the move order: a terminal-free cycle would otherwise leave one arbitrary
    survivor. Components with a terminal are untouched, self-loops included."""
    # --- Input ---
    # component A: terminal 1 with loop; component B: isolated 5; component C: cycle 6-7-8
    graph = MultiGraph([1, 2, 5, 6, 7, 8], {"a": (1, 2), "loop": (1, 1), "x": (6, 7), "y": (7, 8), "z": (8, 6)})

    # --- Execute ---
    reduced = reduce(graph, terminals={1, 2})

    # --- Assert ---
    assert reduced.graph.nodes == [1, 2]
    assert reduced.graph.endpoints == {"a": (1, 2), "loop": (1, 1)}
    assert reduced.folded_nodes == {1: [], 2: []}


def test_folded_material_of_a_series_node_is_kept_in_folded_interior():
    """A leaf folded into a node that is later series-merged must not vanish:
    the series node records the eliminated node and ``folded_interior`` keeps
    what had been folded into it."""
    # --- Input ---
    # 1 -a- 2 -b- 3 with leaf 4 hanging on 2; node 2 has degree 3 until 4 is folded
    graph = MultiGraph([1, 2, 3, 4], {"a": (1, 2), "b": (2, 3), "leaf": (2, 4)})

    # --- Execute ---
    reduced = reduce(graph, terminals={1, 3})
    (tree,) = reduced.provenance.values()

    # --- Assert ---
    assert same_tree(tree, Series((Leaf("a"), Leaf("b")), (2,)))
    assert reduced.folded_interior == {2: [4]}
    assert reduced.folded_nodes == {1: [], 3: []}


def test_pendant_edge_with_a_series_tree_folds_its_interior_nodes():
    """A chain without terminals is series-merged before its end becomes a
    pendant; deleting the pendant edge must pass every node of the chain on,
    and the interior entries leave ``folded_interior`` with the tree."""
    # --- Input ---
    graph = MultiGraph([1, 2, 3, 4], {"a": (1, 2), "b": (2, 3), "c": (3, 4)})

    # --- Execute ---
    reduced = reduce(graph, terminals={1})

    # --- Assert ---
    assert reduced.graph.nodes == [1]
    assert reduced.folded_nodes == {1: [4, 2, 3]}
    assert reduced.folded_interior == {}
    assert [(neighbour, tree_interior_nodes(tree)) for neighbour, tree in reduced.dropped] == [(1, [2, 3])]


def test_chain_payload_reaches_the_terminal_in_either_move_order():
    """Series-then-pendant used to lose the middle node's payload while
    pendant-then-pendant kept it; both orders must fold both nodes into the terminal."""
    # --- Input ---
    # w=0 -- u=1 -- v=2, terminal w
    graph = MultiGraph([0, 1, 2], {"wu": (0, 1), "uv": (1, 2)})

    # --- Execute ---
    series_first = reduce(graph, terminals={0}, order=[1, 2])
    pendant_first = reduce(graph, terminals={0}, order=[2, 1])

    # --- Assert ---
    assert sorted(series_first.folded_nodes[0]) == [1, 2]
    assert sorted(pendant_first.folded_nodes[0]) == [1, 2]
    assert canonical(series_first) == canonical(pendant_first) == canonical(reduce(graph, terminals={0}))


# ── Reference scenarios from the paper ──


def test_parallel_edges_survive_as_one_parallel_node():
    """Two parallel edges between the same terminals must both survive as leaves
    of one parallel node, so opening the residual edge needs both."""
    # --- Input ---
    graph = MultiGraph([1, 2], {"p1": (1, 2), "p2": (1, 2)})

    # --- Execute ---
    reduced = reduce(graph, terminals={1, 2})
    (tree,) = reduced.provenance.values()

    # --- Assert ---
    assert isinstance(tree, Parallel)
    assert leaves(tree) == {"p1", "p2"}
    assert minimal_toggles(tree, {"p1": True, "p2": True}, target_closed=False) == {"p1", "p2"}


def test_long_chain_keeps_all_six_edges():
    """Five interior degree-2 nodes: every edge must appear in the single
    residual tree and the only path uses all six, the case that orphaned a
    group in the pairwise contraction."""
    # --- Input ---
    graph = MultiGraph(range(7), {f"w{index}": (index, index + 1) for index in range(6)})

    # --- Execute ---
    reduced = reduce(graph, terminals={0, 6})
    (tree,) = reduced.provenance.values()
    only_paths = paths(tree)

    # --- Assert ---
    assert len(leaves(tree)) == 6
    assert only_paths == {frozenset(graph.endpoints)}
    assert sorted(interior_nodes(reduced)) == [1, 2, 3, 4, 5]
    assert reduced.folded_interior == {1: [], 2: [], 3: [], 4: [], 5: []}


def test_removing_a_parallel_pair_splits_the_block():
    """A node joined to the rest by two parallel edges, neither a bridge: a
    scenario removing both must still split the block, and the result refines the base."""
    # --- Input ---
    graph = MultiGraph([1, 2, 3, 4], {"e1": (1, 2), "p1": (2, 3), "p2": (2, 3), "e2": (3, 4)})
    active = set(graph.endpoints)

    # --- Execute ---
    base = Partition.from_components(graph, active)
    after = scenario(graph, active, removed={"p1", "p2"})
    single = scenario(graph, active, removed={"p1"})

    # --- Assert ---
    assert len(base.blocks()) == 1
    assert after.blocks() == {1: [1, 2], 3: [3, 4]}
    assert after.refines(base)
    assert single == base


def test_two_level_quotient_keeps_edge_identity():
    """The base mask forms blocks, the crossing class becomes level-one edges and
    a second class level-two edges with identity kept, so the reduced tree lists
    both terminal paths and the minimal toggle set has two edges."""
    # --- Input ---
    endpoints = {
        "c12": (1, 2),
        "c34": (3, 4),
        "c56": (5, 6),
        "c78": (7, 8),
        "p1": (2, 3),
        "p2": (2, 3),
        "x": (4, 5),
        "y": (4, 5),
        "z": (6, 7),
        "loop": (1, 2),
    }
    graph = MultiGraph(range(1, 9), endpoints)
    base_mask = {"c12", "c34", "c56", "c78"}
    crossing = {"p1", "p2"}
    second_class = {"x", "y", "z", "loop"}

    # --- Execute ---
    level_one = Partition.from_components(graph, base_mask)
    meta_one, _ = quotient(level_one, graph, crossing)
    level_two_base = Partition.from_components(meta_one, set(meta_one.endpoints))
    level_two = level_two_base.compose(level_one)
    meta_two, internal = quotient(level_two, graph, second_class)
    reduced = reduce(meta_two, terminals={1, 7})
    (tree,) = reduced.provenance.values()
    state = dict.fromkeys(second_class, False)
    to_close = minimal_toggles(tree, state, target_closed=True)

    # --- Assert ---
    assert len(level_one.blocks()) == 4
    assert len(level_two_base.blocks()) == 3
    assert internal == {1: ["loop"]}
    assert paths(tree) == {frozenset({"x", "z"}), frozenset({"y", "z"})}
    assert to_close == {"x", "z"}
    assert closed(tree, {**state, **dict.fromkeys(to_close, True)})


# ── Properties ──


def test_provenance_round_trips_against_brute_force_paths():
    """The tree loses nothing: the paths expanded from every residual edge equal
    the node-simple paths through the eliminated material of the input graph."""
    # --- Input ---
    rng = random.Random(SEED)
    compared = 0

    # --- Execute ---
    for _ in range(PROPERTY_TRIALS):
        graph = random_multigraph(rng, rng.randrange(4, 10), rng.randrange(4, 16))
        terminals = set(rng.sample(graph.nodes, rng.randrange(1, 4)))
        reduced = reduce(graph, terminals)
        survivors = set(reduced.graph.nodes)
        for edge_id, (from_block, to_block) in reduced.graph.endpoints.items():
            if from_block == to_block:
                continue
            tree = reduced.provenance[edge_id]
            brute = brute_force_paths(graph, leaves(tree), from_block, to_block, survivors)
            compared += 1

            # --- Assert ---
            assert paths(tree) == brute
    assert compared > PROPERTY_TRIALS


def test_reduction_is_deterministic_under_input_shuffling():
    """Tie-breaking uses id order, never hash or input order, so shuffling the
    node and edge input leaves the canonical residual unchanged."""
    # --- Input ---
    rng = random.Random(SEED)

    # --- Execute ---
    for _ in range(PROPERTY_TRIALS):
        graph = random_multigraph(rng, rng.randrange(4, 10), rng.randrange(4, 16))
        terminals = set(rng.sample(graph.nodes, rng.randrange(1, 4)))
        nodes = list(graph.nodes)
        rng.shuffle(nodes)
        edge_ids = list(graph.endpoints)
        rng.shuffle(edge_ids)
        shuffled = MultiGraph(nodes, {edge_id: graph.endpoints[edge_id] for edge_id in edge_ids})

        # --- Assert ---
        assert canonical(reduce(shuffled, terminals)) == canonical(reduce(graph, terminals))


def test_every_eliminated_node_is_accounted_for_exactly_once_when_folding():
    """With folding, node payloads must survive: every node of a component with
    a terminal is a survivor, folded into a survivor, recorded in a series node
    or folded into such a recorded node, exactly once."""
    # --- Input ---
    rng = random.Random(SEED)

    # --- Execute ---
    for trial in range(PROPERTY_TRIALS):
        graph = random_multigraph(rng, rng.randrange(4, 10), rng.randrange(4, 16))
        terminals = set(rng.sample(graph.nodes, rng.randrange(1, 4)))
        protected = set(rng.sample(graph.nodes, trial % 3))
        components = Partition.from_components(graph, set(graph.endpoints))
        terminal_blocks = {components.block_of[node_id] for node_id in terminals}
        expected = [node_id for node_id in graph.nodes if components.block_of[node_id] in terminal_blocks]
        reduced = reduce(graph, terminals, protected)
        folded = [node_id for members in reduced.folded_nodes.values() for node_id in members]
        folded_interior = [node_id for members in reduced.folded_interior.values() for node_id in members]

        # --- Assert ---
        assert sorted([*reduced.graph.nodes, *folded, *interior_nodes(reduced), *folded_interior]) == expected
        assert set(reduced.folded_nodes) == set(reduced.graph.nodes)
        assert set(reduced.folded_interior) == set(interior_nodes(reduced))


def test_reduction_is_confluent_without_protected_nodes():
    """The three moves are locally confluent, so with no protected nodes the
    canonical residual, including the folded material, is the same for every
    candidate processing order."""
    # --- Input ---
    rng = random.Random(SEED)

    # --- Execute ---
    for _ in range(PROPERTY_TRIALS // 10):
        graph = random_multigraph(rng, rng.randrange(4, 10), rng.randrange(4, 16))
        terminals = set(rng.sample(graph.nodes, rng.randrange(1, 4)))
        expected = canonical(reduce(graph, terminals))
        for _ in range(20):
            order = list(graph.nodes)
            rng.shuffle(order)

            # --- Assert ---
            assert canonical(reduce(graph, terminals, order=order)) == expected


def test_protected_node_can_make_the_residual_order_dependent():
    """Documented limit of confluence: on a triangle with the terminal protected,
    the first series move decides which edge pair ends up parallel at the
    protected node, where it cannot merge. Two orders give two residuals; the
    module returns the increasing-id one."""
    # --- Input ---
    graph = MultiGraph([0, 1, 2], {"a": (0, 1), "b": (1, 2), "c": (2, 0)})

    # --- Execute ---
    node_one_first = reduce(graph, terminals={0}, protected={0}, order=[1, 2])
    node_two_first = reduce(graph, terminals={0}, protected={0}, order=[2, 1])
    default = reduce(graph, terminals={0}, protected={0})

    # --- Assert ---
    assert node_one_first.graph.nodes == [0, 2]
    assert node_two_first.graph.nodes == [0, 1]
    assert canonical(node_one_first) != canonical(node_two_first)
    assert canonical(default) == canonical(node_one_first)


def test_scenario_partition_refines_the_base_partition():
    """Removing edges cannot connect nodes, so every scenario partition refines
    the partition of the full active mask."""
    # --- Input ---
    rng = random.Random(SEED)

    # --- Execute ---
    for _ in range(PROPERTY_TRIALS):
        graph = random_multigraph(rng, rng.randrange(4, 10), rng.randrange(4, 16))
        active = set(graph.endpoints)
        removed = set(rng.sample(sorted(active), rng.randrange(len(active) + 1)))
        base = Partition.from_components(graph, active)
        after = scenario(graph, active, removed)

        # --- Assert ---
        assert after.refines(base)
        assert after == Partition.from_components(graph, active - removed)


# ── Tree identity, repr and records ──


def test_series_with_the_same_children_and_different_interior_stay_distinct_in_a_parallel():
    """Two series nodes over the same edges but different eliminated nodes are
    different material; a parallel node keeps both and the records tell them apart."""
    # --- Input ---
    first = Series((Leaf("a"), Leaf("b")), (1,))
    second = Series((Leaf("a"), Leaf("b")), (2,))

    # --- Execute ---
    parallel = Parallel(frozenset({first, second}))

    # --- Assert ---
    assert len(parallel.children) == 2
    assert first != second
    assert tree_records(first) != tree_records(second)


def test_tree_equality_is_identity_and_hashing_is_structural():
    """Series and parallel nodes compare by identity, so a structurally equal
    copy is a different key while its hash, cached from the children, agrees."""
    # --- Input ---
    original = Series((Leaf("a"), Leaf("b")), (1,))
    alias = original
    copy = Series((Leaf("a"), Leaf("b")), (1,))

    # --- Assert ---
    assert alias == original
    assert original != copy
    assert hash(original) == hash(copy)
    assert same_tree(original, copy)


def test_tree_records_list_parallel_children_by_smallest_leaf_and_series_children_in_order():
    """The log is the canonical form: series order is meaning, parallel order is
    fixed by the smallest leaf id so it never follows the frozenset's hash order."""
    # --- Input ---
    tree = Parallel(frozenset({Leaf("b"), Series((Leaf("c"), Leaf("a")), (1,))}))

    # --- Assert ---
    assert tree_records(tree) == [
        (0, "leaf", (), (), "c"),
        (1, "leaf", (), (), "a"),
        (2, "series", (0, 1), (1,), None),
        (3, "leaf", (), (), "b"),
        (4, "parallel", (2, 3), (), None),
    ]


def test_tree_records_do_not_depend_on_the_hash_seed():
    """Cached results are compared across processes, so the log of a tree with
    string edge ids must be byte-identical under different hash seeds."""
    # --- Input ---
    script = (
        "from pygraphc.reduction import MultiGraph, reduce, tree_records\n"
        "edges = {f'p{index}': (1, 2) for index in range(8)}\n"
        "edges.update({'a': (2, 3), 'b': (3, 4), 'c': (1, 4), 'd': (1, 4)})\n"
        "reduced = reduce(MultiGraph([1, 2, 3, 4], edges), terminals={1, 2})\n"
        "print([tree_records(tree) for tree in reduced.provenance.values()])\n"
    )

    # --- Execute ---
    outputs = [
        subprocess.run(
            [sys.executable, "-c", script],
            env={"PYTHONHASHSEED": seed, "PATH": ""},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for seed in ("1", "2", "3")
    ]

    # --- Assert ---
    assert "parallel" in outputs[0]
    assert outputs[0] == outputs[1] == outputs[2]


def test_tree_records_round_trip_a_chain_deeper_than_the_recursion_limit():
    """P6 wants serialisable outputs: the module's own deep fixture must
    compare, print and serialise without recursion, which pickle cannot do."""
    # --- Input ---
    length = 3000
    endpoints = {f"w{index}": (index, index + 1) for index in range(length)}
    endpoints["direct"] = (0, length)
    graph = MultiGraph(range(length + 1), endpoints)

    # --- Execute ---
    reduced = reduce(graph, terminals={0, length})
    ((edge_id, tree),) = reduced.provenance.items()
    records = tree_records(tree)
    rebuilt = tree_from_records(records)

    # --- Assert ---
    assert tree == reduced.provenance[edge_id]
    assert reduced == dataclasses.replace(reduced)
    assert repr(tree) == f"Parallel(children=2, leaves={length + 1})"
    assert len(records) == 2 * length + 1
    assert tree_records(rebuilt) == records
    assert rebuilt != tree
    assert leaves(rebuilt) == leaves(tree)
    assert paths(rebuilt) == paths(tree)


def test_tree_records_round_trip_random_trees():
    """Every residual tree of a random reduction rebuilds to the same log, leaves and paths."""
    # --- Input ---
    rng = random.Random(SEED)

    # --- Execute ---
    for _ in range(200):
        graph = random_multigraph(rng, rng.randrange(4, 10), rng.randrange(4, 16))
        terminals = set(rng.sample(graph.nodes, rng.randrange(1, 4)))
        reduced = reduce(graph, terminals)
        for tree in reduced.provenance.values():
            records = tree_records(tree)
            rebuilt = tree_from_records(records)

            # --- Assert ---
            assert tree_records(rebuilt) == records
            assert leaves(rebuilt) == leaves(tree)
            assert paths(rebuilt) == paths(tree)


def test_tree_from_records_rejects_an_out_of_order_index():
    """Child indices refer to positions, so a log whose indices do not count up is corrupt."""
    with pytest.raises(ValueError, match="record index 1 out of order, expected 0"):
        tree_from_records([(1, "leaf", (), (), "a")])


# ── Reduce: re-reduction, loops in series, unknown ids, dropped trees ──


def test_virtual_edge_ids_are_numbered_above_those_in_the_input():
    """A residual fed back in carries virtual ids; a fresh counter from 1 would
    overwrite a live input edge while its id stayed in the incidence sets."""
    # --- Input ---
    graph = MultiGraph([1, 2, 3], {VirtualEdgeId(1): (1, 2), "b": (2, 3)})

    # --- Execute ---
    reduced = reduce(graph, terminals={1, 3})

    # --- Assert ---
    assert reduced.graph.endpoints == {VirtualEdgeId(2): (3, 1)}
    assert same_tree(reduced.provenance[VirtualEdgeId(2)], Series((Leaf("b"), Leaf(VirtualEdgeId(1))), (2,)))


def test_reducing_the_residual_again_changes_nothing():
    """The residual is a fixpoint of the three moves, so reducing it with the
    same terminals returns it unchanged: same nodes, same edge ids, leaf
    provenance, nothing folded or dropped, and the same canonical form once
    the residual's ids are expanded through the first reduction's trees."""
    # --- Input ---
    rng = random.Random(SEED)

    # --- Execute ---
    for _ in range(200):
        graph = random_multigraph(rng, rng.randrange(4, 10), rng.randrange(4, 16))
        terminals = set(rng.sample(graph.nodes, rng.randrange(1, 4)))
        first = reduce(graph, terminals)
        second = reduce(first.graph, terminals)
        expanded = sorted(
            (tuple(sorted(pair)), tuple(sorted(leaves(first.provenance[residual]))))
            for residual, pair in second.graph.endpoints.items()
        )

        # --- Assert ---
        assert second.graph == first.graph
        assert second.provenance == {edge_id: Leaf(edge_id) for edge_id in first.graph.endpoints}
        assert second.folded_nodes == {node_id: [] for node_id in first.graph.nodes}
        assert second.folded_interior == {}
        assert second.dropped == []
        assert (tuple(second.graph.nodes), tuple(expanded)) == canonical(first)[:2]


def test_node_with_a_loop_and_two_incidences_is_a_series_candidate_and_the_loop_leaves():
    """Degree counts non-loop incidences: the loop neither blocks the series
    move nor enters the tree, it is dropped with its node."""
    # --- Input ---
    graph = MultiGraph([1, 2, 3], {"a": (1, 2), "b": (2, 3), "loop": (2, 2)})

    # --- Execute ---
    reduced = reduce(graph, terminals={1, 3})

    # --- Assert ---
    assert reduced.graph.endpoints == {VirtualEdgeId(1): (1, 3)}
    assert same_tree(reduced.provenance[VirtualEdgeId(1)], Series((Leaf("a"), Leaf("b")), (2,)))
    assert reduced.folded_interior == {2: []}


def test_unknown_terminals_and_protected_nodes_are_rejected():
    """A mistyped terminal would silently let the component it should protect vanish."""
    # --- Input ---
    graph = MultiGraph([1, 2], {"a": (1, 2)})

    # --- Assert ---
    with pytest.raises(ValueError, match=r"unknown: \[99\]"):
        reduce(graph, terminals={1, 99})
    with pytest.raises(ValueError, match=r"unknown: \[7, 99\]"):
        reduce(graph, terminals={1}, protected={7, 99})


def test_fold_leaves_false_drops_the_interior_nodes_with_the_tree():
    """Without folding a dropped series tree takes its interior nodes and their
    folded material with it, leaving no stale ``folded_interior`` entries; the
    tree itself is still reported."""
    # --- Input ---
    graph = MultiGraph([1, 2, 3, 4], {"a": (1, 2), "b": (2, 3), "c": (3, 4)})

    # --- Execute ---
    reduced = reduce(graph, terminals={1}, fold_leaves=False)

    # --- Assert ---
    assert reduced.graph.nodes == [1]
    assert reduced.folded_nodes == {1: []}
    assert reduced.folded_interior == {}
    assert [(neighbour, tree_interior_nodes(tree), leaves(tree)) for neighbour, tree in reduced.dropped] == [
        (1, [2, 3], {"a", "b", "c"})
    ]


def test_dead_end_cycle_at_a_terminal_keeps_its_edges_in_dropped():
    """A cycle returning to one node becomes a parallel edge to a pendant; the
    pendant move used to discard that tree with its leaves. The cycle's
    edges must be reported and the payload of every cycle node folded into the terminal."""
    # --- Input ---
    graph = MultiGraph([0, 1, 2, 3], {"t": (0, 1), "r12": (1, 2), "r23": (2, 3), "r31": (3, 1)})

    # --- Execute ---
    reduced = reduce(graph, terminals={0})
    dropped_leaves = frozenset().union(*(leaves(tree) for _neighbour, tree in reduced.dropped))

    # --- Assert ---
    assert reduced.graph == MultiGraph([0], {})
    assert sorted(reduced.folded_nodes[0]) == [1, 2, 3]
    assert [neighbour for neighbour, _tree in reduced.dropped] == [0]
    assert dropped_leaves == {"t", "r12", "r23", "r31"}
    assert any(kind == "parallel" for _index, kind, *_rest in tree_records(reduced.dropped[0][1]))
