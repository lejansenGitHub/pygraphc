"""Terminal-preserving graph reduction kernel (Python tier).

The partition step and the edge split of the quotient run in the C tier on
int32 label arrays of ``pygraphc.Graph`` (``component_labels`` and
``quotient_edges`` under an edge mask). Everything above them lives here:
quotient with edge identity, lift with a fixed combination order, the fixpoint
reduction with series-parallel provenance, the tree folds and scenario
application.

Node ids are non-negative integers because the C tier interns them. Edge ids
are opaque hashable values and every edge keeps its identity through every
operation. Every operation is deterministic: ties are broken by id order,
never by hash order. Self-loops take part in no move and leave with their node.

The module depends on ``pygraphc`` only for ``pygraphc.Graph`` and its views;
the package is imported as a module so that ``pygraphc/__init__.py`` can
re-export the kernel without a circular import.
"""

from __future__ import annotations

import heapq
from array import array
from bisect import bisect_left
from collections import Counter
from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, field
from functools import cached_property
from itertools import count, product
from typing import Generic, Literal, TypeAlias, TypeVar, cast

import pygraphc

__all__ = [
    "EdgeId",
    "Leaf",
    "MultiGraph",
    "Parallel",
    "Partition",
    "Payload",
    "Reduced",
    "SPTree",
    "Series",
    "SeriesStep",
    "TreeKind",
    "TreeRecord",
    "VirtualEdgeId",
    "closed",
    "leaves",
    "lift",
    "minimal_toggles",
    "paths",
    "quotient",
    "reduce",
    "scenario",
    "series_chain",
    "tree_from_records",
    "tree_records",
]

EdgeId = TypeVar("EdgeId", bound=Hashable)
Payload = TypeVar("Payload")


def _order_key(value: object) -> tuple[int, str, int | str]:
    """Total order over opaque ids of mixed types.

    Integers order numerically and come first, so a block id is the
    numerically smallest member. Everything else orders by type name, then by
    ``repr``. Virtual edge ids come after every input edge id, in creation order.
    """
    if isinstance(value, VirtualEdgeId):
        return (2, "", value.index)
    if isinstance(value, int) and not isinstance(value, bool):
        return (0, "", value)
    return (1, type(value).__name__, repr(value))


def _check_node_ids(node_ids: Iterable[int]) -> None:
    """Node ids must be non-negative ints (``bool`` excluded) so the C tier can intern them.

    A wrong type raises ``TypeError``, a negative id ``ValueError``.
    """
    for node_id in node_ids:
        if isinstance(node_id, bool) or not isinstance(node_id, int):
            message = f"node ids must be ints, got {node_id!r}"
            raise TypeError(message)
        if node_id < 0:
            message = f"node ids must be non-negative, got {node_id}"
            raise ValueError(message)


def _sorted_pair(first_node: int, second_node: int) -> tuple[int, int]:
    return (first_node, second_node) if first_node <= second_node else (second_node, first_node)


# ---------------------------------------------------------------------------
# Multigraph with edge identity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class VirtualEdgeId:
    """Id of an edge produced by a series or parallel move, numbered in creation order.

    A reduction whose input already contains virtual ids (the residual of an
    earlier reduction) numbers its own edges above the largest one present, so
    generated ids never collide with input ids.
    """

    index: int


@dataclass(frozen=True)
class MultiGraph(Generic[EdgeId]):
    """Multigraph whose edges are identified by id, never by endpoint pair.

    Treated as an immutable value: the masked C graph used for partitions
    is built once per instance and reused by every scenario. Node ids are
    validated at construction (``TypeError`` for a non-int, ``ValueError``
    otherwise); ``nodes`` is stored as a list whatever sequence was given.
    """

    nodes: list[int]
    endpoints: dict[EdgeId, tuple[int, int]]

    def __post_init__(self) -> None:
        nodes = list(self.nodes)
        object.__setattr__(self, "nodes", nodes)
        _check_node_ids(nodes)
        node_set = set(nodes)
        if len(node_set) != len(nodes):
            duplicates = sorted(node_id for node_id, occurrences in Counter(nodes).items() if occurrences > 1)
            message = f"node ids must be unique, duplicates: {duplicates}"
            raise ValueError(message)
        for edge_id, (from_node, to_node) in self.endpoints.items():
            if from_node not in node_set or to_node not in node_set:
                message = f"edge {edge_id!r} has an endpoint that is not a node: {(from_node, to_node)}"
                raise ValueError(message)

    @cached_property
    def _kernel(self) -> _KernelGraph[EdgeId]:
        nodes = sorted(self.nodes)
        edge_ids = sorted(self.endpoints, key=_order_key)
        kernel_graph = pygraphc.Graph(nodes, [self.endpoints[edge_id] for edge_id in edge_ids])
        return _KernelGraph(nodes, edge_ids, kernel_graph)


@dataclass(frozen=True)
class _KernelGraph(Generic[EdgeId]):
    """Parsed C graph of a multigraph with the node id at every node index and the edge id at every edge index.

    Node ids are in increasing order, so the smallest node index of a block is
    its smallest member and a C label becomes the block id by one list lookup.
    Edge ids are in ``_order_key`` order, so edge index order is edge id order.
    """

    nodes: list[int]
    edge_ids: list[EdgeId]
    graph: pygraphc.Graph[int, int]

    def restricted_to(self, kept: AbstractSet[EdgeId]) -> pygraphc.GraphView[int, int]:
        """View that masks every edge outside ``kept``; the parsed graph is never rebuilt."""
        return self.graph.without_edges([index for index, edge_id in enumerate(self.edge_ids) if edge_id not in kept])


# ---------------------------------------------------------------------------
# Partition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Partition:
    """Node to block mapping with constant-time lookup through ``block_of``.

    Block ids are the minimum member id, so they are content addressed and
    reproducible regardless of input order.
    """

    block_of: dict[int, int]

    @classmethod
    def from_components(cls, graph: MultiGraph[EdgeId], active: AbstractSet[EdgeId]) -> Partition:
        """Connected components of the graph restricted to the active edges.

        The inactive edges are a byte mask on the cached C graph, so applying
        a different active set never rebuilds the graph. The C tier returns
        one int32 label per node, the node index of the block's smallest
        member, so no set per block is ever built.
        """
        kernel = graph._kernel
        labels = kernel.restricted_to(active).component_labels()
        nodes = kernel.nodes
        return cls({node_id: nodes[label] for node_id, label in zip(nodes, labels, strict=True)})

    @classmethod
    def from_groups(cls, groups: Iterable[Iterable[int]]) -> Partition:
        """Explicit grouping, for callers that already know the blocks."""
        block_of: dict[int, int] = {}
        for group in groups:
            members = list(group)
            _check_node_ids(members)
            representative = min(members)
            for node_id in members:
                block_of[node_id] = representative
        return cls(block_of)

    def blocks(self) -> dict[int, list[int]]:
        """Members per block, blocks and members in increasing id order."""
        members: dict[int, list[int]] = {}
        for node_id in sorted(self.block_of):
            members.setdefault(self.block_of[node_id], []).append(node_id)
        return dict(sorted(members.items()))

    def compose(self, finer: Partition) -> Partition:
        """Partition of the original nodes when self partitions the blocks of the finer partition."""
        return Partition({node_id: self.block_of[block_id] for node_id, block_id in finer.block_of.items()})

    def refines(self, coarser: Partition) -> bool:
        """Every block of self lies inside one block of the coarser partition."""
        image: dict[int, int] = {}
        for node_id, block_id in self.block_of.items():
            target = coarser.block_of[node_id]
            if image.setdefault(block_id, target) != target:
                return False
        return True


def quotient(
    partition: Partition,
    graph: MultiGraph[EdgeId],
    crossing: AbstractSet[EdgeId],
) -> tuple[MultiGraph[EdgeId], dict[int, list[EdgeId]]]:
    """Meta multigraph over the blocks in which every crossing edge keeps its identity.

    Crossing edges with both endpoints in one block are returned separately
    as the internal edges of that block. The C tier splits the edges in one
    pass over an int32 label per node (the position of the node's block in
    the sorted block list); the meta edges come back in edge id order. The
    partition must name a block for every node of the graph.
    """
    block_nodes = sorted(set(partition.block_of.values()))
    block_index = {block_id: index for index, block_id in enumerate(block_nodes)}
    kernel = graph._kernel
    labels = memoryview(array("i", [block_index[partition.block_of[node_id]] for node_id in kernel.nodes]))
    from_labels, to_labels, crossing_indices, internal_indices = kernel.restricted_to(crossing).quotient_edges(labels)
    endpoints: dict[EdgeId, tuple[int, int]] = {
        kernel.edge_ids[index]: (block_nodes[from_label], block_nodes[to_label])
        for from_label, to_label, index in zip(from_labels, to_labels, crossing_indices, strict=True)
    }
    internal: dict[int, list[EdgeId]] = {}
    for index in internal_indices:
        edge_id = kernel.edge_ids[index]
        internal.setdefault(partition.block_of[graph.endpoints[edge_id][0]], []).append(edge_id)
    return MultiGraph(block_nodes, endpoints), internal


def lift(
    partition: Partition,
    attribute: Mapping[int, Payload],
    combine: Callable[[Payload, Payload], Payload],
) -> dict[int, Payload]:
    """Combine node attributes per block in increasing node order."""
    lifted: dict[int, Payload] = {}
    for node_id in sorted(attribute):
        block_id = partition.block_of[node_id]
        value = attribute[node_id]
        lifted[block_id] = value if block_id not in lifted else combine(lifted[block_id], value)
    return lifted


# ---------------------------------------------------------------------------
# Series-parallel provenance tree
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Leaf(Generic[EdgeId]):
    """An edge of the input graph.

    Equality and hashing are the generated ones over the edge id. A leaf has
    no children, so neither can recurse and there is nothing to cache: the
    interior nodes below cache their hash because a structural one would walk
    their subtree. One leaf exists per edge of the input, so this constructor
    is the busiest in the module and stays as thin as the dataclass allows.
    """

    edge_id: EdgeId


@dataclass(frozen=True, eq=False, repr=False)
class Series(Generic[EdgeId]):
    """Children ordered from the from-endpoint to the to-endpoint of the merged edge.

    ``interior_nodes`` records the eliminated node so node payloads can be
    folded; the material folded into it beforehand is in
    ``Reduced.folded_interior``.

    Equality is identity. ``reduce`` creates every interior tree node once, in
    the move that produces it, and the leaf sets of distinct live edges are
    disjoint, so two structurally equal nodes never coexist and a recursive
    structural comparison would only fail on deep chains. The hash is computed
    once from the children's cached hashes, so hashing and ``frozenset``
    membership are constant-time and recursion-free as well. Compare structures
    through ``tree_records``.
    """

    children: tuple[SPTree[EdgeId], ...]
    interior_nodes: tuple[int, ...]
    _hash: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_hash", hash((Series, self.children, self.interior_nodes)))

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return f"Series(children={len(self.children)}, leaves={len(leaves(self))})"


@dataclass(frozen=True, eq=False, repr=False)
class Parallel(Generic[EdgeId]):
    """Unordered children between the same endpoint pair. Equality is identity, see ``Series``."""

    children: frozenset[SPTree[EdgeId]]
    _hash: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_hash", hash((Parallel, self.children)))

    def __hash__(self) -> int:
        return self._hash

    def __repr__(self) -> str:
        return f"Parallel(children={len(self.children)}, leaves={len(leaves(self))})"


SPTree: TypeAlias = Leaf[EdgeId] | Series[EdgeId] | Parallel[EdgeId]

TreeKind: TypeAlias = Literal["leaf", "series", "parallel"]
TreeRecord: TypeAlias = tuple[int, TreeKind, tuple[int, ...], tuple[int, ...], EdgeId | None]


def _children(node: Series[EdgeId] | Parallel[EdgeId]) -> Iterable[SPTree[EdgeId]]:
    return node.children


def _post_order(
    tree: SPTree[EdgeId],
    children_of: Callable[[Series[EdgeId] | Parallel[EdgeId]], Iterable[SPTree[EdgeId]]] = _children,
) -> list[SPTree[EdgeId]]:
    """Every node of the tree once, children before parents in the order ``children_of`` gives, without recursion."""
    order: list[SPTree[EdgeId]] = []
    visited: set[int] = set()
    stack: list[tuple[SPTree[EdgeId], bool]] = [(tree, False)]
    while stack:
        node, children_done = stack.pop()
        if id(node) in visited:
            continue
        if children_done or isinstance(node, Leaf):
            visited.add(id(node))
            order.append(node)
            continue
        stack.append((node, True))
        stack.extend((child, False) for child in reversed(list(children_of(node))))
    return order


def _canonical_post_order(tree: SPTree[EdgeId]) -> list[SPTree[EdgeId]]:
    """Post-order with parallel children in increasing order of their smallest leaf id.

    Series children keep their order. Nothing depends on hash order, so the
    result is the same in every process. Parallel children of a tree produced
    by ``reduce`` have disjoint leaf sets, so their smallest leaves differ.
    """
    smallest_leaf: dict[int, tuple[int, str, int | str]] = {}
    for node in _post_order(tree):
        if isinstance(node, Leaf):
            smallest_leaf[id(node)] = _order_key(node.edge_id)
        else:
            smallest_leaf[id(node)] = min(smallest_leaf[id(child)] for child in node.children)

    def ordered_children(node: Series[EdgeId] | Parallel[EdgeId]) -> Iterable[SPTree[EdgeId]]:
        if isinstance(node, Series):
            return node.children
        return sorted(node.children, key=lambda child: smallest_leaf[id(child)])

    return _post_order(tree, ordered_children)


def tree_records(tree: SPTree[EdgeId]) -> list[TreeRecord[EdgeId]]:
    """Post-order operation log of the tree: one ``(index, kind, child indices, interior nodes, leaf id)`` per node.

    Children come before parents, the last record is the root, parallel
    children are listed by their smallest leaf id and never in hash order. The
    log is built without recursion and is the canonical serialisable form of a
    tree: two trees are structurally equal exactly when their logs are equal.
    ``tree_from_records`` inverts it.
    """
    order = _canonical_post_order(tree)
    index_of = {id(node): index for index, node in enumerate(order)}
    records: list[TreeRecord[EdgeId]] = []
    for index, node in enumerate(order):
        if isinstance(node, Leaf):
            records.append((index, "leaf", (), (), node.edge_id))
        elif isinstance(node, Series):
            child_indices = tuple(index_of[id(child)] for child in node.children)
            records.append((index, "series", child_indices, node.interior_nodes, None))
        else:
            child_indices = tuple(sorted(index_of[id(child)] for child in node.children))
            records.append((index, "parallel", child_indices, (), None))
    return records


def tree_from_records(records: Sequence[TreeRecord[EdgeId]]) -> SPTree[EdgeId]:
    """Rebuild a tree from its ``tree_records`` log without recursion; the last record is the root."""
    nodes: list[SPTree[EdgeId]] = []
    for index, kind, child_indices, interior_nodes, edge_id in records:
        if index != len(nodes):
            message = f"record index {index} out of order, expected {len(nodes)}"
            raise ValueError(message)
        if kind == "leaf":
            nodes.append(Leaf(cast("EdgeId", edge_id)))
        elif kind == "series":
            nodes.append(Series(tuple(nodes[child] for child in child_indices), interior_nodes))
        else:
            nodes.append(Parallel(frozenset(nodes[child] for child in child_indices)))
    if not nodes:
        message = "a tree has at least one record"
        raise ValueError(message)
    return nodes[-1]


def leaves(tree: SPTree[EdgeId]) -> frozenset[EdgeId]:
    """Edge ids of the input graph represented by the tree."""
    return frozenset(node.edge_id for node in _post_order(tree) if isinstance(node, Leaf))


def paths(tree: SPTree[EdgeId], cutoff: int | None = None) -> set[frozenset[EdgeId]]:
    """Edge sets of all simple paths represented by the tree.

    Series is the product of the children, parallel the union. The cutoff
    prunes inside every product, never afterwards.
    """
    order = _post_order(tree)
    pending_parents = Counter(id(child) for node in order if not isinstance(node, Leaf) for child in node.children)
    paths_of: dict[int, set[frozenset[EdgeId]]] = {}
    for node in order:
        if isinstance(node, Leaf):
            paths_of[id(node)] = {frozenset({node.edge_id})}
            continue
        child_paths = [_release(paths_of, pending_parents, id(child)) for child in node.children]
        if isinstance(node, Parallel):
            paths_of[id(node)] = set().union(*child_paths)
            continue
        result: set[frozenset[EdgeId]] = set()
        for combination in product(*child_paths):
            joined: frozenset[EdgeId] = frozenset().union(*combination)
            if cutoff is None or len(joined) <= cutoff:
                result.add(joined)
        paths_of[id(node)] = result
    return paths_of[id(tree)]


def _release(
    paths_of: dict[int, set[frozenset[EdgeId]]],
    pending_parents: Counter[int],
    child_key: int,
) -> set[frozenset[EdgeId]]:
    """Path set of a child, dropped from the table once its last parent has consumed it."""
    pending_parents[child_key] -= 1
    if pending_parents[child_key] == 0:
        return paths_of.pop(child_key)
    return paths_of[child_key]


def _closed_states(tree: SPTree[EdgeId], edge_closed: Mapping[EdgeId, bool]) -> dict[int, bool]:
    """Closed state of every node of the tree, keyed by ``id``. Series is AND, parallel is OR."""
    state: dict[int, bool] = {}
    for node in _post_order(tree):
        if isinstance(node, Leaf):
            state[id(node)] = edge_closed[node.edge_id]
        elif isinstance(node, Series):
            state[id(node)] = all(state[id(child)] for child in node.children)
        else:
            state[id(node)] = any(state[id(child)] for child in node.children)
    return state


def closed(tree: SPTree[EdgeId], edge_closed: Mapping[EdgeId, bool]) -> bool:
    """Boolean state fold. Series is AND, parallel is OR."""
    return _closed_states(tree, edge_closed)[id(tree)]


def _toggle_order(toggle_set: frozenset[EdgeId]) -> tuple[int, list[tuple[int, str, int | str]]]:
    """Tie-break key of a candidate toggle set: fewer leaves first, then leaf id order."""
    return (len(toggle_set), sorted(map(_order_key, toggle_set)))


def minimal_toggles(
    tree: SPTree[EdgeId],
    edge_closed: Mapping[EdgeId, bool],
    *,
    target_closed: bool,
    togglable_leaves: AbstractSet[EdgeId] | None = None,
) -> frozenset[EdgeId] | None:
    """Smallest set of togglable leaves whose toggling makes the tree take the target state.

    Series to closed needs every child closed, series to open needs one child
    open and takes the cheapest. Parallel is the dual. Ties are broken by
    edge id order.

    ``togglable_leaves`` are the leaves that may be flipped; by default every
    leaf may, which is the behaviour of the fold without it. A leaf outside the
    set keeps the state ``edge_closed`` gives it, so a series node that must
    become closed is unreachable as soon as one child is, while a series node
    that must become open picks the cheapest among the reachable children only.
    Parallel is the dual.

    An empty set and ``None`` are different answers. The empty set means the
    tree already takes the target state and nothing has to be toggled. ``None``
    means no subset of ``togglable_leaves`` makes the tree take the target
    state.
    """
    state = _closed_states(tree, edge_closed)
    toggles: dict[int, frozenset[EdgeId] | None] = {}
    for node in _post_order(tree):
        if state[id(node)] == target_closed:
            toggles[id(node)] = frozenset()
        elif isinstance(node, Leaf):
            togglable = togglable_leaves is None or node.edge_id in togglable_leaves
            toggles[id(node)] = frozenset({node.edge_id}) if togglable else None
        elif isinstance(node, Series) == target_closed:
            options = [toggles[id(child)] for child in node.children]
            reachable = [option for option in options if option is not None]
            toggles[id(node)] = frozenset[EdgeId]().union(*reachable) if len(reachable) == len(options) else None
        else:
            candidates = [toggles[id(child)] for child in node.children]
            reachable = [candidate for candidate in candidates if candidate is not None]
            toggles[id(node)] = min(reachable, key=_toggle_order) if reachable else None
    return toggles[id(tree)]


@dataclass(frozen=True)
class SeriesStep(Generic[EdgeId]):
    """One position along a series chain: the node stepped from, the subtree crossed, the node reached."""

    from_node: int
    subtree: SPTree[EdgeId]
    to_node: int


def _has_chain_order(tree: SPTree[EdgeId]) -> bool:
    """A leaf and a series node run along a chain; the children of a parallel node are alternatives."""
    return not isinstance(tree, Parallel)


def _check_interior_count(node: Series[EdgeId]) -> None:
    """A series node has exactly one eliminated node between each pair of neighbouring children."""
    if len(node.interior_nodes) != len(node.children) - 1:
        message = (
            f"a series node with {len(node.children)} children needs {len(node.children) - 1} "
            f"interior nodes, got {len(node.interior_nodes)}"
        )
        raise ValueError(message)


def _chain_boundaries(
    tree: SPTree[EdgeId],
    edge_endpoints: Mapping[EdgeId, tuple[int, int]],
) -> dict[int, frozenset[int]]:
    """The two nodes every subtree spans, by node identity.

    A leaf spans the endpoints of its edge. A parallel node spans what its
    children span, which is the same pair for all of them. A series node spans
    what its children span except the nodes its merges ate, which are exactly
    its interior nodes. Anything else is not a piece of a chain and is an error.
    """
    boundaries: dict[int, frozenset[int]] = {}
    for node in _post_order(tree):
        if isinstance(node, Leaf):
            if node.edge_id not in edge_endpoints:
                message = f"leaf edge {node.edge_id!r} is not an edge of the graph the endpoints come from"
                raise ValueError(message)
            spanned = frozenset(edge_endpoints[node.edge_id])
        else:
            spanned = frozenset[int]().union(*(boundaries[id(child)] for child in node.children))
            if isinstance(node, Series):
                _check_interior_count(node)
                spanned -= frozenset(node.interior_nodes)
        if len(spanned) != 2:
            message = f"{node!r} spans {sorted(spanned)}, a subtree of a chain spans exactly two nodes"
            raise ValueError(message)
        boundaries[id(node)] = spanned
    return boundaries


def _in_walk_order(
    node: Series[EdgeId],
    step_from: int,
    boundaries: Mapping[int, frozenset[int]],
) -> tuple[tuple[SPTree[EdgeId], ...], tuple[int, ...]]:
    """The children and interior nodes of a series node in the direction that leaves ``step_from``.

    The stored order runs between the two nodes the series node spans, and its
    first child touches one of them. The walk crosses the node in stored order
    when that node is the one it steps from and against it otherwise, which is
    unambiguous because an interior node is never an endpoint.
    """
    if step_from in boundaries[id(node.children[0])]:
        return node.children, node.interior_nodes
    if step_from in boundaries[id(node.children[-1])]:
        return tuple(reversed(node.children)), tuple(reversed(node.interior_nodes))
    message = f"node {step_from} is an endpoint of no outer child of {node!r}, so the chain does not run through it"
    raise ValueError(message)


def series_chain(
    tree: SPTree[EdgeId],
    edge_endpoints: Mapping[EdgeId, tuple[int, int]],
    start_node: int,
) -> list[SeriesStep[EdgeId]]:
    """Ordered steps along a series chain, walked from ``start_node``, one step per position.

    ``edge_endpoints`` is the endpoint mapping of the graph the tree was
    reduced from, which is ``MultiGraph.endpoints`` of that graph; every leaf of
    the tree is one of its edges. The nodes a subtree spans follow from it, and
    with them the direction each child runs, which the tree itself does not
    record: the stored order of a series node is the direction of the merge that
    created it and a later merge can cross it either way. A ``Leaf`` is a chain
    of one step.

    A series node nested in a series node is a sub-chain and is flattened into
    the sequence, so index ``i`` addresses the ``i``-th subtree along the whole
    chain and the length is the number of positions on it. A ``Parallel`` child
    is one position: its children carry no order, so the step names the whole
    parallel subtree.

    Walking from the other endpoint returns the reversed sequence with every
    step reversed, so the two walks of a chain are mutual reverses.

    Raises ``ValueError`` for a ``Parallel`` tree, whose children have no order,
    for a start node that is not an endpoint of the tree, for a leaf that is not
    an edge of the graph, for a subtree that does not span exactly two nodes and
    for a series node whose interior nodes do not number one fewer than its
    children, which leaves the chain ambiguous.
    """
    if not _has_chain_order(tree):
        message = "a parallel tree has no chain order: its children are unordered alternatives, not a sequence"
        raise ValueError(message)
    boundaries = _chain_boundaries(tree, edge_endpoints)
    spanned = boundaries[id(tree)]
    if start_node not in spanned:
        endpoint_names = " and ".join(str(node_id) for node_id in sorted(spanned))
        message = f"start node {start_node} is not an endpoint of the tree, the endpoints are {endpoint_names}"
        raise ValueError(message)
    (end_node,) = spanned - {start_node}
    steps: list[SeriesStep[EdgeId]] = []
    pending: list[tuple[SPTree[EdgeId], int, int]] = [(tree, start_node, end_node)]
    while pending:
        subtree, step_from, step_to = pending.pop()
        if not isinstance(subtree, Series):
            steps.append(SeriesStep(step_from, subtree, step_to))
            continue
        children, interior_nodes = _in_walk_order(subtree, step_from, boundaries)
        chain_nodes = [step_from, *interior_nodes, step_to]
        pending.extend(
            (child, chain_nodes[position], chain_nodes[position + 1])
            for position, child in reversed(list(enumerate(children)))
        )
    return steps


def _interior_nodes(tree: SPTree[EdgeId]) -> list[int]:
    """Every node eliminated into the tree by a series move, in increasing id."""
    return sorted(node_id for node in _post_order(tree) if isinstance(node, Series) for node_id in node.interior_nodes)


# ---------------------------------------------------------------------------
# Terminal-preserving reduction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reduced(Generic[EdgeId]):
    """Residual multigraph, the provenance tree of every residual edge and the folded node material.

    ``folded_nodes`` holds, per surviving node, the pendant material folded
    into it. ``folded_interior`` holds, per node recorded in a series node of a
    residual tree, the material folded into it before its series move. With
    ``fold_leaves`` every eliminated node of a component with a terminal
    appears exactly once: as a series interior node, in ``folded_nodes`` or in
    ``folded_interior``. Without it pendant material is dropped.

    ``dropped`` lists, per pendant move in processing order, the neighbour the
    pendant hung on and the provenance tree of the removed edge, so edge
    material merged before its attachment became pendant (a ring returning to
    one node) stays available. Edges of terminal-free components are not
    listed: no node absorbs them.
    """

    graph: MultiGraph[EdgeId | VirtualEdgeId]
    provenance: dict[EdgeId | VirtualEdgeId, SPTree[EdgeId]]
    folded_nodes: dict[int, list[int]]
    folded_interior: dict[int, list[int]]
    dropped: list[tuple[int, SPTree[EdgeId]]]


def _other_end(pair: tuple[int, int], node_id: int) -> int:
    from_node, to_node = pair
    return to_node if from_node == node_id else from_node


class _Reduction(Generic[EdgeId]):
    """Mutable state of one reduction run.

    Incidences are kept per node and the non-loop edges per endpoint pair, both
    updated by every move, so degrees and parallel candidates are read, never
    recomputed. Candidates wait in a min-heap keyed by processing rank and are
    re-validated when popped, which processes the first eligible node exactly
    like a full rescan would, at logarithmic cost per move.
    """

    def __init__(
        self,
        graph: MultiGraph[EdgeId],
        terminals: AbstractSet[int],
        protected: AbstractSet[int],
        *,
        fold_leaves: bool,
        order: Sequence[int] | None,
    ) -> None:
        self.graph = graph
        self.terminals = terminals
        self.protected = protected
        self.fold_leaves = fold_leaves
        self.rank: dict[int, int] = {} if order is None else {node_id: rank for rank, node_id in enumerate(order)}
        self.unranked = len(self.rank)
        self.endpoints: dict[EdgeId | VirtualEdgeId, tuple[int, int]] = {}
        self.provenance: dict[EdgeId | VirtualEdgeId, SPTree[EdgeId]] = {}
        self.incident: dict[int, set[EdgeId | VirtualEdgeId]] = {node_id: set() for node_id in graph.nodes}
        self.pair_edges: dict[tuple[int, int], set[EdgeId | VirtualEdgeId]] = {}
        self.loops: dict[int, list[EdgeId]] = {}
        self.folded_nodes: dict[int, list[int]] = {node_id: [] for node_id in graph.nodes}
        self.folded_interior: dict[int, list[int]] = {}
        self.dropped: list[tuple[int, SPTree[EdgeId]]] = []
        self.alive = set(graph.nodes)
        virtual_indices = (edge_id.index for edge_id in graph.endpoints if isinstance(edge_id, VirtualEdgeId))
        self.fresh = count(1 + max(virtual_indices, default=0))
        for edge_id in sorted(graph.endpoints, key=_order_key):
            from_node, to_node = graph.endpoints[edge_id]
            self.endpoints[edge_id] = (from_node, to_node)
            self.provenance[edge_id] = Leaf(edge_id)
            if from_node == to_node:
                self.loops.setdefault(from_node, []).append(edge_id)
            else:
                self._index_edge(edge_id, (from_node, to_node))

    def run(self) -> Reduced[EdgeId]:
        self._drop_terminal_free_components()
        self._merge_all_parallels()
        candidates = [self._priority(node_id) for node_id in self.alive if node_id not in self.terminals]
        heapq.heapify(candidates)
        while candidates:
            _rank, node_id = heapq.heappop(candidates)
            if node_id not in self.alive:
                continue
            for touched in self._move(node_id):
                if touched not in self.terminals:
                    heapq.heappush(candidates, self._priority(touched))
        residual: MultiGraph[EdgeId | VirtualEdgeId] = MultiGraph(sorted(self.alive), self.endpoints)
        return Reduced(residual, self.provenance, self.folded_nodes, self.folded_interior, self.dropped)

    def _priority(self, node_id: int) -> tuple[int, int]:
        """Heap key: the caller's rank if given (unlisted nodes last), then the node id."""
        return (self.rank.get(node_id, self.unranked), node_id)

    def _drop_terminal_free_components(self) -> None:
        """A component without a terminal carries no question and is removed whole."""
        components = Partition.from_components(self.graph, self.graph.endpoints.keys())
        terminal_blocks = {components.block_of[node_id] for node_id in self.terminals}
        for node_id in self.graph.nodes:
            if components.block_of[node_id] not in terminal_blocks:
                for edge_id in list(self.incident[node_id]):
                    self._remove_edge(edge_id)
                self.folded_nodes.pop(node_id)
                self._eliminate(node_id)

    def _move(self, node_id: int) -> tuple[int, ...]:
        """Apply the pendant or series move at the node if one applies; return the nodes whose incidences changed."""
        incident = self.incident[node_id]
        if len(incident) == 1:
            (edge_id,) = incident
            return (self._pendant(node_id, edge_id),)
        if len(incident) == 2 and node_id not in self.protected:
            first, second = sorted(incident, key=_order_key)
            neighbour_first = _other_end(self.endpoints[first], node_id)
            neighbour_second = _other_end(self.endpoints[second], node_id)
            if neighbour_first != neighbour_second:
                return self._series(node_id, first, second, neighbour_first, neighbour_second)
        return ()

    def _pendant(self, node_id: int, edge_id: EdgeId | VirtualEdgeId) -> int:
        """Remove the pendant node and its edge; with folding, its material moves to the neighbour.

        The material is the node, what was folded into it, and every interior
        node of the removed edge's tree with what was folded into that. The
        tree itself is reported in ``dropped``.
        """
        neighbour = _other_end(self.endpoints[edge_id], node_id)
        tree = self.provenance[edge_id]
        interior = _interior_nodes(tree)
        self._remove_edge(edge_id)
        self.dropped.append((neighbour, tree))
        absorbed = [node_id, *self.folded_nodes.pop(node_id)]
        for interior_node in interior:
            absorbed.extend([interior_node, *self.folded_interior.pop(interior_node)])
        if self.fold_leaves:
            self.folded_nodes[neighbour].extend(absorbed)
        self._eliminate(node_id)
        return neighbour

    def _series(
        self,
        node_id: int,
        first: EdgeId | VirtualEdgeId,
        second: EdgeId | VirtualEdgeId,
        neighbour_first: int,
        neighbour_second: int,
    ) -> tuple[int, int]:
        tree: SPTree[EdgeId] = Series((self.provenance[first], self.provenance[second]), (node_id,))
        self._remove_edge(first)
        self._remove_edge(second)
        self._add_edge(tree, (neighbour_first, neighbour_second))
        self.folded_interior[node_id] = self.folded_nodes.pop(node_id)
        self._eliminate(node_id)
        if neighbour_first not in self.protected and neighbour_second not in self.protected:
            parallel = self.pair_edges[_sorted_pair(neighbour_first, neighbour_second)]
            if len(parallel) >= 2:
                self._merge_parallel(neighbour_first, neighbour_second, sorted(parallel, key=_order_key))
        return neighbour_first, neighbour_second

    def _merge_all_parallels(self) -> None:
        """Initial parallel sweep, pairs in order of their first edge."""
        candidates = [
            (pair, sorted(edge_ids, key=_order_key))
            for pair, edge_ids in self.pair_edges.items()
            if len(edge_ids) >= 2 and pair[0] not in self.protected and pair[1] not in self.protected
        ]
        for (from_node, to_node), edge_ids in candidates:
            self._merge_parallel(from_node, to_node, edge_ids)

    def _merge_parallel(self, from_node: int, to_node: int, edge_ids: list[EdgeId | VirtualEdgeId]) -> None:
        tree: SPTree[EdgeId] = Parallel(frozenset(self.provenance[edge_id] for edge_id in edge_ids))
        for edge_id in edge_ids:
            self._remove_edge(edge_id)
        self._add_edge(tree, _sorted_pair(from_node, to_node))

    def _add_edge(self, tree: SPTree[EdgeId], pair: tuple[int, int]) -> None:
        edge_id = VirtualEdgeId(next(self.fresh))
        self.endpoints[edge_id] = pair
        self.provenance[edge_id] = tree
        self._index_edge(edge_id, pair)

    def _index_edge(self, edge_id: EdgeId | VirtualEdgeId, pair: tuple[int, int]) -> None:
        """Register a non-loop edge in the incidence sets and in the pair index."""
        from_node, to_node = pair
        self.incident[from_node].add(edge_id)
        self.incident[to_node].add(edge_id)
        self.pair_edges.setdefault(_sorted_pair(from_node, to_node), set()).add(edge_id)

    def _remove_edge(self, edge_id: EdgeId | VirtualEdgeId) -> None:
        """Unregister a non-loop edge; loops leave with their node in ``_eliminate``."""
        pair = self.endpoints.pop(edge_id)
        del self.provenance[edge_id]
        for node_id in pair:
            self.incident[node_id].discard(edge_id)
        key = _sorted_pair(*pair)
        between = self.pair_edges[key]
        between.discard(edge_id)
        if not between:
            del self.pair_edges[key]

    def _eliminate(self, node_id: int) -> None:
        """Remove the node together with its self-loops."""
        for loop_id in self.loops.pop(node_id, []):
            del self.endpoints[loop_id]
            del self.provenance[loop_id]
        del self.incident[node_id]
        self.alive.discard(node_id)


def _reduce_python(
    graph: MultiGraph[EdgeId],
    terminals: AbstractSet[int],
    protected: AbstractSet[int],
    *,
    fold_leaves: bool,
    order: Sequence[int] | None,
) -> Reduced[EdgeId]:
    """The worklist over candidate nodes, in Python. The reference semantics of ``reduce``."""
    return _Reduction(graph, terminals, protected, fold_leaves=fold_leaves, order=order).run()


_OPERATION_LEAF = 0
_OPERATION_SERIES = 1
_OPERATION_PARALLEL = 2
_OPERATION_PENDANT = 3


def _structural_log_c(
    kernel: _KernelGraph[EdgeId],
    terminals: AbstractSet[int],
    protected: AbstractSet[int],
) -> pygraphc.ReductionLog:
    """The structural half of the ``"c"`` engine: one crossing for the whole fixpoint loop.

    C receives the cached compressed sparse row graph plus a terminal and a
    protected byte mask over node indices and returns a flat operation log.
    It is a function of its own so that the structural work and the fold over
    its log can be called, and therefore measured, apart.
    """
    terminal_mask = bytearray(len(kernel.nodes))
    protected_mask = bytearray(len(kernel.nodes))
    for node_id in terminals:
        terminal_mask[bisect_left(kernel.nodes, node_id)] = 1
    for node_id in protected:
        protected_mask[bisect_left(kernel.nodes, node_id)] = 1
    return kernel.graph.series_parallel_reduce(terminal_mask, protected_mask)


def _reduce_c(
    graph: MultiGraph[EdgeId],
    terminals: AbstractSet[int],
    protected: AbstractSet[int],
    *,
    fold_leaves: bool,
) -> Reduced[EdgeId]:
    """The same reduction with the structural loop in C and the payload algebra folded here.

    The fold turns the operation log into the very ``Leaf``/``Series``/
    ``Parallel`` trees and the same bookkeeping the Python worklist builds.
    """
    kernel = graph._kernel
    log = _structural_log_c(kernel, terminals, protected)
    return _fold_operation_log(graph, kernel, log, fold_leaves=fold_leaves)


def _fold_operation_log(
    graph: MultiGraph[EdgeId],
    kernel: _KernelGraph[EdgeId],
    log: pygraphc.ReductionLog,
    *,
    fold_leaves: bool,
) -> Reduced[EdgeId]:
    """Build the provenance trees and the folded node material from one pass over the log.

    Every leaf precedes every move, so the leaves are one comprehension and
    the loop runs over the moves only. Moves come in the order the worklist
    applies them, so replaying them reproduces ``folded_nodes``,
    ``folded_interior`` and ``dropped`` exactly, and the virtual edge ids
    handed to the series and parallel operations that produce an edge fall in
    the order the worklist creates them.

    A series operation carries the interior nodes of its subtree along, always
    extending the longer of the two child lists, so a pendant move reads them
    instead of walking its subtree and a chain of series moves stays linear.
    """
    nodes, edge_ids = kernel.nodes, kernel.edge_ids
    kinds = log.op_kind.tolist()
    leaf_edges = log.leaf_edge_index.tolist()
    leaf_count = kinds.count(_OPERATION_LEAF)
    trees: dict[int, SPTree[EdgeId]] = {
        operation: Leaf(edge_ids[index]) for operation, index in enumerate(leaf_edges[:leaf_count])
    }
    moves = zip(
        kinds[leaf_count:],
        log.left[leaf_count:].tolist(),
        log.right[leaf_count:].tolist(),
        log.endpoint_u[leaf_count:].tolist(),
        log.interior_node[leaf_count:].tolist(),
        log.absorber[leaf_count:].tolist(),
        strict=True,
    )

    chain_children: dict[int, list[SPTree[EdgeId]]] = {}
    subtree_interior: dict[int, list[int]] = {}
    virtual_id_of: dict[int, VirtualEdgeId] = {}
    folded_nodes: dict[int, list[int]] = {}
    folded_interior: dict[int, list[int]] = {}
    dropped: list[tuple[int, SPTree[EdgeId]]] = []
    highest = edge_ids[-1] if edge_ids else None
    virtual_ids = count(1 + (highest.index if isinstance(highest, VirtualEdgeId) else 0))

    for operation, (kind, first, second, endpoint, interior_index, absorber) in enumerate(moves, start=leaf_count):
        if kind == _OPERATION_SERIES:
            interior_node = nodes[interior_index]
            trees[operation] = Series((trees[first], trees[second]), (interior_node,))
            carried = _joined(subtree_interior.pop(first, None), subtree_interior.pop(second, None))
            carried.append(interior_node)
            subtree_interior[operation] = carried
            folded_interior[interior_node] = folded_nodes.pop(interior_node, [])
            virtual_id_of[operation] = VirtualEdgeId(next(virtual_ids))
        elif kind == _OPERATION_PARALLEL:
            partial = chain_children.pop(first, None)
            children = [trees[first]] if partial is None else partial
            children.append(trees[second])
            subtree_interior[operation] = _joined(subtree_interior.pop(first, None), subtree_interior.pop(second, None))
            if endpoint < 0:
                chain_children[operation] = children
            else:
                trees[operation] = Parallel(frozenset(children))
                virtual_id_of[operation] = VirtualEdgeId(next(virtual_ids))
        else:
            tree = trees[first]
            removed_node, neighbour = nodes[interior_index], nodes[absorber]
            dropped.append((neighbour, tree))
            absorbed = [removed_node, *folded_nodes.pop(removed_node, ())]
            for interior_node in sorted(subtree_interior.pop(first, ())):
                absorbed.extend([interior_node, *folded_interior.pop(interior_node)])
            if not fold_leaves:
                continue
            material = folded_nodes.get(neighbour)
            if material is None:
                folded_nodes[neighbour] = absorbed
            else:
                material.extend(absorbed)

    residual, provenance = _residual_from_log(kernel, log, trees, virtual_id_of, leaf_edges, leaf_count)
    kept = {node_id: folded_nodes.pop(node_id, []) for node_id in residual.nodes}
    return Reduced(residual, provenance, kept, folded_interior, dropped)


def _residual_from_log(
    kernel: _KernelGraph[EdgeId],
    log: pygraphc.ReductionLog,
    trees: Mapping[int, SPTree[EdgeId]],
    virtual_id_of: Mapping[int, VirtualEdgeId],
    leaf_edges: Sequence[int],
    leaf_count: int,
) -> tuple[MultiGraph[EdgeId | VirtualEdgeId], dict[EdgeId | VirtualEdgeId, SPTree[EdgeId]]]:
    """The surviving nodes and the endpoints and provenance tree of every surviving edge.

    Surviving edges come in slot order, which is the order the worklist leaves
    them in. An operation below ``leaf_count`` is a leaf and its edge keeps its
    input edge id; every other surviving edge is one a move produced.
    """
    nodes, edge_ids = kernel.nodes, kernel.edge_ids
    endpoints: dict[EdgeId | VirtualEdgeId, tuple[int, int]] = {}
    provenance: dict[EdgeId | VirtualEdgeId, SPTree[EdgeId]] = {}
    for operation, from_index, to_index in zip(
        log.residual_op.tolist(), log.residual_u.tolist(), log.residual_v.tolist(), strict=True
    ):
        edge_id: EdgeId | VirtualEdgeId = (
            edge_ids[leaf_edges[operation]] if operation < leaf_count else virtual_id_of[operation]
        )
        endpoints[edge_id] = (nodes[from_index], nodes[to_index])
        provenance[edge_id] = trees[operation]
    surviving = [nodes[index] for index in log.surviving_nodes.tolist()]
    return MultiGraph(surviving, endpoints), provenance


def _joined(first: list[int] | None, second: list[int] | None) -> list[int]:
    """Union of two subtree interior node lists, extending the longer one so a chain stays linear.

    Order does not matter: the pendant move that reads the list sorts it.
    """
    if first is None:
        return [] if second is None else second
    if second is None:
        return first
    if len(first) >= len(second):
        first.extend(second)
        return first
    second.extend(first)
    return second


def reduce(
    graph: MultiGraph[EdgeId],
    terminals: AbstractSet[int],
    protected: AbstractSet[int] = frozenset(),
    *,
    fold_leaves: bool = True,
    order: Sequence[int] | None = None,
    engine: Literal["c", "python"] = "c",
) -> Reduced[EdgeId]:
    """Closure under the three simple reductions with a terminal set.

    Components without a terminal are removed whole first. Pendant deletion
    removes a non-terminal with exactly one non-loop incidence and, with
    ``fold_leaves``, records its material on the neighbour; ``protected``
    does not block that move, only the series and parallel merges. Series merge
    replaces a non-terminal, non-protected node with exactly two non-loop
    incidences to two distinct neighbours by one edge (loops do not count).
    Parallel merge replaces the edges between one endpoint pair, neither
    protected, by one edge. Self-loops take part in no move and leave with
    their node.

    Candidates are processed in increasing node id, or in the given ``order``
    (unlisted nodes last). With no protected nodes the residual and the folded
    material do not depend on that order; protected nodes can make them
    order dependent. Terminals and protected nodes must be nodes of the
    graph. The residual of a reduction is a fixpoint: reducing it again with
    the same terminals and protected nodes changes nothing.

    The ``"c"`` engine runs the structural loop in the C tier and folds its
    operation log here; the ``"python"`` engine runs the worklist in Python.
    Both produce the same ``Reduced``. A caller-supplied ``order`` is a
    Python-engine feature and selects it whatever ``engine`` says, since the
    C work queue is fixed to increasing node index.
    """
    node_set = set(graph.nodes)
    unknown = sorted(node_id for node_id in {*terminals, *protected} if node_id not in node_set)
    if unknown:
        message = f"terminals and protected nodes must be nodes of the graph, unknown: {unknown}"
        raise ValueError(message)
    if engine == "python" or order is not None:
        return _reduce_python(graph, terminals, protected, fold_leaves=fold_leaves, order=order)
    return _reduce_c(graph, terminals, protected, fold_leaves=fold_leaves)


# ---------------------------------------------------------------------------
# Scenario
# ---------------------------------------------------------------------------


def scenario(
    graph: MultiGraph[EdgeId],
    active: AbstractSet[EdgeId],
    removed: AbstractSet[EdgeId],
) -> Partition:
    """Apply a scenario by masking the removed edges and re-partitioning.

    No bridge assumption: parallel edges failing together split the block
    exactly like a single bridge would.
    """
    return Partition.from_components(graph, active - removed)
