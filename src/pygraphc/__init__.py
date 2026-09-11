"""Fast graph algorithms via C extensions: union-find, Tarjan's, BFS, Dijkstra."""

from __future__ import annotations

import types
from collections import deque
from collections.abc import Collection, Generator, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Generic, NewType, TypeVar, overload

from pygraphc._core import all_edge_paths_ctx as _all_edge_paths_ctx
from pygraphc._core import ap_ctx as _ap_ctx
from pygraphc._core import ap_nid as _ap_nid
from pygraphc._core import bcc_ctx as _bcc_ctx
from pygraphc._core import bcc_edge_labels_ctx as _bcc_edge_labels_ctx
from pygraphc._core import bcc_nid as _bcc_nid
from pygraphc._core import bfs_ctx as _bfs_ctx
from pygraphc._core import bfs_nid as _bfs_nid
from pygraphc._core import bridges_as_edge_indices_ctx as _bridges_as_edge_indices_ctx
from pygraphc._core import bridges_ctx as _bridges_ctx
from pygraphc._core import bridges_nid as _bridges_nid
from pygraphc._core import cc_branches_ctx as _cc_branches_ctx
from pygraphc._core import cc_ctx as _cc_ctx
from pygraphc._core import cc_nid as _cc_nid
from pygraphc._core import cc_nid_split as _cc_nid_split
from pygraphc._core import component_labels_ctx as _component_labels_ctx
from pygraphc._core import connected_components_with_branches_remapped as _cc_branches_remapped
from pygraphc._core import cycle_basis_ctx as _cycle_basis_ctx
from pygraphc._core import dag_longest_path_ctx as _dag_longest_path_ctx
from pygraphc._core import degree_ctx as _degree_ctx
from pygraphc._core import degrees_ctx as _degrees_ctx
from pygraphc._core import dijkstra_ctx as _dijkstra_ctx
from pygraphc._core import dijkstra_nid as _dijkstra_nid
from pygraphc._core import edge_indices_ctx as _edge_indices_ctx
from pygraphc._core import graph_edge_count as _graph_edge_count
from pygraphc._core import graph_node_count as _graph_node_count
from pygraphc._core import in_degree_ctx as _in_degree_ctx
from pygraphc._core import incident_edges_ctx as _incident_edges_ctx
from pygraphc._core import incoming_edges_ctx as _incoming_edges_ctx
from pygraphc._core import msdijk_ctx as _msdijk_ctx
from pygraphc._core import msdijk_nid as _msdijk_nid
from pygraphc._core import neighbors_ctx as _neighbors_ctx
from pygraphc._core import parse_graph as _parse_graph
from pygraphc._core import predecessors_ctx as _predecessors_ctx
from pygraphc._core import quotient_edges_ctx as _quotient_edges_ctx
from pygraphc._core import scc_ctx as _scc_ctx
from pygraphc._core import series_parallel_reduce_ctx as _series_parallel_reduce_ctx
from pygraphc._core import sp_apply_batch as _sp_apply_batch
from pygraphc._core import sp_apply_move as _sp_apply_move
from pygraphc._core import sp_apply_parallel as _sp_apply_parallel
from pygraphc._core import sp_batch_moves as _sp_batch_moves
from pygraphc._core import sp_next_move as _sp_next_move
from pygraphc._core import sp_pair_edges as _sp_pair_edges
from pygraphc._core import sp_state_free as _sp_state_free
from pygraphc._core import sp_state_log as _sp_state_log
from pygraphc._core import sp_state_new as _sp_state_new
from pygraphc._core import sssp_ctx as _sssp_ctx
from pygraphc._core import sssp_nid as _sssp_nid
from pygraphc._core import toposort_ctx as _toposort_ctx
from pygraphc._core import toposort_nid as _toposort_nid
from pygraphc._dag_learn import estimate_cpds as _estimate_cpds
from pygraphc._dag_learn import hill_climb_k2 as _hill_climb_k2
from pygraphc._dag_learn import k2_local_score as _k2_local_score
from pygraphc.reduction import (
    Leaf,
    MultiGraph,
    Parallel,
    Partition,
    PendantAction,
    PendantPolicy,
    Reduced,
    Series,
    SeriesStep,
    SPTree,
    TreeRecord,
    VirtualEdgeId,
    closed,
    leaves,
    lift,
    minimal_toggles,
    paths,
    quotient,
    reduce,
    scenario,
    series_chain,
    tree_from_records,
    tree_records,
)

__all__ = [
    "BranchId",
    "BranchIdT",
    "EdgeIndex",
    "Graph",
    "GraphView",
    "Leaf",
    "MultiGraph",
    "NodeId",
    "NodeIdT",
    "NodeMask",
    "Parallel",
    "Partition",
    "PendantAction",
    "PendantPolicy",
    "Reduced",
    "ReductionLog",
    "ReductionState",
    "SPTree",
    "Series",
    "SeriesStep",
    "TreeRecord",
    "VirtualEdgeId",
    "articulation_points",
    "bfs",
    "biconnected_components",
    "bridges",
    "closed",
    "connected_components",
    "connected_components_with_branch_ids",
    "cycle_basis",
    "dag_longest_path",
    "eccentricity",
    "estimate_cpds",
    "for_each_edge_excluded",
    "hill_climb_k2",
    "k2_local_score",
    "leaves",
    "lift",
    "minimal_toggles",
    "multi_source_shortest_path_lengths",
    "nodes_on_simple_paths",
    "paths",
    "quotient",
    "reduce",
    "scenario",
    "series_chain",
    "shortest_path",
    "shortest_path_lengths",
    "strongly_connected_components",
    "topological_sort",
    "tree_from_records",
    "tree_records",
    "two_edge_connected_components",
    "weakly_connected_components",
]

NodeIdT = TypeVar("NodeIdT", bound=int)
"""Node id type of a graph: any ``int`` or ``int``-based ``NewType`` the caller passes in."""

BranchIdT = TypeVar("BranchIdT", bound=int)
"""Branch id type of a graph, inferred from ``branch_ids``; ``int`` when none are given."""

EdgeIndex = NewType("EdgeIndex", int)
"""Position of an edge in the sequence passed to ``Graph()``. Not a branch id."""

NodeId = int
"""Backwards-compatible alias; new code should parameterize ``Graph`` instead."""

BranchId = int
"""Backwards-compatible alias; new code should parameterize ``Graph`` instead."""

NodeMask = bytes | bytearray | memoryview
"""One byte per node index; a non-zero byte marks membership. Read as a buffer, never iterated."""


# ── Connected Components (legacy index-based API kept for branch_ids) ──


def connected_components(
    node_ids: Sequence[NodeIdT],
    edges_or_src: Sequence[tuple[NodeIdT, NodeIdT]] | Sequence[NodeIdT],
    dst: Sequence[NodeIdT] | None = None,
) -> Generator[set[NodeIdT], None, None]:
    """
    Yield each connected component as a set of original node IDs.

    Two calling conventions:
        connected_components(node_ids, edges)       — edges as pairs of node IDs
        connected_components(node_ids, src, dst)     — two flat lists of node IDs
    """
    if dst is not None:
        yield from _cc_nid_split(node_ids, edges_or_src, dst)
    else:
        yield from _cc_nid(node_ids, edges_or_src)


def connected_components_with_branch_ids(
    node_ids: Sequence[NodeIdT],
    edges: Sequence[tuple[NodeIdT, NodeIdT]],
    branch_ids: Sequence[BranchIdT],
) -> Generator[tuple[set[NodeIdT], set[BranchIdT]], None, None]:
    """
    Yield (node_id_set, branch_id_set) with original node IDs.

    Edges are pairs of original node IDs (not indices).
    """
    # Branch variant still uses index-based edges internally
    idx = {nid: i for i, nid in enumerate(node_ids)}
    idx_edges = [(idx[u], idx[v]) for u, v in edges]
    yield from _cc_branches_remapped(node_ids, idx_edges, branch_ids)


def cycle_basis(
    node_ids: Sequence[NodeIdT],
    edges: Sequence[tuple[NodeIdT, NodeIdT]],
) -> list[list[NodeIdT]]:
    """Return a fundamental cycle basis as a list of cycles.

    Each cycle is a list of node IDs. The number of fundamental cycles
    equals m - n + c (circuit rank), where c is the number of connected
    components.
    """
    graph = Graph(node_ids, edges)
    return graph.cycle_basis()


def dag_longest_path(
    node_ids: Sequence[NodeIdT],
    edges: Sequence[tuple[NodeIdT, NodeIdT]],
    weights: list[float] | None = None,
) -> list[NodeIdT]:
    """Return the longest path in a DAG as a list of node IDs.

    Edges are treated as directed: (u, v) means u -> v.
    If weights are provided, edge weights determine path length.
    Raises ValueError if the graph contains a cycle.
    """
    graph = Graph(node_ids, edges, directed=True)
    return graph.dag_longest_path(weights)


# ── Phase 1: Structural graph primitives ──


def bridges(
    node_ids: Sequence[NodeIdT],
    edges: Sequence[tuple[NodeIdT, NodeIdT]],
) -> list[tuple[NodeIdT, NodeIdT]]:
    """Return bridge edges as (node_id, node_id) pairs."""
    result: list[tuple[NodeIdT, NodeIdT]] = _bridges_nid(node_ids, edges)
    return result


def articulation_points(
    node_ids: Sequence[NodeIdT],
    edges: Sequence[tuple[NodeIdT, NodeIdT]],
) -> set[NodeIdT]:
    """Return the set of articulation points."""
    result: set[NodeIdT] = _ap_nid(node_ids, edges)
    return result


def biconnected_components(
    node_ids: Sequence[NodeIdT],
    edges: Sequence[tuple[NodeIdT, NodeIdT]],
) -> Generator[set[NodeIdT], None, None]:
    """Yield each biconnected component as a set of node IDs."""
    yield from _bcc_nid(node_ids, edges)


def bfs(
    node_ids: Sequence[NodeIdT],
    edges: Sequence[tuple[NodeIdT, NodeIdT]],
    source: NodeIdT,
) -> list[NodeIdT]:
    """Return nodes visited in BFS order from source."""
    result: list[NodeIdT] = _bfs_nid(node_ids, edges, source)
    return result


# ── Phase 2: Weighted graph algorithms ──


def shortest_path(
    node_ids: Sequence[NodeIdT],
    edges: Sequence[tuple[NodeIdT, NodeIdT]],
    weights: list[float],
    source: NodeIdT,
    target: NodeIdT,
) -> list[NodeIdT]:
    """Return the shortest weighted path from source to target.

    Runs a bidirectional Dijkstra, so only a small part of a large graph is
    settled. Pass ``weights`` as a float64 buffer (a numpy ``float64`` array or
    ``array.array("d", ...)``) to hand it to C as is; a list of floats is
    converted element by element first, which costs more than the search.
    """
    _dist, path = _dijkstra_nid(node_ids, edges, weights, source, target)
    result: list[NodeIdT] = path
    return result


def shortest_path_lengths(
    node_ids: Sequence[NodeIdT],
    edges: Sequence[tuple[NodeIdT, NodeIdT]],
    weights: list[float],
    source: NodeIdT,
    cutoff: float | None = None,
) -> dict[NodeIdT, float]:
    """Return {node_id: distance} for all nodes reachable from source."""
    c = cutoff if cutoff is not None else -1.0
    result: dict[NodeIdT, float] = _sssp_nid(node_ids, edges, weights, source, c)
    return result


def multi_source_shortest_path_lengths(
    node_ids: Sequence[NodeIdT],
    edges: Sequence[tuple[NodeIdT, NodeIdT]],
    weights: list[float],
    sources: Sequence[NodeIdT],
    cutoff: float | None = None,
) -> dict[NodeIdT, float]:
    """Return {node_id: distance} from nearest source to each reachable node."""
    c = cutoff if cutoff is not None else -1.0
    result: dict[NodeIdT, float] = _msdijk_nid(node_ids, edges, weights, sources, c)
    return result


def eccentricity(
    node_ids: Sequence[NodeIdT],
    edges: Sequence[tuple[NodeIdT, NodeIdT]],
    weights: list[float],
    source: NodeIdT,
) -> float:
    """Return the eccentricity of source (max shortest-path distance)."""
    lengths = shortest_path_lengths(node_ids, edges, weights, source)
    if not lengths:
        return 0.0
    return max(lengths.values())


# ── Phase 3: Composite algorithms ──


def two_edge_connected_components(
    node_ids: Sequence[NodeIdT],
    edges: Sequence[tuple[NodeIdT, NodeIdT]],
) -> Generator[set[NodeIdT], None, None]:
    """Yield 2-edge-connected components (bridges removed, then CC)."""
    bridge_set: set[tuple[NodeIdT, NodeIdT]] = set()
    for u, v in bridges(node_ids, edges):
        bridge_set.add((min(u, v), max(u, v)))

    non_bridge_edges = [(u, v) for u, v in edges if (min(u, v), max(u, v)) not in bridge_set]
    yield from connected_components(node_ids, non_bridge_edges)


# ── Directed graph algorithms ──


def topological_sort(
    node_ids: Sequence[NodeIdT],
    edges: Sequence[tuple[NodeIdT, NodeIdT]],
) -> list[NodeIdT]:
    """Return nodes in topological order (Kahn's algorithm, C implementation).

    Edges are treated as directed: (u, v) means u -> v.
    Raises ValueError if the graph contains a cycle.
    """
    result: list[NodeIdT] = _toposort_nid(node_ids, edges)
    return result


def strongly_connected_components(
    node_ids: Sequence[NodeIdT],
    edges: Sequence[tuple[NodeIdT, NodeIdT]],
) -> Generator[set[NodeIdT], None, None]:
    """Yield each strongly connected component as a set of node IDs.

    Edges are treated as directed: (u, v) means u -> v.
    """
    graph = Graph(node_ids, edges, directed=True)
    yield from graph.strongly_connected_components()


def weakly_connected_components(
    node_ids: Sequence[NodeIdT],
    edges: Sequence[tuple[NodeIdT, NodeIdT]],
) -> Generator[set[NodeIdT], None, None]:
    """Yield each weakly connected component as a set of node IDs.

    Edges are treated as directed but direction is ignored for connectivity.
    """
    graph = Graph(node_ids, edges, directed=True)
    yield from graph.weakly_connected_components()


def nodes_on_simple_paths(
    node_ids: Sequence[NodeIdT],
    edges: Sequence[tuple[NodeIdT, NodeIdT]],
    source: NodeIdT,
    targets: Sequence[NodeIdT],
) -> set[NodeIdT]:
    """Return all nodes on any simple path from source to any target.

    Uses the block-cut tree: finds biconnected components, builds the
    block-cut tree, then collects all nodes in blocks on the tree path
    from source to each target.
    """
    n = len(node_ids)
    if n == 0:
        return set()

    tgts = set(targets)
    result: set[NodeIdT] = set()
    if source in tgts:
        result.add(source)
        tgts.discard(source)
    if not tgts:
        return result

    blocks = list(biconnected_components(node_ids, edges))
    if not blocks:
        return result

    tree = _build_block_cut_tree(node_ids, blocks)
    return _collect_path_nodes(
        node_ids,
        blocks,
        tree,
        source,
        tgts,
        result,
    )


def _build_block_cut_tree(
    node_ids: Sequence[NodeIdT],
    blocks: list[set[NodeIdT]],
) -> tuple[dict[NodeIdT, list[int]], list[list[int]], dict[NodeIdT, int]]:
    """Build block-cut tree from biconnected components.

    Returns (node_blocks, tree_adj, ap_id).
    """
    num_blocks = len(blocks)
    node_blocks: dict[NodeIdT, list[int]] = {}
    for bi, block in enumerate(blocks):
        for v in block:
            node_blocks.setdefault(v, []).append(bi)

    ap_id: dict[NodeIdT, int] = {}
    next_id = num_blocks
    for v, blks in node_blocks.items():
        if len(blks) > 1:
            ap_id[v] = next_id
            next_id += 1

    tree_adj: list[list[int]] = [[] for _ in range(next_id)]
    for bi, block in enumerate(blocks):
        for v in block:
            if v in ap_id:
                tree_adj[bi].append(ap_id[v])
                tree_adj[ap_id[v]].append(bi)

    return node_blocks, tree_adj, ap_id


def _collect_path_nodes(
    node_ids: Sequence[NodeIdT],
    blocks: list[set[NodeIdT]],
    tree: tuple[dict[NodeIdT, list[int]], list[list[int]], dict[NodeIdT, int]],
    src: NodeIdT,
    tgts: set[NodeIdT],
    result: set[NodeIdT],
) -> set[NodeIdT]:
    """BFS on block-cut tree, trace paths, collect nodes."""
    node_blocks, tree_adj, ap_id = tree
    num_blocks = len(blocks)

    def tn(v: NodeIdT) -> int:
        if v in ap_id:
            return ap_id[v]
        blks = node_blocks.get(v)
        return blks[0] if blks else -1

    src_tn = tn(src)
    if src_tn == -1:
        return result

    total = len(tree_adj)
    par = [-1] * total
    par[src_tn] = src_tn
    q: deque[int] = deque([src_tn])
    while q:
        u = q.popleft()
        for v in tree_adj[u]:
            if par[v] == -1:
                par[v] = u
                q.append(v)

    for t in tgts:
        t_tn = tn(t)
        if t_tn == -1 or par[t_tn] == -1:
            continue
        v = t_tn
        while v != src_tn:
            if v < num_blocks:
                result.update(blocks[v])
            v = par[v]
        if src_tn < num_blocks:
            result.update(blocks[src_tn])

    return result


# ── Graph class: parse once, run many algorithms ──


def _merged_branch_ids(
    base_branch_ids: Sequence[BranchIdT] | None,
    added_count: int,
    added_branch_ids: Sequence[BranchIdT] | None,
) -> list[BranchIdT] | None:
    """Branch ids of a rebuilt graph: base ids followed by the added ids."""
    if base_branch_ids is None:
        if added_branch_ids is not None:
            raise ValueError("added_branch_ids given but the base graph has no branch_ids")  # noqa: TRY003 — one clear sentence
        return None
    if added_count == 0:
        return list(base_branch_ids)
    if added_branch_ids is None or len(added_branch_ids) != added_count:
        raise ValueError(  # noqa: TRY003 — the count is the whole message
            f"added_branch_ids must hold {added_count} ids, one per added edge"
        )
    return [*base_branch_ids, *added_branch_ids]


def _rerouted_branch_ids(
    base_branch_ids: Sequence[BranchIdT] | None,
    edge_indices: Collection[int],
) -> list[BranchIdT] | None:
    """A rerouted edge keeps the branch id of the edge it replaces."""
    if base_branch_ids is None:
        return None
    return [base_branch_ids[edge_idx] for edge_idx in edge_indices]


@dataclass(frozen=True)
class ReductionLog:
    """Flat operation log of a terminal-preserving series-parallel reduction, as int32 views.

    The first eight fields hold one entry per operation, in the order the
    moves apply: ``op_kind`` is 0 for a leaf, 1 for a series merge, 2 for a
    parallel merge and 3 for a pendant deletion; ``left`` and ``right`` are
    child operation ids (-1 when absent); ``endpoint_u`` and ``endpoint_v``
    are the oriented endpoints of the virtual edge the operation produces
    (-1 when it produces none); ``interior_node`` is the node a series move
    eliminates or a pendant move removes; ``leaf_edge_index`` is the input
    edge index of a leaf; ``absorber`` is the neighbour a pendant move's
    payload moves to. A parallel merge of more than two edges is a left-deep
    chain of binary operations of which only the last carries endpoints.

    Every leaf precedes every move, one per input edge in edge index order, so
    the leaves are the operations below the first non-leaf one.

    ``residual_op``, ``residual_u`` and ``residual_v`` hold one entry per
    surviving edge: the operation that produced it and its two endpoint node
    indices. ``surviving_nodes`` holds the surviving node indices in
    increasing order.
    """

    op_kind: memoryview
    left: memoryview
    right: memoryview
    endpoint_u: memoryview
    endpoint_v: memoryview
    interior_node: memoryview
    leaf_edge_index: memoryview
    absorber: memoryview
    residual_op: memoryview
    residual_u: memoryview
    residual_v: memoryview
    surviving_nodes: memoryview

    @classmethod
    def from_buffers(cls, raw: tuple[bytes, ...]) -> ReductionLog:
        """The twelve int32 byte buffers of ``series_parallel_reduce_ctx`` as typed views."""
        return cls(*(memoryview(buffer).cast("i") for buffer in raw))


class ReductionState:
    """Opaque handle on the mutable state of one terminal-preserving reduction.

    ``Graph.series_parallel_state`` builds the incidence structure of the
    masked graph once, with the terminal-free components already gone, and
    every method below moves it one step without rebuilding anything. The
    caller drives the fixpoint itself and reads the operation log at the end.

    Move kinds are the ``ReductionLog`` operation kinds, named by the
    ``SERIES``, ``PARALLEL`` and ``PENDANT`` attributes. Nodes and edges are
    the node indices and edge slots of the state, never node ids.

    The handle carries the ``Graph`` it was built from, so it can never be
    paired with another one, and it goes inert on ``free``: every later call
    raises ``ValueError`` instead of touching released memory.
    """

    __slots__ = ("_capsule", "_graph")

    SERIES = 1
    PARALLEL = 2
    PENDANT = 3

    def __init__(self, graph: "Graph", capsule: object) -> None:
        self._graph = graph
        self._capsule = capsule

    @property
    def graph(self) -> "Graph":
        """The graph the state was built from."""
        return self._graph

    def __enter__(self) -> "ReductionState":
        return self

    def __exit__(self, *_exception: object) -> None:
        self.free()

    def free(self) -> None:
        """Release the state. Idempotent only in that a second call raises ``ValueError``."""
        _sp_state_free(self._capsule)

    def next_move(self) -> tuple[int, int, int, int, int, int] | None:
        """The move at the next candidate node, pendant before series, or None when none applies.

        Reports ``(kind, node, edge_a, edge_b, neighbour_a, neighbour_b)``,
        the second edge and neighbour -1 for a pendant move. The node leaves
        the candidate queue whether or not the caller applies the move.
        """
        move: tuple[int, int, int, int, int, int] | None = _sp_next_move(self._capsule)
        return move

    def apply_move(self, move: tuple[int, int, int, int, int, int]) -> tuple[int, int]:
        """Apply one ``next_move`` result, returning ``(created_edge, parallel_pending)``.

        ``created_edge`` is the edge a series move produced and -1 for a
        pendant move; ``parallel_pending`` is 1 when its two endpoints now
        carry a mergeable parallel pair.
        """
        applied: tuple[int, int] = _sp_apply_move(self._capsule, move)
        return applied

    def pair_edges(self, from_node: int, to_node: int) -> bytes | None:
        """The live edges between one unprotected node pair as int32, or None when fewer than two."""
        edges: bytes | None = _sp_pair_edges(self._capsule, from_node, to_node)
        return edges

    def apply_parallel(self, from_node: int, to_node: int, edges: bytes) -> int:
        """Replace the given edges between one node pair by a single edge, returning it."""
        created: int = _sp_apply_parallel(self._capsule, from_node, to_node, edges)
        return created

    def batch_moves(self, kind: int) -> bytes | None:
        """Every currently applicable, mutually independent move of one kind, or None when none applies.

        A pendant or series kind reports five int32 per move, the node and the
        two edges and two neighbours of ``next_move``; a parallel kind reports
        one variable-length record per endpoint pair, the two endpoints, the
        member count and the member edges.
        """
        batch: bytes | None = _sp_batch_moves(self._capsule, kind)
        return batch

    def apply_batch(self, kind: int, batch: bytes) -> None:
        """Apply every move of one ``batch_moves`` buffer, in buffer order."""
        _sp_apply_batch(self._capsule, kind, batch)

    def log(self) -> ReductionLog:
        """The operation log, the residual edges and the surviving nodes so far."""
        raw: tuple[bytes, ...] = _sp_state_log(self._capsule)
        return ReductionLog.from_buffers(raw)


def _quotient_edge_views(
    raw: tuple[bytes, bytes, bytes, bytes],
) -> tuple[memoryview, memoryview, memoryview, memoryview]:
    """The four int32 byte arrays of ``quotient_edges_ctx`` as typed views."""
    src_labels, dst_labels, edge_indices, internal_edge_indices = raw
    return (
        memoryview(src_labels).cast("i"),
        memoryview(dst_labels).cast("i"),
        memoryview(edge_indices).cast("i"),
        memoryview(internal_edge_indices).cast("i"),
    )


_EXCLUDED_LABEL = -1
"""Component label of a node excluded from a view."""

_LARGE_COMPONENT_SHARE = 8
"""Above one member per this many nodes, a full label pass beats a ``find`` per member."""


def _node_index_of(graph: Graph[NodeIdT, BranchIdT], node_id: NodeIdT) -> int:
    """The index of a node id, without building the cached id map for a single lookup.

    The map costs about 79 MiB at 1,000,000 nodes, more than building every
    component peaks at, so a lookup that would be its only user scans the node
    ids instead. Four other methods share it, so a lookup on a graph one of
    them has touched is still a dict hit.
    """
    cached_index_of = graph._node_id_to_idx
    try:
        return graph._node_ids.index(node_id) if cached_index_of is None else cached_index_of[node_id]
    except (ValueError, KeyError):
        raise ValueError(f"node {node_id} is not in the graph") from None  # noqa: TRY003 — the id is the whole message


def _connected_component_of(
    graph: Graph[NodeIdT, BranchIdT],
    node_id: NodeIdT,
    excluded_edges: bytearray | None,
    excluded_nodes: bytearray | None,
) -> set[NodeIdT]:
    """The connected component containing one node, without building any other component.

    ``component_labels_ctx`` labels every node index with the smallest node
    index of its component, so the members are exactly the node indices
    carrying the requested node's label. ``bytes.count`` says in one C pass
    about how many there are, which picks how to collect them: a small
    component is found with ``bytes.find``, keeping the search for the next
    member in C, while a component of more than about an eighth of the graph
    is collected by one pass over every label, which by then is cheaper than a
    ``find`` per member. The count only chooses a strategy, so the straddling
    hits it may include cannot change the answer.

    A ``find`` hit that does not start on a four-byte boundary straddles two
    neighbouring labels and is not a member. The search resumes at the next
    four-byte boundary rather than one byte on: every position it skips sits
    off a boundary, so it can only hold a straddling hit and no member is ever
    passed over.
    """
    node_ids = graph._node_ids
    node_index = _node_index_of(graph, node_id)
    labels: bytes = _component_labels_ctx(graph._ctx, excluded_edges, excluded_nodes)
    label_values = memoryview(labels).cast("i")
    label_value = label_values[node_index]
    if label_value == _EXCLUDED_LABEL:
        raise ValueError(f"node {node_id} is excluded from this view")  # noqa: TRY003 — the id is the whole message
    label = labels[4 * node_index : 4 * node_index + 4]
    if labels.count(label) > len(node_ids) // _LARGE_COMPONENT_SHARE:
        return {node_ids[index] for index, value in enumerate(label_values) if value == label_value}
    component: set[NodeIdT] = set()
    position = labels.find(label)
    while position >= 0:
        if position % 4 == 0:
            component.add(node_ids[position // 4])
        position = labels.find(label, position + 4 - position % 4)
    return component


class Graph(Generic[NodeIdT, BranchIdT]):
    """Parsed graph that supports multiple algorithm calls without re-parsing.

    Parses node IDs and edges once into an internal C structure (IntMap + EdgeList
    + CSR adjacency list), then reuses that parsed state across all algorithm calls.
    Duplicate edges between the same node pair are allowed (multigraph support).

    Two calling conventions:
        Graph(node_ids, edges)                          — edges as pairs of node IDs
        Graph(node_ids, src, dst)                       — two flat lists of node IDs
        Graph(node_ids, edges, branch_ids=branch_ids)   — with branch IDs for exclusion

    The type parameters are the caller's node and branch id types, both bound
    to ``int``. Every result hands back the id objects that were passed in, so
    ``Graph[NodeId, BranchId]`` built from ``NewType`` ids returns those types.
    Without ``branch_ids`` the branch type parameter is ``int``.
    """

    __slots__ = (
        "_ctx",
        "_node_ids",
        "_edges",
        "_branch_ids",
        "_branch_id_to_edge_idx",
        "_repeated_branch_edge_idx",
        "_node_id_to_idx",
        "_directed",
    )

    _node_ids: Sequence[NodeIdT]
    _edges: Sequence[tuple[NodeIdT, NodeIdT]] | None
    _branch_ids: Sequence[BranchIdT] | None
    _branch_id_to_edge_idx: dict[BranchIdT, int] | None
    _repeated_branch_edge_idx: dict[BranchIdT, list[int]] | None
    _node_id_to_idx: dict[NodeIdT, int] | None
    _directed: bool

    @overload
    def __init__(
        self: Graph[NodeIdT, int],
        node_ids: Sequence[NodeIdT],
        edges_or_src: Sequence[tuple[NodeIdT, NodeIdT]] | Sequence[NodeIdT],
        dst: Sequence[NodeIdT] | None = None,
        *,
        branch_ids: None = None,
        directed: bool = False,
    ) -> None: ...

    @overload
    def __init__(
        self,
        node_ids: Sequence[NodeIdT],
        edges_or_src: Sequence[tuple[NodeIdT, NodeIdT]] | Sequence[NodeIdT],
        dst: Sequence[NodeIdT] | None = None,
        *,
        branch_ids: Sequence[BranchIdT] | None = None,
        directed: bool = False,
    ) -> None: ...

    def __init__(
        self,
        node_ids: Sequence[NodeIdT],
        edges_or_src: Sequence[tuple[NodeIdT, NodeIdT]] | Sequence[NodeIdT],
        dst: Sequence[NodeIdT] | None = None,
        *,
        branch_ids: Sequence[BranchIdT] | None = None,
        directed: bool = False,
    ) -> None:
        self._node_ids = node_ids
        self._edges = edges_or_src if dst is None else None  # type: ignore[assignment]  # dst is None selects the edge-pair half of the union
        self._branch_ids = branch_ids
        self._branch_id_to_edge_idx = None
        self._repeated_branch_edge_idx = None
        self._node_id_to_idx = None
        self._directed = directed
        if dst is not None:
            self._ctx = _parse_graph(node_ids, edges_or_src, dst, directed)
        else:
            self._ctx = _parse_graph(node_ids, edges_or_src, None, directed)
        if branch_ids is not None and len(branch_ids) != self.edge_count:
            raise ValueError(  # noqa: TRY003 — the two counts are the whole message
                f"branch_ids length {len(branch_ids)} does not match edge count {self.edge_count}"
            )

    @property
    def directed(self) -> bool:
        """True if the graph is directed."""
        return self._directed

    def _get_node_id_to_idx(self) -> dict[NodeIdT, int]:
        """Lazily build and cache the node_id → internal index mapping."""
        if self._node_id_to_idx is None:
            self._node_id_to_idx = {nid: i for i, nid in enumerate(self._node_ids)}
        return self._node_id_to_idx

    def _get_branch_id_to_edge_idx(self) -> dict[BranchIdT, int]:
        """Lazily build and cache the branch_id → first edge index mapping.

        Several edges may carry the same branch id. The further indices of a
        repeated id are kept in ``_repeated_branch_edge_idx``, which stays
        empty in the common unique case so the build is one dict comprehension.
        """
        if self._branch_id_to_edge_idx is None:
            if self._branch_ids is None:
                raise ValueError("no branch_ids")  # noqa: TRY003 — short, no custom class needed
            first: dict[BranchIdT, int] = {branch_id: edge_idx for edge_idx, branch_id in enumerate(self._branch_ids)}
            repeated: dict[BranchIdT, list[int]] = {}
            if len(first) != len(self._branch_ids):
                for edge_idx, branch_id in enumerate(self._branch_ids):
                    if first[branch_id] > edge_idx:
                        first[branch_id] = edge_idx
                    elif first[branch_id] != edge_idx:
                        repeated.setdefault(branch_id, []).append(edge_idx)
            self._branch_id_to_edge_idx = first
            self._repeated_branch_edge_idx = repeated
        return self._branch_id_to_edge_idx

    def _edge_indices_of_branches(self, branch_ids: Collection[BranchIdT]) -> list[int]:
        """Every edge index carrying one of the given branch ids."""
        first = self._get_branch_id_to_edge_idx()
        edge_indices = [first[branch_id] for branch_id in branch_ids]
        repeated = self._repeated_branch_edge_idx or {}
        if repeated:
            for branch_id in branch_ids:
                edge_indices.extend(repeated.get(branch_id, ()))
        return edge_indices

    @property
    def edge_count(self) -> int:
        """Number of edges in the graph."""
        result: int = _graph_edge_count(self._ctx)
        return result

    @property
    def node_count(self) -> int:
        """Number of nodes in the graph."""
        result: int = _graph_node_count(self._ctx)
        return result

    @property
    def is_multigraph(self) -> bool:
        """True if any node pair has more than one edge (parallel edges)."""
        edges = self._edges
        if edges is None:
            return False
        seen: set[tuple[NodeIdT, NodeIdT]] = set()
        for a, b in edges:
            key = (a, b) if self._directed else (min(a, b), max(a, b))
            if key in seen:
                return True
            seen.add(key)
        return False

    def edge_indices(self, u: NodeIdT, v: NodeIdT) -> list[EdgeIndex]:
        """Return indices of edges between u and v (list, for multigraph support).

        For directed graphs, only matches edges where src=u and dst=v.
        """
        result: list[EdgeIndex] = _edge_indices_ctx(self._ctx, u, v)
        return result

    def incident_edge_indices(self, node_id: NodeIdT) -> list[EdgeIndex]:
        """Return indices of all edges incident to the given node.

        For directed graphs, returns only outgoing edges (src=node_id).
        Use ``incoming_edge_indices`` for incoming edges.
        """
        result: list[EdgeIndex] = _incident_edges_ctx(self._ctx, node_id)
        return result

    def outgoing_edge_indices(self, node_id: NodeIdT) -> list[EdgeIndex]:
        """Return indices of all outgoing edges (src=node_id). Directed graphs only."""
        self._require_directed("outgoing_edge_indices")
        result: list[EdgeIndex] = _incident_edges_ctx(self._ctx, node_id)
        return result

    def incoming_edge_indices(self, node_id: NodeIdT) -> list[EdgeIndex]:
        """Return indices of all incoming edges (dst=node_id). Directed graphs only."""
        self._require_directed("incoming_edge_indices")
        result: list[EdgeIndex] = _incoming_edges_ctx(self._ctx, node_id)
        return result

    def neighbors(self, node_id: NodeIdT) -> set[NodeIdT]:
        """Return the set of neighbor node IDs.

        For directed graphs, returns successors (outgoing neighbors).
        """
        result: set[NodeIdT] = _neighbors_ctx(self._ctx, node_id)
        return result

    def successors(self, node_id: NodeIdT) -> set[NodeIdT]:
        """Return the set of successor node IDs (outgoing neighbors). Directed graphs only."""
        self._require_directed("successors")
        result: set[NodeIdT] = _neighbors_ctx(self._ctx, node_id)
        return result

    def predecessors(self, node_id: NodeIdT) -> set[NodeIdT]:
        """Return the set of predecessor node IDs (incoming neighbors). Directed graphs only."""
        result: set[NodeIdT] = _predecessors_ctx(self._ctx, node_id)
        return result

    def degree(self, node_id: NodeIdT) -> int:
        """Return the number of edges incident to the node.

        For undirected graphs, self-loops are counted twice (via CSR: each direction counted).
        For directed graphs, returns out-degree.
        """
        result: int = _degree_ctx(self._ctx, node_id)
        return result

    def out_degree(self, node_id: NodeIdT) -> int:
        """Return the out-degree of the node. Directed graphs only."""
        self._require_directed("out_degree")
        result: int = _degree_ctx(self._ctx, node_id)
        return result

    def in_degree(self, node_id: NodeIdT) -> int:
        """Return the in-degree of the node. Directed graphs only."""
        result: int = _in_degree_ctx(self._ctx, node_id)
        return result

    def without_edges(
        self,
        edge_indices: Collection[int],
    ) -> GraphView[NodeIdT, BranchIdT]:
        """Create a lightweight view with the given edges excluded."""
        return GraphView(self, edge_indices)

    def with_edges(
        self,
        added_edges: Sequence[tuple[NodeIdT, NodeIdT]],
        added_branch_ids: Sequence[BranchIdT] | None = None,
    ) -> GraphView[NodeIdT, BranchIdT]:
        """Create a view with extra edges added (rebuilds CSR internally).

        The base graph is not modified. The rebuilt graph keeps every base
        edge at its original index and appends ``added_edges`` after them, so
        edge indices held by the caller stay valid. New nodes referenced in
        ``added_edges`` are appended after the base nodes. When the base graph
        carries ``branch_ids``, ``added_branch_ids`` must supply one id per
        added edge. Requires edge-pair construction.
        """
        return GraphView._with_additions(
            self,
            excluded_edges=None,
            excluded_nodes=None,
            added_edges=added_edges,
            added_branch_ids=added_branch_ids,
        )

    def without_branches(
        self,
        branch_ids: Collection[BranchIdT],
    ) -> GraphView[NodeIdT, BranchIdT]:
        """Create a lightweight view with the given branches excluded by ID.

        Every edge carrying one of the ids is excluded. Requires the Graph to
        have been constructed with branch_ids.
        """
        return GraphView(self, self._edge_indices_of_branches(branch_ids))

    def without_nodes(
        self,
        node_ids: Collection[NodeIdT],
    ) -> GraphView[NodeIdT, BranchIdT]:
        """Create a lightweight view with the given nodes excluded.

        All edges incident to excluded nodes are also excluded.
        """
        return GraphView._from_node_exclusion(self, node_ids)

    def split_node(
        self,
        node_id: NodeIdT,
        new_node_id: NodeIdT,
        edge_indices_to_new_node: Collection[int],
    ) -> GraphView[NodeIdT, BranchIdT]:
        """Split a node by rerouting specified edges to a new node.

        Creates a view where the edges identified by ``edge_indices_to_new_node``
        are detached from ``node_id`` and reattached to ``new_node_id``.
        The remaining edges of ``node_id`` stay in place.

        Equivalent to::

            graph.without_edges(edge_indices_to_new_node).with_edges(rerouted)

        where *rerouted* replaces ``node_id`` with ``new_node_id`` in each edge.
        """
        edges = self._edges
        base_branch_ids = self._branch_ids
        if edges is None:
            raise ValueError("split_node requires edge-pair construction")  # noqa: TRY003
        node_id_to_idx = self._get_node_id_to_idx()
        if node_id not in node_id_to_idx:
            raise ValueError(f"node {node_id} is not in the graph")  # noqa: TRY003
        if new_node_id in node_id_to_idx:
            raise ValueError(f"node {new_node_id} already exists in the graph")  # noqa: TRY003
        rerouted_edges: list[tuple[NodeIdT, NodeIdT]] = []
        for edge_idx in edge_indices_to_new_node:
            u, v = edges[edge_idx]
            if u == node_id:
                rerouted_edges.append((new_node_id, v))
            elif v == node_id:
                rerouted_edges.append((u, new_node_id))
            else:
                raise ValueError(  # noqa: TRY003
                    f"edge {edge_idx} ({u}, {v}) is not incident to node {node_id}"
                )
        rerouted_branch_ids = _rerouted_branch_ids(base_branch_ids, edge_indices_to_new_node)
        return self.without_edges(edge_indices_to_new_node).with_edges(rerouted_edges, rerouted_branch_ids)

    def all_edge_paths(
        self,
        source: NodeIdT,
        targets: NodeIdT | Collection[NodeIdT],
        cutoff: int | None = None,
        *,
        node_simple: bool = False,
        ignore_self_loops: bool = False,
    ) -> list[list[EdgeIndex]]:
        """Find all paths from source to targets using each edge at most once.

        Returns a list of paths. Each path is a list of edge indices.

        node_simple: if True, each node may be visited at most once per path
            (source counts as visited at initialization). Default False allows
            node revisits via different edges — relevant for multigraphs.

        ignore_self_loops: if True, self-loops are never traversed, so no
            returned path contains one.

        cutoff: maximum number of edges per path. None = no limit.
        """
        tgt_list = [targets] if isinstance(targets, int) else list(targets)
        c = cutoff if cutoff is not None else -1
        result: list[list[EdgeIndex]] = _all_edge_paths_ctx(
            self._ctx, source, tgt_list, c, None, None, node_simple, ignore_self_loops
        )
        return result

    def _require_undirected(self, method_name: str) -> None:
        if self._directed:
            raise TypeError(f"{method_name} is not defined for directed graphs")  # noqa: TRY003

    def _require_directed(self, method_name: str) -> None:
        if not self._directed:
            raise TypeError(f"{method_name} requires a directed graph")  # noqa: TRY003

    def connected_components(self) -> Generator[set[NodeIdT], None, None]:
        """Yield each connected component as a set of original node IDs."""
        self._require_undirected("connected_components")
        yield from _cc_ctx(self._ctx)

    def connected_component(self, node_id: NodeIdT) -> set[NodeIdT]:
        """Return the connected component containing ``node_id`` as a set of node IDs.

        The single-component counterpart of ``connected_components``, and the
        cheaper call unless the graph is itself a single component. It shares
        the same O(n + m) label pass and then collects only the requested
        component, so at 1,000,000 nodes it runs 103x faster on a graph of
        singletons and 2.3x faster on a sparse graph with a giant component
        and a tail, but 4x slower when the whole graph is one component. Each
        call repeats the label pass, so wanting more than a handful of
        components is a job for ``connected_components()``. Raises
        ``ValueError`` if the node is not in the graph.
        """
        self._require_undirected("connected_component")
        return _connected_component_of(self, node_id, None, None)

    def connected_components_with_branch_ids(self) -> Generator[tuple[set[NodeIdT], set[BranchIdT]], None, None]:
        """Yield (node_id_set, branch_id_set) for each connected component.

        Requires the Graph to have been constructed with branch_ids.
        """
        self._require_undirected("connected_components_with_branch_ids")
        if self._branch_ids is None:
            raise ValueError("no branch_ids")  # noqa: TRY003 — short, no custom class needed
        yield from _cc_branches_ctx(self._ctx, self._branch_ids)

    def strongly_connected_components(self) -> Generator[set[NodeIdT], None, None]:
        """Yield each strongly connected component as a set of node IDs."""
        self._require_directed("strongly_connected_components")
        yield from _scc_ctx(self._ctx)

    def weakly_connected_components(self) -> Generator[set[NodeIdT], None, None]:
        """Yield each weakly connected component as a set of node IDs.

        Ignores edge direction — equivalent to undirected connected components.
        """
        self._require_directed("weakly_connected_components")
        yield from _cc_ctx(self._ctx)

    def topological_sort(self) -> list[NodeIdT]:
        """Return nodes in topological order (Kahn's algorithm).

        Raises ValueError if the graph contains a cycle.
        """
        self._require_directed("topological_sort")
        result: list[NodeIdT] = _toposort_ctx(self._ctx)
        return result

    def bridges(self) -> list[tuple[NodeIdT, NodeIdT]]:
        """Return bridge edges as (node_id, node_id) pairs."""
        self._require_undirected("bridges")
        result: list[tuple[NodeIdT, NodeIdT]] = _bridges_ctx(self._ctx)
        return result

    def bridges_with_branch_ids(self) -> list[tuple[NodeIdT, NodeIdT, BranchIdT]]:
        """Return bridge edges as (node_id, node_id, branch_id) triples.

        Requires the Graph to have been constructed with branch_ids.
        """
        self._require_undirected("bridges_with_branch_ids")
        if self._branch_ids is None:
            raise ValueError("no branch_ids")  # noqa: TRY003 — short, no custom class needed
        bridge_list = self.bridges()
        result: list[tuple[NodeIdT, NodeIdT, BranchIdT]] = []
        for u, v in bridge_list:
            result.extend((u, v, self._branch_ids[edge_idx]) for edge_idx in self.edge_indices(u, v))
        return result

    def articulation_points(self) -> set[NodeIdT]:
        """Return the set of articulation points."""
        self._require_undirected("articulation_points")
        result: set[NodeIdT] = _ap_ctx(self._ctx)
        return result

    def biconnected_components(self) -> Generator[set[NodeIdT], None, None]:
        """Yield each biconnected component as a set of node IDs."""
        self._require_undirected("biconnected_components")
        yield from _bcc_ctx(self._ctx)

    def component_labels(self) -> memoryview:
        """Connected component label per node index as an int32 view, direction ignored.

        The label of a node is the smallest node index of its component, so
        one array replaces one set per component and no Python object is
        created per node.
        """
        result: bytes = _component_labels_ctx(self._ctx)
        return memoryview(result).cast("i")

    def quotient_edges(self, labels: memoryview) -> tuple[memoryview, memoryview, memoryview, memoryview]:
        """Split the edges by the int32 label of their endpoints, one label per node index.

        Returns ``(src_labels, dst_labels, edge_indices, internal_edge_indices)``
        as int32 views in edge index order: the crossing edges (different
        labels) as three parallel arrays and the internal edges (equal labels)
        by index. Edges at a node labelled -1 are neither. Raises ``ValueError``
        when the labels buffer does not hold one entry per node.
        """
        return _quotient_edge_views(_quotient_edges_ctx(self._ctx, labels))

    def degrees(self) -> memoryview:
        """Degree per node index as an int32 view: incidences with a self-loop counted twice, out-degree if directed."""
        result: bytes = _degrees_ctx(self._ctx)
        return memoryview(result).cast("i")

    def series_parallel_reduce(
        self,
        terminal_mask: NodeMask,
        protected_mask: NodeMask,
        *,
        pendant_keep_mask: NodeMask | None = None,
        series_blocked_mask: NodeMask | None = None,
    ) -> ReductionLog:
        """Terminal-preserving series-parallel reduction as a flat operation log.

        Pendant deletion, series merge and parallel merge are applied to a
        fixpoint in increasing node index, self-loops take part in no move
        and leave with their node, and components without a terminal go
        whole before any move. The four masks hold one byte per node index and
        mark membership by a non-zero byte; they are read as buffers, so a list
        of node ids is not a mask. A node in ``pendant_keep_mask`` takes no
        pendant move, one in ``series_blocked_mask`` no series move (parallel
        merges at it stay allowed, unlike ``protected_mask``). A missing
        ``protected_mask``, ``pendant_keep_mask`` or ``series_blocked_mask`` is
        the empty set; ``None`` in place of the terminal mask raises
        ``TypeError``, because a reduction without a terminal deletes every
        component. Undirected graphs only.
        """
        self._require_undirected("series_parallel_reduce")
        raw: tuple[bytes, ...] = _series_parallel_reduce_ctx(
            self._ctx, terminal_mask, protected_mask, None, None, pendant_keep_mask, series_blocked_mask
        )
        return ReductionLog.from_buffers(raw)

    def series_parallel_state(
        self,
        terminal_mask: NodeMask,
        protected_mask: NodeMask,
        *,
        pendant_keep_mask: NodeMask | None = None,
        series_blocked_mask: NodeMask | None = None,
    ) -> ReductionState:
        """Reduction state of this graph, with the terminal-free components already gone.

        The four masks hold one byte per node index and mark membership by a
        non-zero byte, as in ``series_parallel_reduce``: a kept node takes no
        pendant move, a blocked one no series move. No move is applied: the
        caller drives the fixpoint through the returned handle. Undirected
        graphs only.
        """
        self._require_undirected("series_parallel_state")
        return ReductionState(
            self,
            _sp_state_new(self._ctx, terminal_mask, protected_mask, None, None, pendant_keep_mask, series_blocked_mask),
        )

    def bcc_edge_labels(self) -> memoryview:
        """Biconnected component id per edge index as an int32 view.

        Bridges are singleton components and self-loops, which belong to no
        component, get -1.
        """
        self._require_undirected("bcc_edge_labels")
        result: bytes = _bcc_edge_labels_ctx(self._ctx)
        return memoryview(result).cast("i")

    def cycle_basis(self) -> list[list[NodeIdT]]:
        """Return a fundamental cycle basis as a list of cycles.

        Each cycle is a list of node IDs. The number of fundamental cycles
        equals m - n + c (circuit rank), where c is the number of connected
        components.
        """
        self._require_undirected("cycle_basis")
        result: list[list[NodeIdT]] = _cycle_basis_ctx(self._ctx)
        return result

    def dag_longest_path(self, weights: list[float] | None = None) -> list[NodeIdT]:
        """Return the longest path in the DAG as a list of node IDs.

        If weights are provided, edge weights are used to determine path length.
        Otherwise all edges have unit weight.
        Raises ValueError if the graph contains a cycle.
        """
        self._require_directed("dag_longest_path")
        result: list[NodeIdT] = _dag_longest_path_ctx(self._ctx, weights)
        return result

    def bfs(self, source: NodeIdT) -> list[NodeIdT]:
        """Return nodes visited in BFS order from source."""
        result: list[NodeIdT] = _bfs_ctx(self._ctx, source)
        return result

    def shortest_path(
        self,
        weights: list[float],
        source: NodeIdT,
        target: NodeIdT,
    ) -> list[NodeIdT]:
        """Return the shortest weighted path from source to target.

        Runs a bidirectional Dijkstra, so only a small part of a large graph is
        settled. Pass ``weights`` as a float64 buffer (a numpy ``float64`` array
        or ``array.array("d", ...)``) to hand it to C as is; a list of floats is
        converted element by element first, which costs more than the search.
        """
        _dist, path = _dijkstra_ctx(self._ctx, weights, source, target)
        result: list[NodeIdT] = path
        return result

    def shortest_path_lengths(
        self,
        weights: list[float],
        source: NodeIdT,
        cutoff: float | None = None,
    ) -> dict[NodeIdT, float]:
        """Return {node_id: distance} for all nodes reachable from source."""
        c = cutoff if cutoff is not None else -1.0
        result: dict[NodeIdT, float] = _sssp_ctx(self._ctx, weights, source, c)
        return result

    def multi_source_shortest_path_lengths(
        self,
        weights: list[float],
        sources: Sequence[NodeIdT],
        cutoff: float | None = None,
    ) -> dict[NodeIdT, float]:
        """Return {node_id: distance} from nearest source to each reachable node."""
        c = cutoff if cutoff is not None else -1.0
        result: dict[NodeIdT, float] = _msdijk_ctx(self._ctx, weights, sources, c)
        return result

    def eccentricity(self, weights: list[float], source: NodeIdT) -> float:
        """Return the eccentricity of source (max shortest-path distance)."""
        lengths = self.shortest_path_lengths(weights, source)
        if not lengths:
            return 0.0
        return max(lengths.values())

    def two_edge_connected_components(
        self,
    ) -> Generator[set[NodeIdT], None, None]:
        """Yield 2-edge-connected components (bridges removed, then CC)."""
        self._require_undirected("two_edge_connected_components")
        bridge_edge_indices: list[int] = _bridges_as_edge_indices_ctx(self._ctx)
        if not bridge_edge_indices:
            yield from self.connected_components()
            return
        mask = bytearray(self.edge_count)
        for idx in bridge_edge_indices:
            mask[idx] = 1
        yield from _cc_ctx(self._ctx, mask)

    def nodes_on_simple_paths(
        self,
        source: NodeIdT,
        targets: Sequence[NodeIdT],
    ) -> set[NodeIdT]:
        """Return all nodes on any simple path from source to any target."""
        self._require_undirected("nodes_on_simple_paths")
        n = len(self._node_ids)
        if n == 0:
            return set()

        tgts = set(targets)
        result: set[NodeIdT] = set()
        if source in tgts:
            result.add(source)
            tgts.discard(source)
        if not tgts:
            return result

        blocks = list(self.biconnected_components())
        if not blocks:
            return result

        tree = _build_block_cut_tree(self._node_ids, blocks)
        return _collect_path_nodes(
            self._node_ids,
            blocks,
            tree,
            source,
            tgts,
            result,
        )


class GraphView(Generic[NodeIdT, BranchIdT]):
    """Lightweight view of a Graph with excluded edges and/or nodes.

    Shares the base graph's parsed data (IntMap, CSR). Holds a
    bytearray of excluded edges and an optional bytearray of excluded nodes.

    Edges are identified by their index in the original edge list
    (the order in which they were passed to ``Graph()``). A rebuild through
    ``with_edges`` or ``split_node`` keeps every base edge at its index,
    keeps excluded edges and nodes masked, and appends added edges and new
    nodes, so indices and branch ids stay valid across rebuilds.
    """

    __slots__ = ("_graph", "_excluded_edges", "_added_graph", "_excluded_nodes")

    def __init__(
        self,
        graph: Graph[NodeIdT, BranchIdT],
        excluded_edge_indices: Collection[int],
    ) -> None:
        self._graph = graph
        self._excluded_edges = bytearray(graph.edge_count)
        for idx in excluded_edge_indices:
            self._excluded_edges[idx] = 1
        self._added_graph: Graph[NodeIdT, BranchIdT] | None = None
        self._excluded_nodes: bytearray | None = None

    @classmethod
    def _from_excluded_edges(
        cls, graph: Graph[NodeIdT, BranchIdT], excluded_edges: bytearray
    ) -> GraphView[NodeIdT, BranchIdT]:
        """Create a view from an existing excluded-edges bytearray (no copy)."""
        view = object.__new__(cls)
        view._graph = graph
        view._excluded_edges = excluded_edges
        view._added_graph = None
        view._excluded_nodes = None
        return view

    @classmethod
    def _from_node_exclusion(
        cls,
        graph: Graph[NodeIdT, BranchIdT],
        excluded_node_ids: Collection[NodeIdT],
    ) -> GraphView[NodeIdT, BranchIdT]:
        """Create a view excluding the given nodes (and their incident edges)."""
        node_id_to_idx = graph._get_node_id_to_idx()
        excluded_nodes = bytearray(graph.node_count)
        for nid in excluded_node_ids:
            i = node_id_to_idx.get(nid)
            if i is not None:
                excluded_nodes[i] = 1
        view = object.__new__(cls)
        view._graph = graph
        view._excluded_edges = bytearray(graph.edge_count)
        view._added_graph = None
        view._excluded_nodes = excluded_nodes
        return view

    @classmethod
    def _with_additions(
        cls,
        base: Graph[NodeIdT, BranchIdT],
        *,
        excluded_edges: bytearray | None,
        excluded_nodes: bytearray | None,
        added_edges: Sequence[tuple[NodeIdT, NodeIdT]],
        added_branch_ids: Sequence[BranchIdT] | None,
    ) -> GraphView[NodeIdT, BranchIdT]:
        """Rebuild with every base edge kept at its index and added edges appended.

        Excluded edges and nodes stay masked in the new view instead of being
        compacted away, which keeps edge indices, node indices and branch ids
        of the base graph valid on the rebuilt view.
        """
        base_edges = base._edges
        if base_edges is None:
            raise ValueError("with_edges requires edge-pair construction")  # noqa: TRY003 — one clear sentence
        merged_edges: list[tuple[NodeIdT, NodeIdT]] = [*base_edges, *added_edges]

        merged_nodes: list[NodeIdT] = list(base._node_ids)
        known: set[NodeIdT] = set(base._node_ids)
        for u, v in added_edges:
            for node_id in (u, v):
                if node_id not in known:
                    known.add(node_id)
                    merged_nodes.append(node_id)

        merged_branch_ids = _merged_branch_ids(base._branch_ids, len(added_edges), added_branch_ids)
        rebuilt = Graph(merged_nodes, merged_edges, branch_ids=merged_branch_ids, directed=base._directed)

        view = object.__new__(cls)
        view._graph = rebuilt
        base_edge_mask = bytearray(excluded_edges) if excluded_edges is not None else bytearray(base.edge_count)
        view._excluded_edges = base_edge_mask + bytearray(len(added_edges))
        view._added_graph = rebuilt  # prevent GC
        view._excluded_nodes = (
            bytearray(excluded_nodes) + bytearray(len(merged_nodes) - base.node_count)
            if excluded_nodes is not None
            else None
        )
        return view

    def with_edges(
        self,
        added_edges: Sequence[tuple[NodeIdT, NodeIdT]],
        added_branch_ids: Sequence[BranchIdT] | None = None,
    ) -> GraphView[NodeIdT, BranchIdT]:
        """Create a new view adding extra edges to this view.

        The exclusions of this view are kept. Base edges keep their indices
        and the added edges are appended, see ``Graph.with_edges``.
        """
        return GraphView._with_additions(
            self._graph,
            excluded_edges=self._excluded_edges,
            excluded_nodes=self._excluded_nodes,
            added_edges=added_edges,
            added_branch_ids=added_branch_ids,
        )

    def without_nodes(
        self,
        node_ids: Collection[NodeIdT],
    ) -> GraphView[NodeIdT, BranchIdT]:
        """Create a new view also excluding the given nodes."""
        node_id_to_idx = self._graph._get_node_id_to_idx()
        excluded_nodes = bytearray(self._excluded_nodes) if self._excluded_nodes else bytearray(self._graph.node_count)
        for nid in node_ids:
            i = node_id_to_idx.get(nid)
            if i is not None:
                excluded_nodes[i] = 1
        view = object.__new__(GraphView)
        view._graph = self._graph
        view._excluded_edges = bytearray(self._excluded_edges)
        view._added_graph = self._added_graph
        view._excluded_nodes = excluded_nodes
        return view

    def without_branches(
        self,
        branch_ids: Collection[BranchIdT],
    ) -> GraphView[NodeIdT, BranchIdT]:
        """Create a new view also excluding the given branches by ID.

        Every edge carrying one of the ids is excluded. Requires the base
        Graph to have been constructed with branch_ids.
        """
        return self.without_edges(self._graph._edge_indices_of_branches(branch_ids))

    def without_edges(
        self,
        edge_indices: Collection[int],
    ) -> GraphView[NodeIdT, BranchIdT]:
        """Create a new view also excluding the given edges."""
        new_excluded_edges = bytearray(self._excluded_edges)
        for idx in edge_indices:
            new_excluded_edges[idx] = 1
        view = object.__new__(GraphView)
        view._graph = self._graph
        view._excluded_edges = new_excluded_edges
        view._added_graph = self._added_graph
        view._excluded_nodes = bytearray(self._excluded_nodes) if self._excluded_nodes else None
        return view

    def _require_undirected(self, method_name: str) -> None:
        if self._graph._directed:
            raise TypeError(f"{method_name} is not defined for directed graphs")  # noqa: TRY003

    def _require_directed(self, method_name: str) -> None:
        if not self._graph._directed:
            raise TypeError(f"{method_name} requires a directed graph")  # noqa: TRY003

    def incident_edge_indices(self, node_id: NodeIdT) -> list[EdgeIndex]:
        """Return indices of all non-excluded edges incident to the given node.

        For directed graphs, returns only outgoing edges (src=node_id).
        """
        result: list[EdgeIndex] = _incident_edges_ctx(
            self._graph._ctx, node_id, self._excluded_edges, self._excluded_nodes
        )
        return result

    def outgoing_edge_indices(self, node_id: NodeIdT) -> list[EdgeIndex]:
        """Return indices of all non-excluded outgoing edges. Directed graphs only."""
        self._require_directed("outgoing_edge_indices")
        return self.incident_edge_indices(node_id)

    def incoming_edge_indices(self, node_id: NodeIdT) -> list[EdgeIndex]:
        """Return indices of all non-excluded incoming edges. Directed graphs only."""
        result: list[EdgeIndex] = _incoming_edges_ctx(
            self._graph._ctx, node_id, self._excluded_edges, self._excluded_nodes
        )
        return result

    def neighbors(self, node_id: NodeIdT) -> set[NodeIdT]:
        """Return the set of neighbor node IDs, respecting edge and node exclusions.

        For directed graphs, returns successors (outgoing neighbors).
        """
        result: set[NodeIdT] = _neighbors_ctx(self._graph._ctx, node_id, self._excluded_edges, self._excluded_nodes)
        return result

    def successors(self, node_id: NodeIdT) -> set[NodeIdT]:
        """Return the set of successor node IDs. Directed graphs only."""
        self._require_directed("successors")
        return self.neighbors(node_id)

    def predecessors(self, node_id: NodeIdT) -> set[NodeIdT]:
        """Return the set of predecessor node IDs. Directed graphs only."""
        result: set[NodeIdT] = _predecessors_ctx(self._graph._ctx, node_id, self._excluded_edges, self._excluded_nodes)
        return result

    def degree(self, node_id: NodeIdT) -> int:
        """Return the number of non-excluded edges incident to the node.

        For undirected graphs, self-loops are counted twice (via CSR).
        For directed graphs, returns out-degree.
        """
        result: int = _degree_ctx(self._graph._ctx, node_id, self._excluded_edges, self._excluded_nodes)
        return result

    def out_degree(self, node_id: NodeIdT) -> int:
        """Return the out-degree of the node. Directed graphs only."""
        self._require_directed("out_degree")
        return self.degree(node_id)

    def in_degree(self, node_id: NodeIdT) -> int:
        """Return the in-degree of the node. Directed graphs only."""
        result: int = _in_degree_ctx(self._graph._ctx, node_id, self._excluded_edges, self._excluded_nodes)
        return result

    def bridges_with_branch_ids(self) -> list[tuple[NodeIdT, NodeIdT, BranchIdT]]:
        """Return bridge edges as (node_id, node_id, branch_id) triples.

        Requires the base Graph to have been constructed with branch_ids.
        """
        self._require_undirected("bridges_with_branch_ids")
        if self._graph._branch_ids is None:
            raise ValueError("no branch_ids")  # noqa: TRY003 — short, no custom class needed
        bridge_list = self.bridges()
        result: list[tuple[NodeIdT, NodeIdT, BranchIdT]] = []
        for u, v in bridge_list:
            result.extend(
                (u, v, self._graph._branch_ids[edge_idx])
                for edge_idx in self._graph.edge_indices(u, v)
                if not self._excluded_edges[edge_idx]
            )
        return result

    def split_node(
        self,
        node_id: NodeIdT,
        new_node_id: NodeIdT,
        edge_indices_to_new_node: Collection[int],
    ) -> GraphView[NodeIdT, BranchIdT]:
        """Split a node by rerouting specified edges to a new node.

        Creates a view where the edges identified by ``edge_indices_to_new_node``
        are detached from ``node_id`` and reattached to ``new_node_id``.
        The remaining edges of ``node_id`` stay in place.
        """
        edges = self._graph._edges
        base_branch_ids = self._graph._branch_ids
        if edges is None:
            raise ValueError("split_node requires edge-pair construction")  # noqa: TRY003
        node_id_to_idx = self._graph._get_node_id_to_idx()
        if node_id not in node_id_to_idx:
            raise ValueError(f"node {node_id} is not in the graph")  # noqa: TRY003
        if new_node_id in node_id_to_idx:
            raise ValueError(f"node {new_node_id} already exists in the graph")  # noqa: TRY003
        rerouted_edges: list[tuple[NodeIdT, NodeIdT]] = []
        for edge_idx in edge_indices_to_new_node:
            u, v = edges[edge_idx]
            if u == node_id:
                rerouted_edges.append((new_node_id, v))
            elif v == node_id:
                rerouted_edges.append((u, new_node_id))
            else:
                raise ValueError(  # noqa: TRY003
                    f"edge {edge_idx} ({u}, {v}) is not incident to node {node_id}"
                )
        rerouted_branch_ids = _rerouted_branch_ids(base_branch_ids, edge_indices_to_new_node)
        return self.without_edges(edge_indices_to_new_node).with_edges(rerouted_edges, rerouted_branch_ids)

    def all_edge_paths(
        self,
        source: NodeIdT,
        targets: NodeIdT | Collection[NodeIdT],
        cutoff: int | None = None,
        *,
        node_simple: bool = False,
        ignore_self_loops: bool = False,
    ) -> list[list[EdgeIndex]]:
        """Find all paths from source to targets using each edge at most once.

        Returns a list of paths. Each path is a list of edge indices.
        Respects both excluded edges and excluded nodes.

        node_simple: if True, each node may be visited at most once per path.
        ignore_self_loops: if True, self-loops are never traversed.
        """
        tgt_list = [targets] if isinstance(targets, int) else list(targets)
        c = cutoff if cutoff is not None else -1
        result: list[list[EdgeIndex]] = _all_edge_paths_ctx(
            self._graph._ctx,
            source,
            tgt_list,
            c,
            self._excluded_edges,
            self._excluded_nodes,
            node_simple,
            ignore_self_loops,
        )
        return result

    def connected_components(self) -> Generator[set[NodeIdT], None, None]:
        """Yield each connected component as a set of original node IDs."""
        self._require_undirected("connected_components")
        yield from _cc_ctx(self._graph._ctx, self._excluded_edges, self._excluded_nodes)

    def connected_component(self, node_id: NodeIdT) -> set[NodeIdT]:
        """Return the connected component containing ``node_id``, see ``Graph.connected_component``.

        Excluded edges do not connect. A node excluded from the view belongs
        to no component and raises ``ValueError``, as does an unknown node.
        """
        self._require_undirected("connected_component")
        return _connected_component_of(self._graph, node_id, self._excluded_edges, self._excluded_nodes)

    def connected_components_with_branch_ids(self) -> Generator[tuple[set[NodeIdT], set[BranchIdT]], None, None]:
        """Yield (node_id_set, branch_id_set) for each connected component.

        Requires the base Graph to have been constructed with branch_ids.

        Excluded edges are removed from connectivity and their branch IDs
        are dropped. Excluded nodes are removed from the output node sets
        but their edges still contribute to connectivity and branch ID sets.
        """
        self._require_undirected("connected_components_with_branch_ids")
        if self._graph._branch_ids is None:
            raise ValueError("no branch_ids")  # noqa: TRY003 — short, no custom class needed
        yield from _cc_branches_ctx(
            self._graph._ctx,
            self._graph._branch_ids,
            self._excluded_edges,
            self._excluded_nodes,
        )

    def strongly_connected_components(self) -> Generator[set[NodeIdT], None, None]:
        """Yield each strongly connected component as a set of node IDs."""
        self._require_directed("strongly_connected_components")
        yield from _scc_ctx(self._graph._ctx, self._excluded_edges, self._excluded_nodes)

    def weakly_connected_components(self) -> Generator[set[NodeIdT], None, None]:
        """Yield each weakly connected component as a set of node IDs."""
        self._require_directed("weakly_connected_components")
        yield from _cc_ctx(self._graph._ctx, self._excluded_edges, self._excluded_nodes)

    def topological_sort(self) -> list[NodeIdT]:
        """Return nodes in topological order (Kahn's algorithm).

        Raises ValueError if the graph contains a cycle.
        """
        self._require_directed("topological_sort")
        result: list[NodeIdT] = _toposort_ctx(self._graph._ctx, self._excluded_edges, self._excluded_nodes)
        return result

    def bridges(self) -> list[tuple[NodeIdT, NodeIdT]]:
        """Return bridge edges as (node_id, node_id) pairs."""
        self._require_undirected("bridges")
        result: list[tuple[NodeIdT, NodeIdT]] = _bridges_ctx(
            self._graph._ctx, self._excluded_edges, self._excluded_nodes
        )
        return result

    def articulation_points(self) -> set[NodeIdT]:
        """Return the set of articulation points."""
        self._require_undirected("articulation_points")
        result: set[NodeIdT] = _ap_ctx(self._graph._ctx, self._excluded_edges, self._excluded_nodes)
        return result

    def biconnected_components(self) -> Generator[set[NodeIdT], None, None]:
        """Yield each biconnected component as a set of node IDs."""
        self._require_undirected("biconnected_components")
        yield from _bcc_ctx(self._graph._ctx, self._excluded_edges, self._excluded_nodes)

    def component_labels(self) -> memoryview:
        """Connected component label per node index as an int32 view, see ``Graph.component_labels``.

        Excluded edges do not connect and excluded nodes are labelled -1.
        """
        result: bytes = _component_labels_ctx(self._graph._ctx, self._excluded_edges, self._excluded_nodes)
        return memoryview(result).cast("i")

    def quotient_edges(self, labels: memoryview) -> tuple[memoryview, memoryview, memoryview, memoryview]:
        """Split the unmasked edges by endpoint label, see ``Graph.quotient_edges``; excluded edges are skipped."""
        return _quotient_edge_views(_quotient_edges_ctx(self._graph._ctx, labels, self._excluded_edges))

    def series_parallel_reduce(
        self,
        terminal_mask: NodeMask,
        protected_mask: NodeMask,
        *,
        pendant_keep_mask: NodeMask | None = None,
        series_blocked_mask: NodeMask | None = None,
    ) -> ReductionLog:
        """Reduce under the view's masks, see ``Graph.series_parallel_reduce``; excluded nodes and edges never move."""
        self._require_undirected("series_parallel_reduce")
        raw: tuple[bytes, ...] = _series_parallel_reduce_ctx(
            self._graph._ctx,
            terminal_mask,
            protected_mask,
            self._excluded_edges,
            self._excluded_nodes,
            pendant_keep_mask,
            series_blocked_mask,
        )
        return ReductionLog.from_buffers(raw)

    def degrees(self) -> memoryview:
        """Degree per node index as an int32 view under the masks, see ``Graph.degrees``; excluded nodes get 0."""
        result: bytes = _degrees_ctx(self._graph._ctx, self._excluded_edges, self._excluded_nodes)
        return memoryview(result).cast("i")

    def bcc_edge_labels(self) -> memoryview:
        """Biconnected component id per edge index under the masks, see ``Graph.bcc_edge_labels``.

        Excluded edges and edges at an excluded node get -1.
        """
        self._require_undirected("bcc_edge_labels")
        result: bytes = _bcc_edge_labels_ctx(self._graph._ctx, self._excluded_edges, self._excluded_nodes)
        return memoryview(result).cast("i")

    def cycle_basis(self) -> list[list[NodeIdT]]:
        """Return a fundamental cycle basis as a list of cycles."""
        self._require_undirected("cycle_basis")
        result: list[list[NodeIdT]] = _cycle_basis_ctx(self._graph._ctx, self._excluded_edges, self._excluded_nodes)
        return result

    def dag_longest_path(self, weights: list[float] | None = None) -> list[NodeIdT]:
        """Return the longest path in the DAG as a list of node IDs."""
        self._require_directed("dag_longest_path")
        result: list[NodeIdT] = _dag_longest_path_ctx(
            self._graph._ctx, weights, self._excluded_edges, self._excluded_nodes
        )
        return result

    def bfs(self, source: NodeIdT) -> list[NodeIdT]:
        """Return nodes visited in BFS order from source."""
        result: list[NodeIdT] = _bfs_ctx(self._graph._ctx, source, self._excluded_edges, self._excluded_nodes)
        return result

    def shortest_path(
        self,
        weights: list[float],
        source: NodeIdT,
        target: NodeIdT,
    ) -> list[NodeIdT]:
        """Return the shortest weighted path from source to target.

        Runs a bidirectional Dijkstra, so only a small part of a large graph is
        settled. Pass ``weights`` as a float64 buffer (a numpy ``float64`` array
        or ``array.array("d", ...)``) to hand it to C as is; a list of floats is
        converted element by element first, which costs more than the search.
        """
        _dist, path = _dijkstra_ctx(
            self._graph._ctx,
            weights,
            source,
            target,
            self._excluded_edges,
            self._excluded_nodes,
        )
        result: list[NodeIdT] = path
        return result

    def shortest_path_lengths(
        self,
        weights: list[float],
        source: NodeIdT,
        cutoff: float | None = None,
    ) -> dict[NodeIdT, float]:
        """Return {node_id: distance} for all nodes reachable from source."""
        c = cutoff if cutoff is not None else -1.0
        result: dict[NodeIdT, float] = _sssp_ctx(
            self._graph._ctx,
            weights,
            source,
            c,
            self._excluded_edges,
            self._excluded_nodes,
        )
        return result

    def multi_source_shortest_path_lengths(
        self,
        weights: list[float],
        sources: Sequence[NodeIdT],
        cutoff: float | None = None,
    ) -> dict[NodeIdT, float]:
        """Return {node_id: distance} from nearest source to each reachable node."""
        c = cutoff if cutoff is not None else -1.0
        result: dict[NodeIdT, float] = _msdijk_ctx(
            self._graph._ctx,
            weights,
            sources,
            c,
            self._excluded_edges,
            self._excluded_nodes,
        )
        return result

    def eccentricity(self, weights: list[float], source: NodeIdT) -> float:
        """Return the eccentricity of source (max shortest-path distance)."""
        lengths = self.shortest_path_lengths(weights, source)
        if not lengths:
            return 0.0
        return max(lengths.values())


def for_each_edge_excluded(
    graph: Graph[NodeIdT, BranchIdT],
    algorithm: str,
    edge_indices: Iterable[int] | None = None,
    **algorithm_kwargs: object,
) -> Iterator[tuple[EdgeIndex, object]]:
    """Run an algorithm once per excluded edge, yielding (edge_index, result).

    Reuses a single mask bytearray, toggling one bit per iteration.
    If edge_indices is None, iterates over all edges.
    """
    excluded_edges = bytearray(graph.edge_count)
    indices = edge_indices if edge_indices is not None else range(graph.edge_count)
    for idx in indices:
        excluded_edges[idx] = 1
        view = GraphView._from_excluded_edges(graph, excluded_edges)
        result = getattr(view, algorithm)(**algorithm_kwargs)
        if isinstance(result, types.GeneratorType):
            result = list(result)
        yield EdgeIndex(idx), result
        excluded_edges[idx] = 0


# ── DAG structure learning ──


def hill_climb_k2(
    data: list[list[int]],
    cardinalities: list[int],
    *,
    max_indegree: int = 1,
    tabu_length: int = 100,
    epsilon: float = 1e-4,
    max_iter: int = 1_000_000,
) -> list[tuple[int, int]]:
    """Learn DAG structure via greedy hill-climb with K2 scoring.

    Finds the directed acyclic graph that best explains the data according
    to the K2 Bayesian scoring function, using a greedy search over single-edge
    add/remove/flip operations.

    Args:
        data: Dataset as list of rows. Each row is a list of int values
              in range ``[0, cardinality)``. Shape: ``(n_samples, n_vars)``.
        cardinalities: Number of possible values per variable.
        max_indegree: Maximum number of parents per node. Default 1.
        tabu_length: Number of recent operations to forbid (prevents cycling).
        epsilon: Minimum score improvement to continue searching.
        max_iter: Maximum number of hill-climb iterations.

    Returns:
        List of ``(parent, child)`` edge tuples representing the learned DAG.
    """
    result: list[tuple[int, int]] = _hill_climb_k2(
        data,
        cardinalities,
        max_indegree,
        tabu_length,
        epsilon,
        max_iter,
    )
    return result


def estimate_cpds(
    data: list[list[int]],
    cardinalities: list[int],
    edges: list[tuple[int, int]],
) -> dict[int, list[list[float]]]:
    """Estimate CPDs for all variables given DAG edges and data.

    Uses Laplace (add-1) smoothing. Each CPD is a list of probability
    distributions, one per parent configuration. Each distribution sums to 1.0.

    Args:
        data: Dataset as list of rows (n_samples x n_vars).
        cardinalities: Number of possible values per variable.
        edges: List of (parent, child) edge tuples from the learned DAG.

    Returns:
        Dict mapping variable index to its CPD:
        ``{var: [[p(var=0|pa_config), p(var=1|pa_config), ...] for each pa_config]}``
    """
    result: dict[int, list[list[float]]] = _estimate_cpds(data, cardinalities, edges)
    return result


def k2_local_score(
    data: list[list[int]],
    cardinalities: list[int],
    child: int,
    parents: list[int],
) -> float:
    """Compute K2 local score for a variable given its parents.

    Args:
        data: Dataset as list of rows (n_samples x n_vars).
        cardinalities: Number of possible values per variable.
        child: Index of the target variable.
        parents: List of parent variable indices.

    Returns:
        K2 score (higher is better).
    """
    result: float = _k2_local_score(data, cardinalities, child, parents)
    return result
