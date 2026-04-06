"""Fast graph algorithms via C extensions: union-find, Tarjan's, BFS, Dijkstra."""

import types
from collections import deque
from collections.abc import Collection, Generator, Iterable, Iterator
from typing import Any

from cgraph._core import ap_ctx as _ap_ctx
from cgraph._core import ap_nid as _ap_nid
from cgraph._core import bcc_ctx as _bcc_ctx
from cgraph._core import bcc_nid as _bcc_nid
from cgraph._core import bfs_ctx as _bfs_ctx
from cgraph._core import bfs_nid as _bfs_nid
from cgraph._core import bridges_ctx as _bridges_ctx
from cgraph._core import bridges_nid as _bridges_nid
from cgraph._core import cc_ctx as _cc_ctx
from cgraph._core import cc_nid as _cc_nid
from cgraph._core import cc_nid_split as _cc_nid_split
from cgraph._core import connected_components_with_branches_remapped as _cc_branches_remapped
from cgraph._core import dijkstra_ctx as _dijkstra_ctx
from cgraph._core import dijkstra_nid as _dijkstra_nid
from cgraph._core import graph_edge_count as _graph_edge_count
from cgraph._core import graph_node_count as _graph_node_count
from cgraph._core import msdijk_ctx as _msdijk_ctx
from cgraph._core import msdijk_nid as _msdijk_nid
from cgraph._core import parse_graph as _parse_graph
from cgraph._core import sssp_ctx as _sssp_ctx
from cgraph._core import sssp_nid as _sssp_nid

__all__ = [
    "BranchId",
    "Graph",
    "GraphView",
    "NodeId",
    "articulation_points",
    "bfs",
    "biconnected_components",
    "bridges",
    "connected_components",
    "connected_components_with_branch_ids",
    "eccentricity",
    "for_each_edge_excluded",
    "multi_source_shortest_path_lengths",
    "nodes_on_simple_paths",
    "shortest_path",
    "shortest_path_lengths",
    "two_edge_connected_components",
]

NodeId = int
BranchId = int


# ── Connected Components (legacy index-based API kept for branch_ids) ──


def connected_components(
    node_ids: list[NodeId],
    edges_or_src: list[tuple[int, int]] | list[int],
    dst: list[int] | None = None,
) -> Generator[set[NodeId], None, None]:
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
    node_ids: list[NodeId],
    edges: list[tuple[int, int]],
    branch_ids: list[int],
) -> Generator[tuple[set[NodeId], set[BranchId]], None, None]:
    """
    Yield (node_id_set, branch_id_set) with original node IDs.

    Edges are pairs of original node IDs (not indices).
    """
    # Branch variant still uses index-based edges internally
    idx = {nid: i for i, nid in enumerate(node_ids)}
    idx_edges = [(idx[u], idx[v]) for u, v in edges]
    yield from _cc_branches_remapped(node_ids, idx_edges, branch_ids)


# ── Phase 1: Structural graph primitives ──


def bridges(
    node_ids: list[NodeId],
    edges: list[tuple[NodeId, NodeId]],
) -> list[tuple[NodeId, NodeId]]:
    """Return bridge edges as (node_id, node_id) pairs."""
    return _bridges_nid(node_ids, edges)


def articulation_points(
    node_ids: list[NodeId],
    edges: list[tuple[NodeId, NodeId]],
) -> set[NodeId]:
    """Return the set of articulation points."""
    return _ap_nid(node_ids, edges)


def biconnected_components(
    node_ids: list[NodeId],
    edges: list[tuple[NodeId, NodeId]],
) -> Generator[set[NodeId], None, None]:
    """Yield each biconnected component as a set of node IDs."""
    yield from _bcc_nid(node_ids, edges)


def bfs(
    node_ids: list[NodeId],
    edges: list[tuple[NodeId, NodeId]],
    source: NodeId,
) -> list[NodeId]:
    """Return nodes visited in BFS order from source."""
    return _bfs_nid(node_ids, edges, source)


# ── Phase 2: Weighted graph algorithms ──


def shortest_path(
    node_ids: list[NodeId],
    edges: list[tuple[NodeId, NodeId]],
    weights: list[float],
    source: NodeId,
    target: NodeId,
) -> list[NodeId]:
    """Return the shortest weighted path from source to target."""
    _dist, path = _dijkstra_nid(node_ids, edges, weights, source, target)
    return path


def shortest_path_lengths(
    node_ids: list[NodeId],
    edges: list[tuple[NodeId, NodeId]],
    weights: list[float],
    source: NodeId,
    cutoff: float | None = None,
) -> dict[NodeId, float]:
    """Return {node_id: distance} for all nodes reachable from source."""
    c = cutoff if cutoff is not None else -1.0
    return _sssp_nid(node_ids, edges, weights, source, c)


def multi_source_shortest_path_lengths(
    node_ids: list[NodeId],
    edges: list[tuple[NodeId, NodeId]],
    weights: list[float],
    sources: list[NodeId],
    cutoff: float | None = None,
) -> dict[NodeId, float]:
    """Return {node_id: distance} from nearest source to each reachable node."""
    c = cutoff if cutoff is not None else -1.0
    return _msdijk_nid(node_ids, edges, weights, sources, c)


def eccentricity(
    node_ids: list[NodeId],
    edges: list[tuple[NodeId, NodeId]],
    weights: list[float],
    source: NodeId,
) -> float:
    """Return the eccentricity of source (max shortest-path distance)."""
    lengths = shortest_path_lengths(node_ids, edges, weights, source)
    if not lengths:
        return 0.0
    return max(lengths.values())


# ── Phase 3: Composite algorithms ──


def two_edge_connected_components(
    node_ids: list[NodeId],
    edges: list[tuple[NodeId, NodeId]],
) -> Generator[set[NodeId], None, None]:
    """Yield 2-edge-connected components (bridges removed, then CC)."""
    bridge_set: set[tuple[NodeId, NodeId]] = set()
    for u, v in bridges(node_ids, edges):
        bridge_set.add((min(u, v), max(u, v)))

    non_bridge_edges = [(u, v) for u, v in edges if (min(u, v), max(u, v)) not in bridge_set]
    yield from connected_components(node_ids, non_bridge_edges)


def nodes_on_simple_paths(
    node_ids: list[NodeId],
    edges: list[tuple[NodeId, NodeId]],
    source: NodeId,
    targets: list[NodeId],
) -> set[NodeId]:
    """Return all nodes on any simple path from source to any target.

    Uses the block-cut tree: finds biconnected components, builds the
    block-cut tree, then collects all nodes in blocks on the tree path
    from source to each target.
    """
    n = len(node_ids)
    if n == 0:
        return set()

    tgts = set(targets)
    result: set[NodeId] = set()
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
    node_ids: list[NodeId],
    blocks: list[set[NodeId]],
) -> tuple[dict[NodeId, list[int]], list[list[int]], dict[NodeId, int]]:
    """Build block-cut tree from biconnected components.

    Returns (node_blocks, tree_adj, ap_id).
    """
    num_blocks = len(blocks)
    node_blocks: dict[NodeId, list[int]] = {}
    for bi, block in enumerate(blocks):
        for v in block:
            node_blocks.setdefault(v, []).append(bi)

    ap_id: dict[NodeId, int] = {}
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


def _collect_path_nodes(  # noqa: PLR0913
    node_ids: list[NodeId],
    blocks: list[set[NodeId]],
    tree: tuple[dict[NodeId, list[int]], list[list[int]], dict[NodeId, int]],
    src: NodeId,
    tgts: set[NodeId],
    result: set[NodeId],
) -> set[NodeId]:
    """BFS on block-cut tree, trace paths, collect nodes."""
    node_blocks, tree_adj, ap_id = tree
    num_blocks = len(blocks)

    def tn(v: NodeId) -> int:
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


class Graph:
    """Parsed graph that supports multiple algorithm calls without re-parsing.

    Parses node IDs and edges once into an internal C structure (IntMap + EdgeList
    + CSR adjacency list), then reuses that parsed state across all algorithm calls.

    Two calling conventions:
        Graph(node_ids, edges)          — edges as pairs of node IDs
        Graph(node_ids, src, dst)       — two flat lists of node IDs
    """

    __slots__ = ("_ctx", "_node_ids", "_edges")

    def __init__(
        self,
        node_ids: list[NodeId],
        edges_or_src: list[tuple[int, int]] | list[int],
        dst: list[int] | None = None,
    ) -> None:
        self._node_ids = node_ids
        self._edges = edges_or_src if dst is None else None
        if dst is not None:
            self._ctx = _parse_graph(node_ids, edges_or_src, dst)
        else:
            self._ctx = _parse_graph(node_ids, edges_or_src)

    @property
    def edge_count(self) -> int:
        """Number of edges in the graph."""
        return _graph_edge_count(self._ctx)

    @property
    def node_count(self) -> int:
        """Number of nodes in the graph."""
        return _graph_node_count(self._ctx)

    def edge_indices(self, u: NodeId, v: NodeId) -> list[int]:
        """Return indices of edges between u and v (list, for multigraph support)."""
        edges = self._edges
        if edges is None:
            return []
        result = []
        for i, (a, b) in enumerate(edges):
            if (a == u and b == v) or (a == v and b == u):
                result.append(i)
        return result

    def without_edges(
        self,
        edge_indices: Collection[int],
    ) -> "GraphView":
        """Create a lightweight view with the given edges excluded."""
        return GraphView(self, edge_indices)

    def connected_components(self) -> Generator[set[NodeId], None, None]:
        """Yield each connected component as a set of original node IDs."""
        yield from _cc_ctx(self._ctx)

    def bridges(self) -> list[tuple[NodeId, NodeId]]:
        """Return bridge edges as (node_id, node_id) pairs."""
        return _bridges_ctx(self._ctx)

    def articulation_points(self) -> set[NodeId]:
        """Return the set of articulation points."""
        return _ap_ctx(self._ctx)

    def biconnected_components(self) -> Generator[set[NodeId], None, None]:
        """Yield each biconnected component as a set of node IDs."""
        yield from _bcc_ctx(self._ctx)

    def bfs(self, source: NodeId) -> list[NodeId]:
        """Return nodes visited in BFS order from source."""
        return _bfs_ctx(self._ctx, source)

    def shortest_path(
        self,
        weights: list[float],
        source: NodeId,
        target: NodeId,
    ) -> list[NodeId]:
        """Return the shortest weighted path from source to target."""
        _dist, path = _dijkstra_ctx(self._ctx, weights, source, target)
        return path

    def shortest_path_lengths(
        self,
        weights: list[float],
        source: NodeId,
        cutoff: float | None = None,
    ) -> dict[NodeId, float]:
        """Return {node_id: distance} for all nodes reachable from source."""
        c = cutoff if cutoff is not None else -1.0
        return _sssp_ctx(self._ctx, weights, source, c)

    def multi_source_shortest_path_lengths(
        self,
        weights: list[float],
        sources: list[NodeId],
        cutoff: float | None = None,
    ) -> dict[NodeId, float]:
        """Return {node_id: distance} from nearest source to each reachable node."""
        c = cutoff if cutoff is not None else -1.0
        return _msdijk_ctx(self._ctx, weights, sources, c)

    def eccentricity(self, weights: list[float], source: NodeId) -> float:
        """Return the eccentricity of source (max shortest-path distance)."""
        lengths = self.shortest_path_lengths(weights, source)
        if not lengths:
            return 0.0
        return max(lengths.values())

    def two_edge_connected_components(
        self,
    ) -> Generator[set[NodeId], None, None]:
        """Yield 2-edge-connected components (bridges removed, then CC)."""
        bridge_set: set[tuple[NodeId, NodeId]] = set()
        for u, v in self.bridges():
            bridge_set.add((min(u, v), max(u, v)))

        non_bridge_edges = [(u, v) for u, v in (self._edges or []) if (min(u, v), max(u, v)) not in bridge_set]
        yield from connected_components(self._node_ids, non_bridge_edges)

    def nodes_on_simple_paths(
        self,
        source: NodeId,
        targets: list[NodeId],
    ) -> set[NodeId]:
        """Return all nodes on any simple path from source to any target."""
        n = len(self._node_ids)
        if n == 0:
            return set()

        tgts = set(targets)
        result: set[NodeId] = set()
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


class GraphView:
    """Lightweight view of a Graph with excluded edges.

    Shares the base graph's parsed data (IntMap, CSR). Only holds a
    bytearray mask of excluded edge indices. Creating a view is O(m)
    worst case, O(k) if built from scratch with k exclusions.

    Edges are identified by their index in the original edge list
    (the order in which they were passed to ``Graph()``).
    """

    __slots__ = ("_graph", "_mask")

    def __init__(
        self,
        graph: Graph,
        excluded_edge_indices: Collection[int],
    ) -> None:
        self._graph = graph
        self._mask = bytearray(graph.edge_count)
        for idx in excluded_edge_indices:
            self._mask[idx] = 1

    @classmethod
    def _from_mask(cls, graph: Graph, mask: bytearray) -> "GraphView":
        """Create a view from an existing mask (no copy)."""
        view = object.__new__(cls)
        view._graph = graph
        view._mask = mask
        return view

    def connected_components(self) -> Generator[set[NodeId], None, None]:
        """Yield each connected component as a set of original node IDs."""
        yield from _cc_ctx(self._graph._ctx, self._mask)

    def bridges(self) -> list[tuple[NodeId, NodeId]]:
        """Return bridge edges as (node_id, node_id) pairs."""
        return _bridges_ctx(self._graph._ctx, self._mask)

    def articulation_points(self) -> set[NodeId]:
        """Return the set of articulation points."""
        return _ap_ctx(self._graph._ctx, self._mask)

    def biconnected_components(self) -> Generator[set[NodeId], None, None]:
        """Yield each biconnected component as a set of node IDs."""
        yield from _bcc_ctx(self._graph._ctx, self._mask)

    def bfs(self, source: NodeId) -> list[NodeId]:
        """Return nodes visited in BFS order from source."""
        return _bfs_ctx(self._graph._ctx, source, self._mask)

    def shortest_path(
        self,
        weights: list[float],
        source: NodeId,
        target: NodeId,
    ) -> list[NodeId]:
        """Return the shortest weighted path from source to target."""
        _dist, path = _dijkstra_ctx(
            self._graph._ctx,
            weights,
            source,
            target,
            self._mask,
        )
        return path

    def shortest_path_lengths(
        self,
        weights: list[float],
        source: NodeId,
        cutoff: float | None = None,
    ) -> dict[NodeId, float]:
        """Return {node_id: distance} for all nodes reachable from source."""
        c = cutoff if cutoff is not None else -1.0
        return _sssp_ctx(self._graph._ctx, weights, source, c, self._mask)

    def multi_source_shortest_path_lengths(
        self,
        weights: list[float],
        sources: list[NodeId],
        cutoff: float | None = None,
    ) -> dict[NodeId, float]:
        """Return {node_id: distance} from nearest source to each reachable node."""
        c = cutoff if cutoff is not None else -1.0
        return _msdijk_ctx(self._graph._ctx, weights, sources, c, self._mask)

    def eccentricity(self, weights: list[float], source: NodeId) -> float:
        """Return the eccentricity of source (max shortest-path distance)."""
        lengths = self.shortest_path_lengths(weights, source)
        if not lengths:
            return 0.0
        return max(lengths.values())


def for_each_edge_excluded(
    graph: Graph,
    algorithm: str,
    edge_indices: Iterable[int] | None = None,
    **algorithm_kwargs: Any,
) -> Iterator[tuple[int, Any]]:
    """Run an algorithm once per excluded edge, yielding (edge_index, result).

    Reuses a single mask bytearray, toggling one bit per iteration.
    If edge_indices is None, iterates over all edges.
    """
    mask = bytearray(graph.edge_count)
    indices = edge_indices if edge_indices is not None else range(graph.edge_count)
    for idx in indices:
        mask[idx] = 1
        view = GraphView._from_mask(graph, mask)
        result = getattr(view, algorithm)(**algorithm_kwargs)
        # Materialize generators since the mask is shared and will be reset
        if isinstance(result, types.GeneratorType):
            result = list(result)
        yield idx, result
        mask[idx] = 0
