"""Fast graph algorithms via C extensions: union-find, Tarjan's, BFS, Dijkstra."""

from collections import deque
from collections.abc import Generator

from cgraph._core import articulation_points as _articulation_points
from cgraph._core import bfs as _bfs
from cgraph._core import biconnected_components as _biconnected_components
from cgraph._core import bridges as _bridges
from cgraph._core import connected_components_remapped as _cc_remapped
from cgraph._core import connected_components_with_branches_remapped as _cc_branches_remapped
from cgraph._core import dijkstra as _dijkstra
from cgraph._core import multi_source_dijkstra as _multi_source_dijkstra
from cgraph._core import sssp_lengths as _sssp_lengths

__all__ = [
    "BranchId",
    "NodeId",
    "articulation_points",
    "bfs",
    "biconnected_components",
    "bridges",
    "connected_components",
    "connected_components_with_branch_ids",
    "eccentricity",
    "multi_source_shortest_path_lengths",
    "nodes_on_simple_paths",
    "shortest_path",
    "shortest_path_lengths",
    "two_edge_connected_components",
]

NodeId = int
BranchId = int


def connected_components(
    node_ids: list[NodeId],
    edges: list[tuple[int, int]],
) -> Generator[set[NodeId], None, None]:
    """
    Yield each connected component as a set of original node IDs.

    node_ids maps internal index -> original NodeId. The remapping
    happens entirely in C — no Python loop needed.
    """
    yield from _cc_remapped(node_ids, edges)


def connected_components_with_branch_ids(
    node_ids: list[NodeId],
    edges: list[tuple[int, int]],
    branch_ids: list[int],
) -> Generator[tuple[set[NodeId], set[BranchId]], None, None]:
    """
    Yield (node_id_set, branch_id_set) with original node IDs.

    node_ids maps internal index -> original NodeId. The remapping
    happens entirely in C — no Python loop needed.
    """
    yield from _cc_branches_remapped(node_ids, edges, branch_ids)


# ── Phase 1: Structural graph primitives ──


def bridges(
    node_ids: list[NodeId],
    edges: list[tuple[int, int]],
) -> list[tuple[NodeId, NodeId]]:
    """Return bridge edges as (node_id, node_id) pairs using original IDs."""
    result = _bridges(len(node_ids), edges)
    return [(node_ids[u], node_ids[v]) for u, v in result]


def articulation_points(
    node_ids: list[NodeId],
    edges: list[tuple[int, int]],
) -> set[NodeId]:
    """Return the set of articulation points using original node IDs."""
    result = _articulation_points(len(node_ids), edges)
    return {node_ids[i] for i in result}


def biconnected_components(
    node_ids: list[NodeId],
    edges: list[tuple[int, int]],
) -> Generator[set[NodeId], None, None]:
    """Yield each biconnected component as a set of original node IDs."""
    result = _biconnected_components(len(node_ids), edges)
    for comp in result:
        yield {node_ids[i] for i in comp}


def bfs(
    node_ids: list[NodeId],
    edges: list[tuple[int, int]],
    source: NodeId,
) -> list[NodeId]:
    """Return nodes visited in BFS order from source, using original IDs."""
    idx = {nid: i for i, nid in enumerate(node_ids)}
    result = _bfs(len(node_ids), edges, idx[source])
    return [node_ids[i] for i in result]


# ── Phase 2: Weighted graph algorithms ──


def shortest_path(
    node_ids: list[NodeId],
    edges: list[tuple[int, int]],
    weights: list[float],
    source: NodeId,
    target: NodeId,
) -> list[NodeId]:
    """Return the shortest weighted path from source to target as original IDs."""
    idx = {nid: i for i, nid in enumerate(node_ids)}
    _dist, path = _dijkstra(
        len(node_ids), edges, weights, idx[source], idx[target],
    )
    return [node_ids[i] for i in path]


def shortest_path_lengths(
    node_ids: list[NodeId],
    edges: list[tuple[int, int]],
    weights: list[float],
    source: NodeId,
    cutoff: float | None = None,
) -> dict[NodeId, float]:
    """Return {node_id: distance} for all nodes reachable from source."""
    idx = {nid: i for i, nid in enumerate(node_ids)}
    c = cutoff if cutoff is not None else -1.0
    result = _sssp_lengths(len(node_ids), edges, weights, idx[source], c)
    return {node_ids[k]: v for k, v in result.items()}


def multi_source_shortest_path_lengths(
    node_ids: list[NodeId],
    edges: list[tuple[int, int]],
    weights: list[float],
    sources: list[NodeId],
    cutoff: float | None = None,
) -> dict[NodeId, float]:
    """Return {node_id: distance} from nearest source to each reachable node."""
    idx = {nid: i for i, nid in enumerate(node_ids)}
    src_indices = [idx[s] for s in sources]
    c = cutoff if cutoff is not None else -1.0
    result = _multi_source_dijkstra(
        len(node_ids), edges, weights, src_indices, c,
    )
    return {node_ids[k]: v for k, v in result.items()}


def eccentricity(
    node_ids: list[NodeId],
    edges: list[tuple[int, int]],
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
    edges: list[tuple[int, int]],
) -> Generator[set[NodeId], None, None]:
    """Yield 2-edge-connected components (bridges removed, then CC)."""
    n = len(node_ids)
    bridge_set: set[tuple[int, int]] = set()
    for u, v in _bridges(n, edges):
        bridge_set.add((min(u, v), max(u, v)))

    non_bridge_edges = [
        (u, v) for u, v in edges
        if (min(u, v), max(u, v)) not in bridge_set
    ]
    yield from connected_components(node_ids, non_bridge_edges)


def nodes_on_simple_paths(
    node_ids: list[NodeId],
    edges: list[tuple[int, int]],
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

    idx = {nid: i for i, nid in enumerate(node_ids)}
    src = idx[source]
    tgts = {idx[t] for t in targets}

    result_indices: set[int] = set()
    if src in tgts:
        result_indices.add(src)
        tgts.discard(src)
    if not tgts:
        return {node_ids[i] for i in result_indices}

    blocks = _biconnected_components(n, edges)
    if not blocks:
        return {node_ids[i] for i in result_indices}

    tree = _build_block_cut_tree(n, blocks)
    return _collect_path_nodes(
        node_ids, blocks, tree, src, tgts, result_indices,
    )


def _build_block_cut_tree(
    n: int,
    blocks: list[set[int]],
) -> tuple[list[list[int]], list[list[int]], dict[int, int]]:
    """Build block-cut tree from biconnected components.

    Returns (node_blocks, tree_adj, ap_id) where:
    - node_blocks[v] = list of block indices containing v
    - tree_adj[i] = adjacency list for tree node i
    - ap_id = mapping from graph AP nodes to tree node IDs
    """
    num_blocks = len(blocks)
    node_blocks: list[list[int]] = [[] for _ in range(n)]
    for bi, block in enumerate(blocks):
        for v in block:
            node_blocks[v].append(bi)

    ap_id: dict[int, int] = {}
    next_id = num_blocks
    for v in range(n):
        if len(node_blocks[v]) > 1:
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
    blocks: list[set[int]],
    tree: tuple[list[list[int]], list[list[int]], dict[int, int]],
    src: int,
    tgts: set[int],
    result_indices: set[int],
) -> set[NodeId]:
    """BFS on block-cut tree, trace paths, collect nodes."""
    node_blocks, tree_adj, ap_id = tree
    num_blocks = len(blocks)

    def tn(v: int) -> int:
        if v in ap_id:
            return ap_id[v]
        return node_blocks[v][0] if node_blocks[v] else -1

    src_tn = tn(src)
    if src_tn == -1:
        return {node_ids[i] for i in result_indices}

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
                result_indices.update(blocks[v])
            v = par[v]
        if src_tn < num_blocks:
            result_indices.update(blocks[src_tn])

    return {node_ids[i] for i in result_indices}
