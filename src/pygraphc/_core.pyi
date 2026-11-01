"""Type stub of the ``_core`` C extension, read from the ``PyArg_ParseTuple`` formats in ``_core_src``."""

from collections.abc import Sequence
from typing import Generic, TypeVar, final

from pygraphc import EdgeIndex

_NodeIdT = TypeVar("_NodeIdT", bound=int)
_NodeIdT_co = TypeVar("_NodeIdT_co", bound=int, covariant=True)
_BranchIdT = TypeVar("_BranchIdT", bound=int)

_Mask = bytes | bytearray | memoryview
"""One byte per node, edge or terminal index; a non-zero byte marks membership."""

_Edges = Sequence[tuple[_NodeIdT | Sequence[_NodeIdT], ...]]
"""Edges as tuples of node ids, one per terminal; a directed edge may be ``(in_nodes, out_nodes)``."""

@final
class GraphCapsule(Generic[_NodeIdT_co]):
    """Stub-only name of the ``pygraphc.GraphCtx`` capsule, parameterized by the node ids it was parsed from."""

@final
class ReductionStateCapsule:
    """Stub-only name of the ``pygraphc.SPHandle`` capsule of one series-parallel reduction state."""

# ── Index-based, parse per call ──

def connected_components(node_count: int, edges: Sequence[tuple[int, int]], /) -> list[set[int]]: ...
def connected_components_with_branches(
    node_count: int, edges: Sequence[tuple[int, int]], branch_ids: Sequence[_BranchIdT], /
) -> list[tuple[set[int], set[_BranchIdT]]]: ...
def connected_components_remapped(
    node_ids: Sequence[_NodeIdT], edges: Sequence[tuple[int, int]], /
) -> list[set[_NodeIdT]]: ...
def connected_components_with_branches_remapped(
    node_ids: Sequence[_NodeIdT], edges: Sequence[tuple[int, int]], branch_ids: Sequence[_BranchIdT], /
) -> list[tuple[set[_NodeIdT], set[_BranchIdT]]]: ...
def bridges(node_count: int, edges: Sequence[tuple[int, int]], /) -> list[tuple[int, int]]: ...
def articulation_points(node_count: int, edges: Sequence[tuple[int, int]], /) -> set[int]: ...
def biconnected_components(node_count: int, edges: Sequence[tuple[int, int]], /) -> list[set[int]]: ...
def bfs(node_count: int, edges: Sequence[tuple[int, int]], source: int, /) -> list[int]: ...
def dijkstra(
    node_count: int, edges: Sequence[tuple[int, int]], weights: Sequence[float], source: int, target: int, /
) -> tuple[float, list[int]]: ...
def shortest_path_lengths(
    node_count: int, edges: Sequence[tuple[int, int]], weights: Sequence[float], source: int, cutoff: float, /
) -> dict[int, float]: ...
def multi_source_dijkstra(
    node_count: int,
    edges: Sequence[tuple[int, int]],
    weights: Sequence[float],
    sources: Sequence[int],
    cutoff: float,
    /,
) -> dict[int, float]: ...

# ── Node-id-based, parse per call ──

def translate_edges(
    node_ids: Sequence[_NodeIdT], node_id_edges: Sequence[tuple[_NodeIdT, _NodeIdT]], /
) -> list[tuple[int, int]]: ...
def connected_components_by_node_id(
    node_ids: Sequence[_NodeIdT], node_id_edges: Sequence[tuple[_NodeIdT, _NodeIdT]], /
) -> list[set[_NodeIdT]]: ...
def connected_components_by_node_id_split(
    node_ids: Sequence[_NodeIdT], first_endpoints: Sequence[_NodeIdT], second_endpoints: Sequence[_NodeIdT], /
) -> list[set[_NodeIdT]]: ...
def bridges_by_node_id(
    node_ids: Sequence[_NodeIdT], node_id_edges: Sequence[tuple[_NodeIdT, _NodeIdT]], /
) -> list[tuple[_NodeIdT, _NodeIdT]]: ...
def articulation_points_by_node_id(
    node_ids: Sequence[_NodeIdT], node_id_edges: Sequence[tuple[_NodeIdT, _NodeIdT]], /
) -> set[_NodeIdT]: ...
def biconnected_components_by_node_id(
    node_ids: Sequence[_NodeIdT], node_id_edges: Sequence[tuple[_NodeIdT, _NodeIdT]], /
) -> list[set[_NodeIdT]]: ...
def bfs_by_node_id(
    node_ids: Sequence[_NodeIdT], node_id_edges: Sequence[tuple[_NodeIdT, _NodeIdT]], source_node_id: _NodeIdT, /
) -> list[_NodeIdT]: ...
def dijkstra_by_node_id(
    node_ids: Sequence[_NodeIdT],
    node_id_edges: Sequence[tuple[_NodeIdT, _NodeIdT]],
    weights: Sequence[float],
    source_node_id: _NodeIdT,
    target_node_id: _NodeIdT,
    /,
) -> tuple[float, list[_NodeIdT]]: ...
def shortest_path_lengths_by_node_id(
    node_ids: Sequence[_NodeIdT],
    node_id_edges: Sequence[tuple[_NodeIdT, _NodeIdT]],
    weights: Sequence[float],
    source_node_id: _NodeIdT,
    cutoff: float,
    /,
) -> dict[_NodeIdT, float]: ...
def multi_source_dijkstra_by_node_id(
    node_ids: Sequence[_NodeIdT],
    node_id_edges: Sequence[tuple[_NodeIdT, _NodeIdT]],
    weights: Sequence[float],
    source_node_ids: Sequence[_NodeIdT],
    cutoff: float,
    /,
) -> dict[_NodeIdT, float]: ...
def topological_sort_by_node_id(
    node_ids: Sequence[_NodeIdT], node_id_edges: Sequence[tuple[_NodeIdT, _NodeIdT]], /
) -> list[_NodeIdT]: ...

# ── Parsed graph ──

def parse_graph(
    node_ids: Sequence[_NodeIdT],
    edges_or_src: _Edges[_NodeIdT] | Sequence[_NodeIdT],
    dst: Sequence[_NodeIdT] | None = None,
    directed: bool = False,
    /,
) -> GraphCapsule[_NodeIdT]: ...
def graph_shape(capsule: GraphCapsule[int], /) -> tuple[int, int, int]: ...
def has_multi_terminal_edges(capsule: GraphCapsule[int], /) -> bool: ...
def terminal_offsets(capsule: GraphCapsule[int], /) -> bytes: ...
def out_terminal_offsets(capsule: GraphCapsule[int], /) -> bytes: ...
def terminal_nodes(capsule: GraphCapsule[int], /) -> bytes: ...
def graph_is_directed(capsule: GraphCapsule[int], /) -> bool: ...
def graph_edge_count(capsule: GraphCapsule[int], /) -> int: ...
def graph_node_count(capsule: GraphCapsule[int], /) -> int: ...

# ── Algorithms on a parsed graph ──

def connected_components_ctx(
    capsule: GraphCapsule[_NodeIdT],
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    terminal_mask: _Mask | None = None,
    /,
) -> list[set[_NodeIdT]]: ...
def connected_components_with_branches_ctx(
    capsule: GraphCapsule[_NodeIdT],
    branch_ids: Sequence[_BranchIdT],
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    terminal_mask: _Mask | None = None,
    /,
) -> list[tuple[set[_NodeIdT], set[_BranchIdT]]]: ...
def strongly_connected_components_ctx(
    capsule: GraphCapsule[_NodeIdT],
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    terminal_mask: _Mask | None = None,
    /,
) -> list[set[_NodeIdT]]: ...
def bridges_ctx(
    capsule: GraphCapsule[_NodeIdT],
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    terminal_mask: _Mask | None = None,
    /,
) -> list[tuple[_NodeIdT, _NodeIdT]]: ...
def critical_edges_ctx(
    capsule: GraphCapsule[int],
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    terminal_mask: _Mask | None = None,
    /,
) -> list[EdgeIndex]: ...
def articulation_points_ctx(
    capsule: GraphCapsule[_NodeIdT],
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    terminal_mask: _Mask | None = None,
    /,
) -> set[_NodeIdT]: ...
def biconnected_components_ctx(
    capsule: GraphCapsule[_NodeIdT],
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    terminal_mask: _Mask | None = None,
    /,
) -> list[set[_NodeIdT]]: ...
def simple_path_blocks_ctx(
    capsule: GraphCapsule[_NodeIdT],
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    terminal_mask: _Mask | None = None,
    /,
) -> tuple[list[set[_NodeIdT]], dict[int, set[EdgeIndex]]]: ...
def bfs_ctx(
    capsule: GraphCapsule[_NodeIdT],
    source: _NodeIdT,
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    terminal_mask: _Mask | None = None,
    /,
) -> list[_NodeIdT]: ...
def all_edge_paths_ctx(
    capsule: GraphCapsule[_NodeIdT],
    source: _NodeIdT,
    targets: Sequence[_NodeIdT],
    cutoff: int = -1,
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    node_simple: bool = False,
    ignore_self_loops: bool = False,
    /,
) -> list[list[EdgeIndex]]: ...
def dijkstra_ctx(
    capsule: GraphCapsule[_NodeIdT],
    weights: Sequence[float],
    source: _NodeIdT,
    target: _NodeIdT,
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    terminal_mask: _Mask | None = None,
    /,
) -> tuple[float, list[_NodeIdT]]: ...
def shortest_path_lengths_ctx(
    capsule: GraphCapsule[_NodeIdT],
    weights: Sequence[float],
    source: _NodeIdT,
    cutoff: float,
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    terminal_mask: _Mask | None = None,
    /,
) -> dict[_NodeIdT, float]: ...
def multi_source_dijkstra_ctx(
    capsule: GraphCapsule[_NodeIdT],
    weights: Sequence[float],
    sources: Sequence[_NodeIdT],
    cutoff: float,
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    terminal_mask: _Mask | None = None,
    /,
) -> dict[_NodeIdT, float]: ...
def topological_sort_ctx(
    capsule: GraphCapsule[_NodeIdT],
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    terminal_mask: _Mask | None = None,
    /,
) -> list[_NodeIdT]: ...
def cycle_basis_ctx(
    capsule: GraphCapsule[_NodeIdT],
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    terminal_mask: _Mask | None = None,
    /,
) -> list[list[_NodeIdT]]: ...
def dag_longest_path_ctx(
    capsule: GraphCapsule[_NodeIdT],
    weights: Sequence[float] | None = None,
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    terminal_mask: _Mask | None = None,
    /,
) -> list[_NodeIdT]: ...

# ── Neighborhood queries on a parsed graph ──

def neighbors_ctx(
    capsule: GraphCapsule[_NodeIdT],
    node_id: _NodeIdT,
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    terminal_mask: _Mask | None = None,
    /,
) -> set[_NodeIdT]: ...
def predecessors_ctx(
    capsule: GraphCapsule[_NodeIdT],
    node_id: _NodeIdT,
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    terminal_mask: _Mask | None = None,
    /,
) -> set[_NodeIdT]: ...
def degree_ctx(
    capsule: GraphCapsule[_NodeIdT],
    node_id: _NodeIdT,
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    terminal_mask: _Mask | None = None,
    /,
) -> int: ...
def in_degree_ctx(
    capsule: GraphCapsule[_NodeIdT],
    node_id: _NodeIdT,
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    terminal_mask: _Mask | None = None,
    /,
) -> int: ...
def incident_edges_ctx(
    capsule: GraphCapsule[_NodeIdT],
    node_id: _NodeIdT,
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    terminal_mask: _Mask | None = None,
    /,
) -> list[EdgeIndex]: ...
def incoming_edges_ctx(
    capsule: GraphCapsule[_NodeIdT],
    node_id: _NodeIdT,
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    terminal_mask: _Mask | None = None,
    /,
) -> list[EdgeIndex]: ...
def edge_indices_ctx(
    capsule: GraphCapsule[_NodeIdT], u: _NodeIdT, v: _NodeIdT, edge_mask: _Mask | None = None, /
) -> list[EdgeIndex]: ...

# ── int32 byte arrays over a parsed graph ──

def component_labels_ctx(
    capsule: GraphCapsule[int],
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    terminal_mask: _Mask | None = None,
    /,
) -> bytes: ...
def quotient_edges_ctx(
    capsule: GraphCapsule[int], labels: _Mask, edge_mask: _Mask | None = None, /
) -> tuple[bytes, bytes, bytes, bytes]: ...
def degrees_ctx(
    capsule: GraphCapsule[int],
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    terminal_mask: _Mask | None = None,
    /,
) -> bytes: ...
def biconnected_component_edge_labels_ctx(
    capsule: GraphCapsule[int],
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    terminal_mask: _Mask | None = None,
    /,
) -> bytes: ...

# ── Series-parallel reduction ──

def series_parallel_reduce_ctx(
    capsule: GraphCapsule[int],
    preserved_mask: _Mask,
    protected_mask: _Mask,
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    pendant_keep_mask: _Mask | None = None,
    series_blocked_mask: _Mask | None = None,
) -> tuple[bytes, ...]: ...
def sp_state_new(
    capsule: GraphCapsule[int],
    preserved_mask: _Mask,
    protected_mask: _Mask,
    edge_mask: _Mask | None = None,
    node_mask: _Mask | None = None,
    pendant_keep_mask: _Mask | None = None,
    series_blocked_mask: _Mask | None = None,
    /,
) -> ReductionStateCapsule: ...
def sp_state_free(state: ReductionStateCapsule, /) -> None: ...
def sp_state_graph(state: ReductionStateCapsule, /) -> GraphCapsule[int]: ...
def sp_next_move(state: ReductionStateCapsule, /) -> tuple[int, int, int, int, int, int] | None: ...
def sp_apply_move(state: ReductionStateCapsule, move: tuple[int, int, int, int, int, int], /) -> tuple[int, int]: ...
def sp_pair_edges(state: ReductionStateCapsule, from_node: int, to_node: int, /) -> bytes | None: ...
def sp_apply_parallel(state: ReductionStateCapsule, from_node: int, to_node: int, edges: _Mask, /) -> int: ...
def sp_batch_moves(state: ReductionStateCapsule, kind: int, /) -> bytes | None: ...
def sp_apply_batch(state: ReductionStateCapsule, kind: int, batch: _Mask, /) -> None: ...
def sp_state_log(state: ReductionStateCapsule, /) -> tuple[bytes, ...]: ...
