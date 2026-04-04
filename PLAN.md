# cgraph — Expansion Plan

## Vision

cgraph is a two-tier graph algorithm library:

- **C tier**: Primitive graph algorithms implemented in C for speed. These are the hot inner loops — union-find, BFS, biconnected components, etc. Exposed as private `_core` functions.
- **Python tier**: Meta/composite algorithms that orchestrate multiple C primitives. Written in Python, calling into C kernels. Part of the public API but not performance-critical themselves.

## API Design Principle: Mapped IDs Only

The public API always works with the caller's original node IDs. Users pass `node_ids: list[int]` and `edges: list[tuple[int, int]]` (where edges reference indices into `node_ids`). The C layer handles the internal 0..n-1 mapping — callers never see it.

This was a deliberate decision: the old igp-mono code required callers to build and maintain their own index mappings. cgraph eliminates that entire class of bugs.

## Current State (v0.1.0)

### C tier (`_core.c`)
- `connected_components(num_nodes, edges)` — union-find with path compression + rank
- `connected_components_remapped(node_ids, edges)` — same, returns original IDs
- `connected_components_with_branches(num_nodes, edges, branch_ids)` — with branch grouping
- `connected_components_with_branches_remapped(node_ids, edges, branch_ids)` — same, returns original IDs

### Python tier (`__init__.py`)
- `connected_components(node_ids, edges)` — wraps `_cc_remapped`
- `connected_components_with_branch_ids(node_ids, edges, branch_ids)` — wraps `_cc_branches_remapped`

## Algorithms Needed (from igp-mono audit)

### From `backend/topology/` (current igp-mono, scipy-based)
- **Connected components** ✅ (done — drop-in replacement for scipy version)

### From `backend/switch_state_optimization/` (SSO, networkx-based)
- **Connected components** — 8+ uses via `nx.connected_components()`
- **Shortest path** (weighted) — `nx.shortest_path(G, u, v, weight=...)`, `nx.single_source_shortest_path_length(G, source, cutoff=...)`
- **Eccentricity** (weighted) — `nx.eccentricity(G, v=u, weight=...)`
- **Simple paths** — `nodes_on_any_simple_paths(G, source, targets)` (custom library function)
- **Graph utilities** — `neighbors()`, `degree()`, `subgraph()`, `pairwise()`

### Not yet needed but anticipated
- **Biconnected components** — protection zone analysis (not in SSO but expected for grid topology)
- **Articulation points / cut vertices** — same context
- **BFS** — general traversal primitive, useful as building block

## Implementation Phases

### Phase 1: SSO Migration (C tier additions)
Priority: the algorithms SSO currently gets from networkx.

| Algorithm | C tier | Python tier | Notes |
|-----------|--------|-------------|-------|
| BFS (unweighted) | `_bfs(num_nodes, edges, source)` | `bfs(node_ids, edges, source)` | Foundation for shortest path |
| Shortest path (weighted) | `_dijkstra(num_nodes, edges, weights, source, target)` | `shortest_path(node_ids, edges, weights, source, target)` | Replaces `nx.shortest_path` |
| Single-source shortest path lengths | `_sssp_lengths(num_nodes, edges, weights, source, cutoff)` | `shortest_path_lengths(node_ids, edges, weights, source, cutoff)` | Replaces `nx.single_source_shortest_path_length` |
| Eccentricity | — | `eccentricity(node_ids, edges, weights, source)` | Python tier: max of single-source distances |

### Phase 2: Topology Primitives (C tier)
| Algorithm | C tier | Python tier | Notes |
|-----------|--------|-------------|-------|
| Biconnected components | `_biconnected_components(num_nodes, edges)` | `biconnected_components(node_ids, edges)` | Tarjan's algorithm in C |
| Articulation points | `_articulation_points(num_nodes, edges)` | `articulation_points(node_ids, edges)` | Falls out of biconnected components |
| Bridge edges | `_bridges(num_nodes, edges)` | `bridges(node_ids, edges)` | Falls out of biconnected components |

### Phase 3: Composite Algorithms (Python tier only)
| Algorithm | Python tier | Notes |
|-----------|-------------|-------|
| `nodes_on_any_simple_paths(node_ids, edges, source, targets)` | Orchestrates BFS + connected components | Replaces custom `graph_traversal` library |
| Protection zone detection | Combines articulation points + connected components | Future |

## File Structure (target)

```
src/cgraph/
├── __init__.py          # Public API — all user-facing symbols
├── _core.c              # C primitives (union-find, BFS, Dijkstra, Tarjan)
├── _paths.py            # Python tier: shortest_path, eccentricity
├── _traversal.py        # Python tier: simple paths, composite traversals
├── py.typed
```

## Migration Path for igp-mono

1. cgraph replaces `backend/topology/src/graphs/connected_components.py` (scipy version) — already possible
2. SSO migrates from networkx to cgraph as algorithms are added
3. The `node_ids` API means SSO just passes its ID lists directly — no index mapping code needed
