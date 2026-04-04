# cgraph — Expansion Plan

## Vision

cgraph is a two-tier graph algorithm library optimized for **performance and low memory consumption**. Every public API function must beat or match the best available Python alternative (networkx, scipy) while using less memory.

- **C tier**: Primitive graph algorithms implemented in C for speed. These are the hot inner loops — union-find, BFS, biconnected components, etc. Exposed as private `_core` functions. Memory is allocated once per call, freed on return — no long-lived allocations.
- **Python tier**: Meta/composite algorithms that orchestrate multiple C primitives. Written in Python, calling into C kernels. Part of the public API but not performance-critical themselves.

## Performance Requirements

- **Every public API function gets a benchmark test** in `tests/performance_tests/` — no exceptions.
- Benchmarks compare against the reference implementation (scipy or networkx) across scaling curves (10^3 to 10^7 nodes).
- Benchmarks measure both **runtime** and **peak memory** (via `tracemalloc` or `/proc/self/status`).
- C tier functions must not allocate more than O(V + E) memory.
- No Python object creation in hot loops — build result sets in C, return them to Python in one shot.

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

| Algorithm | C tier | Python tier | Benchmark vs | Notes |
|-----------|--------|-------------|-------------|-------|
| BFS (unweighted) | `_bfs(num_nodes, edges, source)` | `bfs(node_ids, edges, source)` | `nx.bfs_tree` | Foundation for shortest path |
| Shortest path (weighted) | `_dijkstra(num_nodes, edges, weights, source, target)` | `shortest_path(node_ids, edges, weights, source, target)` | `nx.shortest_path` | Binary heap in C |
| Single-source shortest path lengths | `_sssp_lengths(num_nodes, edges, weights, source, cutoff)` | `shortest_path_lengths(node_ids, edges, weights, source, cutoff)` | `nx.single_source_shortest_path_length` | With cutoff support |
| Eccentricity | — | `eccentricity(node_ids, edges, weights, source)` | `nx.eccentricity` | Python tier: max of single-source distances |

### Phase 2: Topology Primitives (C tier)
| Algorithm | C tier | Python tier | Benchmark vs | Notes |
|-----------|--------|-------------|-------------|-------|
| Biconnected components | `_biconnected_components(num_nodes, edges)` | `biconnected_components(node_ids, edges)` | `nx.biconnected_components` | Tarjan's algorithm in C |
| Articulation points | `_articulation_points(num_nodes, edges)` | `articulation_points(node_ids, edges)` | `nx.articulation_points` | Falls out of biconnected components |
| Bridge edges | `_bridges(num_nodes, edges)` | `bridges(node_ids, edges)` | `nx.bridges` | Falls out of biconnected components |

### Phase 3: Composite Algorithms (Python tier only)
| Algorithm | Python tier | Benchmark vs | Notes |
|-----------|-------------|-------------|-------|
| `nodes_on_any_simple_paths(node_ids, edges, source, targets)` | Orchestrates BFS + connected components | `libraries.graph_functions.graph_traversal` | Replaces custom igp-mono function |
| Protection zone detection | Combines articulation points + connected components | — | Future |

## Benchmark Test Pattern

Every public API function gets two benchmark tests:

1. **Absolute performance** — must complete within a time limit per graph size (10^3 to 10^7 nodes). Catches regressions.
2. **Speedup vs reference** — measures and prints speedup vs networkx/scipy. Not a pass/fail gate (CI environments vary), but tracked for visibility.

Both tests use sparse random graphs with ~3 edges per node (typical grid topology).

Memory benchmarks use `tracemalloc` to measure peak allocation during the call. The C tier must stay within O(V + E) — no hidden quadratic blowups.

```python
# Pattern for new algorithm benchmarks:
# tests/performance_tests/test_<algorithm>.py

def test_<algorithm>_performance(exponent, time_limit_seconds):
    """Absolute time limit per graph size."""

def test_<algorithm>_speedup_vs_<reference>(exponent):
    """Print speedup vs networkx/scipy."""

def test_<algorithm>_memory(exponent):
    """Peak memory must be O(V + E)."""
```

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
