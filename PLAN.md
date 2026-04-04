# cgraph — Expansion Plan

## Vision

cgraph is a two-tier graph algorithm library optimized for **performance and low memory consumption**. Every public API function must beat or match the best available Python alternative (networkx, scipy) while using less memory.

- **C tier**: Primitive graph algorithms implemented in C for speed. These are the hot inner loops — union-find, BFS, biconnected components, etc. Exposed as private `_core` functions. Memory is allocated once per call, freed on return — no long-lived allocations.
- **Python tier**: Meta/composite algorithms that orchestrate multiple C primitives. Written in Python, calling into C kernels. Part of the public API but not performance-critical themselves.

### When to collapse tiers

The two-tier split favors maintainability — Python composites are easier to read, test, and change than C. But **performance wins**. If benchmarks show a Python-tier composite loses meaningful performance vs what a single C function could achieve (e.g. repeated Python↔C boundary crossings, intermediate list allocations, or missed cache locality), push it down to C. The benchmark tests are the decision mechanism: prototype in Python first, benchmark it, and only write C if the numbers demand it.

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

### From `backend/topology/` — connected components + bridges
- **Connected components** ✅ (done — drop-in replacement for scipy version in `connected_components.py`)
- **Bridges** — `unversioned-models-no-dump-prep` branch has a Python Tarjan's implementation in `backend/topology/src/graphs/bridges.py` with `find_bridges()` + `find_two_edge_connected_components()`. This is the exact algorithm cgraph should provide in C. The Python version is the correctness reference and test suite to port.

### From `backend/topology/src/graphs/n_1/` — N-1 contingency analysis

The N-1 analysis is the most complex and performance-critical graph code in igp-mono (1583-line `graph_n_1.py` + 627-line `get_locations.py`). It uses a **two-layer connected component abstraction** that exists specifically to work around slow networkx operations:

1. **Layer 1 (CC Graph)**: Filters open switches + faulty edges → `nx.connected_components()` → re-adds fault edges between component IDs → finds **bridges** via `nx.bridges()` → custom DFS per bridge to find split components
2. **Layer 2 (CC2 Graph)**: Takes Layer 1 components as nodes → `nx.connected_components()` again → re-adds open switches → iteratively removes parallel/series edges → removes leaf nodes
3. **Path finding**: Custom `all_edge_paths_multigraph()` — modified DFS tracking visited **edges** (not nodes) with node visit limits and cut-off depth. Called per fault × per participant. Exponential worst case.

**The core hypothesis**: much of this two-layer complexity exists because the underlying graph operations (connected components, bridges, path enumeration) were too slow in networkx. With fast C primitives, simpler algorithms might be fast enough — or at minimum, the existing approach gets dramatically faster.

**Graph operations used by N-1:**
- `nx.connected_components()` — called multiple times per analysis
- `nx.bridges()` — critical edge identification (Python-only in networkx)
- Custom stack-based DFS (`_bridge_dfs`) — find components split by a bridge
- Custom edge-path DFS (`all_edge_paths_multigraph`) — enumerate switching paths
- Degree queries, neighbor iteration, subgraph creation

### From `backend/topology/src/graphs/find_primary_protection_zones/`
- Connected components (multiple filter variants: no feeding transformers, no open switches, no fuses)
- Custom BFS with visited tracking
- Component neighbor mapping via fuse edges
- Set operations on node/edge IDs

### From `backend/switch_state_optimization/` (SSO, networkx-based)
- **Connected components** — 8+ uses via `nx.connected_components()`
- **Shortest path** (weighted) — `nx.shortest_path(G, u, v, weight=...)`, `nx.single_source_shortest_path_length(G, source, cutoff=...)`
- **Multi-source Dijkstra** — `nx.multi_source_dijkstra_path_length()` for electrical distance from multiple sources
- **Eccentricity** (weighted) — `nx.eccentricity(G, v=u, weight=...)`
- **Simple paths** — `nodes_on_any_simple_paths(G, source, targets)` (custom library function)
- **Graph utilities** — `neighbors()`, `degree()`, `subgraph()`, `pairwise()`

### Across igp-mono (106 files use networkx)
- `nx.connected_components()` — ~50 files
- `nx.shortest_path()` / `nx.shortest_simple_paths()` — ~15 files
- `nx.multi_source_dijkstra_path_length()` — 3 files
- `nx.bridges()` — 1 file (N-1, but critical)

## Implementation Phases

### Phase 1: N-1 Primitives (highest impact)
The N-1 analysis is the most performance-sensitive consumer. Bridges and connected components are called repeatedly.

| Algorithm | C tier | Python tier | Benchmark vs | Notes |
|-----------|--------|-------------|-------------|-------|
| Bridge edges | `_bridges(num_nodes, edges)` | `bridges(node_ids, edges)` | `nx.bridges` | Tarjan's in C. N-1 bottleneck — called once but on full grid |
| Biconnected components | `_biconnected_components(num_nodes, edges)` | `biconnected_components(node_ids, edges)` | `nx.biconnected_components` | Falls out of same Tarjan traversal |
| Articulation points | `_articulation_points(num_nodes, edges)` | `articulation_points(node_ids, edges)` | `nx.articulation_points` | Same traversal, different output |
| BFS (unweighted) | `_bfs(num_nodes, edges, source)` | `bfs(node_ids, edges, source)` | `nx.bfs_tree` | Used by protection zones + N-1 component splitting |

### Phase 2: SSO Migration (weighted graph algorithms)
| Algorithm | C tier | Python tier | Benchmark vs | Notes |
|-----------|--------|-------------|-------------|-------|
| Shortest path (weighted) | `_dijkstra(num_nodes, edges, weights, source, target)` | `shortest_path(node_ids, edges, weights, source, target)` | `nx.shortest_path` | Binary heap in C |
| Single-source shortest path lengths | `_sssp_lengths(num_nodes, edges, weights, source, cutoff)` | `shortest_path_lengths(node_ids, edges, weights, source, cutoff)` | `nx.single_source_shortest_path_length` | With cutoff support |
| Multi-source Dijkstra | `_multi_source_dijkstra(...)` | `multi_source_shortest_path_lengths(...)` | `nx.multi_source_dijkstra_path_length` | Electrical distance from multiple feed-ins |
| Eccentricity | — | `eccentricity(node_ids, edges, weights, source)` | `nx.eccentricity` | Python tier: max of single-source distances |

### Phase 3: Composite Algorithms (Python tier, collapse to C if benchmarks demand it)
| Algorithm | Python tier | Benchmark vs | Notes |
|-----------|-------------|-------------|-------|
| `nodes_on_any_simple_paths(node_ids, edges, source, targets)` | Orchestrates BFS + connected components | `libraries.graph_functions.graph_traversal` | Replaces custom igp-mono function |
| Protection zone detection | Combines articulation points + connected components | Current igp-mono implementation | BFS + component neighbor mapping |

### Future: N-1 simplification investigation
With fast C-backed bridges + connected components + BFS, investigate whether the two-layer CC abstraction in `graph_n_1.py` can be simplified. The layers exist to reduce graph size before expensive path enumeration — but if the primitives are 10-50x faster, a simpler single-pass approach might be fast enough. This is an investigation, not a commitment.

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

1. **Connected components** — cgraph replaces `backend/topology/src/graphs/connected_components.py` (scipy version). Already possible today.
2. **N-1 analysis** — replace `nx.bridges()` + `nx.connected_components()` calls in `graph_n_1.py` with cgraph equivalents. Benchmark the N-1 analysis end-to-end before and after.
3. **Protection zones** — replace BFS + connected component calls in `find_primary_protection_zones.py`.
4. **SSO** — migrate from networkx to cgraph for connected components + Dijkstra + eccentricity.
5. **Broader networkx elimination** — ~106 files use networkx. As cgraph covers more algorithms, systematically replace. The `node_ids` API means callers just pass their ID lists directly — no index mapping code needed.
6. **N-1 simplification** — once primitives are fast, investigate whether the two-layer CC abstraction can be replaced with a simpler approach.
