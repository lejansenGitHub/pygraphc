# cgraph — Expansion Plan

## Vision

cgraph is a general-purpose, domain-free graph algorithm library optimized for **performance and low memory consumption**. Zero runtime dependencies. Public on PyPI.

No domain knowledge — cgraph knows about nodes, edges, and weights. What those represent (electrical grids, social networks, road maps) is the caller's business.

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

This eliminates an entire class of index-mapping bugs that callers would otherwise need to manage.

## Current State (v0.1.0)

### C tier (`_core.c`)
- `connected_components(num_nodes, edges)` — union-find with path compression + rank
- `connected_components_remapped(node_ids, edges)` — same, returns original IDs
- `connected_components_with_branches(num_nodes, edges, branch_ids)` — with edge grouping
- `connected_components_with_branches_remapped(node_ids, edges, branch_ids)` — same, returns original IDs

### Python tier (`__init__.py`)
- `connected_components(node_ids, edges)` — wraps `_cc_remapped`
- `connected_components_with_branch_ids(node_ids, edges, branch_ids)` — wraps `_cc_branches_remapped`

## Target Algorithms

### Phase 1: Structural graph primitives

| Algorithm | C tier | Python tier | Benchmark vs | Notes |
|-----------|--------|-------------|-------------|-------|
| Bridge edges | `_bridges(num_nodes, edges)` | `bridges(node_ids, edges)` | `nx.bridges` | Iterative Tarjan's in C |
| Biconnected components | `_biconnected_components(num_nodes, edges)` | `biconnected_components(node_ids, edges)` | `nx.biconnected_components` | Same Tarjan traversal, different output |
| Articulation points | `_articulation_points(num_nodes, edges)` | `articulation_points(node_ids, edges)` | `nx.articulation_points` | Same traversal, different output |
| BFS (unweighted) | `_bfs(num_nodes, edges, source)` | `bfs(node_ids, edges, source)` | `nx.bfs_tree` | Returns visited set or parent map |

### Phase 2: Weighted graph algorithms

**Why Dijkstra, not A*:** A* only helps for single source → single target with an admissible heuristic (h(n) ≤ actual cost). Most practical use cases are single-source-all-targets (SSSP with cutoff, eccentricity, multi-source) where A* doesn't apply. Even for source→target, A* needs a domain-specific heuristic that cgraph can't provide as a generic library. The real speedup is C binary heap vs Python heapq, not algorithmic.

| Algorithm | C tier | Python tier | Benchmark vs | Notes |
|-----------|--------|-------------|-------------|-------|
| Shortest path (weighted) | `_dijkstra(num_nodes, edges, weights, source, target)` | `shortest_path(node_ids, edges, weights, source, target)` | `nx.shortest_path` | Binary heap in C |
| Single-source shortest path lengths | `_sssp_lengths(num_nodes, edges, weights, source, cutoff)` | `shortest_path_lengths(node_ids, edges, weights, source, cutoff)` | `nx.single_source_shortest_path_length` | With cutoff support |
| Multi-source shortest path lengths | `_multi_source_dijkstra(...)` | `multi_source_shortest_path_lengths(...)` | `nx.multi_source_dijkstra_path_length` | Multiple start nodes, one pass |
| Eccentricity | — | `eccentricity(node_ids, edges, weights, source)` | `nx.eccentricity` | Python tier: max of single-source distances |

### Phase 3: Composite algorithms (Python tier, collapse to C if benchmarks demand it)

| Algorithm | Python tier | Benchmark vs | Notes |
|-----------|-------------|-------------|-------|
| 2-edge-connected components | `two_edge_connected_components(node_ids, edges)` | `nx.k_edge_components` | Bridges + connected components |
| Nodes on any simple path | `nodes_on_simple_paths(node_ids, edges, source, targets)` | — | BFS + connected components |

## Benchmark Test Pattern

Every public API function gets two benchmark tests:

1. **Absolute performance** — must complete within a time limit per graph size (10^3 to 10^7 nodes). Catches regressions.
2. **Speedup vs reference** — measures and prints speedup vs networkx/scipy. Not a pass/fail gate (CI environments vary), but tracked for visibility.

Both tests use sparse random graphs with ~3 edges per node (typical sparse graph topology).

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
├── _composites.py       # Python tier: 2-edge-connected, simple paths
├── py.typed
```
