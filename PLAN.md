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

## Phase 4: Graph Class (DONE)

### Problem

Every cgraph algorithm call pays the cost of:
1. **Node-ID → index mapping**: Building an IntMap hash table from node_ids
2. **Edge translation**: Looking up both endpoints per edge in the hash map
3. **CSR construction**: Building the adjacency list from translated edges

When running multiple algorithms on the same graph (e.g. bridges + articulation
points + connected components), this parsing is repeated redundantly.

### Implementation

A `Graph` class that parses input once into a `GraphCtx` C structure (IntMap +
EdgeList + CSR adjacency list), stored as a `PyCapsule`. All algorithm methods
reuse the cached context.

```python
from cgraph import Graph

g = Graph(node_ids, edges)          # parse once
g = Graph(node_ids, src, dst)       # split-list variant

# All calls reuse cached C structures — no re-parsing
g.connected_components()
g.bridges()
g.articulation_points()
g.biconnected_components()
g.bfs(source)
g.shortest_path(weights, source, target)
g.shortest_path_lengths(weights, source, cutoff=None)
g.multi_source_shortest_path_lengths(weights, sources, cutoff=None)
g.eccentricity(weights, source)
g.two_edge_connected_components()
g.nodes_on_simple_paths(source, targets)
```

**C tier**: `GraphCtx` struct holds `NidContext` + `AdjList`. Returned as
`PyCapsule` with destructor. Eight `_ctx` algorithm variants (`cc_ctx`,
`bridges_ctx`, `ap_ctx`, `bcc_ctx`, `bfs_ctx`, `dijkstra_ctx`, `sssp_ctx`,
`msdijk_ctx`) skip parsing and use the cached context directly.

**Performance**: ~2x speedup when running 3 algorithms on the same 100K-node
graph vs 3 separate free-function calls.

**Migration**: All existing free functions remain unchanged. `Graph` is additive.

## Phase 5: Edge-Masked Graph Views

### Problem

Many graph analysis workflows follow a **modify → compute → restore** loop: take a
base graph, temporarily remove (or add) a small number of edges, run an algorithm,
then repeat with different modifications. Examples:

- **Resilience analysis**: For each critical edge, remove it and check if the graph
  stays connected (or find alternative paths).
- **Sensitivity analysis**: How does removing edge X change shortest paths / component
  structure?
- **What-if scenarios**: Add a candidate edge, measure improvement, discard.
- **Iterative optimization**: Toggle edges on/off, evaluate a metric, repeat.

Without library support, callers must manage this loop manually: deep-copy the graph
state before modification, carefully reverse every change after computing, and verify
that restoration succeeded. This is error-prone, verbose (hundreds of lines of
bookkeeping in real-world code), and wastes memory on full copies.

### Design: Immutable Views with Edge Masks

The core idea: **never mutate the base graph**. Instead, create lightweight *views*
that overlay an edge mask on the existing CSR structure. Algorithms traverse the CSR
as before, but skip masked-out edges. Creating and discarding views is O(m/8) (one
bit per edge), while the base graph's CSR is shared and never rebuilt.

```
┌──────────────────────────────────────────────────────┐
│  GraphCtx (base, immutable, shared)                  │
│  ┌────────────┐ ┌──────────────────────────────────┐ │
│  │ NidContext  │ │ AdjList (CSR)                    │ │
│  │ IntMap      │ │ offset[n+1], adj[2m], eid[2m]   │ │
│  └────────────┘ └──────────────────────────────────┘ │
└──────────────────────────┬───────────────────────────┘
                           │ shared pointer
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ View A   │ │ View B   │ │ View C   │
        │ mask: {3}│ │ mask: {7}│ │ mask: {}  │
        └──────────┘ └──────────┘ └──────────┘
        (edge 3 off) (edge 7 off)  (= base)
```

Each view holds a reference to the base `GraphCtx` plus a bitmask of excluded edge
indices. Algorithms receive the mask alongside the CSR and check it in the inner loop.

### Public API

```python
# Exclude edges — O(m/8) to create, O(1) per edge to set
view = graph.without_edges([edge_id, ...])

# Include extra edges — requires CSR rebuild, see Phase 5c
view = graph.with_edges([(u, v), ...])

# Combine: exclude some, add others
view = graph.without_edges([...]).with_edges([...])

# All existing algorithms work on views
components = view.connected_components()
bridge_list = view.bridges()
path = view.shortest_path(weights, source, target)

# Views are disposable — no restore step needed
# The base graph is never modified
```

#### Edge Identification

Edges are identified by their **index in the original edge list** (the order in which
they were passed to `Graph()`). This is the same index stored in `AdjList.eid` and
already used internally. The caller can maintain a mapping from domain-specific edge
identifiers to these indices.

```python
# Edges passed at construction time:
edges = [(10, 20), (20, 30), (30, 40)]
#         idx 0      idx 1      idx 2

g = Graph(node_ids, edges)

# Exclude edge (20, 30) by its index:
view = g.without_edges([1])
```

To support lookup by node pair, a convenience method resolves `(u, v)` to edge
indices:

```python
edge_indices = g.edge_indices(20, 30)   # -> [1]  (list, because multigraphs)
view = g.without_edges(edge_indices)
```

### Implementation Plan

#### Phase 5a: Edge-Exclusion Views (C tier)

**Goal**: Run any algorithm on a graph with a subset of edges removed, without
rebuilding the CSR.

**C changes** (`_core.c`):

1. Add a `uint8_t *edge_mask` field alongside `AdjList`. Convention: `edge_mask[i] = 1`
   means edge `i` is **excluded** (skipped during traversal). `NULL` means no
   exclusion (all edges active) — this preserves the fast path for non-masked graphs.

2. Add mask-aware variants of the inner-loop pattern. The current traversal:

   ```c
   // Current: no mask
   for (int i = al->offset[u]; i < al->offset[u+1]; i++) {
       int v = al->adj[i];
       int eid = al->eid[i];
       // ... process edge (u, v)
   }
   ```

   Becomes:

   ```c
   // With mask (NULL-check outside the hot loop)
   for (int i = al->offset[u]; i < al->offset[u+1]; i++) {
       if (mask && mask[al->eid[i]]) continue;  // skip excluded
       int v = al->adj[i];
       int eid = al->eid[i];
       // ... process edge (u, v)
   }
   ```

   The `mask && ...` check compiles to a single branch that is trivially predicted
   when `mask` is NULL (no-mask fast path) or non-NULL (mask path). The per-edge
   cost is one byte load + compare — negligible relative to the rest of the loop body.

   **Alternative** (if benchmarks show the branch matters): split into two code paths
   via a macro:

   ```c
   #define TRAVERSE(u, BODY) \
       for (int _i = al->offset[u]; _i < al->offset[u+1]; _i++) { \
           int _eid = al->eid[_i]; \
           int v = al->adj[_i]; \
           BODY \
       }
   #define TRAVERSE_MASKED(u, mask, BODY) \
       for (int _i = al->offset[u]; _i < al->offset[u+1]; _i++) { \
           if (mask[al->eid[_i]]) continue; \
           int _eid = al->eid[_i]; \
           int v = al->adj[_i]; \
           BODY \
       }
   ```

   Decision: start with the single-branch approach. Only split if benchmarks show
   >5% regression on non-masked hot paths.

3. Affected C functions (all CSR-based algorithms):

   | Function | Traversal pattern | Mask integration |
   |----------|-------------------|------------------|
   | `compute_bridges` | Tarjan DFS over CSR | Skip masked edges in DFS |
   | `compute_ap` | Tarjan DFS over CSR | Same |
   | `compute_bcc` | Tarjan DFS over CSR | Same |
   | BFS (in `py_bfs_ctx`) | Queue + CSR neighbors | Skip masked edges |
   | Dijkstra (in `py_dijkstra_ctx`) | Heap + CSR neighbors | Skip masked edges |
   | SSSP (in `py_sssp_ctx`) | Dijkstra all-targets | Skip masked edges |
   | Multi-source Dijkstra | Multi-start Dijkstra | Skip masked edges |

   Connected components (union-find) iterate over the EdgeList, not the CSR. Here
   the mask check is on the edge index directly:

   ```c
   for (Py_ssize_t i = 0; i < m; i++) {
       if (mask && mask[i]) continue;
       uf_union(&uf, src[i], dst[i]);
   }
   ```

4. New C entry points (context-based):

   ```c
   py_cc_ctx_masked(capsule, mask_array)        -> list[set]
   py_bridges_ctx_masked(capsule, mask_array)    -> list[tuple]
   py_bfs_ctx_masked(capsule, source, mask_array) -> list
   // ... etc for all _ctx functions
   ```

   Alternatively (preferred): add an **optional** `mask` parameter to the existing
   `_ctx` functions. When NULL/None, no mask is applied. This avoids doubling the
   function count.

**Python changes** (`__init__.py`):

1. New `GraphView` class:

   ```python
   class GraphView:
       """Lightweight view of a Graph with excluded edges.

       Shares the base graph's parsed data (IntMap, CSR). Only holds a
       bitmask of excluded edge indices. Creating a view is O(m) in the
       worst case (copying the mask), O(k) if built from scratch with k
       exclusions.
       """

       def __init__(self, base: Graph, excluded_edge_indices: Collection[int]):
           self._base = base
           self._mask = bytearray(base.edge_count)  # 1 byte per edge
           for idx in excluded_edge_indices:
               self._mask[idx] = 1

       def connected_components(self) -> list[set[int]]: ...
       def bridges(self) -> list[tuple[int, int]]: ...
       def bfs(self, source: int) -> list[int]: ...
       def shortest_path(self, weights, source, target): ...
       # ... all algorithms, passing self._mask to C
   ```

2. `Graph.without_edges(edge_indices) -> GraphView` factory method.

3. `Graph.edge_indices(u, v) -> list[int]` convenience for resolving node pairs.

**Memory cost**: 1 byte per edge. For a 1M-edge graph: 1 MB per view. For a 10M-edge
graph: 10 MB per view. Views are short-lived (created, used, discarded), so this is
acceptable. If memory becomes a concern, a bitset (1 bit per edge) reduces this 8x
at the cost of slightly more complex C code (`mask[eid >> 3] & (1 << (eid & 7))`).

**Performance expectation**: The mask check adds ~1 ns per edge (one byte load +
branch). For a 1M-edge graph, this is ~1 ms overhead per full traversal — negligible
compared to the algorithm's own work. The key win is avoiding O(V + E) CSR rebuild
per modification.

#### Phase 5b: Edge-Addition Views

**Goal**: Create views that add edges not present in the base graph.

This is fundamentally more expensive than exclusion because new edges create new
neighbors in the CSR. Two approaches:

**Approach 1: Full CSR rebuild** (simple, correct)

Build a new CSR from the base graph's edge list plus the added edges. The IntMap and
node ID mapping are reused from the base graph — only the CSR is rebuilt.

```python
class GraphViewWithAdditions(GraphView):
    def __init__(self, base, excluded, added_edges):
        # Merge base edges (minus excluded) + added edges
        # Build new CSR from merged edge list
        # Reuse base.node_ids and IntMap
```

Cost: O(V + E) per view creation. Acceptable when additions are infrequent or the
graph is small. The caller still avoids manual state management.

**Approach 2: Auxiliary adjacency list** (optimized, deferred)

Maintain a small secondary CSR for added edges only. Algorithms query both the base
CSR (with mask) and the auxiliary CSR. This avoids rebuilding the full CSR but
increases algorithm complexity.

Decision: implement Approach 1 first. Only pursue Approach 2 if benchmarks show CSR
rebuild is a bottleneck for real workloads.

#### Phase 5c: Benchmarks

New benchmark tests following the existing pattern:

```python
# tests/performance_tests/test_masked_views.py

def test_cc_masked_vs_rebuild(exponent):
    """Compare masked CC vs full graph rebuild for single-edge exclusion."""
    # Build base graph, then for each edge:
    #   - Masked: graph.without_edges([edge_idx]).connected_components()
    #   - Rebuild: Graph(node_ids, edges_without_one).connected_components()
    # Report: speedup of masked over rebuild across all edges

def test_bridges_masked_vs_rebuild(exponent):
    """Same comparison for bridge detection."""

def test_mask_overhead_vs_no_mask(exponent):
    """Measure overhead of mask=NULL vs no mask parameter (fast-path cost)."""
    # This ensures the mask check doesn't regress non-masked performance.

def test_view_creation_cost(exponent):
    """Measure time to create a GraphView with k exclusions."""
```

**Key metric**: For the "exclude 1 edge, run CC" use case at 1M nodes, the masked
approach should be >10x faster than rebuilding a full Graph, because it skips:
- Edge list construction: O(E)
- IntMap rebuild: O(V)
- CSR construction: O(V + E)

#### Phase 5d: Composite View Operations

Build higher-level Python utilities on top of the view primitive:

```python
def for_each_edge_excluded(
    graph: Graph,
    algorithm: str,
    edge_indices: Iterable[int] | None = None,
    **algorithm_kwargs,
) -> Iterator[tuple[int, Any]]:
    """Run an algorithm once per excluded edge, yielding (edge_index, result).

    If edge_indices is None, iterates over all edges.
    Reuses a single mask bytearray, toggling one bit per iteration.
    """
    mask = bytearray(graph.edge_count)
    indices = edge_indices if edge_indices is not None else range(graph.edge_count)
    for idx in indices:
        mask[idx] = 1
        view = GraphView._from_mask(graph, mask)
        result = getattr(view, algorithm)(**algorithm_kwargs)
        yield idx, result
        mask[idx] = 0  # reset for next iteration
```

This encapsulates the entire "for each modification, run algorithm" loop in a single
call. The mask is allocated once and toggled in-place — no per-iteration allocation.

### Design Decisions

**Why byte-mask, not bitset?** A byte array (`uint8_t[]`) is simpler in C (direct
indexing: `mask[eid]`) and avoids bit-shifting overhead in the inner loop. The 8x
memory cost (1 byte vs 1 bit per edge) is acceptable for graphs up to ~100M edges
(100 MB). If cgraph grows to handle billion-edge graphs, switch to bitset.

**Why not modify the CSR in-place?** Removing an edge from CSR requires either
rebuilding the offset array (O(V)) or marking a sentinel value in the adj array. Both
are more complex than a side-channel mask and risk corrupting the shared base graph if
a bug occurs. The mask approach is safer (base graph is truly immutable) and just as
fast.

**Why not a copy-on-write graph?** COW would share the CSR until a modification,
then copy the entire structure. For the common case (exclude 1-3 edges out of
thousands), a full copy is wasteful. The mask approach costs O(E) bytes regardless
of how many edges are excluded, but avoids any O(V + E) copy.

**Thread safety**: Views hold a read-only reference to the base graph. Multiple views
can be used concurrently (e.g. from Python threads with GIL released in C) as long as
the base Graph outlives all views. No locking needed since views never write to the
base graph's memory.

### Migration Path

All existing APIs remain unchanged. `GraphView` is opt-in — callers who don't need
masked views never see the mask parameter. The `_ctx` C functions gain an optional
mask parameter that defaults to NULL, preserving the current fast path.

### Suggested Implementation Order

1. **Phase 5a** — edge-exclusion views (highest impact, enables the core use case)
2. **Phase 5c** — benchmarks (validate performance claims before proceeding)
3. **Phase 5d** — composite `for_each_edge_excluded` helper
4. **Phase 5b** — edge-addition views (lower priority, implement on demand)

## File Structure (target)

```
src/cgraph/
├── __init__.py          # Public API — all user-facing symbols
├── _core.c              # C primitives (union-find, BFS, Dijkstra, Tarjan)
├── _paths.py            # Python tier: shortest_path, eccentricity
├── _composites.py       # Python tier: 2-edge-connected, simple paths
├── py.typed
```
