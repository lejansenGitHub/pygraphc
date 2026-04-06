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

## Phase 6: MultiGraph, Node Masks, and Edge-Path Enumeration

### Motivation

Many real-world graph problems operate on **multigraphs** — graphs where multiple
edges connect the same pair of nodes (parallel cables, redundant links, alternative
routes). These problems also need:

- **Node-masked views**: exclude nodes without copying the graph
- **Edge-path enumeration**: find all paths where each physical edge is used at most
  once (edge-disjoint path exploration)

These three features are independent but reinforce each other. Together they cover the
full spectrum of "what-if" analysis on multigraphs.

### Key Insight: cgraph Already Handles Parallel Edges Internally

The C layer imposes **no uniqueness constraint** on edges:

1. **EdgeList** stores edges by index. `edges = [(1,2), (1,2), (2,3)]` creates three
   edges with indices 0, 1, 2. No deduplication.

2. **CSR (`build_adj`)** counts every edge occurrence. Node 1 gets two adjacency
   entries pointing to node 2, each with a distinct `eid`.

3. **Tarjan's bridge/AP/BCC** uses `peid` (parent edge ID) to avoid the parent edge.
   With parallel edges `(u,v,eid=0)` and `(u,v,eid=1)`:
   - Traverse eid=0 to reach v → `peid[v] = 0`
   - At v, encounter eid=1 back to u → `eid=1 != peid[v]=0` → treat as back-edge
   - This gives `low[v] <= disc[u]`, so eid=0 is **not** a bridge (correct!)
   - If there's only one edge between u and v, it may be a bridge (also correct)

4. **Union-find (CC)** calls `uf_union(src, dst)` per edge. Duplicate edges just
   union the same pair twice — no-op. Correct.

5. **BFS** marks nodes as visited. Multiple edges to the same neighbor are harmless —
   the neighbor is visited via whichever edge comes first.

6. **Dijkstra** relaxes via the minimum weight. Parallel edges with different weights
   are handled correctly — the lighter edge wins.

**Conclusion**: Phase 6a requires **no C changes** for core algorithms. The work is
Python-level: documentation, tests, API clarity, and the `edge_indices(u, v)` method
(already implemented) returning all edge indices between a node pair.

#### Phase 6a: MultiGraph Support (Formalization)

**Goal**: Officially support parallel edges. Document the behavior, add tests, ensure
all algorithms produce correct results on multigraphs.

**Python changes** (`__init__.py`):

1. Update `Graph.__init__` docstring to state that duplicate edges are allowed.
   Each duplicate gets a unique edge index (its position in the input list).

2. Add `Graph.is_multigraph` property — True if any node pair has >1 edge.

3. Verify `Graph.edge_indices(u, v)` returns all indices for parallel edges (it
   should already, since it iterates the CSR and collects all `eid` values for
   matching `adj` entries).

4. Update `GraphView.without_edges()` docstring to clarify behavior with parallel
   edges: masking edge index 0 still leaves edge index 1 active.

**Test changes** (`tests/`):

New test file `test_multigraph.py` covering:

```python
def test_cc_parallel_edges():
    """Parallel edges don't affect component count."""
    g = Graph([1, 2, 3], [(1,2), (1,2), (2,3)])
    assert g.connected_components() == [{1, 2, 3}]

def test_bridges_parallel_edges_not_bridges():
    """An edge between u,v is not a bridge if a parallel edge exists."""
    g = Graph([1, 2, 3], [(1,2), (1,2), (2,3)])
    assert g.bridges() == [(2, 3)]  # (1,2) is NOT a bridge

def test_bridges_single_edge_is_bridge():
    """Single edge between u,v is a bridge if removal disconnects."""
    g = Graph([1, 2, 3], [(1,2), (2,3)])
    assert set(map(frozenset, g.bridges())) == {frozenset({1,2}), frozenset({2,3})}

def test_edge_indices_parallel():
    """edge_indices returns all indices for parallel edges."""
    g = Graph([1, 2, 3], [(1,2), (1,2), (2,3)])
    assert sorted(g.edge_indices(1, 2)) == [0, 1]
    assert g.edge_indices(2, 3) == [2]

def test_mask_one_of_parallel():
    """Masking one parallel edge keeps the other active."""
    g = Graph([1, 2, 3], [(1,2), (1,2), (2,3)])
    view = g.without_edges([0])
    assert view.connected_components() == [{1, 2, 3}]  # still connected via edge 1

def test_mask_all_parallel():
    """Masking all parallel edges disconnects."""
    g = Graph([1, 2, 3], [(1,2), (1,2), (2,3)])
    view = g.without_edges([0, 1])
    components = view.connected_components()
    assert len(components) == 2

def test_bfs_parallel_edges():
    """BFS visits node once regardless of parallel edges."""
    g = Graph([1, 2, 3], [(1,2), (1,2), (2,3)])
    assert set(g.bfs(1)) == {1, 2, 3}

def test_dijkstra_parallel_edges_picks_lighter():
    """Shortest path picks the lighter of parallel edges."""
    g = Graph([1, 2, 3], [(1,2), (1,2), (2,3)])
    dist, path = g.shortest_path([10.0, 1.0, 5.0], 1, 3)
    assert dist == 6.0  # via edge 1 (weight 1.0) + edge 2 (weight 5.0)

def test_edge_count_with_parallel():
    """edge_count includes all parallel edges."""
    g = Graph([1, 2], [(1,2), (1,2), (1,2)])
    assert g.edge_count == 3
```

**C changes**: None expected. If any algorithm incorrectly deduplicates or crashes
with parallel edges, fix as a bug.

**Migration path**: Purely additive. Existing users see no change.

#### Phase 6b: Node-Masked Views

**Goal**: Exclude nodes (and all their incident edges) from a graph view without
copying the graph.

**Why general**: Subgraph extraction, iterative pruning ("remove all leaves"), node
failure analysis ("what if node X goes down"). The node-mask counterpart to Phase 5's
edge masks.

**C changes** (`_core.c`):

1. All CSR-traversing functions gain an optional `uint8_t *node_mask` parameter
   (alongside the existing `edge_mask`). Convention: `node_mask[v] = 1` means node
   `v` is excluded.

2. Inner loop modification:

   ```c
   // Current (with edge mask only):
   for (int i = al->offset[u]; i < al->offset[u+1]; i++) {
       if (mask && mask[al->eid[i]]) continue;
       int v = al->adj[i];
       // ...
   }

   // With node mask:
   for (int i = al->offset[u]; i < al->offset[u+1]; i++) {
       int v = al->adj[i];
       if (node_mask && node_mask[v]) continue;      // skip excluded node
       if (edge_mask && edge_mask[al->eid[i]]) continue;  // skip excluded edge
       // ...
   }
   ```

3. Outer loops (BFS queue, DFS start nodes, CC iteration) must also skip excluded
   nodes:

   ```c
   // CC union-find:
   for (int i = 0; i < n; i++) {
       if (node_mask && node_mask[i]) continue;  // exclude from components
   }

   // BFS/DFS start:
   if (node_mask && node_mask[source]) → error or empty result
   ```

4. Affected functions: same set as Phase 5a (CC, bridges, AP, BCC, BFS, Dijkstra,
   SSSP, multi-source Dijkstra). Each gains a `node_mask` parameter.

**Python changes** (`__init__.py`):

```python
class GraphView:
    # Existing: _edge_mask
    # New: _node_mask

    def __init__(self, graph, excluded_edge_indices=None, excluded_node_ids=None):
        self._base = graph
        self._edge_mask = ...  # as before
        self._node_mask = bytearray(graph.node_count) if excluded_node_ids else None
        if excluded_node_ids:
            for nid in excluded_node_ids:
                idx = self._base._node_index(nid)
                self._node_mask[idx] = 1

# On Graph:
def without_nodes(self, node_ids: Collection[int]) -> GraphView:
    """View excluding specified nodes and all their incident edges."""

# On GraphView (chainable):
def without_nodes(self, node_ids: Collection[int]) -> GraphView:
    """Further exclude nodes from this view."""
```

**API composition** — views are chainable:
```python
view = graph.without_edges([3, 7]).without_nodes([42])
components = view.connected_components()
```

**Memory cost**: 1 byte per node. Negligible (1M nodes = 1 MB).

**Tests**: Parallel structure to `test_graph_view.py`:
- CC/bridges/BFS/Dijkstra with nodes excluded
- Excluding source node in BFS/Dijkstra → error
- Excluding all nodes → empty result
- Combined edge + node masks
- Multigraph + node masks

#### Phase 6c: All-Edge-Paths Enumeration

**Goal**: Find all paths from source to targets where each **edge** is traversed at
most once. Nodes may be visited multiple times (important for multigraphs where
multiple edges connect the same pair).

**Why general**: This is the edge-disjoint path exploration problem. Applications:

- **Network reliability**: How many independent routes exist between A and B?
- **Routing with redundancy**: Find all ways to route through a network without
  reusing any physical link.
- **Switch configuration**: Find all switch-opening sequences that connect a source
  to a target (the N-1 use case, but the algorithm is domain-free).

**Algorithm**: Stack-based DFS with per-edge visit tracking.

```
visited_edges = bitset(edge_count)  // 1 bit per edge
stack = [(source, iterator over source's CSR neighbors)]

while stack not empty:
    u, neighbors = stack.top()
    edge = next(neighbors)
    if edge is None:
        stack.pop()
        if stack not empty:
            unmark last visited edge  // backtrack
        continue
    if visited_edges[edge.eid]:
        continue
    v = edge.adj
    if v in targets:
        yield path  // found a complete path
    visited_edges[edge.eid] = 1
    if len(stack) < cutoff:
        stack.push((v, iterator over v's CSR neighbors))
```

**Key properties**:
- Edge-uniqueness (not node-uniqueness): each edge index used at most once per path
- Node revisits allowed: critical for multigraphs where u→v→u→w is valid if different
  edges are used for each traversal
- Cutoff: maximum path length (number of edges). Prevents combinatorial explosion.
- Yields paths lazily (generator): caller can stop early

**C implementation** (`_core.c`):

```c
typedef struct {
    int source;
    int *targets;        // sorted array for binary search
    int n_targets;
    int cutoff;
    AdjList *al;
    uint8_t *edge_mask;  // from GraphView (NULL if no mask)
    uint8_t *node_mask;  // from GraphView (NULL if no mask)
} EdgePathCtx;
```

Two implementation approaches:

**Approach 1: Batch** — Enumerate all paths in C, return as list of lists.
Simpler C code, but may use too much memory if path count is large.

**Approach 2: Iterator** — Return a Python iterator backed by C state. Each
`__next__` call resumes the DFS from where it left off. More complex C (must
save/restore stack state), but memory-efficient.

Decision: Start with Approach 1 (batch). If memory is a concern for large graphs,
add Approach 2 as an optimization.

**Python API**:

```python
# On Graph:
def all_edge_paths(
    self,
    source: int,
    targets: int | Collection[int],
    cutoff: int | None = None,
) -> list[list[int]]:
    """Find all paths from source to targets using each edge at most once.

    Returns a list of paths. Each path is a list of edge indices.
    Nodes may appear multiple times in a path (relevant for multigraphs).

    cutoff: maximum number of edges per path. None = no limit (use with
            caution on dense graphs).
    """

# On GraphView (same signature, respects masks):
def all_edge_paths(self, source, targets, cutoff=None) -> list[list[int]]:
    """Same, but skips masked edges/nodes."""
```

**Return format**: `list[list[int]]` where each inner list contains **edge indices**
(not node IDs). The caller can recover node sequences from edge indices if needed:

```python
g = Graph([1, 2, 3, 4], [(1,2), (2,3), (2,3), (3,4)])
paths = g.all_edge_paths(source=1, targets=4, cutoff=3)
# Example result: [[0, 1, 3], [0, 2, 3]]
# Path 1: edge 0 (1→2), edge 1 (2→3), edge 3 (3→4)
# Path 2: edge 0 (1→2), edge 2 (2→3), edge 3 (3→4)  — uses other parallel edge
```

**Interaction with masks**: Masked edges are never traversed. Masked nodes are never
entered. This allows:
```python
view = graph.without_edges(faulty_edges)
paths = view.all_edge_paths(source, targets, cutoff=5)
```
No need to rebuild the graph — just mask the faulty edges and enumerate paths.

**Performance expectation**: The algorithm is inherently exponential in the worst case
(all paths in a complete graph). The cutoff parameter bounds this. For sparse graphs
with small cutoff (typical use case), the C implementation should be 10-50x faster
than the equivalent Python DFS due to:
- No Python object creation per edge traversal
- Bitset for visited edges (vs Python set/list)
- Direct CSR array access (vs `graph.edges(node, keys=True)` iterator)

**Benchmarks**:
```python
def test_all_edge_paths_vs_networkx(exponent):
    """Compare against nx.all_simple_edge_paths on simple graphs."""

def test_all_edge_paths_multigraph_scaling(parallel_factor):
    """Measure path count growth with increasing parallel edges."""

def test_all_edge_paths_cutoff_effectiveness():
    """Verify cutoff prevents combinatorial explosion."""
```

### Phase 6 Dependencies

```
Phase 5a (edge masks)     ── done
Phase 5b (edge addition)  ── independent, deferred
Phase 6a (multigraph)     ── no C changes, tests + docs only
Phase 6b (node masks)     ── C changes, independent of 6a
Phase 6c (all-edge-paths) ── C changes, benefits from 6a (parallel edges)
                              and 6b (node masks in path enumeration)
```

6a can ship immediately (test + document existing behavior).
6b and 6c are independent and can be developed in parallel.

### Suggested Implementation Order

1. **Phase 6a** — multigraph formalization (smallest effort, unblocks 6c testing)
2. **Phase 6b** — node-masked views (moderate effort, reuses Phase 5a patterns)
3. **Phase 6c** — all-edge-paths (largest effort, biggest performance impact)

### What Remains Outside cgraph After Phase 6

Two N-1 operations are too domain-specific for a generic graph library:

1. **Edge contraction with nesting** (`concentrate_parallel_and_series_edges`):
   Iteratively collapses parallel and series edges while building a recursive
   NestedEdgeIds structure for path reconstruction. This is a domain-specific
   graph simplification — the "nesting" semantics (parallel = OR, series = AND)
   have no general graph-theory equivalent.

2. **Node splitting with edge redistribution** (`_update_graph` / `_revert_graph`):
   Replaces one node with two nodes and redistributes edges based on partition
   membership. This is a graph mutation that conflicts with cgraph's immutable-view
   philosophy.

Both operations would be **significantly simpler to implement on top of cgraph** than
on networkx, because the underlying primitives (CC, bridges, views) are faster and
the edge-index model eliminates the need for manual bookkeeping. But the operations
themselves are application-level logic, not library primitives.

## File Structure (target)

```
src/cgraph/
├── __init__.py          # Public API — all user-facing symbols
├── _core.c              # C primitives (union-find, BFS, Dijkstra, Tarjan,
│                        #   edge-path enumeration, masked variants)
├── _paths.py            # Python tier: shortest_path, eccentricity
├── _composites.py       # Python tier: 2-edge-connected, simple paths
├── py.typed

tests/
├── unit_tests/
│   ├── test_correctness.py
│   ├── test_graph_class.py
│   ├── test_graph_view.py       # Phase 5a edge-mask tests
│   ├── test_multigraph.py       # Phase 6a parallel-edge tests
│   ├── test_node_mask.py        # Phase 6b node-mask tests
│   └── test_edge_paths.py       # Phase 6c all-edge-paths tests
├── performance_tests/
│   ├── test_masked_views.py     # Phase 5c benchmarks
│   ├── test_edge_paths_perf.py  # Phase 6c benchmarks
│   └── ...
```
