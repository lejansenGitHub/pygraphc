# Multiport edges: decision record

Every decision behind edges with ports, with its date and the test that checks
it. The implementation follows this record and the README section *Edges with
ports*; a deviation is proposed and decided here first, never built in
quietly. Each implementation stage lists in its pull request which entries it
touches and how their tests came out.

| # | Decision | Decided | Checked by |
|---|----------|---------|------------|
| D1 | **Edges are first-class.** An edge is its list of ports, in the C core as well: no auxiliary node, no replacement edge, no star expansion. The algorithms work on port incidence (node → its (edge, port) pairs, edge → its ports). | 2026-10-30, reaffirmed 2026-10-31 | `test_multiport_graph_holds_exactly_the_given_nodes_and_edges` (internal node and edge counts equal the given ones) |
| D2 | An edge is a tuple of node ids: a pair is a two-port edge, a longer tuple an edge with that many ports, the position is the port index. Ports are addressed as `(edge_index, port_index)`, never by node. | 2026-10-31 | `test_port_index_is_the_position_in_the_tuple`, `test_two_ports_at_one_node_stay_distinguishable` |
| D3 | **Graphs of two-port edges behave exactly as before**: every existing call, input format and result keeps its meaning and its speed. | 2026-10-31 | the existing unit and integration suites unchanged; instruction counts of all existing scenarios against `main` within layout noise |
| D4 | Directed: `(u, v)` stays an edge from `u` to `v` with no extra input; a multiport edge is `(in-nodes, out-nodes)`, port indices run through the in-nodes first; every edge has at least one in-port and one out-port. | 2026-10-31 | `test_directed_pair_keeps_its_meaning`, `test_directed_multiport_needs_an_in_and_an_out_port` |
| D5 | Directed reach is **OR**: reaching any active in-port reaches every active out-port. AND semantics is out; it would be a separate edge type if ever needed. | 2026-10-31 | `test_one_active_in_port_reaches_every_out_port` |
| D6 | **One rule decides activity:** a port is active iff its edge, the port and its node are unmasked. Every algorithm asks only this rule. | 2026-10-31 | the twelve masking-case tests (D9) on every algorithm family |
| D7 | Connection: an edge joins all of its active ports (directed: every active in-port to every active out-port). | 2026-10-31 | masking cases 2, 3, 8 |
| D8 | **Membership** (user rule): an edge belongs to a component iff all of its active ports are at nodes of that component; with no active port it belongs to none. | 2026-10-31 | masking cases 1–6, membership column |
| D9 | The twelve masking cases of the README table, each with a named test, connection and membership. | 2026-10-31 | `test_masking_case_<n>_<name>` for n = 1…12 |
| D10 | Masks are chainable views: `without_edges`, `without_ports`, `without_nodes`; they combine as a union, order does not matter. | 2026-10-31 | `test_masks_combine_as_a_union_in_any_order` |
| D11 | Weights: one per edge for graphs of two-port edges, as before; one per port as soon as an edge has more than two ports. Passing through an edge costs the entry port's weight plus the exit port's weight. | 2026-10-31 | `test_path_through_a_multiport_edge_costs_entry_plus_exit`, D3 |
| D12 | An undirected edge may have one port (joins nothing, belongs to its node's component); a directed edge has at least two. | 2026-10-31 | `test_single_port_edge_joins_nothing_and_belongs_to_its_node` |
| D13 | Pair-based methods (`bridges()`, `edge_indices(u, v)`) keep their meaning on graphs of two-port edges and raise, naming `critical_edges()` and `edge_indices_at()`, on a graph with a multiport edge. `*_with_branch_ids` stay as they are. | 2026-10-31 | `test_pair_methods_raise_on_multiport_graphs`, D3 |
| D14 | A tree-shaped edge with k ports is **one** biconnected block holding its k port nodes. | 2026-10-31 | masking case 12 |
| D15 | Scope of the first pull request: the graph core including directed edges (storage, connectivity, masks, critical edges and nodes, blocks, weighted paths). The contraction logic (reduction with port partitions) is the next pull request. | 2026-10-31 | — |
| D16 | pygraphc stays domain-agnostic: it knows ports by index only; which port is a transformer's primary is the caller's mapping. | 2026-10-31 | no domain terms in the API (review) |
| D17 | **One fast kernel per algorithm.** One adjacency list whose entries are either a neighbor node (a two-port edge, as before) or `-1 - e`, a reference to edge `e` with three or more ports, whose port list is expanded once per traversal. It replaces separate kernels for two-port and multiport edges: graphs of two-port edges keep their speed, mixed graphs avoid the indirection for their two-port edges. Measured on a BFS prototype at 1M nodes: two-port graph 29–34 ms against 30–31 ms for the old kernel, mixed graph 44–48 ms against 94–98 ms for a port-only kernel, one edge with 1M ports linear. | 2026-10-31 | instruction counts of every two-port scenario against `main` within noise; complexity tier on the multiport shapes |

## Open

| # | Question | State |
|---|----------|-------|
| O1 | Ports whose node is unknown: rejected at construction, as edges with an unknown endpoint are today; "not connected" is a mask. | recommended, not decided |
| O2 | A "touching" relation (an edge with a port at a node of a component it is not a member of). | not part of the first pull request |
| O3 | What these mean on edges with ports: `neighbors`, `successors`, `predecessors`, `degree`, `in_degree`, `out_degree`, `incident_edge_indices`, `outgoing_edge_indices`, `incoming_edge_indices`, `is_multigraph`, `with_edges`, `split_node`, `all_edge_paths`, `cycle_basis`, `quotient_edges`, `degrees`, `bcc_edge_labels`, `series_parallel_reduce`, `series_parallel_state`. | open; until decided they raise `NotImplementedError` on a graph with multiport edges instead of answering |

## Implementation notes

Choices inside the decisions above, recorded so that they can be questioned.

| Note | Serves |
|------|--------|
| **Being resolved by D17.** The multiport kernels lived in their own extension, `pygraphc._ports`, sharing only the data-model header with `_core`. Inside `_core` they changed how link-time optimization inlined the shared union-find, costing graphs of two-port edges 1 to 3% in the component kernels; apart, `_core` compiled exactly as before. `_core` now parses graphs with multiport edges and runs the unified BFS; the components and the weighted paths still run the stage-1 kernels of `_ports`, which read the `_core` graph and fill their own fields (edge per port, port direction, node incidence) on first use. `_ports` goes once those are unified. | D3; superseded by D17, under which one kernel serves both |
| `Graph()` tries the unchanged two-port parser first and switches to the port parser (`parse_port_graph` in `_core`) only when that rejects the input; the port parser shares the node id parser, so errors on graphs of two-port edges keep their message. A graph with a multiport edge is a `PortGraph`, a subclass of `Graph`, so the two-port methods stay as they were. | D3 |
| A graph of two-port edges keeps its layout and `build_adjacency` exactly. A graph with an edge of other than two ports adds a port list (`port_offsets` numbering the ports of every edge, a two-port edge's port 0 at its first endpoint and port 1 at its second; `port_nodes`; `out_port_offsets` when directed) and is built by `build_port_adjacency`: a two-port edge as before, an edge with three or more ports as `-1 - e` at the node of each of its ports (directed: in-ports in the forward lists, out-ports in the reverse lists) with that slot's port in `edge_indices`, and a single-port edge with no slot, since it joins nothing. In the edge list every edge of other than two ports sits at its own index as (node of its first port, same node), which union-find joins to itself, so the edge-list kernels need no per-edge test. | D1, D12, D17 |
| One kernel source per algorithm, compiled twice: the BFS loop is an always-inline function with a constant `has_multiport_edges`, instantiated for both values. The instantiation for graphs of two-port edges contains no test for negative entries, which keeps its instruction count at `main`'s; the other expands an edge with three or more ports once per traversal. | D3, D17 |
| A port mask is checked only when one is given. A two-port edge with a masked port joins nothing (case 5), so it is folded into a copy of the edge mask once per call and the loop checks two-port edges by edge mask alone; the ports of an edge with three or more ports are checked by `port_is_active`, entry and exit alike. | D6, D9 |
| The C kernels that read only two-port edges (Tarjan family, strongly connected components, topological order, the queries, quotient edges, reduction) raise `NotImplementedError` on a graph with multiport edges themselves, so a direct call into `_core` can never read `-1 - e` as a node. | O3, D13 |

## Rejected

| What | Why |
|------|-----|
| Hidden hub nodes (star expansion) as the internal model | Rejected 2026-10-30 when edges became first-class; the algorithms would need work-arounds for every case the spec defines directly. |
| Results naming edges only by id, breaking the pair methods | Replaced 2026-10-31 by D13: graphs of two-port edges keep their interface. |
