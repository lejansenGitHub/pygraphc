# Multiport edges: decision record

Every decision behind edges with ports, with its date and the test that checks
it. The implementation follows this record and the README section *Edges with
ports*; a deviation is proposed and decided here first, never built in
quietly. Each implementation stage lists in its pull request which entries it
touches and how their tests came out.

| # | Decision | Decided | Checked by |
|---|----------|---------|------------|
| D1 | **Edges are first-class.** An edge is its list of ports, in the C core as well: no auxiliary node, no replacement edge, no star expansion. The algorithms work on port incidence (node → its (edge, port) pairs, edge → its ports). | 2026-10-30, reaffirmed 2026-10-31 | `test_multiport_graph_holds_exactly_the_given_nodes_and_edges` (internal node and edge counts equal the given ones) |
| D2 | An edge is a tuple of node ids: a pair is a line, a longer tuple an edge with that many ports, the position is the port index. Ports are addressed as `(edge_index, port_index)`, never by node. | 2026-10-31 | `test_port_index_is_the_position_in_the_tuple`, `test_two_ports_at_one_node_stay_distinguishable` |
| D3 | **Graphs of lines behave exactly as before**: every existing call, input format and result keeps its meaning and its speed. | 2026-10-31 | the existing unit and integration suites unchanged; instruction counts of all existing scenarios against `main` within layout noise |
| D4 | Directed: `(u, v)` stays an edge from `u` to `v` with no extra input; a multiport edge is `(in-nodes, out-nodes)`, port indices run through the in-nodes first; every edge has at least one in-port and one out-port. | 2026-10-31 | `test_directed_pair_keeps_its_meaning`, `test_directed_multiport_needs_an_in_and_an_out_port` |
| D5 | Directed reach is **OR**: reaching any active in-port reaches every active out-port. AND semantics is out; it would be a separate edge type if ever needed. | 2026-10-31 | `test_one_active_in_port_reaches_every_out_port` |
| D6 | **One rule decides activity:** a port is active iff its edge, the port and its node are unmasked. Every algorithm asks only this rule. | 2026-10-31 | the twelve masking-case tests (D9) on every algorithm family |
| D7 | Connection: an edge joins all of its active ports (directed: every active in-port to every active out-port). | 2026-10-31 | masking cases 2, 3, 8 |
| D8 | **Membership** (user rule): an edge belongs to a component iff all of its active ports are at nodes of that component; with no active port it belongs to none. | 2026-10-31 | masking cases 1–6, membership column |
| D9 | The twelve masking cases of the README table, each with a named test, connection and membership. | 2026-10-31 | `test_masking_case_<n>_<name>` for n = 1…12 |
| D10 | Masks are chainable views: `without_edges`, `without_ports`, `without_nodes`; they combine as a union, order does not matter. | 2026-10-31 | `test_masks_combine_as_a_union_in_any_order` |
| D11 | Weights: one per edge for graphs of lines, as before; one per port as soon as an edge has more than two ports. Passing through an edge costs the entry port's weight plus the exit port's weight. | 2026-10-31 | `test_path_through_a_multiport_edge_costs_entry_plus_exit`, D3 |
| D12 | An undirected edge may have one port (joins nothing, belongs to its node's component); a directed edge has at least two. | 2026-10-31 | `test_single_port_edge_joins_nothing_and_belongs_to_its_node` |
| D13 | Pair-based methods (`bridges()`, `edge_indices(u, v)`) keep their meaning on graphs of lines and raise, naming `critical_edges()` and `edge_indices_at()`, on a graph with a multiport edge. `*_with_branch_ids` stay as they are. | 2026-10-31 | `test_pair_methods_raise_on_multiport_graphs`, D3 |
| D14 | A tree-shaped edge with k ports is **one** biconnected block holding its k port nodes. | 2026-10-31 | masking case 12 |
| D15 | Scope of the first pull request: the graph core including directed edges (storage, connectivity, masks, critical edges and nodes, blocks, weighted paths). The contraction logic (reduction with port partitions) is the next pull request. | 2026-10-31 | — |
| D16 | pygraphc stays domain-agnostic: it knows ports by index only; which port is a transformer's primary is the caller's mapping. | 2026-10-31 | no domain terms in the API (review) |

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
| The multiport kernels live in their own extension, `pygraphc._ports`, sharing only the data-model header with `_core`. Inside `_core` they changed how link-time optimization inlined the shared union-find, costing graphs of lines 1 to 3% in the component kernels; apart, `_core` compiles exactly as before. | D3 |
| `Graph()` tries the unchanged line parser first and switches to the port parser only when that rejects the input; the port parser shares the node id parser, so errors on graphs of lines keep their message. A graph with a multiport edge is a `PortGraph`, a subclass of `Graph`, so the line methods stay as they were. | D3 |

## Rejected

| What | Why |
|------|-----|
| Hidden hub nodes (star expansion) as the internal model | Rejected 2026-10-30 when edges became first-class; the algorithms would need work-arounds for every case the spec defines directly. |
| Results naming edges only by id, breaking the pair methods | Replaced 2026-10-31 by D13: graphs of lines keep their interface. |
