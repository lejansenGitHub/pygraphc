#include "entry_points.h"
#include "../conversion/parse.h"
#include "../algorithms/terminals.h"

/* ── Neighborhood queries over the adjacency list (O(degree) instead of O(m)) ──
 *
 * Every adjacency entry of a node is one terminal of an edge at that node: a
 * two-terminal edge as its other endpoint, an edge with three or more terminals
 * as -1 - e with the terminal in edge_indices (directed: in-terminals in the
 * forward list, out-terminals in the reverse list). An entry joins the node to
 * other nodes when its terminal is active and its edge has an active terminal
 * on the other side: any other terminal undirected, an out-terminal from an
 * in-terminal, an in-terminal from an out-terminal (D5, D24). Degrees count
 * such terminals, so a self-loop and an edge with two terminals at the node
 * count twice; neighbor sets leave the node itself out.
 */

/* Helper: resolve a node ID to its internal index via the IntMap.
 * Returns -1 if not found. Sets *found to 0 if not found, 1 if found.
 * Does NOT set a Python exception on not-found (callers return empty). */
static int resolve_node(GraphContext *graph, PyObject *node_id_obj, int *found) {
    long node_id = PyLong_AsLong(node_id_obj);
    if (node_id == -1 && PyErr_Occurred()) { *found = -1; return -1; }
    if (graph->node_ids.node_count == 0 || !graph->node_ids.node_index_by_node_id.keys) { *found = 0; return -1; }
    int idx = intmap_get(&graph->node_ids.node_index_by_node_id, node_id);
    *found = (idx >= 0) ? 1 : 0;
    return idx;
}

/*
 * A parsed query at one node: the graph, the node (-1 when the answer is empty)
 * and the masks. The terminal mask is read as given: folding it into the edge
 * mask, as the whole-graph kernels do, would cost O(m) per query.
 */
typedef struct {
    GraphContext *graph;
    int node;
    const uint8_t *edge_mask, *node_mask, *terminal_mask;
    Py_buffer edge_mask_buffer, node_mask_buffer, terminal_mask_buffer;
} NodeQuery;

/*
 * Parse (capsule, node_id[, edge_mask, node_mask, terminal_mask]). The node is -1
 * when it is unknown, masked or the graph has no edges. directed_method names
 * the method for the TypeError on an undirected graph, NULL when any graph will
 * do. Returns 0, or -1 with an exception set and nothing to release.
 */
static int open_node_query(PyObject *args, const char *directed_method, NodeQuery *query) {
    PyObject *capsule, *node_id_obj, *edge_mask_obj = Py_None, *node_mask_obj = Py_None, *terminal_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OO|OOO", &capsule, &node_id_obj, &edge_mask_obj, &node_mask_obj, &terminal_mask_obj))
        return -1;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return -1;
    if (directed_method && !graph->directed) {
        PyErr_Format(PyExc_TypeError, "%s requires a directed graph", directed_method);
        return -1;
    }
    int found;
    int node = resolve_node(graph, node_id_obj, &found);
    if (found < 0) return -1;
    query->graph = graph;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &query->edge_mask, &query->edge_mask_buffer) < 0)
        return -1;
    if (parse_mask(node_mask_obj, graph->node_ids.node_count, &query->node_mask, &query->node_mask_buffer) < 0) {
        release_mask(&query->edge_mask_buffer);
        return -1;
    }
    Py_ssize_t terminal_count = graph->terminals ? graph->terminals->terminal_count : 2 * graph->node_ids.edges.edge_count;
    if (parse_mask(terminal_mask_obj, terminal_count, &query->terminal_mask, &query->terminal_mask_buffer) < 0) {
        release_mask(&query->node_mask_buffer); release_mask(&query->edge_mask_buffer);
        return -1;
    }
    query->node = (found && graph->has_adjacency && !(query->node_mask && query->node_mask[node])) ? node : -1;
    return 0;
}

static void close_node_query(NodeQuery *query) {
    release_mask(&query->terminal_mask_buffer);
    release_mask(&query->node_mask_buffer);
    release_mask(&query->edge_mask_buffer);
}

/* The forward list (undirected, or the out-side of a directed graph) or the reverse list of a node. */
typedef struct {
    const int *offset, *neighbors, *edge_indices;
} EntryList;

static EntryList entry_list(const GraphContext *graph, int reverse) {
    const AdjacencyList *adjacency = &graph->adjacency;
    EntryList entries = {adjacency->offset, adjacency->neighbors, adjacency->edge_indices};
    if (reverse) {
        entries.offset = adjacency->reverse_offset;
        entries.neighbors = adjacency->reverse_neighbors;
        entries.edge_indices = adjacency->reverse_edge_indices;
    }
    return entries;
}

/*
 * The terminals of a multi-terminal edge that an entry joins its node to: every
 * terminal undirected, the out-terminals from an in-terminal (forward), the
 * in-terminals from an out-terminal (reverse).
 */
static void joined_terminals(const GraphContext *graph, int edge, int reverse, int *first, int *end) {
    const TerminalList *terminals = graph->terminals;
    *first = terminals->terminal_offsets[edge];
    *end = terminals->terminal_offsets[edge + 1];
    if (!graph->directed) return;
    if (reverse) *end = terminals->out_terminal_offsets[edge];
    else *first = terminals->out_terminal_offsets[edge];
}

/*
 * Whether the entry at slot joins the query's node to another active terminal. A
 * two-terminal edge joins when it, both its terminals and the other endpoint are
 * unmasked (case 5); its terminals are 2e and 2e + 1 without a terminal list.
 */
static int entry_is_joined(const NodeQuery *query, const EntryList *entries, int slot, int reverse) {
    int neighbor = entries->neighbors[slot], index = entries->edge_indices[slot];
    const TerminalList *terminals = query->graph->terminals;
    const uint8_t *terminal_mask = query->terminal_mask;
    if (neighbor >= 0) {
        if (query->edge_mask && query->edge_mask[index]) return 0;
        if (query->node_mask && query->node_mask[neighbor]) return 0;
        if (!terminal_mask) return 1;
        int first_terminal = terminals ? terminals->terminal_offsets[index] : 2 * index;
        return !terminal_mask[first_terminal] && !terminal_mask[first_terminal + 1];
    }
    int edge = -1 - neighbor;
    if (!terminal_is_active(terminals, edge, index, query->edge_mask, terminal_mask, query->node_mask)) return 0;
    int first, end;
    joined_terminals(query->graph, edge, reverse, &first, &end);
    for (int terminal = first; terminal < end; terminal++)
        if (terminal != index && terminal_is_active(terminals, edge, terminal, query->edge_mask, terminal_mask, query->node_mask))
            return 1;
    return 0;
}

/* Whether the query has no mask: then every entry is joined, since every edge in the list has an active terminal on the other side. */
static int has_no_mask(const NodeQuery *query) {
    return !query->edge_mask && !query->node_mask && !query->terminal_mask;
}

/* The number of the node's entries that are joined: its degree on that side. */
static long joined_entry_count(const NodeQuery *query, int reverse) {
    if (query->node < 0) return 0;
    EntryList entries = entry_list(query->graph, reverse);
    if (has_no_mask(query)) return entries.offset[query->node + 1] - entries.offset[query->node];
    long count = 0;
    for (int slot = entries.offset[query->node]; slot < entries.offset[query->node + 1]; slot++)
        count += entry_is_joined(query, &entries, slot, reverse);
    return count;
}

/* The nodes the query's node is joined to on that side, the node itself left out. NULL with an exception set. */
static PyObject *joined_nodes(const NodeQuery *query, int reverse) {
    PyObject *result = PySet_New(NULL);
    if (!result || query->node < 0) return result;
    const GraphContext *graph = query->graph;
    EntryList entries = entry_list(graph, reverse);
    for (int slot = entries.offset[query->node]; slot < entries.offset[query->node + 1]; slot++) {
        if (!entry_is_joined(query, &entries, slot, reverse)) continue;
        int neighbor = entries.neighbors[slot];
        if (neighbor >= 0) {
            if (neighbor != query->node && PySet_Add(result, graph->node_ids.node_id_items[neighbor]) < 0) goto fail;
            continue;
        }
        const TerminalList *terminals = graph->terminals;
        int edge = -1 - neighbor, first, end;
        joined_terminals(graph, edge, reverse, &first, &end);
        for (int terminal = first; terminal < end; terminal++) {
            int terminal_node = terminals->terminal_nodes[terminal];
            if (terminal_node == query->node) continue;
            if (!terminal_is_active(terminals, edge, terminal, query->edge_mask, query->terminal_mask, query->node_mask))
                continue;
            if (PySet_Add(result, graph->node_ids.node_id_items[terminal_node]) < 0) goto fail;
        }
    }
    return result;
fail:
    Py_DECREF(result);
    return NULL;
}

/*
 * The edges with a joined entry at the query's node on that side, each once. The
 * entries of one edge at a node are adjacent in its list. NULL with an exception set.
 */
static PyObject *joined_edges(const NodeQuery *query, int reverse) {
    PyObject *result = PyList_New(0);
    if (!result || query->node < 0) return result;
    EntryList entries = entry_list(query->graph, reverse);
    int previous_edge = -1, no_mask = has_no_mask(query);
    for (int slot = entries.offset[query->node]; slot < entries.offset[query->node + 1]; slot++) {
        int edge = entries.edge_indices[slot];
        if (query->graph->terminals && entries.neighbors[slot] < 0) edge = -1 - entries.neighbors[slot];
        if (edge == previous_edge || !(no_mask || entry_is_joined(query, &entries, slot, reverse))) continue;
        previous_edge = edge;
        PyObject *edge_index = PyLong_FromLong(edge);
        if (!edge_index || PyList_Append(result, edge_index) < 0) { Py_XDECREF(edge_index); Py_DECREF(result); return NULL; }
        Py_DECREF(edge_index);
    }
    return result;
}

/* neighbors_ctx: the joined nodes; for directed graphs the successors. */
PyObject *py_neighbors_ctx(PyObject *self, PyObject *args) {
    NodeQuery query;
    if (open_node_query(args, NULL, &query) < 0) return NULL;
    PyObject *result = joined_nodes(&query, 0);
    close_node_query(&query);
    return result;
}

/* predecessors_ctx: the in-terminal nodes of the edges at the node's out-terminals. Directed only. */
PyObject *py_predecessors_ctx(PyObject *self, PyObject *args) {
    NodeQuery query;
    if (open_node_query(args, "predecessors", &query) < 0) return NULL;
    PyObject *result = joined_nodes(&query, 1);
    close_node_query(&query);
    return result;
}

/* degree_ctx: the joined terminals at the node; for directed graphs the out-degree. */
PyObject *py_degree_ctx(PyObject *self, PyObject *args) {
    NodeQuery query;
    if (open_node_query(args, NULL, &query) < 0) return NULL;
    long count = joined_entry_count(&query, 0);
    close_node_query(&query);
    return PyLong_FromLong(count);
}

/* in_degree_ctx: the joined out-terminals at the node. Directed only. */
PyObject *py_in_degree_ctx(PyObject *self, PyObject *args) {
    NodeQuery query;
    if (open_node_query(args, "in_degree", &query) < 0) return NULL;
    long count = joined_entry_count(&query, 1);
    close_node_query(&query);
    return PyLong_FromLong(count);
}

/* incident_edges_ctx: the edges joined at the node; for directed graphs the outgoing ones. */
PyObject *py_incident_edges_ctx(PyObject *self, PyObject *args) {
    NodeQuery query;
    if (open_node_query(args, NULL, &query) < 0) return NULL;
    PyObject *result = joined_edges(&query, 0);
    close_node_query(&query);
    return result;
}

/* incoming_edges_ctx: the edges joined at the node's out-terminals. Directed only. */
PyObject *py_incoming_edges_ctx(PyObject *self, PyObject *args) {
    NodeQuery query;
    if (open_node_query(args, "incoming_edge_indices", &query) < 0) return NULL;
    PyObject *result = joined_edges(&query, 1);
    close_node_query(&query);
    return result;
}

/* edge_indices_ctx: find edge indices between first_node and second_node via CSR. */
PyObject *py_edge_indices_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *first_node_obj, *second_node_obj, *edge_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OOO|O", &capsule, &first_node_obj, &second_node_obj, &edge_mask_obj)) return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    if (graph->terminals) {
        PyErr_SetString(PyExc_NotImplementedError,
                        "edge_indices names edges by node pair, which cannot name an edge with more than two terminals; "
                        "use edge_indices_at()");
        return NULL;
    }

    int first_found, second_found;
    int first_node = resolve_node(graph, first_node_obj, &first_found);
    if (first_found < 0) return NULL;
    int second_node = resolve_node(graph, second_node_obj, &second_found);
    if (second_found < 0) return NULL;

    PyObject *result = PyList_New(0);
    if (!result) return NULL;
    if (!first_found || !second_found || !graph->has_adjacency) return result;

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) { Py_DECREF(result); return NULL; }

    AdjacencyList *adjacency = &graph->adjacency;
    int previous_edge_index = -1;
    for (int slot = adjacency->offset[first_node]; slot < adjacency->offset[first_node + 1]; slot++) {
        if (adjacency->neighbors[slot] != second_node) continue;
        int edge_index = adjacency->edge_indices[slot];
        if (edge_index == previous_edge_index) continue;  /* deduplicate self-loops in undirected CSR */
        previous_edge_index = edge_index;
        if (edge_mask && edge_mask[edge_index]) continue;
        PyObject *edge_index_obj = PyLong_FromLong(edge_index);
        PyList_Append(result, edge_index_obj);
        Py_DECREF(edge_index_obj);
    }
    release_mask(&edge_mask_buffer);
    return result;
}
