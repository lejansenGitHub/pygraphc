/* Breadth-first search and edge-simple path enumeration. */
#include "entry_points.h"
#include "../helpers/parse.h"
#include "../helpers/terminals.h"

/* ── BFS (unweighted, from source) ── */

PyObject *py_bfs(PyObject *self, PyObject *args) {
    int node_count, source;
    PyObject *edges_obj;
    if (!PyArg_ParseTuple(args, "iOi", &node_count, &edges_obj, &source)) return NULL;

    if (source < 0 || source >= node_count) {
        PyErr_SetString(PyExc_ValueError, "source out of range");
        return NULL;
    }

    EdgeList edges;
    if (parse_edges(edges_obj, &edges) < 0) return NULL;

    AdjacencyList adjacency;
    if (build_adjacency(node_count, &edges, NULL, &adjacency, 0) < 0) { free_edges(&edges); return NULL; }
    free_edges(&edges);

    int *visited = (int *)calloc((size_t)node_count, sizeof(int));
    int *queue   = (int *)malloc((size_t)node_count * sizeof(int));
    if (!visited || !queue) {
        free(visited); free(queue);
        free_adjacency(&adjacency);
        PyErr_NoMemory();
        return NULL;
    }

    int head = 0, tail = 0;
    queue[tail++] = source;
    visited[source] = 1;

    while (head < tail) {
        int node = queue[head++];
        for (int slot = adjacency.offset[node]; slot < adjacency.offset[node + 1]; slot++) {
            int neighbor = adjacency.neighbors[slot];
            if (!visited[neighbor]) {
                visited[neighbor] = 1;
                queue[tail++] = neighbor;
            }
        }
    }

    /* Build result list from queue (BFS order) */
    PyObject *result = PyList_New(tail);
    if (!result) {
        free(visited); free(queue); free_adjacency(&adjacency);
        return NULL;
    }
    for (int position = 0; position < tail; position++)
        PyList_SET_ITEM(result, position, PyLong_FromLong(queue[position]));

    free(visited); free(queue);
    free_adjacency(&adjacency);
    return result;
}

/* ── bfs (node-ID edges) ── */

PyObject *py_bfs_by_node_id(PyObject *self, PyObject *args) {
    PyObject *node_ids_obj, *edges_obj, *source_obj;
    if (!PyArg_ParseTuple(args, "OOO", &node_ids_obj, &edges_obj, &source_obj)) return NULL;

    NodeIdContext context;
    if (node_id_parse(node_ids_obj, edges_obj, &context) < 0) return NULL;
    int node_count = context.node_count;

    long source_node_id = PyLong_AsLong(source_obj);
    if (source_node_id == -1 && PyErr_Occurred()) { node_id_context_free(&context); return NULL; }
    int source = intmap_get(&context.node_index_by_node_id, source_node_id);
    if (source < 0) {
        node_id_context_free(&context);
        PyErr_SetString(PyExc_ValueError, "source not in node_ids");
        return NULL;
    }

    AdjacencyList adjacency;
    if (build_adjacency(node_count, &context.edges, NULL, &adjacency, 0) < 0) { node_id_context_free(&context); return NULL; }

    int *visited = calloc(node_count, sizeof(int)), *queue = malloc(node_count * sizeof(int));
    if (!visited || !queue) {
        free(visited); free(queue); free_adjacency(&adjacency); node_id_context_free(&context);
        PyErr_NoMemory(); return NULL;
    }

    int head = 0, tail = 0;
    queue[tail++] = source; visited[source] = 1;
    while (head < tail) {
        int node = queue[head++];
        for (int slot = adjacency.offset[node]; slot < adjacency.offset[node+1]; slot++) {
            int neighbor = adjacency.neighbors[slot];
            if (!visited[neighbor]) { visited[neighbor] = 1; queue[tail++] = neighbor; }
        }
    }

    PyObject *result = PyList_New(tail);
    if (result) {
        for (int position = 0; position < tail; position++) {
            Py_INCREF(context.node_id_items[queue[position]]);
            PyList_SET_ITEM(result, position, context.node_id_items[queue[position]]);
        }
    }

    free(visited); free(queue); free_adjacency(&adjacency); node_id_context_free(&context);
    return result;
}

/*
 * Enter an edge with three or more terminals through the terminal of its slot and, the
 * first time that terminal is active, add the nodes of all its active exit terminals
 * (directed: out-terminals) to the queue. Returns the new queue tail.
 */
static inline int expand_multi_terminal_edge(const GraphContext *graph, int slot, const uint8_t *edge_mask,
                                        const uint8_t *terminal_mask, const uint8_t *node_mask, int *visited,
                                        int *queue, int tail, uint8_t *expanded) {
    const TerminalList *terminals = graph->terminals;
    int edge = -1 - graph->adjacency.neighbors[slot];
    if (expanded[edge] || !terminal_is_active(terminals, edge, graph->adjacency.edge_indices[slot], edge_mask, terminal_mask, node_mask))
        return tail;
    expanded[edge] = 1;
    int first_exit = graph->directed ? terminals->out_terminal_offsets[edge] : terminals->terminal_offsets[edge];
    for (int exit = first_exit; exit < terminals->terminal_offsets[edge + 1]; exit++) {
        if (!terminal_is_active(terminals, edge, exit, edge_mask, terminal_mask, node_mask)) continue;
        int reached = terminals->terminal_nodes[exit];
        if (!visited[reached]) { visited[reached] = 1; queue[tail++] = reached; }
    }
    return tail;
}

/*
 * Breadth-first order from source into queue; returns the number of nodes
 * visited. A two-terminal edge leads to its neighbor, an edge with three or more
 * terminals is expanded once (expand_multi_terminal_edge). The loop is instantiated for
 * both values of has_multi_terminal_edges, so on a graph of two-terminal edges it runs
 * without the branch for multi-terminal entries.
 */
static inline __attribute__((always_inline)) int breadth_first_order(
    const GraphContext *graph, int source, const uint8_t *edge_mask, const uint8_t *terminal_mask,
    const uint8_t *node_mask, int *visited, int *queue, uint8_t *expanded, const int has_multi_terminal_edges) {
    const AdjacencyList *adjacency = &graph->adjacency;
    int head = 0, tail = 0;
    queue[tail++] = source; visited[source] = 1;
    while (head < tail) {
        int node = queue[head++];
        for (int slot = adjacency->offset[node]; slot < adjacency->offset[node+1]; slot++) {
            if (has_multi_terminal_edges && adjacency->neighbors[slot] < 0) {
                tail = expand_multi_terminal_edge(graph, slot, edge_mask, terminal_mask, node_mask, visited, queue, tail, expanded);
                continue;
            }
            if (edge_mask && edge_mask[adjacency->edge_indices[slot]]) continue;
            int neighbor = adjacency->neighbors[slot];
            if (node_mask && node_mask[neighbor]) continue;
            if (!visited[neighbor]) { visited[neighbor] = 1; queue[tail++] = neighbor; }
        }
    }
    return tail;
}

PyObject *py_bfs_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *source_obj, *edge_mask_obj = Py_None, *node_mask_obj = Py_None, *terminal_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OO|OOO", &capsule, &source_obj, &edge_mask_obj, &node_mask_obj, &terminal_mask_obj)) return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    int node_count = graph->node_ids.node_count;

    long source_node_id = PyLong_AsLong(source_obj);
    if (source_node_id == -1 && PyErr_Occurred()) return NULL;
    int source = intmap_get(&graph->node_ids.node_index_by_node_id, source_node_id);
    if (source < 0) {
        PyErr_SetString(PyExc_ValueError, "source not in node_ids");
        return NULL;
    }

    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) return NULL;
    if (node_mask && node_mask[source]) {
        release_mask(&node_mask_buffer);
        PyErr_SetString(PyExc_ValueError, "source node is excluded by node mask");
        return NULL;
    }

    if (!graph->has_adjacency) { release_mask(&node_mask_buffer); return PyList_New(0); }

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) { release_mask(&node_mask_buffer); return NULL; }
    TerminalMask terminal_mask;
    if (parse_terminal_mask(graph, terminal_mask_obj, edge_mask, &terminal_mask) < 0) { release_mask(&edge_mask_buffer); release_mask(&node_mask_buffer); return NULL; }

    int *visited = calloc(node_count, sizeof(int)), *queue = malloc(node_count * sizeof(int));
    uint8_t *expanded = graph->terminals ? (uint8_t *)calloc((size_t)graph->node_ids.edges.edge_count, 1) : NULL;
    if (!visited || !queue || (graph->terminals && !expanded)) {
        free(visited); free(queue); free(expanded);
        release_terminal_mask(&terminal_mask); release_mask(&edge_mask_buffer); release_mask(&node_mask_buffer);
        PyErr_NoMemory(); return NULL;
    }

    const uint8_t *traversal_edge_mask = two_terminal_edge_mask(&terminal_mask, edge_mask);
    int tail = graph->terminals
        ? breadth_first_order(graph, source, traversal_edge_mask, terminal_mask.terminals, node_mask, visited, queue, expanded, 1)
        : breadth_first_order(graph, source, traversal_edge_mask, NULL, node_mask, visited, queue, NULL, 0);

    PyObject *result = PyList_New(tail);
    if (result) {
        for (int position = 0; position < tail; position++) {
            Py_INCREF(graph->node_ids.node_id_items[queue[position]]);
            PyList_SET_ITEM(result, position, graph->node_ids.node_id_items[queue[position]]);
        }
    }
    free(visited); free(queue); free(expanded);
    release_terminal_mask(&terminal_mask); release_mask(&edge_mask_buffer); release_mask(&node_mask_buffer);
    return result;
}

/* ── All-edge-paths (stack-based DFS, edge-simple: no edge repeats within a path) ── */

/* Flags per terminal of a multi-terminal edge, fixed for one call: which terminals a path enters and leaves by. */
enum { ENTRY_TERMINAL = 1, EXIT_TERMINAL = 2, SHARES_ITS_NODE = 4 };

/*
 * Flag the terminals of every multi-terminal edge a path steps through (D28). A path
 * enters at an active terminal and leaves at another (directed: an in-terminal to an
 * out-terminal); terminals of one edge at one node lead to the same node, so only the
 * first active one per node and side enters or leaves, and SHARES_ITS_NODE marks a
 * first terminal with another active one at its node, the step back to it. O(terminals).
 */
static int flag_path_terminals(const GraphContext *graph, const uint8_t *edge_mask, const uint8_t *terminal_mask,
                               const uint8_t *node_mask, uint8_t *flags) {
    const TerminalList *terminals = graph->terminals;
    int node_count = graph->node_ids.node_count, edge_count = (int)graph->node_ids.edges.edge_count;
    int *stamp = (int *)malloc((size_t)(node_count > 0 ? node_count : 1) * sizeof(int));
    int *first = (int *)malloc((size_t)(node_count > 0 ? node_count : 1) * sizeof(int));
    if (!stamp || !first) { free(stamp); free(first); PyErr_NoMemory(); return -1; }
    memset(stamp, -1, (size_t)node_count * sizeof(int));
    for (int edge = 0; edge < edge_count; edge++) {
        if (!is_multi_terminal_edge(terminals, edge)) continue;
        int out_start = graph->directed ? terminals->out_terminal_offsets[edge] : terminals->terminal_offsets[edge + 1];
        for (int terminal = terminals->terminal_offsets[edge]; terminal < terminals->terminal_offsets[edge + 1]; terminal++) {
            if (!terminal_is_active(terminals, edge, terminal, edge_mask, terminal_mask, node_mask)) continue;
            int is_out = terminal >= out_start, node = terminals->terminal_nodes[terminal];
            int key = graph->directed ? 2 * edge + is_out : edge;
            if (stamp[node] == key) { flags[first[node]] |= SHARES_ITS_NODE; continue; }
            stamp[node] = key; first[node] = terminal;
            flags[terminal] = graph->directed ? (is_out ? EXIT_TERMINAL : ENTRY_TERMINAL) : ENTRY_TERMINAL | EXIT_TERMINAL;
        }
    }
    free(stamp); free(first);
    return 0;
}

/* Append the path of edges[0 .. length - 1] to result, once: with multi-terminal edges two node paths can share it. */
static inline __attribute__((always_inline)) int report_edge_path(const int *edges, int length, PyObject *result,
                                                                  PyObject *reported) {
    PyObject *path = PyList_New(length);
    if (!path) return -1;
    for (int depth = 0; depth < length; depth++) {
        PyObject *edge_index_obj = PyLong_FromLong(edges[depth]);
        if (!edge_index_obj) { Py_DECREF(path); return -1; }
        PyList_SET_ITEM(path, depth, edge_index_obj);
    }
    if (reported) {
        PyObject *key = PyList_AsTuple(path);
        int seen = key ? PySet_Contains(reported, key) : -1;
        if (seen == 0) seen = PySet_Add(reported, key) < 0 ? -1 : 0;
        Py_XDECREF(key);
        if (seen != 0) { Py_DECREF(path); return seen < 0 ? -1 : 0; }
    }
    int status = PyList_Append(result, path);
    Py_DECREF(path);
    return status;
}

typedef struct {
    int source, max_depth, node_simple, ignore_self_loops;
    const uint8_t *is_target, *two_terminal_edge_mask, *node_mask;
    const uint8_t *terminal_flags;   /* flag_path_terminals, with multi-terminal edges only */
} EdgePathQuery;

/*
 * Depth-first search over the edge-simple paths from the source, appending each
 * path that reaches a target. Instantiated per value of has_multi_terminal_edges
 * (D17): a frame at a node walks its adjacency slots, and with multi-terminal
 * edges a slot entering such an edge first walks the edge's exit terminals, the
 * edge staying used until they are done. Returns 0, or -1 with an exception set.
 */
static inline __attribute__((always_inline)) int edge_path_search(const GraphContext *graph, const EdgePathQuery *query,
                                                                  PyObject *result, PyObject *reported,
                                                                  const int has_multi_terminal_edges) {
    const AdjacencyList *adjacency = &graph->adjacency;
    const TerminalList *terminals = graph->terminals;
    int node_count = graph->node_ids.node_count, max_depth = query->max_depth;
    Py_ssize_t edge_count = graph->node_ids.edges.edge_count;
    const uint8_t *edge_mask = query->two_terminal_edge_mask, *node_mask = query->node_mask, *is_target = query->is_target;
    const uint8_t *terminal_flags = query->terminal_flags;
    const int ignore_self_loops = query->ignore_self_loops, directed = graph->directed;

    uint8_t *visited_edges = (uint8_t *)calloc((size_t)edge_count, sizeof(uint8_t));
    uint8_t *visited_nodes = query->node_simple ? (uint8_t *)calloc((size_t)node_count, sizeof(uint8_t)) : NULL;
    int *stack_node = (int *)malloc((size_t)(max_depth + 1) * sizeof(int));
    int *stack_slot = (int *)malloc((size_t)(max_depth + 1) * sizeof(int));
    int *path_edges = (int *)malloc((size_t)max_depth * sizeof(int));
    int *stack_edge = has_multi_terminal_edges ? (int *)malloc((size_t)(max_depth + 1) * sizeof(int)) : NULL;
    int *stack_entry = has_multi_terminal_edges ? (int *)malloc((size_t)(max_depth + 1) * sizeof(int)) : NULL;
    int *stack_exit = has_multi_terminal_edges ? (int *)malloc((size_t)(max_depth + 1) * sizeof(int)) : NULL;
    int status = 0;
    if (!visited_edges || !stack_node || !stack_slot || !path_edges || (query->node_simple && !visited_nodes) ||
        (has_multi_terminal_edges && (!stack_edge || !stack_entry || !stack_exit))) {
        PyErr_NoMemory(); status = -1; goto done;
    }
    if (visited_nodes) visited_nodes[query->source] = 1;  /* source counts as visited */

    int stack_top = 0;
    stack_node[0] = query->source;
    stack_slot[0] = adjacency->offset[query->source];
    if (has_multi_terminal_edges) stack_edge[0] = -1;

    while (stack_top >= 0) {
        int node = stack_node[stack_top];
        int edge_index, neighbor, in_multi_terminal_edge = 0;
        if (has_multi_terminal_edges && stack_edge[stack_top] >= 0) {
            /* Walk the exits of the multi-terminal edge this frame entered; the
             * edge stays used until they are done. */
            int edge = stack_edge[stack_top], entry = stack_entry[stack_top], end = terminals->terminal_offsets[edge + 1];
            int exit = -1;
            while (stack_exit[stack_top] < end) {
                int terminal = stack_exit[stack_top]++;
                if ((terminal_flags[terminal] & EXIT_TERMINAL) &&
                    (terminal != entry || (terminal_flags[entry] & SHARES_ITS_NODE))) {
                    exit = terminal;
                    break;
                }
            }
            if (exit < 0) {
                visited_edges[edge] = 0;
                stack_edge[stack_top] = -1;
                continue;
            }
            in_multi_terminal_edge = 1;
            edge_index = edge;
            neighbor = terminals->terminal_nodes[exit];
            if (neighbor == node && ignore_self_loops) continue;
            goto step;
        }
        if (stack_slot[stack_top] < adjacency->offset[node + 1]) {
            int idx = stack_slot[stack_top]++;
            edge_index = adjacency->edge_indices[idx];
            neighbor = adjacency->neighbors[idx];
            if (has_multi_terminal_edges && neighbor < 0) {
                int edge = -1 - neighbor;
                if (visited_edges[edge] || !(terminal_flags[edge_index] & ENTRY_TERMINAL)) continue;
                visited_edges[edge] = 1;
                stack_edge[stack_top] = edge;
                stack_entry[stack_top] = edge_index;
                stack_exit[stack_top] = directed ? terminals->out_terminal_offsets[edge] : terminals->terminal_offsets[edge];
                continue;
            }

            /* Self-loops: optionally never traversed. Otherwise an undirected
             * self-loop occupies two consecutive CSR slots and is visited
             * through the first slot only. */
            if (neighbor == node && ignore_self_loops) continue;
            if (neighbor == node && idx > adjacency->offset[node] && adjacency->edge_indices[idx - 1] == edge_index &&
                (!has_multi_terminal_edges || adjacency->neighbors[idx - 1] >= 0)) continue;

            /* Skip masked/visited edges */
            if (visited_edges[edge_index]) continue;
            if (edge_mask && edge_mask[edge_index]) continue;
            if (node_mask && node_mask[neighbor]) continue;

            /* Mark edge as visited, record in path */
            visited_edges[edge_index] = 1;
        step:
            path_edges[stack_top] = edge_index;

            /* Check if neighbor is a target BEFORE checking node_simple.
             * This matches the Python all_edge_paths_multigraph which yields
             * the path before checking if the node can be entered. This is
             * critical for self-loops: source -> self-loop -> source(=target)
             * should yield even though source is already visited. */
            if (is_target[neighbor] && report_edge_path(path_edges, stack_top + 1, result, reported) < 0) {
                status = -1;
                goto done;
            }

            /* Check node_simple constraint: skip entering already-visited nodes;
             * a multi-terminal edge stays used for its other exits. */
            if (visited_nodes && visited_nodes[neighbor]) {
                if (!in_multi_terminal_edge) visited_edges[edge_index] = 0;
                continue;
            }

            /* Mark node as visited (only when actually entering) */
            if (visited_nodes) visited_nodes[neighbor] = 1;

            /* Push neighbor if depth allows */
            if (stack_top + 1 < max_depth) {
                stack_top++;
                stack_node[stack_top] = neighbor;
                stack_slot[stack_top] = adjacency->offset[neighbor];
                if (has_multi_terminal_edges) stack_edge[stack_top] = -1;
            } else {
                /* At max depth, don't descend, unmark edge and node */
                if (!in_multi_terminal_edge) visited_edges[edge_index] = 0;
                if (visited_nodes) visited_nodes[neighbor] = 0;
            }
        } else {
            /* Pop: unmark the node and the edge that brought us here, unless that
             * edge still walks its exits in the frame below. */
            if (stack_top > 0 && visited_nodes) {
                visited_nodes[stack_node[stack_top]] = 0;
            }
            stack_top--;
            if (stack_top >= 0 && !(has_multi_terminal_edges && stack_edge[stack_top] >= 0)) {
                visited_edges[path_edges[stack_top]] = 0;
            }
        }
    }

done:
    free(visited_edges); free(visited_nodes);
    free(stack_node); free(stack_slot); free(path_edges);
    free(stack_edge); free(stack_entry); free(stack_exit);
    return status;
}

/*
 * Each instantiation in its own non-inlined function: inlined together into the entry,
 * they share one register allocation and the two-terminal search pays for the other.
 */
static __attribute__((noinline)) int search_two_terminal_edge_paths(const GraphContext *graph,
                                                                    const EdgePathQuery *query, PyObject *result) {
    return edge_path_search(graph, query, result, NULL, 0);
}

static __attribute__((noinline)) int search_multi_terminal_edge_paths(const GraphContext *graph,
                                                                      const EdgePathQuery *query, PyObject *result,
                                                                      PyObject *reported) {
    return edge_path_search(graph, query, result, reported, 1);
}

PyObject *py_all_edge_paths_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *source_obj, *targets_obj;
    int cutoff = -1;
    PyObject *edge_mask_obj = Py_None, *node_mask_obj = Py_None, *terminal_mask_obj = Py_None;
    int node_simple = 0;
    int ignore_self_loops = 0;
    if (!PyArg_ParseTuple(args, "OOO|iOOppO", &capsule, &source_obj, &targets_obj,
                          &cutoff, &edge_mask_obj, &node_mask_obj, &node_simple, &ignore_self_loops, &terminal_mask_obj))
        return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    int node_count = graph->node_ids.node_count;
    Py_ssize_t edge_count = graph->node_ids.edges.edge_count;

    /* Parse source */
    long source_node_id = PyLong_AsLong(source_obj);
    if (source_node_id == -1 && PyErr_Occurred()) return NULL;
    int source = node_count > 0 ? intmap_get(&graph->node_ids.node_index_by_node_id, source_node_id) : -1;
    if (source < 0) {
        PyErr_SetString(PyExc_ValueError, "source not in node_ids");
        return NULL;
    }

    /* Parse targets */
    PyObject *targets_fast = PySequence_Fast(targets_obj, "targets must be a sequence");
    if (!targets_fast) return NULL;
    Py_ssize_t target_count = PySequence_Fast_GET_SIZE(targets_fast);
    uint8_t *is_target = (uint8_t *)calloc(node_count, sizeof(uint8_t));
    if (!is_target) { Py_DECREF(targets_fast); PyErr_NoMemory(); return NULL; }

    PyObject **target_items = PySequence_Fast_ITEMS(targets_fast);
    for (Py_ssize_t position = 0; position < target_count; position++) {
        long target_node_id = PyLong_AsLong(target_items[position]);
        if (target_node_id == -1 && PyErr_Occurred()) {
            free(is_target); Py_DECREF(targets_fast); return NULL;
        }
        int target_index = intmap_get(&graph->node_ids.node_index_by_node_id, target_node_id);
        if (target_index >= 0) is_target[target_index] = 1;
    }
    Py_DECREF(targets_fast);

    /* Parse masks */
    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, edge_count, &edge_mask, &edge_mask_buffer) < 0) { free(is_target); return NULL; }
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); free(is_target); return NULL; }
    TerminalMask terminal_mask;
    if (parse_terminal_mask(graph, terminal_mask_obj, edge_mask, &terminal_mask) < 0) {
        release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); free(is_target); return NULL;
    }

    PyObject *result = NULL, *reported = NULL;
    uint8_t *terminal_flags = NULL;
    if (node_mask && node_mask[source]) {
        PyErr_SetString(PyExc_ValueError, "source node is excluded by node mask");
        goto done;
    }
    result = PyList_New(0);
    if (!result || !graph->has_adjacency || edge_count == 0) goto done;

    EdgePathQuery query = {source, (cutoff > 0) ? cutoff : (int)(2 * edge_count) /* safety bound */, node_simple,
                           ignore_self_loops, is_target, two_terminal_edge_mask(&terminal_mask, edge_mask), node_mask, NULL};
    int status;
    if (graph->terminals) {
        terminal_flags = (uint8_t *)calloc((size_t)graph->terminals->terminal_count + 1, 1);
        reported = PySet_New(NULL);
        if (!terminal_flags || !reported) {
            if (!PyErr_Occurred()) PyErr_NoMemory();
            status = -1;
        } else {
            status = flag_path_terminals(graph, edge_mask, terminal_mask.terminals, node_mask, terminal_flags);
        }
        query.terminal_flags = terminal_flags;
        if (status == 0) status = search_multi_terminal_edge_paths(graph, &query, result, reported);
    } else {
        status = search_two_terminal_edge_paths(graph, &query, result);
    }
    if (status < 0) Py_CLEAR(result);

done:
    free(terminal_flags); Py_XDECREF(reported);
    release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); free(is_target);
    return result;
}
