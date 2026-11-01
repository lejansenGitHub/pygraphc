/* Breadth-first search and edge-simple path enumeration. */
#include "entry_points.h"
#include "../conversion/parse.h"
#include "../conversion/parse_terminals.h"
#include "../algorithms/adjacency.h"

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

PyObject *py_all_edge_paths_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *source_obj, *targets_obj;
    int cutoff = -1;
    PyObject *edge_mask_obj = Py_None, *node_mask_obj = Py_None;
    int node_simple = 0;
    int ignore_self_loops = 0;
    if (!PyArg_ParseTuple(args, "OOO|iOOpp", &capsule, &source_obj, &targets_obj,
                          &cutoff, &edge_mask_obj, &node_mask_obj, &node_simple, &ignore_self_loops))
        return NULL;
    GraphContext *graph = require_two_terminal_edges(capsule);
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

    if (node_mask && node_mask[source]) {
        release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); free(is_target);
        PyErr_SetString(PyExc_ValueError, "source node is excluded by node mask");
        return NULL;
    }

    PyObject *result = PyList_New(0);
    if (!result || !graph->has_adjacency || edge_count == 0) {
        if (!result) { release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); free(is_target); return NULL; }
        release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); free(is_target);
        return result;
    }

    AdjacencyList *adjacency = &graph->adjacency;
    int max_depth = (cutoff > 0) ? cutoff : (int)(2 * edge_count);  /* safety bound */

    /* Allocate DFS state */
    uint8_t *visited_edges = (uint8_t *)calloc(edge_count, sizeof(uint8_t));
    uint8_t *visited_nodes = node_simple ? (uint8_t *)calloc(node_count, sizeof(uint8_t)) : NULL;
    int *stack_node = (int *)malloc((size_t)(max_depth + 1) * sizeof(int));
    int *stack_slot = (int *)malloc((size_t)(max_depth + 1) * sizeof(int));
    int *path_edges = (int *)malloc((size_t)max_depth * sizeof(int));
    if (!visited_edges || !stack_node || !stack_slot || !path_edges ||
        (node_simple && !visited_nodes)) {
        free(visited_edges); free(visited_nodes);
        free(stack_node); free(stack_slot); free(path_edges);
        release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); free(is_target);
        Py_DECREF(result); PyErr_NoMemory(); return NULL;
    }
    if (visited_nodes) visited_nodes[source] = 1;  /* source counts as visited */

    /* DFS */
    int stack_top = 0;
    stack_node[0] = source;
    stack_slot[0] = adjacency->offset[source];

    while (stack_top >= 0) {
        int node = stack_node[stack_top];
        if (stack_slot[stack_top] < adjacency->offset[node + 1]) {
            int idx = stack_slot[stack_top]++;
            int edge_index = adjacency->edge_indices[idx];
            int neighbor = adjacency->neighbors[idx];

            /* Self-loops: optionally never traversed. Otherwise an undirected
             * self-loop occupies two consecutive CSR slots and is visited
             * through the first slot only. */
            if (neighbor == node && ignore_self_loops) continue;
            if (neighbor == node && idx > adjacency->offset[node] && adjacency->edge_indices[idx - 1] == edge_index) continue;

            /* Skip masked/visited edges */
            if (visited_edges[edge_index]) continue;
            if (edge_mask && edge_mask[edge_index]) continue;
            if (node_mask && node_mask[neighbor]) continue;

            /* Mark edge as visited, record in path */
            visited_edges[edge_index] = 1;
            path_edges[stack_top] = edge_index;

            /* Check if neighbor is a target BEFORE checking node_simple.
             * This matches the Python all_edge_paths_multigraph which yields
             * the path before checking if the node can be entered. This is
             * critical for self-loops: source -> self-loop -> source(=target)
             * should yield even though source is already visited. */
            if (is_target[neighbor]) {
                /* Build path list */
                PyObject *path = PyList_New(stack_top + 1);
                if (!path) {
                    visited_edges[edge_index] = 0;
                    goto error;
                }
                for (int depth = 0; depth <= stack_top; depth++)
                    PyList_SET_ITEM(path, depth, PyLong_FromLong(path_edges[depth]));
                if (PyList_Append(result, path) < 0) {
                    Py_DECREF(path);
                    visited_edges[edge_index] = 0;
                    goto error;
                }
                Py_DECREF(path);
            }

            /* Check node_simple constraint: skip entering already-visited nodes */
            if (visited_nodes && visited_nodes[neighbor]) {
                visited_edges[edge_index] = 0;
                continue;
            }

            /* Mark node as visited (only when actually entering) */
            if (visited_nodes) visited_nodes[neighbor] = 1;

            /* Push neighbor if depth allows */
            if (stack_top + 1 < max_depth) {
                stack_top++;
                stack_node[stack_top] = neighbor;
                stack_slot[stack_top] = adjacency->offset[neighbor];
            } else {
                /* At max depth, don't descend, unmark edge and node */
                visited_edges[edge_index] = 0;
                if (visited_nodes) visited_nodes[neighbor] = 0;
            }
        } else {
            /* Pop: unmark the edge that brought us here and the node */
            if (stack_top > 0 && visited_nodes) {
                visited_nodes[stack_node[stack_top]] = 0;
            }
            stack_top--;
            if (stack_top >= 0) {
                visited_edges[path_edges[stack_top]] = 0;
            }
        }
    }

    free(visited_edges); free(visited_nodes);
    free(stack_node); free(stack_slot); free(path_edges);
    release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); free(is_target);
    return result;

error:
    free(visited_edges); free(visited_nodes);
    free(stack_node); free(stack_slot); free(path_edges);
    release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); free(is_target);
    Py_DECREF(result);
    return NULL;
}
