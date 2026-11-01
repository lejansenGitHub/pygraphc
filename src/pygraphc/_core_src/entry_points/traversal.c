/* Breadth-first search and edge-simple path enumeration. */
#include "entry_points.h"
#include "../conversion/parse.h"
#include "../conversion/parse_terminals.h"
#include "../algorithms/adjacency.h"
#include "../algorithms/traversal.h"

/* The node ids of the first count queued nodes, in queue order. */
static PyObject *node_ids_in_queue_order(PyObject **node_id_items, const int *queue, int count) {
    PyObject *result = PyList_New(count);
    if (!result) return NULL;
    for (int position = 0; position < count; position++) {
        PyObject *node_id = node_id_items[queue[position]];
        Py_INCREF(node_id);
        PyList_SET_ITEM(result, position, node_id);
    }
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

    int tail = breadth_first_order(&adjacency, NULL, 0, source, NULL, NULL, NULL, visited, queue, NULL);
    PyObject *result = node_ids_in_queue_order(context.node_id_items, queue, tail);

    free(visited); free(queue); free_adjacency(&adjacency); node_id_context_free(&context);
    return result;
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

    int tail = breadth_first_order(&graph->adjacency, graph->terminals, graph->directed, source,
                                   two_terminal_edge_mask(&terminal_mask, edge_mask), terminal_mask.terminals, node_mask,
                                   visited, queue, expanded);
    PyObject *result = node_ids_in_queue_order(graph->node_ids.node_id_items, queue, tail);

    free(visited); free(queue); free(expanded);
    release_terminal_mask(&terminal_mask); release_mask(&edge_mask_buffer); release_mask(&node_mask_buffer);
    return result;
}

/* ── All-edge-paths (stack-based DFS, edge-simple: no edge repeats within a path) ── */

/* The path as a new list of edge indices, or NULL with an exception set. */
static inline __attribute__((always_inline)) PyObject *edge_path_list(const int *path_edges, int length) {
    PyObject *path = PyList_New(length);
    if (!path) return NULL;
    int allocated = 1;
    for (int depth = 0; depth < length; depth++) {
        PyObject *edge_index_obj = PyLong_FromLong(path_edges[depth]);
        allocated &= edge_index_obj != NULL;
        PyList_SET_ITEM(path, depth, edge_index_obj);
    }
    if (!allocated) { Py_DECREF(path); return NULL; }
    return path;
}

/* Append each path to the result list as a list of edge indices. */
static inline __attribute__((always_inline)) int append_edge_path(void *result, const int *path_edges, int length) {
    PyObject *path = edge_path_list(path_edges, length);
    if (!path) return -1;
    int status = PyList_Append((PyObject *)result, path);
    Py_DECREF(path);
    return status;
}

typedef struct {
    PyObject *result, *reported;
} EdgePathsOnce;

/* Append each path once: with multi-terminal edges two node paths can share one edge sequence. */
static inline __attribute__((always_inline)) int append_edge_path_once(void *visitor_state, const int *path_edges,
                                                                       int length) {
    EdgePathsOnce *paths = (EdgePathsOnce *)visitor_state;
    PyObject *path = edge_path_list(path_edges, length);
    if (!path) return -1;
    PyObject *key = PyList_AsTuple(path);
    int seen = key ? PySet_Contains(paths->reported, key) : -1;
    if (seen == 0) seen = PySet_Add(paths->reported, key) < 0 ? -1 : 0;
    Py_XDECREF(key);
    int status = seen < 0 ? -1 : seen ? 0 : PyList_Append(paths->result, path);
    Py_DECREF(path);
    return status;
}

/*
 * Each instantiation in its own non-inlined function: inlined together into the entry,
 * they share one register allocation and the two-terminal search pays for the other.
 */
static __attribute__((noinline)) int search_two_terminal_edge_paths(const GraphContext *graph,
                                                                    const EdgePathQuery *query, PyObject *result) {
    return all_edge_paths(graph, query, append_edge_path, result, 0);
}

static __attribute__((noinline)) int search_multi_terminal_edge_paths(const GraphContext *graph,
                                                                      const EdgePathQuery *query, EdgePathsOnce *paths) {
    return all_edge_paths(graph, query, append_edge_path_once, paths, 1);
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

    long source_node_id = PyLong_AsLong(source_obj);
    if (source_node_id == -1 && PyErr_Occurred()) return NULL;
    int source = node_count > 0 ? intmap_get(&graph->node_ids.node_index_by_node_id, source_node_id) : -1;
    if (source < 0) {
        PyErr_SetString(PyExc_ValueError, "source not in node_ids");
        return NULL;
    }

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

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, edge_count, &edge_mask, &edge_mask_buffer) < 0) { free(is_target); return NULL; }
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); free(is_target); return NULL; }
    TerminalMask terminal_mask;
    if (parse_terminal_mask(graph, terminal_mask_obj, edge_mask, &terminal_mask) < 0) {
        release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); free(is_target); return NULL;
    }

    PyObject *result = NULL;
    EdgePathsOnce paths = {NULL, NULL};
    uint8_t *terminal_flags = NULL;
    if (node_mask && node_mask[source]) {
        PyErr_SetString(PyExc_ValueError, "source node is excluded by node mask");
        goto done;
    }
    result = PyList_New(0);
    if (!result || !graph->has_adjacency || edge_count == 0) goto done;

    EdgePathQuery query = {source, cutoff, node_simple, ignore_self_loops, is_target,
                           two_terminal_edge_mask(&terminal_mask, edge_mask), node_mask, NULL};
    int status;
    if (graph->terminals) {
        terminal_flags = (uint8_t *)calloc((size_t)graph->terminals->terminal_count + 1, 1);
        paths = (EdgePathsOnce){result, PySet_New(NULL)};
        if (!terminal_flags || !paths.reported) {
            if (!PyErr_Occurred()) PyErr_NoMemory();
            status = -1;
        } else {
            status = flag_path_terminals(graph, edge_mask, terminal_mask.terminals, node_mask, terminal_flags);
        }
        query.terminal_flags = terminal_flags;
        if (status == 0) status = search_multi_terminal_edge_paths(graph, &query, &paths);
    } else {
        status = search_two_terminal_edge_paths(graph, &query, result);
    }
    if (status < 0) Py_CLEAR(result);

done:
    free(terminal_flags); Py_XDECREF(paths.reported);
    release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); free(is_target);
    return result;
}
