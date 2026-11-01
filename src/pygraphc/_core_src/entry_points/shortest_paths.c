/* Dijkstra: source-target paths, single-source and multi-source lengths. */
#include "entry_points.h"
#include "../conversion/parse_terminals.h"
#include "../algorithms/adjacency.h"
#include "../algorithms/arrival_labels.h"
#include "../conversion/parse.h"
#include "../algorithms/min_heap.h"
#include "../algorithms/bidirectional_dijkstra.h"

/* Turn the node indices of a bidirectional result into a list of node IDs. */
static PyObject *bidirectional_path_to_list(const BidirectionalPath *bidirectional_path, const NodeIdContext *context) {
    PyObject *path = PyList_New(bidirectional_path->path_length);
    if (!path) return NULL;
    for (int position = 0; position < bidirectional_path->path_length; position++) {
        PyObject *item = context->node_id_items[bidirectional_path->path[position]];
        Py_INCREF(item);
        PyList_SET_ITEM(path, position, item);
    }
    return path;
}

/* ── Dijkstra (source -> target, returns distance + path) ── */

PyObject *py_dijkstra(PyObject *self, PyObject *args) {
    int node_count, source, target;
    PyObject *edges_obj, *weights_obj;
    if (!PyArg_ParseTuple(args, "iOOii", &node_count, &edges_obj, &weights_obj,
                          &source, &target))
        return NULL;

    if (node_count == 0 || source < 0 || source >= node_count || target < 0 || target >= node_count) {
        PyObject *path = PyList_New(0);
        if (!path) return NULL;
        return Py_BuildValue("(dN)", HUGE_VAL, path);
    }
    if (source == target) {
        PyObject *path = PyList_New(1);
        if (!path) return NULL;
        PyList_SET_ITEM(path, 0, PyLong_FromLong(source));
        return Py_BuildValue("(dN)", 0.0, path);
    }

    EdgeList edges;
    if (parse_edges(edges_obj, &edges) < 0) return NULL;
    WeightList weights;
    if (parse_weights(weights_obj, &weights, edges.edge_count) < 0) { free_edges(&edges); return NULL; }

    AdjacencyList adjacency;
    if (build_adjacency(node_count, &edges, NULL, &adjacency, 0) < 0) {
        free_edges(&edges); free_weights(&weights); return NULL;
    }
    free_edges(&edges);

    double *distance   = (double *)malloc((size_t)node_count * sizeof(double));
    int *predecessor   = (int *)malloc((size_t)node_count * sizeof(int));
    int *heap          = (int *)malloc((size_t)node_count * sizeof(int));
    int *heap_position = (int *)malloc((size_t)node_count * sizeof(int));

    if (!distance || !predecessor || !heap || !heap_position) {
        free(distance); free(predecessor); free(heap); free(heap_position);
        free_adjacency(&adjacency); free_weights(&weights);
        PyErr_NoMemory();
        return NULL;
    }

    for (int node = 0; node < node_count; node++) {
        distance[node] = HUGE_VAL;
        predecessor[node] = -1;
        heap_position[node] = -1;
    }

    MinHeap min_heap = {distance, heap, heap_position, 0};
    distance[source] = 0.0;
    heap[0] = source;
    heap_position[source] = 0;
    min_heap.size = 1;

    while (min_heap.size > 0) {
        int node = min_heap_pop(&min_heap);
        if (node == target) break;
        if (distance[node] == HUGE_VAL) break;

        for (int slot = adjacency.offset[node]; slot < adjacency.offset[node + 1]; slot++) {
            int neighbor = adjacency.neighbors[slot];
            double weight = weights.values[adjacency.edge_indices[slot]];
            double candidate_distance = distance[node] + weight;
            if (candidate_distance < distance[neighbor]) {
                predecessor[neighbor] = node;
                if (heap_position[neighbor] == -1) {
                    distance[neighbor] = candidate_distance;
                    int heap_slot = min_heap.size++;
                    heap[heap_slot] = neighbor;
                    heap_position[neighbor] = heap_slot;
                    min_heap_sift_up(&min_heap, heap_slot);
                } else {
                    distance[neighbor] = candidate_distance;
                    min_heap_sift_up(&min_heap, heap_position[neighbor]);
                }
            }
        }
    }

    /* Build result */
    double final_distance = distance[target];
    PyObject *path_list;

    if (final_distance < HUGE_VAL) {
        int path_length = 0;
        for (int path_node = target; path_node != -1; path_node = predecessor[path_node]) path_length++;
        path_list = PyList_New(path_length);
        if (!path_list) {
            free(distance); free(predecessor); free(heap); free(heap_position);
            free_adjacency(&adjacency); free_weights(&weights);
            return NULL;
        }
        int path_node = target;
        for (int position = path_length - 1; position >= 0; position--) {
            PyList_SET_ITEM(path_list, position, PyLong_FromLong(path_node));
            path_node = predecessor[path_node];
        }
    } else {
        path_list = PyList_New(0);
        if (!path_list) {
            free(distance); free(predecessor); free(heap); free(heap_position);
            free_adjacency(&adjacency); free_weights(&weights);
            return NULL;
        }
    }

    PyObject *result = Py_BuildValue("(dN)", final_distance, path_list);
    free(distance); free(predecessor); free(heap); free(heap_position);
    free_adjacency(&adjacency); free_weights(&weights);
    return result;
}

/* ── SSSP lengths (single source, all reachable, optional cutoff) ── */

PyObject *py_shortest_path_lengths(PyObject *self, PyObject *args) {
    int node_count, source;
    double cutoff;
    PyObject *edges_obj, *weights_obj;
    if (!PyArg_ParseTuple(args, "iOOid", &node_count, &edges_obj, &weights_obj,
                          &source, &cutoff))
        return NULL;

    int use_cutoff = (cutoff >= 0.0);

    PyObject *result_dict = PyDict_New();
    if (!result_dict) return NULL;
    if (node_count == 0 || source < 0 || source >= node_count) return result_dict;

    EdgeList edges;
    if (parse_edges(edges_obj, &edges) < 0) { Py_DECREF(result_dict); return NULL; }
    WeightList weights;
    if (parse_weights(weights_obj, &weights, edges.edge_count) < 0) {
        free_edges(&edges); Py_DECREF(result_dict); return NULL;
    }

    AdjacencyList adjacency;
    if (build_adjacency(node_count, &edges, NULL, &adjacency, 0) < 0) {
        free_edges(&edges); free_weights(&weights);
        Py_DECREF(result_dict); return NULL;
    }
    free_edges(&edges);

    double *distance   = (double *)malloc((size_t)node_count * sizeof(double));
    int *heap          = (int *)malloc((size_t)node_count * sizeof(int));
    int *heap_position = (int *)malloc((size_t)node_count * sizeof(int));

    if (!distance || !heap || !heap_position) {
        free(distance); free(heap); free(heap_position);
        free_adjacency(&adjacency); free_weights(&weights); Py_DECREF(result_dict);
        PyErr_NoMemory();
        return NULL;
    }

    for (int node = 0; node < node_count; node++) {
        distance[node] = HUGE_VAL;
        heap_position[node] = -1;
    }

    MinHeap min_heap = {distance, heap, heap_position, 0};
    distance[source] = 0.0;
    heap[0] = source;
    heap_position[source] = 0;
    min_heap.size = 1;

    while (min_heap.size > 0) {
        int node = min_heap_pop(&min_heap);
        if (distance[node] == HUGE_VAL) break;
        if (use_cutoff && distance[node] > cutoff) break;

        for (int slot = adjacency.offset[node]; slot < adjacency.offset[node + 1]; slot++) {
            int neighbor = adjacency.neighbors[slot];
            double weight = weights.values[adjacency.edge_indices[slot]];
            double candidate_distance = distance[node] + weight;
            if (candidate_distance < distance[neighbor] && (!use_cutoff || candidate_distance <= cutoff)) {
                if (heap_position[neighbor] == -1) {
                    distance[neighbor] = candidate_distance;
                    int heap_slot = min_heap.size++;
                    heap[heap_slot] = neighbor;
                    heap_position[neighbor] = heap_slot;
                    min_heap_sift_up(&min_heap, heap_slot);
                } else {
                    distance[neighbor] = candidate_distance;
                    min_heap_sift_up(&min_heap, heap_position[neighbor]);
                }
            }
        }
    }

    for (int node = 0; node < node_count; node++) {
        if (distance[node] < HUGE_VAL && (!use_cutoff || distance[node] <= cutoff)) {
            PyObject *key = PyLong_FromLong(node);
            PyObject *value = PyFloat_FromDouble(distance[node]);
            PyDict_SetItem(result_dict, key, value);
            Py_DECREF(key); Py_DECREF(value);
        }
    }

    free(distance); free(heap); free(heap_position);
    free_adjacency(&adjacency); free_weights(&weights);
    return result_dict;
}

/* ── Multi-source Dijkstra (multiple start nodes, optional cutoff) ── */

PyObject *py_multi_source_dijkstra(PyObject *self, PyObject *args) {
    int node_count;
    double cutoff;
    PyObject *edges_obj, *weights_obj, *sources_obj;
    if (!PyArg_ParseTuple(args, "iOOOd", &node_count, &edges_obj, &weights_obj,
                          &sources_obj, &cutoff))
        return NULL;

    int use_cutoff = (cutoff >= 0.0);

    PyObject *result_dict = PyDict_New();
    if (!result_dict) return NULL;
    if (node_count == 0) return result_dict;

    PyObject *sources_fast = PySequence_Fast(sources_obj, "sources must be a sequence");
    if (!sources_fast) { Py_DECREF(result_dict); return NULL; }
    Py_ssize_t source_count = PySequence_Fast_GET_SIZE(sources_fast);

    EdgeList edges;
    if (parse_edges(edges_obj, &edges) < 0) {
        Py_DECREF(sources_fast); Py_DECREF(result_dict); return NULL;
    }
    WeightList weights;
    if (parse_weights(weights_obj, &weights, edges.edge_count) < 0) {
        free_edges(&edges); Py_DECREF(sources_fast); Py_DECREF(result_dict);
        return NULL;
    }

    AdjacencyList adjacency;
    if (build_adjacency(node_count, &edges, NULL, &adjacency, 0) < 0) {
        free_edges(&edges); free_weights(&weights);
        Py_DECREF(sources_fast); Py_DECREF(result_dict);
        return NULL;
    }
    free_edges(&edges);

    double *distance   = (double *)malloc((size_t)node_count * sizeof(double));
    int *heap          = (int *)malloc((size_t)node_count * sizeof(int));
    int *heap_position = (int *)malloc((size_t)node_count * sizeof(int));

    if (!distance || !heap || !heap_position) {
        free(distance); free(heap); free(heap_position);
        free_adjacency(&adjacency); free_weights(&weights);
        Py_DECREF(sources_fast); Py_DECREF(result_dict);
        PyErr_NoMemory();
        return NULL;
    }

    for (int node = 0; node < node_count; node++) {
        distance[node] = HUGE_VAL;
        heap_position[node] = -1;
    }

    MinHeap min_heap = {distance, heap, heap_position, 0};

    /* Initialize all source nodes at distance 0 */
    PyObject **source_items = PySequence_Fast_ITEMS(sources_fast);
    for (Py_ssize_t source_position = 0; source_position < source_count; source_position++) {
        int source = (int)PyLong_AsLong(source_items[source_position]);
        if (PyErr_Occurred()) {
            free(distance); free(heap); free(heap_position);
            free_adjacency(&adjacency); free_weights(&weights);
            Py_DECREF(sources_fast); Py_DECREF(result_dict);
            return NULL;
        }
        if (source >= 0 && source < node_count && distance[source] > 0.0) {
            distance[source] = 0.0;
            int heap_slot = min_heap.size++;
            heap[heap_slot] = source;
            heap_position[source] = heap_slot;
            min_heap_sift_up(&min_heap, heap_slot);
        }
    }
    Py_DECREF(sources_fast);

    while (min_heap.size > 0) {
        int node = min_heap_pop(&min_heap);
        if (distance[node] == HUGE_VAL) break;
        if (use_cutoff && distance[node] > cutoff) break;

        for (int slot = adjacency.offset[node]; slot < adjacency.offset[node + 1]; slot++) {
            int neighbor = adjacency.neighbors[slot];
            double weight = weights.values[adjacency.edge_indices[slot]];
            double candidate_distance = distance[node] + weight;
            if (candidate_distance < distance[neighbor] && (!use_cutoff || candidate_distance <= cutoff)) {
                if (heap_position[neighbor] == -1) {
                    distance[neighbor] = candidate_distance;
                    int heap_slot = min_heap.size++;
                    heap[heap_slot] = neighbor;
                    heap_position[neighbor] = heap_slot;
                    min_heap_sift_up(&min_heap, heap_slot);
                } else {
                    distance[neighbor] = candidate_distance;
                    min_heap_sift_up(&min_heap, heap_position[neighbor]);
                }
            }
        }
    }

    for (int node = 0; node < node_count; node++) {
        if (distance[node] < HUGE_VAL && (!use_cutoff || distance[node] <= cutoff)) {
            PyObject *key = PyLong_FromLong(node);
            PyObject *value = PyFloat_FromDouble(distance[node]);
            PyDict_SetItem(result_dict, key, value);
            Py_DECREF(key); Py_DECREF(value);
        }
    }

    free(distance); free(heap); free(heap_position);
    free_adjacency(&adjacency); free_weights(&weights);
    return result_dict;
}

/* ── dijkstra (node-ID edges) ── */

PyObject *py_dijkstra_by_node_id(PyObject *self, PyObject *args) {
    PyObject *node_ids_obj, *edges_obj, *weights_obj, *source_obj, *target_obj;
    if (!PyArg_ParseTuple(args, "OOOOO", &node_ids_obj, &edges_obj, &weights_obj, &source_obj, &target_obj))
        return NULL;

    NodeIdContext context;
    if (node_id_parse(node_ids_obj, edges_obj, &context) < 0) return NULL;
    int node_count = context.node_count;

    long source_node_id = PyLong_AsLong(source_obj), target_node_id = PyLong_AsLong(target_obj);
    if (PyErr_Occurred()) { node_id_context_free(&context); return NULL; }
    int source = node_count > 0 ? intmap_get(&context.node_index_by_node_id, source_node_id) : -1;
    int target = node_count > 0 ? intmap_get(&context.node_index_by_node_id, target_node_id) : -1;

    if (source == target && source >= 0) {
        PyObject *path = PyList_New(1);
        Py_INCREF(context.node_id_items[source]);
        PyList_SET_ITEM(path, 0, context.node_id_items[source]);
        node_id_context_free(&context);
        return Py_BuildValue("(dN)", 0.0, path);
    }
    if (node_count == 0 || source < 0 || target < 0) {
        node_id_context_free(&context);
        PyObject *empty_path = PyList_New(0);
        return Py_BuildValue("(dN)", HUGE_VAL, empty_path);
    }

    WeightList weights;
    if (parse_weights(weights_obj, &weights, context.edges.edge_count) < 0) { node_id_context_free(&context); return NULL; }
    AdjacencyList adjacency;
    if (build_adjacency(node_count, &context.edges, NULL, &adjacency, 0) < 0) { node_id_context_free(&context); free_weights(&weights); return NULL; }

    BidirectionalPath bidirectional_path;
    int failed = bidirectional_dijkstra(node_count, 0, &adjacency, weights.values, NULL, NULL, NULL, NULL, NULL, NULL, 0,
                                        source, target, &bidirectional_path) < 0;
    PyObject *path = failed ? NULL : bidirectional_path_to_list(&bidirectional_path, &context);
    free(bidirectional_path.path);
    free_adjacency(&adjacency); free_weights(&weights);
    if (!path) { node_id_context_free(&context); return NULL; }
    PyObject *result = Py_BuildValue("(dN)", bidirectional_path.weight, path);
    node_id_context_free(&context);
    return result;
}

/* ── shortest_path_lengths (node-ID edges) ── */

PyObject *py_shortest_path_lengths_by_node_id(PyObject *self, PyObject *args) {
    PyObject *node_ids_obj, *edges_obj, *weights_obj, *source_obj;
    double cutoff;
    if (!PyArg_ParseTuple(args, "OOOOd", &node_ids_obj, &edges_obj, &weights_obj, &source_obj, &cutoff))
        return NULL;

    int use_cutoff = (cutoff >= 0.0);
    NodeIdContext context;
    if (node_id_parse(node_ids_obj, edges_obj, &context) < 0) return NULL;
    int node_count = context.node_count;

    PyObject *result = PyDict_New();
    if (!result) { node_id_context_free(&context); return NULL; }

    long source_node_id = PyLong_AsLong(source_obj);
    if (PyErr_Occurred()) { node_id_context_free(&context); Py_DECREF(result); return NULL; }
    int source = node_count > 0 ? intmap_get(&context.node_index_by_node_id, source_node_id) : -1;
    if (node_count == 0 || source < 0) { node_id_context_free(&context); return result; }

    WeightList weights;
    if (parse_weights(weights_obj, &weights, context.edges.edge_count) < 0) { node_id_context_free(&context); Py_DECREF(result); return NULL; }
    AdjacencyList adjacency;
    if (build_adjacency(node_count, &context.edges, NULL, &adjacency, 0) < 0) { node_id_context_free(&context); free_weights(&weights); Py_DECREF(result); return NULL; }

    double *distance = malloc(node_count*sizeof(double));
    int *heap = malloc(node_count*sizeof(int)), *heap_position = malloc(node_count*sizeof(int));
    if (!distance||!heap||!heap_position) {
        free(distance);free(heap);free(heap_position);
        free_adjacency(&adjacency); free_weights(&weights); node_id_context_free(&context); Py_DECREF(result);
        PyErr_NoMemory(); return NULL;
    }
    for (int node=0;node<node_count;node++) { distance[node]=HUGE_VAL; heap_position[node]=-1; }
    MinHeap min_heap = {distance, heap, heap_position, 0};
    distance[source]=0.0; heap[0]=source; heap_position[source]=0; min_heap.size=1;

    while (min_heap.size > 0) {
        int node = min_heap_pop(&min_heap);
        if (distance[node]==HUGE_VAL) break;
        if (use_cutoff && distance[node]>cutoff) break;
        for (int slot=adjacency.offset[node]; slot<adjacency.offset[node+1]; slot++) {
            int neighbor=adjacency.neighbors[slot]; double candidate_distance=distance[node]+weights.values[adjacency.edge_indices[slot]];
            if (candidate_distance < distance[neighbor] && (!use_cutoff || candidate_distance <= cutoff)) {
                if (heap_position[neighbor]==-1) { distance[neighbor]=candidate_distance; int heap_slot=min_heap.size++; heap[heap_slot]=neighbor; heap_position[neighbor]=heap_slot; min_heap_sift_up(&min_heap,heap_slot); }
                else { distance[neighbor]=candidate_distance; min_heap_sift_up(&min_heap, heap_position[neighbor]); }
            }
        }
    }

    for (int node=0;node<node_count;node++) {
        if (distance[node] < HUGE_VAL && (!use_cutoff || distance[node] <= cutoff)) {
            PyObject *value = PyFloat_FromDouble(distance[node]);
            PyDict_SetItem(result, context.node_id_items[node], value);
            Py_DECREF(value);
        }
    }

    free(distance);free(heap);free(heap_position);
    free_adjacency(&adjacency); free_weights(&weights); node_id_context_free(&context);
    return result;
}

/* ── multi_source_dijkstra (node-ID edges) ── */

PyObject *py_multi_source_dijkstra_by_node_id(PyObject *self, PyObject *args) {
    PyObject *node_ids_obj, *edges_obj, *weights_obj, *sources_obj;
    double cutoff;
    if (!PyArg_ParseTuple(args, "OOOOd", &node_ids_obj, &edges_obj, &weights_obj, &sources_obj, &cutoff))
        return NULL;

    int use_cutoff = (cutoff >= 0.0);
    NodeIdContext context;
    if (node_id_parse(node_ids_obj, edges_obj, &context) < 0) return NULL;
    int node_count = context.node_count;

    PyObject *result = PyDict_New();
    if (!result) { node_id_context_free(&context); return NULL; }
    if (node_count == 0) { node_id_context_free(&context); return result; }

    PyObject *sources_fast = PySequence_Fast(sources_obj, "sources must be a sequence");
    if (!sources_fast) { node_id_context_free(&context); Py_DECREF(result); return NULL; }

    WeightList weights;
    if (parse_weights(weights_obj, &weights, context.edges.edge_count) < 0) { Py_DECREF(sources_fast); node_id_context_free(&context); Py_DECREF(result); return NULL; }
    AdjacencyList adjacency;
    if (build_adjacency(node_count, &context.edges, NULL, &adjacency, 0) < 0) { Py_DECREF(sources_fast); free_weights(&weights); node_id_context_free(&context); Py_DECREF(result); return NULL; }

    double *distance = malloc(node_count*sizeof(double));
    int *heap = malloc(node_count*sizeof(int)), *heap_position = malloc(node_count*sizeof(int));
    if (!distance||!heap||!heap_position) {
        free(distance);free(heap);free(heap_position);
        free_adjacency(&adjacency); free_weights(&weights); Py_DECREF(sources_fast); node_id_context_free(&context); Py_DECREF(result);
        PyErr_NoMemory(); return NULL;
    }
    for (int node=0;node<node_count;node++) { distance[node]=HUGE_VAL; heap_position[node]=-1; }
    MinHeap min_heap = {distance, heap, heap_position, 0};

    Py_ssize_t source_count = PySequence_Fast_GET_SIZE(sources_fast);
    PyObject **source_items = PySequence_Fast_ITEMS(sources_fast);
    for (Py_ssize_t source_position=0;source_position<source_count;source_position++) {
        long source_node_id = PyLong_AsLong(source_items[source_position]);
        if (PyErr_Occurred()) {
            free(distance);free(heap);free(heap_position);
            free_adjacency(&adjacency); free_weights(&weights); Py_DECREF(sources_fast); node_id_context_free(&context); Py_DECREF(result);
            return NULL;
        }
        int source = intmap_get(&context.node_index_by_node_id, source_node_id);
        if (source >= 0 && distance[source] > 0.0) {
            distance[source]=0.0; int heap_slot=min_heap.size++; heap[heap_slot]=source; heap_position[source]=heap_slot; min_heap_sift_up(&min_heap,heap_slot);
        }
    }
    Py_DECREF(sources_fast);

    while (min_heap.size > 0) {
        int node = min_heap_pop(&min_heap);
        if (distance[node]==HUGE_VAL) break;
        if (use_cutoff && distance[node]>cutoff) break;
        for (int slot=adjacency.offset[node]; slot<adjacency.offset[node+1]; slot++) {
            int neighbor=adjacency.neighbors[slot]; double candidate_distance=distance[node]+weights.values[adjacency.edge_indices[slot]];
            if (candidate_distance < distance[neighbor] && (!use_cutoff || candidate_distance <= cutoff)) {
                if (heap_position[neighbor]==-1) { distance[neighbor]=candidate_distance; int heap_slot=min_heap.size++; heap[heap_slot]=neighbor; heap_position[neighbor]=heap_slot; min_heap_sift_up(&min_heap,heap_slot); }
                else { distance[neighbor]=candidate_distance; min_heap_sift_up(&min_heap, heap_position[neighbor]); }
            }
        }
    }

    for (int node=0;node<node_count;node++) {
        if (distance[node] < HUGE_VAL && (!use_cutoff || distance[node] <= cutoff)) {
            PyObject *value = PyFloat_FromDouble(distance[node]);
            PyDict_SetItem(result, context.node_id_items[node], value);
            Py_DECREF(value);
        }
    }

    free(distance);free(heap);free(heap_position);
    free_adjacency(&adjacency); free_weights(&weights); node_id_context_free(&context);
    return result;
}

static int labeled_shortest_path(const GraphContext *graph, const double *weights, const PassWeights *pass_weights,
                                 const uint8_t *two_terminal_edge_mask, const uint8_t *edge_mask, const uint8_t *terminal_mask,
                                 const uint8_t *node_mask, int source, int target, BidirectionalPath *out);

PyObject *py_dijkstra_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *weights_obj, *source_obj, *target_obj, *edge_mask_obj = Py_None, *node_mask_obj = Py_None, *terminal_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OOOO|OOO", &capsule, &weights_obj, &source_obj, &target_obj, &edge_mask_obj, &node_mask_obj, &terminal_mask_obj))
        return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    int node_count = graph->node_ids.node_count;
    int edge_count = (int)graph->node_ids.edges.edge_count;
    const TerminalList *terminals = graph->terminals;

    long source_node_id = PyLong_AsLong(source_obj), target_node_id = PyLong_AsLong(target_obj);
    if (PyErr_Occurred()) return NULL;
    int source = node_count > 0 ? intmap_get(&graph->node_ids.node_index_by_node_id, source_node_id) : -1;
    int target = node_count > 0 ? intmap_get(&graph->node_ids.node_index_by_node_id, target_node_id) : -1;

    if (source == target && source >= 0) {
        PyObject *path = PyList_New(1);
        Py_INCREF(graph->node_ids.node_id_items[source]);
        PyList_SET_ITEM(path, 0, graph->node_ids.node_id_items[source]);
        return Py_BuildValue("(dN)", 0.0, path);
    }
    if (node_count == 0 || source < 0 || target < 0) {
        PyObject *empty_path = PyList_New(0);
        return Py_BuildValue("(dN)", HUGE_VAL, empty_path);
    }

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, edge_count, &edge_mask, &edge_mask_buffer) < 0) return NULL;
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); return NULL; }
    TerminalMask terminal_mask;
    if (parse_terminal_mask(graph, terminal_mask_obj, edge_mask, &terminal_mask) < 0) { release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); return NULL; }

    if (node_mask && (node_mask[source] || node_mask[target])) {
        release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
        PyObject *empty_path = PyList_New(0);
        return Py_BuildValue("(dN)", HUGE_VAL, empty_path);
    }

    WeightList weights;
    if (parse_counted_weights(weights_obj, &weights, terminals ? terminals->terminal_count : edge_count, terminals ? "terminal" : "edge", terminals != NULL) < 0) { release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); return NULL; }
    if (!graph->has_adjacency) { free_weights(&weights); release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); PyObject *empty_path = PyList_New(0); return Py_BuildValue("(dN)", HUGE_VAL, empty_path); }

    PassWeights pass_weights = {NULL, NULL};
    if (terminals && weights.has_sign_bit && parse_pass_weights(terminals, edge_count, weights.values, graph->directed, &pass_weights) < 0) {
        free_weights(&weights); release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
        return NULL;
    }
    BidirectionalPath bidirectional_path;
    int failed = (pass_weights.shifts && !graph->directed)
        ? labeled_shortest_path(graph, weights.values, &pass_weights, two_terminal_edge_mask(&terminal_mask, edge_mask), edge_mask,
                                terminal_mask.terminals, node_mask, source, target, &bidirectional_path) < 0
        : bidirectional_dijkstra(node_count, edge_count, &graph->adjacency, weights.values, terminals, &pass_weights,
                                 two_terminal_edge_mask(&terminal_mask, edge_mask), edge_mask, terminal_mask.terminals,
                                 node_mask, graph->directed, source, target, &bidirectional_path) < 0;
    PyObject *path = failed ? NULL : bidirectional_path_to_list(&bidirectional_path, &graph->node_ids);
    free(bidirectional_path.path);
    release_pass_weights(&pass_weights);
    free_weights(&weights); release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    if (!path) return NULL;
    return Py_BuildValue("(dN)", bidirectional_path.weight, path);
}

/* Lower a state's distance to candidate_distance when that is shorter and within the cutoff. */
static inline __attribute__((always_inline)) void relax_state(MinHeap *min_heap, int state, double candidate_distance,
                                                              int use_cutoff, double cutoff) {
    double *distance = min_heap->distance;
    if (candidate_distance >= distance[state] || (use_cutoff && candidate_distance > cutoff)) return;
    distance[state] = candidate_distance;
    if (min_heap->heap_position[state] == -1) {
        int heap_slot = min_heap->size++;
        min_heap->nodes[heap_slot] = state;
        min_heap->heap_position[state] = heap_slot;
        min_heap_sift_up(min_heap, heap_slot);
    } else {
        min_heap_sift_up(min_heap, min_heap->heap_position[state]);
    }
}

/*
 * Relax a node from state `from`: plainly, or keeping its arrival labels when the
 * search has them; on a side of a point-to-point search, offer it as a meeting.
 */
static inline __attribute__((always_inline)) void relax_node(MinHeap *min_heap, ArrivalLabels *labels, int node,
                                                             double candidate_distance, int arrival, int from,
                                                             int use_cutoff, double cutoff, const int offers_meetings) {
    if (!labels) { relax_state(min_heap, node, candidate_distance, use_cutoff, cutoff); return; }
    if (use_cutoff && candidate_distance > cutoff) return;
    if (offers_meetings)
        record_meeting(labels->meeting, labels->meeting->far_labels->arrival_edges[node], node,
                       labels->second_node_states + node, arrival, candidate_distance, from);
    relax_labeled(min_heap, &labels->arrival_edges[node], labels->predecessors, node, labels->second_node_states + node,
                  candidate_distance, arrival, from);
}

/*
 * The meetings of a pass entered through `entry` at candidate_distance: with the
 * far side's edge state, both sides having added the shift on entry, and, since
 * leaving through the negative terminal costs nothing after the shift, with the
 * far side at that terminal's node. The two sides represent a pass touching the
 * negative terminal differently (a direct pass on one, an edge state on the other),
 * so the second meeting is what lets them meet on such a pass.
 */
static inline __attribute__((always_inline)) void record_edge_state_meetings(
    const GraphContext *graph, const double *weights, const PassWeights *pass_weights, const uint8_t *edge_mask,
    const uint8_t *terminal_mask, const uint8_t *node_mask, ArrivalLabels *labels, int edge, int entry,
    double candidate_distance, int from) {
    LabelledMeeting *meeting = labels->meeting;
    double shift = pass_weights->shifts[edge];
    record_meeting(meeting, meeting->far_labels->entry_terminals[edge], graph->node_ids.node_count + edge,
                   labels->second_edge_states + edge, entry, candidate_distance - 2.0 * shift, from);
    int negative_terminal = pass_weights->negative_terminals[edge];
    if (negative_terminal < 0 ||
        !terminal_is_active(graph->terminals, edge, negative_terminal, edge_mask, terminal_mask, node_mask)) return;
    int node = graph->terminals->terminal_nodes[negative_terminal];
    record_meeting(meeting, meeting->far_labels->arrival_edges[node], node, labels->second_node_states + node, edge,
                   candidate_distance + weights[negative_terminal] - shift, from);
}

/*
 * The passes of an undirected edge out of its negative terminal n, from search
 * state `from` (node n, or its second label): each other active terminal at
 * w_n + w_q, untagged.
 */
static inline __attribute__((always_inline)) void relax_negative_terminal_pass(const GraphContext *graph, const double *weights, const PassWeights *pass_weights,
                                     const uint8_t *edge_mask, const uint8_t *terminal_mask, const uint8_t *node_mask,
                                     int use_cutoff, double cutoff, MinHeap *min_heap, ArrivalLabels *labels, int edge,
                                     int from, const int offers_meetings) {
    const TerminalList *terminals = graph->terminals;
    int negative_terminal = pass_weights->negative_terminals[edge];
    if (!terminal_is_active(terminals, edge, negative_terminal, edge_mask, terminal_mask, node_mask)) return;
    double base = min_heap->distance[from] + weights[negative_terminal];
    for (int exit = terminals->terminal_offsets[edge]; exit < terminals->terminal_offsets[edge + 1]; exit++) {
        if (exit == negative_terminal || !terminal_is_active(terminals, edge, exit, edge_mask, terminal_mask, node_mask)) continue;
        relax_node(min_heap, labels, terminals->terminal_nodes[exit], base + weights[exit], -1, from, use_cutoff, cutoff,
                   offers_meetings);
    }
}

/*
 * Leave an edge state through its active exit terminals (directed: out-terminals);
 * with labels through every terminal but `excluded_exit`, or only through
 * `only_exit` when that is not -1. Leaving through the negative terminal tags the
 * arrival with the edge.
 */
static inline __attribute__((always_inline)) void leave_edge_state(
    const GraphContext *graph, const double *weights, const PassWeights *pass_weights, const uint8_t *edge_mask,
    const uint8_t *terminal_mask, const uint8_t *node_mask, int use_cutoff, double cutoff, MinHeap *min_heap,
    ArrivalLabels *labels, const double *shifts, int edge, int from, int excluded_exit, int only_exit,
    const int offers_meetings) {
    const TerminalList *terminals = graph->terminals;
    int directed = graph->directed;
    double exit_shift = shifts ? (directed ? shifts[edge] : -shifts[edge]) : 0.0;
    int tagged_exit = labels ? pass_weights->negative_terminals[edge] : -1;
    int first_exit = only_exit >= 0 ? only_exit : directed ? terminals->out_terminal_offsets[edge] : terminals->terminal_offsets[edge];
    int end_exit = only_exit >= 0 ? only_exit + 1 : terminals->terminal_offsets[edge + 1];
    for (int exit = first_exit; exit < end_exit; exit++) {
        if (exit == excluded_exit || !terminal_is_active(terminals, edge, exit, edge_mask, terminal_mask, node_mask)) continue;
        relax_node(min_heap, labels, terminals->terminal_nodes[exit], min_heap->distance[from] + weights[exit] + exit_shift,
                   exit == tagged_exit ? edge : -1, from, use_cutoff, cutoff, offers_meetings);
    }
}

/*
 * Dijkstra from the sources already in the heap until it is empty or, with
 * is_point_to_point_side, for one state, offering every relaxation to the other
 * side as a meeting. Search states: the nodes, with
 * has_multi_terminal_edges one state per edge with three or more terminals
 * (node_count + e), and with labels the second labels of nodes and edges.
 * A two-terminal edge relaxes its neighbor directly, at its edge weight or, with
 * terminal weights, at the sum of its two terminal weights. An edge with more terminals
 * is entered through an active terminal and left through each active exit terminal
 * (directed: out-terminal), so an edge with k terminals costs O(k); with a negative
 * terminal weight the steps are shifted as PassWeights describes, and an
 * undirected search keeps ArrivalLabels. The function is instantiated per
 * value of has_multi_terminal_edges, has_negative_terminal_weights and
 * is_point_to_point_side, so graphs of two-terminal edges, weights without a
 * negative terminal and single-source searches run the loop as before.
 */
static inline __attribute__((always_inline)) void relax_until_settled(
    const GraphContext *graph, const double *weights, const uint8_t *two_terminal_edge_mask, const uint8_t *edge_mask,
    const uint8_t *terminal_mask, const uint8_t *node_mask, int use_cutoff, double cutoff, MinHeap *min_heap,
    uint8_t *settled_edges, const PassWeights *pass_weights, ArrivalLabels *labels, const int is_point_to_point_side,
    const int has_multi_terminal_edges, const int has_negative_terminal_weights) {
    const double *shifts = has_negative_terminal_weights ? pass_weights->shifts : NULL;
    if (!has_negative_terminal_weights) labels = NULL;
    const int directed = graph->directed;
    const AdjacencyList *adjacency = &graph->adjacency;
    const TerminalList *terminals = graph->terminals;
    int node_count = graph->node_ids.node_count;
    double *distance = min_heap->distance;
    int *heap = min_heap->nodes, *heap_position = min_heap->heap_position;
    for (int settled_one = 0; min_heap->size > 0 && !(is_point_to_point_side && settled_one); settled_one = 1) {
        int state = min_heap_pop(min_heap);
        if (distance[state]==HUGE_VAL) break;
        if (use_cutoff && distance[state]>cutoff) break;
        if (has_multi_terminal_edges && labels && state >= labels->second_edge_states) {
            int edge = state - labels->second_edge_states;
            leave_edge_state(graph, weights, pass_weights, edge_mask, terminal_mask, node_mask, use_cutoff, cutoff, min_heap,
                             labels, shifts, edge, state, -1, labels->entry_terminals[edge], is_point_to_point_side);
            continue;
        }
        if (has_multi_terminal_edges && labels && state >= labels->second_node_states) {
            int node = state - labels->second_node_states, edge = labels->arrival_edges[node];
            if (edge >= 0) relax_negative_terminal_pass(graph, weights, pass_weights, edge_mask, terminal_mask, node_mask,
                                                    use_cutoff, cutoff, min_heap, labels, edge, state,
                                                    is_point_to_point_side);
            continue;
        }
        if (has_multi_terminal_edges && state >= node_count) {
            int edge = state - node_count;
            if (!labels) settled_edges[edge] = 1;
            leave_edge_state(graph, weights, pass_weights, edge_mask, terminal_mask, node_mask, use_cutoff, cutoff, min_heap,
                             labels, shifts, edge, state, labels ? labels->entry_terminals[edge] : -1, -1,
                             is_point_to_point_side);
            continue;
        }
        int node = state;
        for (int slot=adjacency->offset[node]; slot<adjacency->offset[node+1]; slot++) {
            int neighbor=adjacency->neighbors[slot];
            if (has_multi_terminal_edges && neighbor < 0) {
                int edge = -1 - neighbor, entry = adjacency->edge_indices[slot];
                if (!terminal_is_active(terminals, edge, entry, edge_mask, terminal_mask, node_mask)) continue;
                double entry_shift = shifts ? (directed ? -shifts[edge] : shifts[edge]) : 0.0;
                if (!labels) {
                    if (!settled_edges[edge])
                        relax_state(min_heap, node_count + edge, distance[node] + weights[entry] + entry_shift, use_cutoff, cutoff);
                    continue;
                }
                if (pass_weights->negative_terminals[edge] == entry) {
                    if (labels->arrival_edges[node] != edge)
                        relax_negative_terminal_pass(graph, weights, pass_weights, edge_mask, terminal_mask, node_mask, use_cutoff,
                                                 cutoff, min_heap, labels, edge, node, is_point_to_point_side);
                    continue;
                }
                double candidate_distance = distance[node] + weights[entry] + entry_shift;
                if (use_cutoff && candidate_distance > cutoff) continue;
                if (is_point_to_point_side)
                    record_edge_state_meetings(graph, weights, pass_weights, edge_mask, terminal_mask, node_mask, labels,
                                               edge, entry, candidate_distance, node);
                relax_labeled(min_heap, &labels->entry_terminals[edge], labels->predecessors, node_count + edge,
                              labels->second_edge_states + edge, candidate_distance, entry, node);
                continue;
            }
            int edge_index = adjacency->edge_indices[slot];
            if (two_terminal_edge_mask && two_terminal_edge_mask[edge_index]) continue;
            if (node_mask && node_mask[neighbor]) continue;
            double edge_weight = has_multi_terminal_edges
                ? weights[terminals->terminal_offsets[edge_index]] + weights[terminals->terminal_offsets[edge_index] + 1]
                : weights[edge_index];
            double candidate_distance=distance[node]+edge_weight;
            if (has_multi_terminal_edges && labels) {
                int negative_terminal = pass_weights->negative_terminals[edge_index];
                if (negative_terminal < 0) {
                    relax_node(min_heap, labels, neighbor, candidate_distance, -1, node, use_cutoff, cutoff,
                               is_point_to_point_side);
                    continue;
                }
                for (int entry = terminals->terminal_offsets[edge_index]; entry < terminals->terminal_offsets[edge_index] + 2; entry++) {
                    if (terminals->terminal_nodes[entry] != node) continue;
                    if (entry == negative_terminal && labels->arrival_edges[node] == edge_index) continue;
                    int exit = 2 * terminals->terminal_offsets[edge_index] + 1 - entry;
                    relax_node(min_heap, labels, terminals->terminal_nodes[exit], candidate_distance,
                               exit == negative_terminal ? edge_index : -1, node, use_cutoff, cutoff,
                               is_point_to_point_side);
                }
                continue;
            }
            if (candidate_distance < distance[neighbor] && (!use_cutoff || candidate_distance <= cutoff)) {
                if (heap_position[neighbor]==-1) { distance[neighbor]=candidate_distance; int heap_slot=min_heap->size++; heap[heap_slot]=neighbor; heap_position[neighbor]=heap_slot; min_heap_sift_up(min_heap,heap_slot); }
                else { distance[neighbor]=candidate_distance; min_heap_sift_up(min_heap, heap_position[neighbor]); }
            }
        }
    }
}

/* One side of a labelled point-to-point search: its labels and heap over every search state. */
typedef struct {
    ArrivalLabels labels;
    MinHeap heap;
    LabelledMeeting meeting;
} LabelledSearchSide;

/* Allocate a side whose search starts at `start`. Returns -1 with MemoryError set. */
static int start_labelled_search_side(LabelledSearchSide *side, int node_count, int edge_count, int start) {
    int state_count = allocate_arrival_labels(&side->labels, node_count, edge_count, node_count + edge_count, 1);
    if (state_count < 0) return -1;
    double *distance = malloc((size_t)state_count * sizeof(double));
    int *heap = malloc((size_t)state_count * sizeof(int)), *heap_position = malloc((size_t)state_count * sizeof(int));
    side->heap = (MinHeap){distance, heap, heap_position, 1};
    if (!distance || !heap || !heap_position) {
        free(distance); free(heap); free(heap_position); free_arrival_labels(&side->labels);
        PyErr_NoMemory();
        return -1;
    }
    for (int state = 0; state < state_count; state++) { distance[state] = HUGE_VAL; heap_position[state] = -1; }
    distance[start] = 0.0; heap[0] = start; heap_position[start] = 0;
    return 0;
}

static void free_labelled_search_side(LabelledSearchSide *side) {
    free(side->heap.distance); free(side->heap.nodes); free(side->heap.heap_position); free_arrival_labels(&side->labels);
}

/*
 * Length of a side's predecessor chain from `state`; with path, also write its
 * nodes there, in chain order or reversed, an edge state as -1.
 */
static int write_labelled_chain(const LabelledSearchSide *side, int node_count, int state, int *path, int reversed) {
    int length = 0;
    for (int chain_state = state; chain_state != -1; chain_state = side->labels.predecessors[chain_state]) length++;
    if (!path) return length;
    int position = reversed ? length : 0;
    for (int chain_state = state; chain_state != -1; chain_state = side->labels.predecessors[chain_state]) {
        int node = chain_state < node_count ? chain_state
                 : (chain_state >= side->labels.second_node_states && chain_state < side->labels.second_edge_states)
                   ? chain_state - side->labels.second_node_states : -1;
        path[reversed ? --position : position++] = node;
    }
    return length;
}

/*
 * Cheapest path from source to target on an undirected graph with a negative
 * terminal weight: bidirectional, both sides settling states with the labelled
 * search of relax_until_settled (the backward side from the target, see
 * LabelledMeeting), each relaxation offering the other side a meeting. The side
 * with the lower top expands, the forward one on a tie; the search stops once the
 * two tops add up to the best meeting, or once a side has run out without reaching
 * the other end (the rules being the same reversed, then no path exists either
 * way). A path may visit a node twice (it may come back to the negative
 * terminal's node by another way and then pass the edge). Returns -1 with an
 * exception set.
 */
static int labeled_shortest_path(const GraphContext *graph, const double *weights, const PassWeights *pass_weights,
                                 const uint8_t *two_terminal_edge_mask, const uint8_t *edge_mask, const uint8_t *terminal_mask,
                                 const uint8_t *node_mask, int source, int target, BidirectionalPath *out) {
    int node_count = graph->node_ids.node_count, edge_count = (int)graph->node_ids.edges.edge_count;
    out->weight = HUGE_VAL; out->path = NULL; out->path_length = 0;
    LabelledSearchSide forward, backward;
    if (start_labelled_search_side(&forward, node_count, edge_count, source) < 0) return -1;
    if (start_labelled_search_side(&backward, node_count, edge_count, target) < 0) { free_labelled_search_side(&forward); return -1; }
    PathMeeting best = {HUGE_VAL, -1, -1};
    forward.meeting = (LabelledMeeting){&backward.heap, &backward.labels, &best, 0};
    backward.meeting = (LabelledMeeting){&forward.heap, &forward.labels, &best, 1};
    forward.labels.meeting = &forward.meeting;
    backward.labels.meeting = &backward.meeting;
    while (forward.heap.size > 0 || backward.heap.size > 0) {
        double forward_top = forward.heap.size > 0 ? forward.heap.distance[forward.heap.nodes[0]] : HUGE_VAL;
        double backward_top = backward.heap.size > 0 ? backward.heap.distance[backward.heap.nodes[0]] : HUGE_VAL;
        if (best.weight < HUGE_VAL && forward_top + backward_top >= best.weight) break;
        if ((forward.heap.size == 0 && forward.heap.distance[target] == HUGE_VAL) ||
            (backward.heap.size == 0 && backward.heap.distance[source] == HUGE_VAL)) break;
        LabelledSearchSide *side = forward_top <= backward_top ? &forward : &backward;
        relax_until_settled(graph, weights, two_terminal_edge_mask, edge_mask, terminal_mask, node_mask, 0, 0.0, &side->heap,
                            NULL, pass_weights, &side->labels, 1, 1, 1);
    }
    int status = 0;
    if (best.weight < HUGE_VAL) {
        int forward_length = write_labelled_chain(&forward, node_count, best.forward_state, NULL, 1);
        int backward_length = write_labelled_chain(&backward, node_count, best.backward_state, NULL, 0);
        out->path = (int *)malloc((size_t)(forward_length + backward_length) * sizeof(int));
        if (!out->path) { status = -1; PyErr_NoMemory(); }
        else {
            out->weight = best.weight;
            write_labelled_chain(&forward, node_count, best.forward_state, out->path, 1);
            write_labelled_chain(&backward, node_count, best.backward_state, out->path + forward_length, 0);
            int kept = 0;
            for (int position = 0; position < forward_length + backward_length; position++)
                if (out->path[position] >= 0) out->path[kept++] = out->path[position];
            out->path_length = kept;
        }
    }
    free_labelled_search_side(&forward); free_labelled_search_side(&backward);
    return status;
}

/*
 * {node id: distance} from the given source indices (masked or repeated
 * sources skipped): one weight per edge on a graph of two-terminal edges, one per
 * terminal on a graph with multi-terminal edges. NULL with an exception set on error.
 */
static PyObject *shortest_path_lengths_from(const GraphContext *graph, PyObject *weights_obj, const int *sources,
                                            Py_ssize_t source_count, double cutoff, PyObject *edge_mask_obj,
                                            PyObject *node_mask_obj, PyObject *terminal_mask_obj) {
    int node_count = graph->node_ids.node_count;
    int use_cutoff = (cutoff >= 0.0);
    PyObject *result = PyDict_New();
    if (!result) return NULL;
    if (node_count == 0 || !graph->has_adjacency) return result;
    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }
    TerminalMask terminal_mask;
    if (parse_terminal_mask(graph, terminal_mask_obj, edge_mask, &terminal_mask) < 0) { release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }
    const TerminalList *terminals = graph->terminals;
    WeightList weights;
    if (parse_counted_weights(weights_obj, &weights, terminals ? terminals->terminal_count : graph->node_ids.edges.edge_count,
                              terminals ? "terminal" : "edge", terminals != NULL) < 0) {
        release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL;
    }
    PassWeights pass_weights = {NULL, NULL};
    if (terminals && weights.has_sign_bit &&
        parse_pass_weights(terminals, graph->node_ids.edges.edge_count, weights.values, graph->directed, &pass_weights) < 0) {
        free_weights(&weights); release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result);
        return NULL;
    }
    int edge_count = (int)graph->node_ids.edges.edge_count;
    int uses_labels = pass_weights.shifts && !graph->directed;
    int state_count = node_count + (terminals ? edge_count : 0);
    ArrivalLabels labels = {NULL, NULL, NULL, 0, 0, NULL};
    if (uses_labels && (state_count = allocate_arrival_labels(&labels, node_count, edge_count, state_count, 0)) < 0) {
        release_pass_weights(&pass_weights);
        free_weights(&weights); release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result);
        return NULL;
    }
    double *distance = malloc((size_t)state_count*sizeof(double));
    int *heap = malloc((size_t)state_count*sizeof(int)), *heap_position = malloc((size_t)state_count*sizeof(int));
    uint8_t *settled_edges = terminals ? calloc((size_t)edge_count + 1, 1) : NULL;
    if (!distance||!heap||!heap_position||(terminals && !settled_edges)) {
        release_pass_weights(&pass_weights); free_arrival_labels(&labels);
        free(distance);free(heap);free(heap_position);free(settled_edges);
        free_weights(&weights); release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result);
        PyErr_NoMemory(); return NULL;
    }
    for (int state=0;state<state_count;state++) { distance[state]=HUGE_VAL; heap_position[state]=-1; }
    MinHeap min_heap = {distance, heap, heap_position, 0};
    for (Py_ssize_t position=0; position<source_count; position++) {
        int source = sources[position];
        if (source >= 0 && !(node_mask && node_mask[source]) && distance[source] > 0.0) {
            distance[source]=0.0; int heap_slot=min_heap.size++; heap[heap_slot]=source; heap_position[source]=heap_slot; min_heap_sift_up(&min_heap,heap_slot);
        }
    }
    const uint8_t *traversal_edge_mask = two_terminal_edge_mask(&terminal_mask, edge_mask);
    if (terminals && pass_weights.shifts)
        relax_until_settled(graph, weights.values, traversal_edge_mask, edge_mask, terminal_mask.terminals, node_mask,
                            use_cutoff, cutoff, &min_heap, settled_edges, &pass_weights, uses_labels ? &labels : NULL, 0, 1, 1);
    else if (terminals)
        relax_until_settled(graph, weights.values, traversal_edge_mask, edge_mask, terminal_mask.terminals, node_mask,
                            use_cutoff, cutoff, &min_heap, settled_edges, &pass_weights, NULL, 0, 1, 0);
    else
        relax_until_settled(graph, weights.values, traversal_edge_mask, NULL, NULL, node_mask, use_cutoff, cutoff,
                            &min_heap, NULL, &pass_weights, NULL, 0, 0, 0);
    release_pass_weights(&pass_weights); free_arrival_labels(&labels);
    for (int node=0;node<node_count;node++) {
        if (node_mask && node_mask[node]) continue;
        if (distance[node] < HUGE_VAL && (!use_cutoff || distance[node] <= cutoff)) {
            PyObject *value = PyFloat_FromDouble(distance[node]);
            PyDict_SetItem(result, graph->node_ids.node_id_items[node], value);
            Py_DECREF(value);
        }
    }
    free(distance);free(heap);free(heap_position);free(settled_edges);
    free_weights(&weights); release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    return result;
}

PyObject *py_shortest_path_lengths_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *weights_obj, *source_obj, *edge_mask_obj = Py_None, *node_mask_obj = Py_None, *terminal_mask_obj = Py_None;
    double cutoff;
    if (!PyArg_ParseTuple(args, "OOOd|OOO", &capsule, &weights_obj, &source_obj, &cutoff, &edge_mask_obj, &node_mask_obj, &terminal_mask_obj))
        return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    long source_node_id = PyLong_AsLong(source_obj);
    if (PyErr_Occurred()) return NULL;
    int source = graph->node_ids.node_count > 0 ? intmap_get(&graph->node_ids.node_index_by_node_id, source_node_id) : -1;
    if (source < 0) return PyDict_New();
    return shortest_path_lengths_from(graph, weights_obj, &source, 1, cutoff, edge_mask_obj, node_mask_obj, terminal_mask_obj);
}

PyObject *py_multi_source_dijkstra_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *weights_obj, *sources_obj, *edge_mask_obj = Py_None, *node_mask_obj = Py_None, *terminal_mask_obj = Py_None;
    double cutoff;
    if (!PyArg_ParseTuple(args, "OOOd|OOO", &capsule, &weights_obj, &sources_obj, &cutoff, &edge_mask_obj, &node_mask_obj, &terminal_mask_obj))
        return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    PyObject *sources_fast = PySequence_Fast(sources_obj, "sources must be a sequence");
    if (!sources_fast) return NULL;
    Py_ssize_t source_count = PySequence_Fast_GET_SIZE(sources_fast);
    PyObject **source_items = PySequence_Fast_ITEMS(sources_fast);
    int *sources = malloc(((size_t)source_count + 1) * sizeof(int));
    if (!sources) { Py_DECREF(sources_fast); return PyErr_NoMemory(); }
    for (Py_ssize_t position = 0; position < source_count; position++) {
        long source_node_id = PyLong_AsLong(source_items[position]);
        if (PyErr_Occurred()) { free(sources); Py_DECREF(sources_fast); return NULL; }
        sources[position] = graph->node_ids.node_count > 0 ? intmap_get(&graph->node_ids.node_index_by_node_id, source_node_id) : -1;
    }
    Py_DECREF(sources_fast);
    PyObject *result = shortest_path_lengths_from(graph, weights_obj, sources, source_count, cutoff,
                                                  edge_mask_obj, node_mask_obj, terminal_mask_obj);
    free(sources);
    return result;
}
