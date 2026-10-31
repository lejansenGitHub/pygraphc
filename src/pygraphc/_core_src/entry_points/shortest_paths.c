/* Dijkstra: source-target paths, single-source and multi-source lengths. */
#include "entry_points.h"
#include "../helpers/parse.h"
#include "../helpers/min_heap.h"
#include "../helpers/bidirectional_dijkstra.h"

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
    if (build_adjacency(node_count, &edges, &adjacency, 0) < 0) {
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
    if (build_adjacency(node_count, &edges, &adjacency, 0) < 0) {
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
    if (build_adjacency(node_count, &edges, &adjacency, 0) < 0) {
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
    if (build_adjacency(node_count, &context.edges, &adjacency, 0) < 0) { node_id_context_free(&context); free_weights(&weights); return NULL; }

    BidirectionalPath bidirectional_path;
    int failed = bidirectional_dijkstra(node_count, &adjacency, weights.values, NULL, NULL, 0, source, target, &bidirectional_path) < 0;
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
    if (build_adjacency(node_count, &context.edges, &adjacency, 0) < 0) { node_id_context_free(&context); free_weights(&weights); Py_DECREF(result); return NULL; }

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
    if (build_adjacency(node_count, &context.edges, &adjacency, 0) < 0) { Py_DECREF(sources_fast); free_weights(&weights); node_id_context_free(&context); Py_DECREF(result); return NULL; }

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

PyObject *py_dijkstra_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *weights_obj, *source_obj, *target_obj, *edge_mask_obj = Py_None, *node_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OOOO|OO", &capsule, &weights_obj, &source_obj, &target_obj, &edge_mask_obj, &node_mask_obj))
        return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    int node_count = graph->node_ids.node_count;

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
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) return NULL;
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); return NULL; }

    if (node_mask && (node_mask[source] || node_mask[target])) {
        release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
        PyObject *empty_path = PyList_New(0);
        return Py_BuildValue("(dN)", HUGE_VAL, empty_path);
    }

    WeightList weights;
    if (parse_weights(weights_obj, &weights, graph->node_ids.edges.edge_count) < 0) { release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); return NULL; }
    if (!graph->has_adjacency) { free_weights(&weights); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); PyObject *empty_path = PyList_New(0); return Py_BuildValue("(dN)", HUGE_VAL, empty_path); }

    BidirectionalPath bidirectional_path;
    int failed = bidirectional_dijkstra(node_count, &graph->adjacency, weights.values, edge_mask, node_mask, graph->directed, source, target, &bidirectional_path) < 0;
    PyObject *path = failed ? NULL : bidirectional_path_to_list(&bidirectional_path, &graph->node_ids);
    free(bidirectional_path.path);
    free_weights(&weights); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    if (!path) return NULL;
    return Py_BuildValue("(dN)", bidirectional_path.weight, path);
}

PyObject *py_shortest_path_lengths_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *weights_obj, *source_obj, *edge_mask_obj = Py_None, *node_mask_obj = Py_None;
    double cutoff;
    if (!PyArg_ParseTuple(args, "OOOd|OO", &capsule, &weights_obj, &source_obj, &cutoff, &edge_mask_obj, &node_mask_obj))
        return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    int node_count = graph->node_ids.node_count;
    int use_cutoff = (cutoff >= 0.0);

    PyObject *result = PyDict_New();
    if (!result) return NULL;

    long source_node_id = PyLong_AsLong(source_obj);
    if (PyErr_Occurred()) { Py_DECREF(result); return NULL; }
    int source = node_count > 0 ? intmap_get(&graph->node_ids.node_index_by_node_id, source_node_id) : -1;
    if (node_count == 0 || source < 0 || !graph->has_adjacency) return result;

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }

    if (node_mask && node_mask[source]) {
        release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
        return result;  /* source masked out — empty result */
    }

    WeightList weights;
    if (parse_weights(weights_obj, &weights, graph->node_ids.edges.edge_count) < 0) { release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }
    AdjacencyList *adjacency = &graph->adjacency;

    double *distance = malloc(node_count*sizeof(double));
    int *heap = malloc(node_count*sizeof(int)), *heap_position = malloc(node_count*sizeof(int));
    if (!distance||!heap||!heap_position) {
        free(distance);free(heap);free(heap_position);
        free_weights(&weights); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result); PyErr_NoMemory(); return NULL;
    }
    for (int node=0;node<node_count;node++) { distance[node]=HUGE_VAL; heap_position[node]=-1; }
    MinHeap min_heap = {distance, heap, heap_position, 0};
    distance[source]=0.0; heap[0]=source; heap_position[source]=0; min_heap.size=1;

    while (min_heap.size > 0) {
        int node = min_heap_pop(&min_heap);
        if (distance[node]==HUGE_VAL) break;
        if (use_cutoff && distance[node]>cutoff) break;
        for (int slot=adjacency->offset[node]; slot<adjacency->offset[node+1]; slot++) {
            if (edge_mask && edge_mask[adjacency->edge_indices[slot]]) continue;
            int neighbor=adjacency->neighbors[slot];
            if (node_mask && node_mask[neighbor]) continue;
            double candidate_distance=distance[node]+weights.values[adjacency->edge_indices[slot]];
            if (candidate_distance < distance[neighbor] && (!use_cutoff || candidate_distance <= cutoff)) {
                if (heap_position[neighbor]==-1) { distance[neighbor]=candidate_distance; int heap_slot=min_heap.size++; heap[heap_slot]=neighbor; heap_position[neighbor]=heap_slot; min_heap_sift_up(&min_heap,heap_slot); }
                else { distance[neighbor]=candidate_distance; min_heap_sift_up(&min_heap, heap_position[neighbor]); }
            }
        }
    }

    for (int node=0;node<node_count;node++) {
        if (node_mask && node_mask[node]) continue;
        if (distance[node] < HUGE_VAL && (!use_cutoff || distance[node] <= cutoff)) {
            PyObject *value = PyFloat_FromDouble(distance[node]);
            PyDict_SetItem(result, graph->node_ids.node_id_items[node], value);
            Py_DECREF(value);
        }
    }
    free(distance);free(heap);free(heap_position);
    free_weights(&weights); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    return result;
}

PyObject *py_multi_source_dijkstra_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *weights_obj, *sources_obj, *edge_mask_obj = Py_None, *node_mask_obj = Py_None;
    double cutoff;
    if (!PyArg_ParseTuple(args, "OOOd|OO", &capsule, &weights_obj, &sources_obj, &cutoff, &edge_mask_obj, &node_mask_obj))
        return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    int node_count = graph->node_ids.node_count;
    int use_cutoff = (cutoff >= 0.0);

    PyObject *result = PyDict_New();
    if (!result) return NULL;
    if (node_count == 0 || !graph->has_adjacency) return result;

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }

    PyObject *sources_fast = PySequence_Fast(sources_obj, "sources must be a sequence");
    if (!sources_fast) { release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }

    WeightList weights;
    if (parse_weights(weights_obj, &weights, graph->node_ids.edges.edge_count) < 0) { Py_DECREF(sources_fast); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }
    AdjacencyList *adjacency = &graph->adjacency;

    double *distance = malloc(node_count*sizeof(double));
    int *heap = malloc(node_count*sizeof(int)), *heap_position = malloc(node_count*sizeof(int));
    if (!distance||!heap||!heap_position) {
        free(distance);free(heap);free(heap_position);
        free_weights(&weights); Py_DECREF(sources_fast); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result);
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
            free_weights(&weights); Py_DECREF(sources_fast); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result);
            return NULL;
        }
        int source = intmap_get(&graph->node_ids.node_index_by_node_id, source_node_id);
        if (source >= 0 && !(node_mask && node_mask[source]) && distance[source] > 0.0) {
            distance[source]=0.0; int heap_slot=min_heap.size++; heap[heap_slot]=source; heap_position[source]=heap_slot; min_heap_sift_up(&min_heap,heap_slot);
        }
    }
    Py_DECREF(sources_fast);

    while (min_heap.size > 0) {
        int node = min_heap_pop(&min_heap);
        if (distance[node]==HUGE_VAL) break;
        if (use_cutoff && distance[node]>cutoff) break;
        for (int slot=adjacency->offset[node]; slot<adjacency->offset[node+1]; slot++) {
            if (edge_mask && edge_mask[adjacency->edge_indices[slot]]) continue;
            int neighbor=adjacency->neighbors[slot];
            if (node_mask && node_mask[neighbor]) continue;
            double candidate_distance=distance[node]+weights.values[adjacency->edge_indices[slot]];
            if (candidate_distance < distance[neighbor] && (!use_cutoff || candidate_distance <= cutoff)) {
                if (heap_position[neighbor]==-1) { distance[neighbor]=candidate_distance; int heap_slot=min_heap.size++; heap[heap_slot]=neighbor; heap_position[neighbor]=heap_slot; min_heap_sift_up(&min_heap,heap_slot); }
                else { distance[neighbor]=candidate_distance; min_heap_sift_up(&min_heap, heap_position[neighbor]); }
            }
        }
    }

    for (int node=0;node<node_count;node++) {
        if (node_mask && node_mask[node]) continue;
        if (distance[node] < HUGE_VAL && (!use_cutoff || distance[node] <= cutoff)) {
            PyObject *value = PyFloat_FromDouble(distance[node]);
            PyDict_SetItem(result, graph->node_ids.node_id_items[node], value);
            Py_DECREF(value);
        }
    }
    free(distance);free(heap);free(heap_position);
    free_weights(&weights); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    return result;
}
