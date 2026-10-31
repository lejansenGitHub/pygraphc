/* Dijkstra: source-target paths, single-source and multi-source lengths. */
#include "entry_points.h"
#include "../helpers/ports.h"
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
    PyObject *capsule, *weights_obj, *source_obj, *target_obj, *edge_mask_obj = Py_None, *node_mask_obj = Py_None, *port_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OOOO|OOO", &capsule, &weights_obj, &source_obj, &target_obj, &edge_mask_obj, &node_mask_obj, &port_mask_obj))
        return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    int node_count = graph->node_ids.node_count;
    int edge_count = (int)graph->node_ids.edges.edge_count;
    const PortList *ports = graph->ports;

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
    PortMask port_mask;
    if (parse_port_mask(graph, port_mask_obj, edge_mask, &port_mask) < 0) { release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); return NULL; }

    if (node_mask && (node_mask[source] || node_mask[target])) {
        release_port_mask(&port_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
        PyObject *empty_path = PyList_New(0);
        return Py_BuildValue("(dN)", HUGE_VAL, empty_path);
    }

    WeightList weights;
    if (parse_weights(weights_obj, &weights, ports ? ports->port_count : edge_count) < 0) { release_port_mask(&port_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); return NULL; }
    if (!graph->has_adjacency) { free_weights(&weights); release_port_mask(&port_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); PyObject *empty_path = PyList_New(0); return Py_BuildValue("(dN)", HUGE_VAL, empty_path); }

    BidirectionalPath bidirectional_path;
    int failed = (ports
        ? bidirectional_dijkstra_with_ports(node_count, edge_count, &graph->adjacency, weights.values, ports,
                                            two_port_edge_mask(&port_mask, edge_mask), edge_mask, port_mask.ports,
                                            node_mask, graph->directed, source, target, &bidirectional_path)
        : bidirectional_dijkstra(node_count, &graph->adjacency, weights.values, edge_mask, node_mask, graph->directed,
                                 source, target, &bidirectional_path)) < 0;
    PyObject *path = failed ? NULL : bidirectional_path_to_list(&bidirectional_path, &graph->node_ids);
    free(bidirectional_path.path);
    free_weights(&weights); release_port_mask(&port_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    if (!path) return NULL;
    return Py_BuildValue("(dN)", bidirectional_path.weight, path);
}

/*
 * Dijkstra from the sources already in the heap, over the search states of a
 * graph: its nodes, and with has_multiport_edges also one state per edge with
 * three or more ports (node_count + e). A two-port edge relaxes its neighbor
 * directly, at its edge weight or, with port weights, at the sum of its two
 * port weights. An edge with more ports is entered through an active port at
 * that port's weight, settled once at its cheapest entry and then relaxes each
 * active exit port (directed: out-port) at the exit port's weight, so an edge
 * with k ports costs O(k). The function is instantiated for both values of
 * has_multiport_edges, so graphs of two-port edges run the loop as before.
 */
static inline __attribute__((always_inline)) void relax_until_settled(
    const GraphContext *graph, const double *weights, const uint8_t *two_port_edge_mask, const uint8_t *edge_mask,
    const uint8_t *port_mask, const uint8_t *node_mask, int use_cutoff, double cutoff, MinHeap *min_heap,
    uint8_t *settled_edges, const int has_multiport_edges) {
    const AdjacencyList *adjacency = &graph->adjacency;
    const PortList *ports = graph->ports;
    int node_count = graph->node_ids.node_count;
    double *distance = min_heap->distance;
    int *heap = min_heap->nodes, *heap_position = min_heap->heap_position;
    while (min_heap->size > 0) {
        int state = min_heap_pop(min_heap);
        if (distance[state]==HUGE_VAL) break;
        if (use_cutoff && distance[state]>cutoff) break;
        if (has_multiport_edges && state >= node_count) {
            int edge = state - node_count;
            settled_edges[edge] = 1;
            int first_exit = graph->directed ? ports->out_port_offsets[edge] : ports->port_offsets[edge];
            for (int exit = first_exit; exit < ports->port_offsets[edge + 1]; exit++) {
                if (!port_is_active(ports, edge, exit, edge_mask, port_mask, node_mask)) continue;
                int reached = ports->port_nodes[exit];
                double candidate_distance = distance[state] + weights[exit];
                if (candidate_distance < distance[reached] && (!use_cutoff || candidate_distance <= cutoff)) {
                    distance[reached] = candidate_distance;
                    if (heap_position[reached]==-1) { int heap_slot=min_heap->size++; heap[heap_slot]=reached; heap_position[reached]=heap_slot; min_heap_sift_up(min_heap,heap_slot); }
                    else min_heap_sift_up(min_heap, heap_position[reached]);
                }
            }
            continue;
        }
        int node = state;
        for (int slot=adjacency->offset[node]; slot<adjacency->offset[node+1]; slot++) {
            int neighbor=adjacency->neighbors[slot];
            if (has_multiport_edges && neighbor < 0) {
                int edge = -1 - neighbor, entry = adjacency->edge_indices[slot];
                if (settled_edges[edge] || !port_is_active(ports, edge, entry, edge_mask, port_mask, node_mask)) continue;
                int edge_state = node_count + edge;
                double candidate_distance = distance[node] + weights[entry];
                if (candidate_distance < distance[edge_state] && (!use_cutoff || candidate_distance <= cutoff)) {
                    distance[edge_state] = candidate_distance;
                    if (heap_position[edge_state]==-1) { int heap_slot=min_heap->size++; heap[heap_slot]=edge_state; heap_position[edge_state]=heap_slot; min_heap_sift_up(min_heap,heap_slot); }
                    else min_heap_sift_up(min_heap, heap_position[edge_state]);
                }
                continue;
            }
            int edge_index = adjacency->edge_indices[slot];
            if (two_port_edge_mask && two_port_edge_mask[edge_index]) continue;
            if (node_mask && node_mask[neighbor]) continue;
            double edge_weight = has_multiport_edges
                ? weights[ports->port_offsets[edge_index]] + weights[ports->port_offsets[edge_index] + 1]
                : weights[edge_index];
            double candidate_distance=distance[node]+edge_weight;
            if (candidate_distance < distance[neighbor] && (!use_cutoff || candidate_distance <= cutoff)) {
                if (heap_position[neighbor]==-1) { distance[neighbor]=candidate_distance; int heap_slot=min_heap->size++; heap[heap_slot]=neighbor; heap_position[neighbor]=heap_slot; min_heap_sift_up(min_heap,heap_slot); }
                else { distance[neighbor]=candidate_distance; min_heap_sift_up(min_heap, heap_position[neighbor]); }
            }
        }
    }
}

/*
 * {node id: distance} from the given source indices (masked or repeated
 * sources skipped): one weight per edge on a graph of two-port edges, one per
 * port on a graph with multiport edges. NULL with an exception set on error.
 */
static PyObject *shortest_path_lengths_from(const GraphContext *graph, PyObject *weights_obj, const int *sources,
                                            Py_ssize_t source_count, double cutoff, PyObject *edge_mask_obj,
                                            PyObject *node_mask_obj, PyObject *port_mask_obj) {
    int node_count = graph->node_ids.node_count;
    int use_cutoff = (cutoff >= 0.0);
    PyObject *result = PyDict_New();
    if (!result) return NULL;
    if (node_count == 0 || !graph->has_adjacency) return result;
    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }
    PortMask port_mask;
    if (parse_port_mask(graph, port_mask_obj, edge_mask, &port_mask) < 0) { release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }
    const PortList *ports = graph->ports;
    WeightList weights;
    if (parse_weights(weights_obj, &weights, ports ? ports->port_count : graph->node_ids.edges.edge_count) < 0) {
        release_port_mask(&port_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL;
    }
    int state_count = node_count + (ports ? (int)graph->node_ids.edges.edge_count : 0);
    double *distance = malloc((size_t)state_count*sizeof(double));
    int *heap = malloc((size_t)state_count*sizeof(int)), *heap_position = malloc((size_t)state_count*sizeof(int));
    uint8_t *settled_edges = ports ? calloc((size_t)graph->node_ids.edges.edge_count + 1, 1) : NULL;
    if (!distance||!heap||!heap_position||(ports && !settled_edges)) {
        free(distance);free(heap);free(heap_position);free(settled_edges);
        free_weights(&weights); release_port_mask(&port_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result);
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
    if (ports)
        relax_until_settled(graph, weights.values, two_port_edge_mask(&port_mask, edge_mask), edge_mask, port_mask.ports,
                            node_mask, use_cutoff, cutoff, &min_heap, settled_edges, 1);
    else
        relax_until_settled(graph, weights.values, edge_mask, edge_mask, NULL, node_mask, use_cutoff, cutoff, &min_heap,
                            NULL, 0);
    for (int node=0;node<node_count;node++) {
        if (node_mask && node_mask[node]) continue;
        if (distance[node] < HUGE_VAL && (!use_cutoff || distance[node] <= cutoff)) {
            PyObject *value = PyFloat_FromDouble(distance[node]);
            PyDict_SetItem(result, graph->node_ids.node_id_items[node], value);
            Py_DECREF(value);
        }
    }
    free(distance);free(heap);free(heap_position);free(settled_edges);
    free_weights(&weights); release_port_mask(&port_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    return result;
}

PyObject *py_shortest_path_lengths_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *weights_obj, *source_obj, *edge_mask_obj = Py_None, *node_mask_obj = Py_None, *port_mask_obj = Py_None;
    double cutoff;
    if (!PyArg_ParseTuple(args, "OOOd|OOO", &capsule, &weights_obj, &source_obj, &cutoff, &edge_mask_obj, &node_mask_obj, &port_mask_obj))
        return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    long source_node_id = PyLong_AsLong(source_obj);
    if (PyErr_Occurred()) return NULL;
    int source = graph->node_ids.node_count > 0 ? intmap_get(&graph->node_ids.node_index_by_node_id, source_node_id) : -1;
    if (source < 0) return PyDict_New();
    return shortest_path_lengths_from(graph, weights_obj, &source, 1, cutoff, edge_mask_obj, node_mask_obj, port_mask_obj);
}

PyObject *py_multi_source_dijkstra_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *weights_obj, *sources_obj, *edge_mask_obj = Py_None, *node_mask_obj = Py_None, *port_mask_obj = Py_None;
    double cutoff;
    if (!PyArg_ParseTuple(args, "OOOd|OOO", &capsule, &weights_obj, &sources_obj, &cutoff, &edge_mask_obj, &node_mask_obj, &port_mask_obj))
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
                                                  edge_mask_obj, node_mask_obj, port_mask_obj);
    free(sources);
    return result;
}
