/* Directed acyclic graphs and cycles: topological sort, longest path, cycle basis. */
#include "entry_points.h"
#include "../helpers/parse.h"
#include "../helpers/toposort.h"

PyObject *py_topological_sort_by_node_id(PyObject *self, PyObject *args) {
    PyObject *node_ids_obj, *edges_obj;
    if (!PyArg_ParseTuple(args, "OO", &node_ids_obj, &edges_obj)) return NULL;

    NodeIdContext context;
    if (node_id_parse(node_ids_obj, edges_obj, &context) < 0) return NULL;
    int node_count = context.node_count;

    if (node_count == 0) {
        node_id_context_free(&context);
        return PyList_New(0);
    }

    int *order;
    int count = topological_order(node_count, &context.edges, &order);
    if (count < 0) { node_id_context_free(&context); return NULL; }

    if (count < node_count) {
        free(order); node_id_context_free(&context);
        PyErr_Format(PyExc_ValueError,
            "Graph contains a cycle - topological sort is undefined "
            "(%d of %d nodes processed)", count, node_count);
        return NULL;
    }

    PyObject *result = PyList_New(node_count);
    if (result) {
        for (int position = 0; position < node_count; position++) {
            Py_INCREF(context.node_id_items[order[position]]);
            PyList_SET_ITEM(result, position, context.node_id_items[order[position]]);
        }
    }

    free(order); node_id_context_free(&context);
    return result;
}

PyObject *py_topological_sort_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *edge_mask_obj = Py_None, *node_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OO", &capsule, &edge_mask_obj, &node_mask_obj)) return NULL;
    GraphContext *graph = get_two_port_graph_context(capsule);
    if (!graph) return NULL;
    int node_count = graph->node_ids.node_count;

    if (!graph->directed) {
        PyErr_SetString(PyExc_TypeError,
            "topological_sort requires a directed graph");
        return NULL;
    }
    if (node_count == 0) return PyList_New(0);

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) return NULL;
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); return NULL; }

    /* Count active nodes */
    int active_node_count = node_count;
    if (node_mask) { active_node_count = 0; for (int node = 0; node < node_count; node++) if (!node_mask[node]) active_node_count++; }

    int *order;
    int count = topological_order_masked(node_count, &graph->node_ids.edges, edge_mask, node_mask, &order);
    release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    if (count < 0) return NULL;

    if (count < active_node_count) {
        free(order);
        PyErr_Format(PyExc_ValueError,
            "Graph contains a cycle - topological sort is undefined "
            "(%d of %d nodes processed)", count, active_node_count);
        return NULL;
    }

    PyObject *result = PyList_New(count);
    if (result) {
        for (int position = 0; position < count; position++) {
            Py_INCREF(graph->node_ids.node_id_items[order[position]]);
            PyList_SET_ITEM(result, position, graph->node_ids.node_id_items[order[position]]);
        }
    }
    free(order);
    return result;
}

/* ── Cycle basis (fundamental cycles via BFS spanning tree) ── */

/*
 * Find a fundamental cycle basis of an undirected graph.
 * For each non-tree edge of the BFS tree, traces the cycle through the tree.
 * Returns a Python list of lists of original node IDs.
 * Supports edge mask and node mask.
 */
PyObject *py_cycle_basis_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *edge_mask_obj = Py_None, *node_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OO", &capsule, &edge_mask_obj, &node_mask_obj)) return NULL;
    GraphContext *graph = get_two_port_graph_context(capsule);
    if (!graph) return NULL;
    int node_count = graph->node_ids.node_count;

    PyObject *result = PyList_New(0);
    if (!result) return NULL;
    if (node_count == 0 || !graph->has_adjacency) return result;

    if (graph->directed) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_TypeError,
            "cycle_basis is not defined for directed graphs");
        return NULL;
    }

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }

    AdjacencyList *adjacency = &graph->adjacency;
    EdgeList *edges = &graph->node_ids.edges;

    /* Self-loops are trivial cycles */
    for (Py_ssize_t edge_index = 0; edge_index < edges->edge_count; edge_index++) {
        if (edge_mask && edge_mask[edge_index]) continue;
        int first_node = EDGE_FIRST_ENDPOINT(edges, edge_index);
        if (node_mask && node_mask[first_node]) continue;
        if (first_node == EDGE_SECOND_ENDPOINT(edges, edge_index)) {
            PyObject *cycle = PyList_New(1);
            if (!cycle) { release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }
            Py_INCREF(graph->node_ids.node_id_items[first_node]);
            PyList_SET_ITEM(cycle, 0, graph->node_ids.node_id_items[first_node]);
            PyList_Append(result, cycle);
            Py_DECREF(cycle);
        }
    }

    /* Track which non-tree edges have been processed (one bit per edge) */
    Py_ssize_t edge_count = edges->edge_count;
    uint8_t *edge_seen = (uint8_t *)calloc(edge_count > 0 ? (size_t)edge_count : 1, 1);
    if (!edge_seen) {
        release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result);
        PyErr_NoMemory(); return NULL;
    }

    /* Allocate: parent, depth, queue, parent_edge, path_a, path_b; node_count entries each */
    int *scratch = (int *)malloc(6 * (size_t)node_count * sizeof(int));
    if (!scratch) {
        free(edge_seen);
        release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result);
        PyErr_NoMemory(); return NULL;
    }
    int *parent      = scratch;
    int *depth       = scratch + node_count;
    int *queue       = scratch + 2 * node_count;
    int *parent_edge = scratch + 3 * node_count;  /* to skip the reverse tree edge */
    int *path_a      = scratch + 4 * node_count;
    int *path_b      = scratch + 5 * node_count;
    memset(depth, -1, (size_t)node_count * sizeof(int));

    /* BFS spanning tree + non-tree edge cycle extraction.
     * BFS trees are shallow (O(log n) for random graphs), making the
     * lowest-common-ancestor cycle trace O(log n) per non-tree edge instead of O(n). */
    for (int start = 0; start < node_count; start++) {
        if (depth[start] != -1) continue;
        if (node_mask && node_mask[start]) continue;

        /* BFS to build spanning tree */
        int head = 0, tail = 0;
        queue[tail++] = start;
        depth[start] = 0;
        parent[start] = -1;
        parent_edge[start] = -1;

        while (head < tail) {
            int node = queue[head++];
            for (int slot = adjacency->offset[node]; slot < adjacency->offset[node + 1]; slot++) {
                int edge_index = adjacency->edge_indices[slot];
                if (edge_mask && edge_mask[edge_index]) continue;
                int neighbor = adjacency->neighbors[slot];
                if (node_mask && node_mask[neighbor]) continue;
                if (neighbor == node) continue;  /* skip self-loops (handled above) */
                if (depth[neighbor] == -1) {
                    /* Tree edge */
                    depth[neighbor] = depth[node] + 1;
                    parent[neighbor] = node;
                    parent_edge[neighbor] = edge_index;
                    queue[tail++] = neighbor;
                } else if (edge_index != parent_edge[node] && !edge_seen[edge_index]) {
                    /* Non-tree edge: extract cycle via the lowest common ancestor.
                     * edge_seen prevents processing each undirected
                     * edge twice (CSR has both node->neighbor and neighbor->node). */
                    edge_seen[edge_index] = 1;
                    int walker_a = node, walker_b = neighbor;
                    int path_a_length = 0, path_b_length = 0;

                    /* Walk walker_a and walker_b up to the same depth */
                    while (depth[walker_a] > depth[walker_b]) { path_a[path_a_length++] = walker_a; walker_a = parent[walker_a]; }
                    while (depth[walker_b] > depth[walker_a]) { path_b[path_b_length++] = walker_b; walker_b = parent[walker_b]; }
                    /* Walk both up to the lowest common ancestor */
                    while (walker_a != walker_b) {
                        path_a[path_a_length++] = walker_a; walker_a = parent[walker_a];
                        path_b[path_b_length++] = walker_b; walker_b = parent[walker_b];
                    }
                    /* walker_a == walker_b == lowest common ancestor */
                    int lowest_common_ancestor = walker_a;
                    int cycle_length = path_a_length + path_b_length + 1;

                    PyObject *cycle = PyList_New(cycle_length);
                    if (!cycle) {
                        free(scratch); free(edge_seen);
                        release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
                        Py_DECREF(result); return NULL;
                    }
                    int idx = 0;
                    for (int position = 0; position < path_a_length; position++) {
                        Py_INCREF(graph->node_ids.node_id_items[path_a[position]]);
                        PyList_SET_ITEM(cycle, idx++, graph->node_ids.node_id_items[path_a[position]]);
                    }
                    Py_INCREF(graph->node_ids.node_id_items[lowest_common_ancestor]);
                    PyList_SET_ITEM(cycle, idx++, graph->node_ids.node_id_items[lowest_common_ancestor]);
                    for (int position = path_b_length - 1; position >= 0; position--) {
                        Py_INCREF(graph->node_ids.node_id_items[path_b[position]]);
                        PyList_SET_ITEM(cycle, idx++, graph->node_ids.node_id_items[path_b[position]]);
                    }
                    PyList_Append(result, cycle);
                    Py_DECREF(cycle);
                }
            }
        }
    }

    free(scratch); free(edge_seen);
    release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    return result;
}

/* ── DAG longest path (topological sort + DP) ── */

/*
 * Find the longest path in a weighted or unweighted DAG.
 * Uses topological sort + dynamic programming.
 * weights: if NULL, all edges have weight 1.
 * Returns Python list of original node IDs forming the longest path.
 */
PyObject *py_dag_longest_path_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *weights_obj = Py_None, *edge_mask_obj = Py_None, *node_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OOO", &capsule, &weights_obj, &edge_mask_obj, &node_mask_obj)) return NULL;
    GraphContext *graph = get_two_port_graph_context(capsule);
    if (!graph) return NULL;
    int node_count = graph->node_ids.node_count;

    if (!graph->directed) {
        PyErr_SetString(PyExc_TypeError,
            "dag_longest_path requires a directed graph");
        return NULL;
    }
    if (node_count == 0) return PyList_New(0);

    /* Parse optional weights */
    WeightList weights = {NULL, 0, 0};
    int has_weights = (weights_obj != Py_None);
    if (has_weights) {
        if (parse_weights(weights_obj, &weights, graph->node_ids.edges.edge_count) < 0) return NULL;
    }

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) {
        if (has_weights) free_weights(&weights); return NULL;
    }
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) {
        release_mask(&edge_mask_buffer); if (has_weights) free_weights(&weights); return NULL;
    }

    /* Get topological order (reuse masked toposort) */
    int *order;
    int active_node_count = node_count;
    if (node_mask) { active_node_count = 0; for (int node = 0; node < node_count; node++) if (!node_mask[node]) active_node_count++; }

    int count = topological_order_masked(node_count, &graph->node_ids.edges, edge_mask, node_mask, &order);
    if (count < 0) {
        release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
        if (has_weights) free_weights(&weights);
        return NULL;
    }
    if (count < active_node_count) {
        free(order);
        release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
        if (has_weights) free_weights(&weights);
        PyErr_SetString(PyExc_ValueError,
            "Graph contains a cycle - dag_longest_path is undefined");
        return NULL;
    }

    /* DP: distance[node] = longest distance to reach node, predecessor[node] = predecessor on longest path */
    double *distance = (double *)malloc((size_t)node_count * sizeof(double));
    int *predecessor = (int *)malloc((size_t)node_count * sizeof(int));
    if (!distance || !predecessor) {
        free(distance); free(predecessor); free(order);
        release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
        if (has_weights) free_weights(&weights);
        PyErr_NoMemory(); return NULL;
    }
    for (int node = 0; node < node_count; node++) { distance[node] = 0.0; predecessor[node] = -1; }

    /* Use forward CSR to iterate outgoing edges */
    AdjacencyList *adjacency = &graph->adjacency;
    int has_adjacency = graph->has_adjacency;

    /* Process nodes in topological order */
    for (int order_position = 0; order_position < count; order_position++) {
        int node = order[order_position];
        if (has_adjacency) {
            for (int slot = adjacency->offset[node]; slot < adjacency->offset[node + 1]; slot++) {
                int edge_index = adjacency->edge_indices[slot];
                if (edge_mask && edge_mask[edge_index]) continue;
                int neighbor = adjacency->neighbors[slot];
                if (node_mask && node_mask[neighbor]) continue;
                double weight = has_weights ? weights.values[edge_index] : 1.0;
                if (distance[node] + weight > distance[neighbor]) {
                    distance[neighbor] = distance[node] + weight;
                    predecessor[neighbor] = node;
                }
            }
        } else {
            /* No CSR (no edges) — nothing to relax */
        }
    }

    /* Find node with maximum distance */
    int best_node = -1;
    double best_distance = -1.0;
    for (int order_position = 0; order_position < count; order_position++) {
        int node = order[order_position];
        if (distance[node] > best_distance) {
            best_distance = distance[node];
            best_node = node;
        }
    }

    /* Trace back the path */
    if (best_node < 0 || best_distance == 0.0) {
        /* No edges or all isolated: return first active node */
        PyObject *single_node_path;
        if (count > 0) {
            single_node_path = PyList_New(1);
            if (single_node_path) {
                Py_INCREF(graph->node_ids.node_id_items[order[0]]);
                PyList_SET_ITEM(single_node_path, 0, graph->node_ids.node_id_items[order[0]]);
            }
        } else {
            single_node_path = PyList_New(0);
        }
        free(distance); free(predecessor); free(order);
        release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
        if (has_weights) free_weights(&weights);
        return single_node_path;
    }

    /* Count path length by tracing back */
    int path_length = 0;
    for (int node = best_node; node >= 0; node = predecessor[node]) path_length++;

    PyObject *result = PyList_New(path_length);
    if (!result) {
        free(distance); free(predecessor); free(order);
        release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
        if (has_weights) free_weights(&weights);
        return NULL;
    }
    /* Fill in reverse */
    int idx = path_length - 1;
    for (int node = best_node; node >= 0; node = predecessor[node]) {
        Py_INCREF(graph->node_ids.node_id_items[node]);
        PyList_SET_ITEM(result, idx--, graph->node_ids.node_id_items[node]);
    }

    free(distance); free(predecessor); free(order);
    release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    if (has_weights) free_weights(&weights);
    return result;
}
