/* Bridges, articulation points and biconnected components (iterative Tarjan). */
#include "entry_points.h"
#include "../helpers/parse.h"

/* ── Bridges (iterative Tarjan's) ── */

PyObject *py_bridges(PyObject *self, PyObject *args) {
    int node_count;
    PyObject *edges_obj;
    if (!PyArg_ParseTuple(args, "iO", &node_count, &edges_obj)) return NULL;

    PyObject *result = PyList_New(0);
    if (!result) return NULL;
    if (node_count == 0) return result;

    EdgeList edges;
    if (parse_edges(edges_obj, &edges) < 0) { Py_DECREF(result); return NULL; }
    if (edges.edge_count == 0) { free_edges(&edges); return result; }

    AdjacencyList adjacency;
    if (build_adjacency(node_count, &edges, &adjacency, 0) < 0) {
        free_edges(&edges); Py_DECREF(result); return NULL;
    }

    int *scratch = (int *)malloc(5 * (size_t)node_count * sizeof(int));
    if (!scratch) {
        free_adjacency(&adjacency); free_edges(&edges); Py_DECREF(result);
        PyErr_NoMemory();
        return NULL;
    }
    int *discovery_time = scratch, *low_link = scratch + node_count, *stack = scratch + 2*node_count;
    int *stack_next_slot = scratch + 3*node_count, *parent_edge = scratch + 4*node_count;
    memset(discovery_time, -1, (size_t)node_count * sizeof(int));

    int timer = 0;
    for (int start = 0; start < node_count; start++) {
        if (discovery_time[start] != -1) continue;
        int stack_top = 0;
        stack[0] = start;
        stack_next_slot[0] = adjacency.offset[start];
        discovery_time[start] = low_link[start] = timer++;
        parent_edge[start] = -1;

        while (stack_top >= 0) {
            int node = stack[stack_top];
            if (stack_next_slot[stack_top] < adjacency.offset[node + 1]) {
                int slot = stack_next_slot[stack_top]++;
                int neighbor = adjacency.neighbors[slot];
                int edge_index = adjacency.edge_indices[slot];
                if (edge_index == parent_edge[node]) continue;
                if (discovery_time[neighbor] == -1) {
                    discovery_time[neighbor] = low_link[neighbor] = timer++;
                    parent_edge[neighbor] = edge_index;
                    stack_top++;
                    stack[stack_top] = neighbor;
                    stack_next_slot[stack_top] = adjacency.offset[neighbor];
                } else {
                    if (low_link[node] > discovery_time[neighbor]) low_link[node] = discovery_time[neighbor];
                }
            } else {
                if (stack_top > 0) {
                    int parent = stack[stack_top - 1];
                    if (low_link[parent] > low_link[node]) low_link[parent] = low_link[node];
                    if (low_link[node] > discovery_time[parent]) {
                        int bridge_edge = parent_edge[node];
                        PyObject *bridge_pair = Py_BuildValue(
                            "(ii)", EDGE_FIRST_ENDPOINT(&edges, bridge_edge), EDGE_SECOND_ENDPOINT(&edges, bridge_edge));
                        if (!bridge_pair) {
                            free(scratch);
                            free_adjacency(&adjacency); free_edges(&edges);
                            Py_DECREF(result);
                            return NULL;
                        }
                        PyList_Append(result, bridge_pair);
                        Py_DECREF(bridge_pair);
                    }
                }
                stack_top--;
            }
        }
    }

    free(scratch);
    free_adjacency(&adjacency); free_edges(&edges);
    return result;
}

/* ── Articulation points (iterative Tarjan's) ── */

PyObject *py_articulation_points(PyObject *self, PyObject *args) {
    int node_count;
    PyObject *edges_obj;
    if (!PyArg_ParseTuple(args, "iO", &node_count, &edges_obj)) return NULL;

    PyObject *result = PySet_New(NULL);
    if (!result) return NULL;
    if (node_count == 0) return result;

    EdgeList edges;
    if (parse_edges(edges_obj, &edges) < 0) { Py_DECREF(result); return NULL; }

    AdjacencyList adjacency;
    if (build_adjacency(node_count, &edges, &adjacency, 0) < 0) {
        free_edges(&edges); Py_DECREF(result); return NULL;
    }
    free_edges(&edges);

    int *scratch = (int *)malloc(6 * (size_t)node_count * sizeof(int));
    if (!scratch) {
        free_adjacency(&adjacency); Py_DECREF(result);
        PyErr_NoMemory();
        return NULL;
    }
    int *discovery_time = scratch, *low_link = scratch + node_count, *stack = scratch + 2*node_count;
    int *stack_next_slot = scratch + 3*node_count, *parent_edge = scratch + 4*node_count, *child_count = scratch + 5*node_count;
    memset(discovery_time, -1, (size_t)node_count * sizeof(int));
    memset(child_count, 0, (size_t)node_count * sizeof(int));

    int timer = 0;
    for (int start = 0; start < node_count; start++) {
        if (discovery_time[start] != -1) continue;
        int stack_top = 0;
        stack[0] = start;
        stack_next_slot[0] = adjacency.offset[start];
        discovery_time[start] = low_link[start] = timer++;
        parent_edge[start] = -1;
        child_count[start] = 0;

        while (stack_top >= 0) {
            int node = stack[stack_top];
            if (stack_next_slot[stack_top] < adjacency.offset[node + 1]) {
                int slot = stack_next_slot[stack_top]++;
                int neighbor = adjacency.neighbors[slot];
                int edge_index = adjacency.edge_indices[slot];
                if (edge_index == parent_edge[node]) continue;
                if (discovery_time[neighbor] == -1) {
                    discovery_time[neighbor] = low_link[neighbor] = timer++;
                    parent_edge[neighbor] = edge_index;
                    child_count[neighbor] = 0;
                    child_count[node]++;
                    stack_top++;
                    stack[stack_top] = neighbor;
                    stack_next_slot[stack_top] = adjacency.offset[neighbor];
                } else {
                    if (low_link[node] > discovery_time[neighbor]) low_link[node] = discovery_time[neighbor];
                }
            } else {
                if (stack_top > 0) {
                    int parent = stack[stack_top - 1];
                    if (low_link[parent] > low_link[node]) low_link[parent] = low_link[node];
                    /* Non-root articulation point: low_link[child] >= discovery_time[parent] */
                    if (parent_edge[parent] != -1 && low_link[node] >= discovery_time[parent]) {
                        PyObject *articulation_point_obj = PyLong_FromLong(parent);
                        PySet_Add(result, articulation_point_obj);
                        Py_DECREF(articulation_point_obj);
                    }
                }
                stack_top--;
            }
        }

        /* Root articulation point: 2+ DFS tree children */
        if (child_count[start] >= 2) {
            PyObject *articulation_point_obj = PyLong_FromLong(start);
            PySet_Add(result, articulation_point_obj);
            Py_DECREF(articulation_point_obj);
        }
    }

    free(scratch);
    free_adjacency(&adjacency);
    return result;
}

/* ── Biconnected components (iterative Tarjan's with edge stack) ── */

PyObject *py_biconnected_components(PyObject *self, PyObject *args) {
    int node_count;
    PyObject *edges_obj;
    if (!PyArg_ParseTuple(args, "iO", &node_count, &edges_obj)) return NULL;

    PyObject *result = PyList_New(0);
    if (!result) return NULL;
    if (node_count == 0) return result;

    EdgeList edges;
    if (parse_edges(edges_obj, &edges) < 0) { Py_DECREF(result); return NULL; }
    if (edges.edge_count == 0) { free_edges(&edges); return result; }

    Py_ssize_t edge_count = edges.edge_count;
    AdjacencyList adjacency;
    if (build_adjacency(node_count, &edges, &adjacency, 0) < 0) {
        free_edges(&edges); Py_DECREF(result); return NULL;
    }
    free_edges(&edges);

    int *scratch = (int *)malloc(5 * (size_t)node_count * sizeof(int));
    int *edge_stack_scratch = (int *)malloc(2 * (size_t)edge_count * sizeof(int));
    if (!scratch || !edge_stack_scratch) {
        free(scratch); free(edge_stack_scratch);
        free_adjacency(&adjacency); Py_DECREF(result);
        PyErr_NoMemory();
        return NULL;
    }
    int *discovery_time = scratch, *low_link = scratch + node_count, *stack = scratch + 2*node_count;
    int *stack_next_slot = scratch + 3*node_count, *parent_edge = scratch + 4*node_count;
    int *edge_stack_from_node = edge_stack_scratch, *edge_stack_to_node = edge_stack_scratch + edge_count;
    memset(discovery_time, -1, (size_t)node_count * sizeof(int));

    int timer = 0;
    int edge_stack_top = 0;

    for (int start = 0; start < node_count; start++) {
        if (discovery_time[start] != -1) continue;
        int stack_top = 0;
        stack[0] = start;
        stack_next_slot[0] = adjacency.offset[start];
        discovery_time[start] = low_link[start] = timer++;
        parent_edge[start] = -1;

        while (stack_top >= 0) {
            int node = stack[stack_top];
            if (stack_next_slot[stack_top] < adjacency.offset[node + 1]) {
                int slot = stack_next_slot[stack_top]++;
                int neighbor = adjacency.neighbors[slot];
                int edge_index = adjacency.edge_indices[slot];
                if (edge_index == parent_edge[node]) continue;
                if (discovery_time[neighbor] == -1) {
                    /* Tree edge */
                    edge_stack_from_node[edge_stack_top] = node; edge_stack_to_node[edge_stack_top] = neighbor; edge_stack_top++;
                    discovery_time[neighbor] = low_link[neighbor] = timer++;
                    parent_edge[neighbor] = edge_index;
                    stack_top++;
                    stack[stack_top] = neighbor;
                    stack_next_slot[stack_top] = adjacency.offset[neighbor];
                } else if (discovery_time[neighbor] < discovery_time[node]) {
                    /* Back edge to ancestor */
                    edge_stack_from_node[edge_stack_top] = node; edge_stack_to_node[edge_stack_top] = neighbor; edge_stack_top++;
                    if (low_link[node] > discovery_time[neighbor]) low_link[node] = discovery_time[neighbor];
                }
            } else {
                if (stack_top > 0) {
                    int parent = stack[stack_top - 1];
                    if (low_link[parent] > low_link[node]) low_link[parent] = low_link[node];
                    if (low_link[node] >= discovery_time[parent]) {
                        /* Pop edges to form biconnected component */
                        PyObject *component = PySet_New(NULL);
                        if (!component) {
                            free(scratch); free(edge_stack_scratch);
                            free_adjacency(&adjacency); Py_DECREF(result);
                            return NULL;
                        }
                        while (edge_stack_top > 0) {
                            edge_stack_top--;
                            PyObject *from_node_obj = PyLong_FromLong(edge_stack_from_node[edge_stack_top]);
                            PyObject *to_node_obj = PyLong_FromLong(edge_stack_to_node[edge_stack_top]);
                            PySet_Add(component, from_node_obj);
                            PySet_Add(component, to_node_obj);
                            Py_DECREF(from_node_obj);
                            Py_DECREF(to_node_obj);
                            if (edge_stack_from_node[edge_stack_top] == parent && edge_stack_to_node[edge_stack_top] == node) break;
                        }
                        PyList_Append(result, component);
                        Py_DECREF(component);
                    }
                }
                stack_top--;
            }
        }
    }

    free(scratch); free(edge_stack_scratch);
    free_adjacency(&adjacency);
    return result;
}

/* ── bridges (node-ID edges) ── */

PyObject *py_bridges_by_node_id(PyObject *self, PyObject *args) {
    PyObject *node_ids_obj, *edges_obj;
    if (!PyArg_ParseTuple(args, "OO", &node_ids_obj, &edges_obj)) return NULL;

    NodeIdContext context;
    if (node_id_parse(node_ids_obj, edges_obj, &context) < 0) return NULL;
    int node_count = context.node_count;

    PyObject *result = PyList_New(0);
    if (!result || node_count == 0 || context.edges.edge_count == 0) { node_id_context_free(&context); return result; }

    AdjacencyList adjacency;
    if (build_adjacency(node_count, &context.edges, &adjacency, 0) < 0) { node_id_context_free(&context); Py_DECREF(result); return NULL; }

    int *scratch = (int *)malloc(5 * (size_t)node_count * sizeof(int));
    if (!scratch) {
        free_adjacency(&adjacency); node_id_context_free(&context); Py_DECREF(result);
        PyErr_NoMemory(); return NULL;
    }
    int *discovery_time = scratch, *low_link = scratch + node_count, *stack = scratch + 2*node_count;
    int *stack_next_slot = scratch + 3*node_count, *parent_edge = scratch + 4*node_count;
    memset(discovery_time, -1, node_count * sizeof(int));
    int timer = 0;

    for (int start = 0; start < node_count; start++) {
        if (discovery_time[start] != -1) continue;
        int stack_top = 0;
        stack[0] = start; stack_next_slot[0] = adjacency.offset[start];
        discovery_time[start] = low_link[start] = timer++; parent_edge[start] = -1;
        while (stack_top >= 0) {
            int node = stack[stack_top];
            if (stack_next_slot[stack_top] < adjacency.offset[node + 1]) {
                int slot = stack_next_slot[stack_top]++;
                int neighbor = adjacency.neighbors[slot], edge_index = adjacency.edge_indices[slot];
                if (edge_index == parent_edge[node]) continue;
                if (discovery_time[neighbor] == -1) {
                    discovery_time[neighbor] = low_link[neighbor] = timer++; parent_edge[neighbor] = edge_index;
                    stack[++stack_top] = neighbor; stack_next_slot[stack_top] = adjacency.offset[neighbor];
                } else { if (low_link[node] > discovery_time[neighbor]) low_link[node] = discovery_time[neighbor]; }
            } else {
                if (stack_top > 0) {
                    int parent = stack[stack_top-1];
                    if (low_link[parent] > low_link[node]) low_link[parent] = low_link[node];
                    if (low_link[node] > discovery_time[parent]) {
                        int bridge_edge = parent_edge[node];
                        int bridge_first_node = EDGE_FIRST_ENDPOINT(&context.edges, bridge_edge), bridge_second_node = EDGE_SECOND_ENDPOINT(&context.edges, bridge_edge);
                        PyObject *bridge_pair = PyTuple_New(2);
                        Py_INCREF(context.node_id_items[bridge_first_node]); Py_INCREF(context.node_id_items[bridge_second_node]);
                        PyTuple_SET_ITEM(bridge_pair, 0, context.node_id_items[bridge_first_node]);
                        PyTuple_SET_ITEM(bridge_pair, 1, context.node_id_items[bridge_second_node]);
                        PyList_Append(result, bridge_pair); Py_DECREF(bridge_pair);
                    }
                }
                stack_top--;
            }
        }
    }

    free(scratch);
    free_adjacency(&adjacency); node_id_context_free(&context);
    return result;
}

/* ── articulation_points (node-ID edges) ── */

PyObject *py_articulation_points_by_node_id(PyObject *self, PyObject *args) {
    PyObject *node_ids_obj, *edges_obj;
    if (!PyArg_ParseTuple(args, "OO", &node_ids_obj, &edges_obj)) return NULL;

    NodeIdContext context;
    if (node_id_parse(node_ids_obj, edges_obj, &context) < 0) return NULL;
    int node_count = context.node_count;

    PyObject *result = PySet_New(NULL);
    if (!result || node_count == 0) { node_id_context_free(&context); return result; }

    AdjacencyList adjacency;
    if (build_adjacency(node_count, &context.edges, &adjacency, 0) < 0) { node_id_context_free(&context); Py_DECREF(result); return NULL; }

    int *scratch = (int *)malloc(6 * (size_t)node_count * sizeof(int));
    if (!scratch) {
        free_adjacency(&adjacency); node_id_context_free(&context); Py_DECREF(result);
        PyErr_NoMemory(); return NULL;
    }
    int *discovery_time = scratch, *low_link = scratch + node_count, *stack = scratch + 2*node_count;
    int *stack_next_slot = scratch + 3*node_count, *parent_edge = scratch + 4*node_count, *child_count = scratch + 5*node_count;
    memset(discovery_time, -1, node_count * sizeof(int));
    memset(child_count, 0, node_count * sizeof(int));
    int timer = 0;

    for (int start = 0; start < node_count; start++) {
        if (discovery_time[start] != -1) continue;
        int stack_top = 0;
        stack[0] = start; stack_next_slot[0] = adjacency.offset[start];
        discovery_time[start] = low_link[start] = timer++; parent_edge[start] = -1; child_count[start] = 0;
        while (stack_top >= 0) {
            int node = stack[stack_top];
            if (stack_next_slot[stack_top] < adjacency.offset[node + 1]) {
                int slot = stack_next_slot[stack_top]++;
                int neighbor = adjacency.neighbors[slot], edge_index = adjacency.edge_indices[slot];
                if (edge_index == parent_edge[node]) continue;
                if (discovery_time[neighbor] == -1) {
                    discovery_time[neighbor] = low_link[neighbor] = timer++; parent_edge[neighbor] = edge_index; child_count[neighbor] = 0; child_count[node]++;
                    stack[++stack_top] = neighbor; stack_next_slot[stack_top] = adjacency.offset[neighbor];
                } else { if (low_link[node] > discovery_time[neighbor]) low_link[node] = discovery_time[neighbor]; }
            } else {
                if (stack_top > 0) {
                    int parent = stack[stack_top-1];
                    if (low_link[parent] > low_link[node]) low_link[parent] = low_link[node];
                    if (parent_edge[parent] != -1 && low_link[node] >= discovery_time[parent])
                        PySet_Add(result, context.node_id_items[parent]);
                }
                stack_top--;
            }
        }
        if (child_count[start] >= 2)
            PySet_Add(result, context.node_id_items[start]);
    }

    free(scratch);
    free_adjacency(&adjacency); node_id_context_free(&context);
    return result;
}

/* ── biconnected_components (node-ID edges) ── */

PyObject *py_biconnected_components_by_node_id(PyObject *self, PyObject *args) {
    PyObject *node_ids_obj, *edges_obj;
    if (!PyArg_ParseTuple(args, "OO", &node_ids_obj, &edges_obj)) return NULL;

    NodeIdContext context;
    if (node_id_parse(node_ids_obj, edges_obj, &context) < 0) return NULL;
    int node_count = context.node_count;

    PyObject *result = PyList_New(0);
    if (!result || node_count == 0 || context.edges.edge_count == 0) { node_id_context_free(&context); return result; }

    Py_ssize_t edge_count = context.edges.edge_count;
    AdjacencyList adjacency;
    if (build_adjacency(node_count, &context.edges, &adjacency, 0) < 0) { node_id_context_free(&context); Py_DECREF(result); return NULL; }

    int *scratch = (int *)malloc(5 * (size_t)node_count * sizeof(int));
    int *edge_stack_scratch = (int *)malloc(2 * (size_t)edge_count * sizeof(int));
    if (!scratch || !edge_stack_scratch) {
        free(scratch); free(edge_stack_scratch);
        free_adjacency(&adjacency); node_id_context_free(&context); Py_DECREF(result);
        PyErr_NoMemory(); return NULL;
    }
    int *discovery_time = scratch, *low_link = scratch + node_count, *stack = scratch + 2*node_count;
    int *stack_next_slot = scratch + 3*node_count, *parent_edge = scratch + 4*node_count;
    int *edge_stack_from_node = edge_stack_scratch, *edge_stack_to_node = edge_stack_scratch + edge_count;
    memset(discovery_time, -1, node_count * sizeof(int));
    int timer = 0, edge_stack_top = 0;

    for (int start = 0; start < node_count; start++) {
        if (discovery_time[start] != -1) continue;
        int stack_top = 0;
        stack[0] = start; stack_next_slot[0] = adjacency.offset[start];
        discovery_time[start] = low_link[start] = timer++; parent_edge[start] = -1;
        while (stack_top >= 0) {
            int node = stack[stack_top];
            if (stack_next_slot[stack_top] < adjacency.offset[node + 1]) {
                int slot = stack_next_slot[stack_top]++;
                int neighbor = adjacency.neighbors[slot], edge_index = adjacency.edge_indices[slot];
                if (edge_index == parent_edge[node]) continue;
                if (discovery_time[neighbor] == -1) {
                    edge_stack_from_node[edge_stack_top] = node; edge_stack_to_node[edge_stack_top] = neighbor; edge_stack_top++;
                    discovery_time[neighbor] = low_link[neighbor] = timer++; parent_edge[neighbor] = edge_index;
                    stack[++stack_top] = neighbor; stack_next_slot[stack_top] = adjacency.offset[neighbor];
                } else if (discovery_time[neighbor] < discovery_time[node]) {
                    edge_stack_from_node[edge_stack_top] = node; edge_stack_to_node[edge_stack_top] = neighbor; edge_stack_top++;
                    if (low_link[node] > discovery_time[neighbor]) low_link[node] = discovery_time[neighbor];
                }
            } else {
                if (stack_top > 0) {
                    int parent = stack[stack_top-1];
                    if (low_link[parent] > low_link[node]) low_link[parent] = low_link[node];
                    if (low_link[node] >= discovery_time[parent]) {
                        PyObject *component = PySet_New(NULL);
                        while (edge_stack_top > 0) {
                            edge_stack_top--;
                            PySet_Add(component, context.node_id_items[edge_stack_from_node[edge_stack_top]]);
                            PySet_Add(component, context.node_id_items[edge_stack_to_node[edge_stack_top]]);
                            if (edge_stack_from_node[edge_stack_top] == parent && edge_stack_to_node[edge_stack_top] == node) break;
                        }
                        PyList_Append(result, component); Py_DECREF(component);
                    }
                }
                stack_top--;
            }
        }
    }

    free(scratch); free(edge_stack_scratch);
    free_adjacency(&adjacency); node_id_context_free(&context);
    return result;
}

PyObject *py_bridges_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *edge_mask_obj = Py_None, *node_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OO", &capsule, &edge_mask_obj, &node_mask_obj)) return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    int node_count = graph->node_ids.node_count;

    PyObject *result = PyList_New(0);
    if (!result || node_count == 0 || graph->node_ids.edges.edge_count == 0 || !graph->has_adjacency) return result;

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }

    AdjacencyList *adjacency = &graph->adjacency;
    int *scratch = (int *)malloc(5 * (size_t)node_count * sizeof(int));
    if (!scratch) { release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result); PyErr_NoMemory(); return NULL; }
    int *discovery_time = scratch, *low_link = scratch + node_count, *stack = scratch + 2*node_count;
    int *stack_next_slot = scratch + 3*node_count, *parent_edge = scratch + 4*node_count;
    memset(discovery_time, -1, node_count * sizeof(int));
    int timer = 0;

    for (int start = 0; start < node_count; start++) {
        if (discovery_time[start] != -1) continue;
        if (node_mask && node_mask[start]) continue;
        int stack_top = 0;
        stack[0] = start; stack_next_slot[0] = adjacency->offset[start];
        discovery_time[start] = low_link[start] = timer++; parent_edge[start] = -1;
        while (stack_top >= 0) {
            int node = stack[stack_top];
            if (stack_next_slot[stack_top] < adjacency->offset[node + 1]) {
                int slot = stack_next_slot[stack_top]++;
                int neighbor = adjacency->neighbors[slot], edge_index = adjacency->edge_indices[slot];
                if (edge_mask && edge_mask[edge_index]) continue;
                if (node_mask && node_mask[neighbor]) continue;
                if (edge_index == parent_edge[node]) continue;
                if (discovery_time[neighbor] == -1) {
                    discovery_time[neighbor] = low_link[neighbor] = timer++; parent_edge[neighbor] = edge_index;
                    stack[++stack_top] = neighbor; stack_next_slot[stack_top] = adjacency->offset[neighbor];
                } else { if (low_link[node] > discovery_time[neighbor]) low_link[node] = discovery_time[neighbor]; }
            } else {
                if (stack_top > 0) {
                    int parent = stack[stack_top-1];
                    if (low_link[parent] > low_link[node]) low_link[parent] = low_link[node];
                    if (low_link[node] > discovery_time[parent]) {
                        int bridge_edge = parent_edge[node];
                        int bridge_first_node = EDGE_FIRST_ENDPOINT(&graph->node_ids.edges, bridge_edge), bridge_second_node = EDGE_SECOND_ENDPOINT(&graph->node_ids.edges, bridge_edge);
                        PyObject *bridge_pair = PyTuple_New(2);
                        Py_INCREF(graph->node_ids.node_id_items[bridge_first_node]); Py_INCREF(graph->node_ids.node_id_items[bridge_second_node]);
                        PyTuple_SET_ITEM(bridge_pair, 0, graph->node_ids.node_id_items[bridge_first_node]);
                        PyTuple_SET_ITEM(bridge_pair, 1, graph->node_ids.node_id_items[bridge_second_node]);
                        PyList_Append(result, bridge_pair); Py_DECREF(bridge_pair);
                    }
                }
                stack_top--;
            }
        }
    }
    free(scratch);
    release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    return result;
}

PyObject *py_articulation_points_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *edge_mask_obj = Py_None, *node_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OO", &capsule, &edge_mask_obj, &node_mask_obj)) return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    int node_count = graph->node_ids.node_count;

    PyObject *result = PySet_New(NULL);
    if (!result || node_count == 0 || !graph->has_adjacency) return result;

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }

    AdjacencyList *adjacency = &graph->adjacency;
    int *scratch = (int *)malloc(6 * (size_t)node_count * sizeof(int));
    if (!scratch) { release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result); PyErr_NoMemory(); return NULL; }
    int *discovery_time = scratch, *low_link = scratch + node_count, *stack = scratch + 2*node_count;
    int *stack_next_slot = scratch + 3*node_count, *parent_edge = scratch + 4*node_count, *child_count = scratch + 5*node_count;
    memset(discovery_time, -1, node_count * sizeof(int));
    memset(child_count, 0, node_count * sizeof(int));
    int timer = 0;

    for (int start = 0; start < node_count; start++) {
        if (discovery_time[start] != -1) continue;
        if (node_mask && node_mask[start]) continue;
        int stack_top = 0;
        stack[0] = start; stack_next_slot[0] = adjacency->offset[start];
        discovery_time[start] = low_link[start] = timer++; parent_edge[start] = -1; child_count[start] = 0;
        while (stack_top >= 0) {
            int node = stack[stack_top];
            if (stack_next_slot[stack_top] < adjacency->offset[node + 1]) {
                int slot = stack_next_slot[stack_top]++;
                int neighbor = adjacency->neighbors[slot], edge_index = adjacency->edge_indices[slot];
                if (edge_mask && edge_mask[edge_index]) continue;
                if (node_mask && node_mask[neighbor]) continue;
                if (edge_index == parent_edge[node]) continue;
                if (discovery_time[neighbor] == -1) {
                    discovery_time[neighbor] = low_link[neighbor] = timer++; parent_edge[neighbor] = edge_index; child_count[neighbor] = 0; child_count[node]++;
                    stack[++stack_top] = neighbor; stack_next_slot[stack_top] = adjacency->offset[neighbor];
                } else { if (low_link[node] > discovery_time[neighbor]) low_link[node] = discovery_time[neighbor]; }
            } else {
                if (stack_top > 0) {
                    int parent = stack[stack_top-1];
                    if (low_link[parent] > low_link[node]) low_link[parent] = low_link[node];
                    if (parent_edge[parent] != -1 && low_link[node] >= discovery_time[parent])
                        PySet_Add(result, graph->node_ids.node_id_items[parent]);
                }
                stack_top--;
            }
        }
        if (child_count[start] >= 2)
            PySet_Add(result, graph->node_ids.node_id_items[start]);
    }
    free(scratch);
    release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    return result;
}

PyObject *py_biconnected_components_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *edge_mask_obj = Py_None, *node_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OO", &capsule, &edge_mask_obj, &node_mask_obj)) return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    int node_count = graph->node_ids.node_count;

    PyObject *result = PyList_New(0);
    if (!result || node_count == 0 || graph->node_ids.edges.edge_count == 0 || !graph->has_adjacency) return result;

    Py_ssize_t edge_count = graph->node_ids.edges.edge_count;
    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, edge_count, &edge_mask, &edge_mask_buffer) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }

    AdjacencyList *adjacency = &graph->adjacency;
    int *scratch = (int *)malloc(5 * (size_t)node_count * sizeof(int));
    int *edge_stack_scratch = (int *)malloc(2 * (size_t)edge_count * sizeof(int));
    if (!scratch || !edge_stack_scratch) {
        free(scratch); free(edge_stack_scratch); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result);
        PyErr_NoMemory(); return NULL;
    }
    int *discovery_time = scratch, *low_link = scratch + node_count, *stack = scratch + 2*node_count;
    int *stack_next_slot = scratch + 3*node_count, *parent_edge = scratch + 4*node_count;
    int *edge_stack_from_node = edge_stack_scratch, *edge_stack_to_node = edge_stack_scratch + edge_count;
    memset(discovery_time, -1, node_count * sizeof(int));
    int timer = 0, edge_stack_top = 0;

    for (int start = 0; start < node_count; start++) {
        if (discovery_time[start] != -1) continue;
        if (node_mask && node_mask[start]) continue;
        int stack_top = 0;
        stack[0] = start; stack_next_slot[0] = adjacency->offset[start];
        discovery_time[start] = low_link[start] = timer++; parent_edge[start] = -1;
        while (stack_top >= 0) {
            int node = stack[stack_top];
            if (stack_next_slot[stack_top] < adjacency->offset[node + 1]) {
                int slot = stack_next_slot[stack_top]++;
                int neighbor = adjacency->neighbors[slot], edge_index = adjacency->edge_indices[slot];
                if (edge_mask && edge_mask[edge_index]) continue;
                if (node_mask && node_mask[neighbor]) continue;
                if (edge_index == parent_edge[node]) continue;
                if (discovery_time[neighbor] == -1) {
                    edge_stack_from_node[edge_stack_top] = node; edge_stack_to_node[edge_stack_top] = neighbor; edge_stack_top++;
                    discovery_time[neighbor] = low_link[neighbor] = timer++; parent_edge[neighbor] = edge_index;
                    stack[++stack_top] = neighbor; stack_next_slot[stack_top] = adjacency->offset[neighbor];
                } else if (discovery_time[neighbor] < discovery_time[node]) {
                    edge_stack_from_node[edge_stack_top] = node; edge_stack_to_node[edge_stack_top] = neighbor; edge_stack_top++;
                    if (low_link[node] > discovery_time[neighbor]) low_link[node] = discovery_time[neighbor];
                }
            } else {
                if (stack_top > 0) {
                    int parent = stack[stack_top-1];
                    if (low_link[parent] > low_link[node]) low_link[parent] = low_link[node];
                    if (low_link[node] >= discovery_time[parent]) {
                        PyObject *component = PySet_New(NULL);
                        while (edge_stack_top > 0) {
                            edge_stack_top--;
                            PySet_Add(component, graph->node_ids.node_id_items[edge_stack_from_node[edge_stack_top]]);
                            PySet_Add(component, graph->node_ids.node_id_items[edge_stack_to_node[edge_stack_top]]);
                            if (edge_stack_from_node[edge_stack_top] == parent && edge_stack_to_node[edge_stack_top] == node) break;
                        }
                        PyList_Append(result, component); Py_DECREF(component);
                    }
                }
                stack_top--;
            }
        }
    }
    free(scratch); free(edge_stack_scratch);
    release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    return result;
}

/* bridges_as_edge_indices_ctx: return bridge edge indices for use in masking. */
PyObject *py_bridges_as_edge_indices_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *edge_mask_obj = Py_None, *node_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OO", &capsule, &edge_mask_obj, &node_mask_obj)) return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    int node_count = graph->node_ids.node_count;

    PyObject *result = PyList_New(0);
    if (!result || node_count == 0 || graph->node_ids.edges.edge_count == 0 || !graph->has_adjacency) return result;

    if (graph->directed) {
        Py_DECREF(result);
        PyErr_SetString(PyExc_TypeError, "bridges is not defined for directed graphs");
        return NULL;
    }

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }

    AdjacencyList *adjacency = &graph->adjacency;
    int *scratch = (int *)malloc(5 * (size_t)node_count * sizeof(int));
    if (!scratch) { release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result); PyErr_NoMemory(); return NULL; }
    int *discovery_time = scratch, *low_link = scratch + node_count, *stack = scratch + 2*node_count;
    int *stack_next_slot = scratch + 3*node_count, *parent_edge = scratch + 4*node_count;
    memset(discovery_time, -1, node_count * sizeof(int));
    int timer = 0;

    for (int start = 0; start < node_count; start++) {
        if (discovery_time[start] != -1) continue;
        if (node_mask && node_mask[start]) continue;
        int stack_top = 0;
        stack[0] = start; stack_next_slot[0] = adjacency->offset[start];
        discovery_time[start] = low_link[start] = timer++; parent_edge[start] = -1;
        while (stack_top >= 0) {
            int node = stack[stack_top];
            if (stack_next_slot[stack_top] < adjacency->offset[node + 1]) {
                int slot = stack_next_slot[stack_top]++;
                int neighbor = adjacency->neighbors[slot], edge_index = adjacency->edge_indices[slot];
                if (edge_mask && edge_mask[edge_index]) continue;
                if (node_mask && node_mask[neighbor]) continue;
                if (edge_index == parent_edge[node]) continue;
                if (discovery_time[neighbor] == -1) {
                    discovery_time[neighbor] = low_link[neighbor] = timer++; parent_edge[neighbor] = edge_index;
                    stack[++stack_top] = neighbor; stack_next_slot[stack_top] = adjacency->offset[neighbor];
                } else { if (low_link[node] > discovery_time[neighbor]) low_link[node] = discovery_time[neighbor]; }
            } else {
                if (stack_top > 0) {
                    int parent = stack[stack_top-1];
                    if (low_link[parent] > low_link[node]) low_link[parent] = low_link[node];
                    if (low_link[node] > discovery_time[parent]) {
                        PyObject *bridge_edge_obj = PyLong_FromLong(parent_edge[node]);
                        PyList_Append(result, bridge_edge_obj);
                        Py_DECREF(bridge_edge_obj);
                    }
                }
                stack_top--;
            }
        }
    }
    free(scratch);
    release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    return result;
}
