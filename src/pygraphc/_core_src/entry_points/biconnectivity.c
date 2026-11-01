/* Bridges, articulation points and biconnected components (iterative Tarjan). */
#include "entry_points.h"
#include "../conversion/parse.h"
#include "../conversion/parse_terminals.h"
#include "../algorithms/adjacency.h"

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
    if (build_adjacency(node_count, &edges, NULL, &adjacency, 0) < 0) {
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
    if (build_adjacency(node_count, &edges, NULL, &adjacency, 0) < 0) {
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
    if (build_adjacency(node_count, &edges, NULL, &adjacency, 0) < 0) {
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
    if (build_adjacency(node_count, &context.edges, NULL, &adjacency, 0) < 0) { node_id_context_free(&context); Py_DECREF(result); return NULL; }

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
    if (build_adjacency(node_count, &context.edges, NULL, &adjacency, 0) < 0) { node_id_context_free(&context); Py_DECREF(result); return NULL; }

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
    if (build_adjacency(node_count, &context.edges, NULL, &adjacency, 0) < 0) { node_id_context_free(&context); Py_DECREF(result); return NULL; }

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

/*
 * ── One Tarjan search for the parsed graph ──
 *
 * Depth-first search over the search states of a graph: its nodes and, with
 * has_multi_terminal_edges, one state per edge with three or more terminals
 * (node_count + e), linked to the node of each of its active terminals. A link is
 * a two-terminal edge (its edge index) or one terminal (edge_count + terminal), so a
 * search never walks back over the link it came in by, while a second terminal of
 * one edge at the same node is a back link (case 7). The mode decides what is
 * reported; every mode and both values of has_multi_terminal_edges are separate
 * instantiations, so graphs of two-terminal edges run the search as before.
 *
 * - critical two-terminal edge: a tree link between two nodes with
 *   low[child] > discovery[parent] (a bridge);
 * - critical multi-terminal edge: an edge state with a child node c and
 *   low[c] >= discovery[edge state], i.e. removing the edge separates its terminals;
 * - critical node: as in Tarjan, counting only children whose subtree holds a
 *   node, so an edge with one active terminal never makes its node critical;
 * - blocks: closed only below a node, never below an edge state, so the blocks
 *   meeting at a multi-terminal edge merge into one (D14); a block with fewer than
 *   two nodes joins nothing and is not reported;
 * - simple path blocks: the blocks of the search states themselves, closed
 *   below every state, reported as (list of node sets, block index -> its
 *   multi-terminal edges), so a multi-terminal edge can separate them as a node does;
 *   without multi-terminal edges they are the blocks, collected the same way. A simple path visits
 *   each node and each multi-terminal edge at most once, so the block-cut tree of
 *   these blocks gives the nodes on simple paths.
 */

enum { REPORT_BRIDGE_PAIRS, REPORT_CRITICAL_EDGES, REPORT_CRITICAL_NODES, REPORT_BLOCKS, REPORT_SIMPLE_PATH_BLOCKS };

static inline __attribute__((always_inline)) int report_bridge_pair(const GraphContext *graph, int edge_index,
                                                                     PyObject *result) {
    int first_node = EDGE_FIRST_ENDPOINT(&graph->node_ids.edges, edge_index);
    int second_node = EDGE_SECOND_ENDPOINT(&graph->node_ids.edges, edge_index);
    PyObject *pair = PyTuple_Pack(2, graph->node_ids.node_id_items[first_node], graph->node_ids.node_id_items[second_node]);
    if (!pair) return -1;
    int status = PyList_Append(result, pair);
    Py_DECREF(pair);
    return status;
}

static inline __attribute__((always_inline)) int report_edge_index(int edge_index, PyObject *result) {
    PyObject *edge_index_obj = PyLong_FromLong(edge_index);
    if (!edge_index_obj) return -1;
    int status = PyList_Append(result, edge_index_obj);
    Py_DECREF(edge_index_obj);
    return status;
}

/*
 * Pop the links of the block below (parent, state), append its node ids as a
 * set to the list result[0] and, when it holds multi-terminal edges, map its index
 * to their edge indices in the dict result[1]. Returns 0, or -1 with an
 * exception set.
 */
static int report_simple_path_block(const GraphContext *graph, const int *link_from, const int *link_to,
                                    int *link_top, int parent, int state, PyObject *result) {
    int node_count = graph->node_ids.node_count;
    PyObject *blocks = PyTuple_GET_ITEM(result, 0), *edge_indices_by_block = PyTuple_GET_ITEM(result, 1);
    PyObject *block_nodes = PySet_New(NULL), *block_edges = NULL;
    if (!block_nodes) return -1;
    while (*link_top > 0) {
        (*link_top)--;
        int from = link_from[*link_top], to = link_to[*link_top];
        int ends[2] = {from, to};
        for (int end = 0; end < 2; end++) {
            if (ends[end] < node_count) {
                if (PySet_Add(block_nodes, graph->node_ids.node_id_items[ends[end]]) < 0) goto failed;
                continue;
            }
            if (!block_edges && !(block_edges = PySet_New(NULL))) goto failed;
            PyObject *edge_index_obj = PyLong_FromLong(ends[end] - node_count);
            if (!edge_index_obj) goto failed;
            int status = PySet_Add(block_edges, edge_index_obj);
            Py_DECREF(edge_index_obj);
            if (status < 0) goto failed;
        }
        if (from == parent && to == state) break;
    }
    if (PyList_Append(blocks, block_nodes) < 0) goto failed;
    Py_DECREF(block_nodes);
    if (!block_edges) return 0;
    PyObject *block_index_obj = PyLong_FromSsize_t(PyList_GET_SIZE(blocks) - 1);
    int status = block_index_obj ? PyDict_SetItem(edge_indices_by_block, block_index_obj, block_edges) : -1;
    Py_XDECREF(block_index_obj); Py_DECREF(block_edges);
    return status;
failed:
    Py_DECREF(block_nodes); Py_XDECREF(block_edges);
    return -1;
}

static inline __attribute__((always_inline)) int tarjan_search(
    const GraphContext *graph, const uint8_t *two_terminal_edge_mask, const uint8_t *edge_mask, const uint8_t *terminal_mask,
    const uint8_t *node_mask, PyObject *result, const int mode, const int has_multi_terminal_edges) {
    const int *slot_offsets = graph->adjacency.offset, *slot_neighbors = graph->adjacency.neighbors;
    const int *slot_edge_indices = graph->adjacency.edge_indices;
    const TerminalList *terminals = graph->terminals;
    int node_count = graph->node_ids.node_count;
    int edge_count = (int)graph->node_ids.edges.edge_count;
    size_t state_count = (size_t)node_count + (has_multi_terminal_edges ? (size_t)edge_count : 0);
    size_t link_capacity = (size_t)edge_count + (has_multi_terminal_edges ? (size_t)terminals->terminal_count : 0);

    int *scratch = (int *)malloc(6 * state_count * sizeof(int));
    const int collects_links = mode == REPORT_BLOCKS || mode == REPORT_SIMPLE_PATH_BLOCKS;
    int *link_stack = collects_links ? (int *)malloc(2 * (link_capacity > 0 ? link_capacity : 1) * sizeof(int)) : NULL;
    uint8_t *critical_edges = (has_multi_terminal_edges && mode == REPORT_CRITICAL_EDGES) ? (uint8_t *)calloc((size_t)edge_count + 1, 1) : NULL;
    if (!scratch || (collects_links && !link_stack) || (has_multi_terminal_edges && mode == REPORT_CRITICAL_EDGES && !critical_edges)) {
        free(scratch); free(link_stack); free(critical_edges);
        PyErr_NoMemory();
        return -1;
    }
    int *discovery_time = scratch, *low_link = scratch + state_count, *stack = scratch + 2 * state_count;
    int *stack_next_slot = scratch + 3 * state_count, *parent_link = scratch + 4 * state_count;
    int *child_count = scratch + 5 * state_count;
    int *link_from = link_stack, *link_to = link_stack ? link_stack + link_capacity : NULL;
    memset(discovery_time, -1, state_count * sizeof(int));
    const int counts_children = mode == REPORT_CRITICAL_NODES || has_multi_terminal_edges;
    int timer = 0, link_top = 0;

    for (int start = 0; start < node_count; start++) {
        if (discovery_time[start] != -1) continue;
        if (node_mask && node_mask[start]) continue;
        int stack_top = 0;
        stack[0] = start; stack_next_slot[0] = slot_offsets[start];
        discovery_time[start] = low_link[start] = timer++; parent_link[start] = -1;
        if (counts_children) child_count[start] = 0;
        while (stack_top >= 0) {
            int state = stack[stack_top];
            int is_edge_state = has_multi_terminal_edges && state >= node_count;
            int end_slot = is_edge_state ? terminals->terminal_offsets[state - node_count + 1] : slot_offsets[state + 1];
            if (stack_next_slot[stack_top] < end_slot) {
                int slot = stack_next_slot[stack_top]++;
                int neighbor, link;
                if (is_edge_state) {
                    int edge = state - node_count;
                    if (!terminal_is_active(terminals, edge, slot, edge_mask, terminal_mask, node_mask)) continue;
                    neighbor = terminals->terminal_nodes[slot];
                    link = edge_count + slot;
                } else {
                    neighbor = slot_neighbors[slot];
                    int edge_index = slot_edge_indices[slot];
                    if (has_multi_terminal_edges && neighbor < 0) {
                        int edge = -1 - neighbor;
                        if (!terminal_is_active(terminals, edge, edge_index, edge_mask, terminal_mask, node_mask)) continue;
                        neighbor = node_count + edge;
                        link = edge_count + edge_index;
                    } else {
                        if (two_terminal_edge_mask && two_terminal_edge_mask[edge_index]) continue;
                        if (node_mask && node_mask[neighbor]) continue;
                        link = edge_index;
                    }
                }
                if (link == parent_link[state]) continue;
                if (discovery_time[neighbor] == -1) {
                    if (collects_links) { link_from[link_top] = state; link_to[link_top] = neighbor; link_top++; }
                    discovery_time[neighbor] = low_link[neighbor] = timer++; parent_link[neighbor] = link;
                    if (counts_children) { child_count[neighbor] = 0; if (!has_multi_terminal_edges) child_count[state]++; }
                    stack[++stack_top] = neighbor;
                    stack_next_slot[stack_top] = (has_multi_terminal_edges && neighbor >= node_count)
                        ? terminals->terminal_offsets[neighbor - node_count] : slot_offsets[neighbor];
                } else if (!collects_links || discovery_time[neighbor] < discovery_time[state]) {
                    if (collects_links) { link_from[link_top] = state; link_to[link_top] = neighbor; link_top++; }
                    if (low_link[state] > discovery_time[neighbor]) low_link[state] = discovery_time[neighbor];
                }
                continue;
            }
            /* state is finished */
            if (stack_top == 0) {
                if (mode == REPORT_CRITICAL_NODES && child_count[state] >= 2 &&
                    PySet_Add(result, graph->node_ids.node_id_items[state]) < 0) goto failed;
                stack_top--;
                continue;
            }
            int parent = stack[stack_top - 1];
            int parent_is_edge_state = has_multi_terminal_edges && parent >= node_count;
            int subtree_holds_node = !is_edge_state || child_count[state] > 0;
            if (low_link[parent] > low_link[state]) low_link[parent] = low_link[state];
            if (has_multi_terminal_edges && subtree_holds_node) child_count[parent]++;
            if (mode == REPORT_BRIDGE_PAIRS || mode == REPORT_CRITICAL_EDGES) {
                if (!is_edge_state && !parent_is_edge_state && low_link[state] > discovery_time[parent] &&
                    (mode == REPORT_BRIDGE_PAIRS ? report_bridge_pair(graph, parent_link[state], result)
                                                 : report_edge_index(parent_link[state], result)) < 0) goto failed;
                if (has_multi_terminal_edges && mode == REPORT_CRITICAL_EDGES) {
                    if (parent_is_edge_state && low_link[state] >= discovery_time[parent]) critical_edges[parent - node_count] = 1;
                    if (is_edge_state && critical_edges[state - node_count] &&
                        report_edge_index(state - node_count, result) < 0) goto failed;
                }
            } else if (mode == REPORT_CRITICAL_NODES) {
                if (!parent_is_edge_state && parent_link[parent] != -1 && subtree_holds_node &&
                    low_link[state] >= discovery_time[parent] &&
                    PySet_Add(result, graph->node_ids.node_id_items[parent]) < 0) goto failed;
            } else if (mode == REPORT_SIMPLE_PATH_BLOCKS && has_multi_terminal_edges) {
                if (low_link[state] >= discovery_time[parent] &&
                    report_simple_path_block(graph, link_from, link_to, &link_top, parent, state, result) < 0) goto failed;
            } else if (!parent_is_edge_state && low_link[state] >= discovery_time[parent]) {
                PyObject *block = PySet_New(NULL);
                if (!block) goto failed;
                PyObject *const *node_id_items = graph->node_ids.node_id_items;
                while (link_top > 0) {
                    link_top--;
                    int from = link_from[link_top], to = link_to[link_top];
                    if (!has_multi_terminal_edges || from < node_count) PySet_Add(block, node_id_items[from]);
                    if (!has_multi_terminal_edges || to < node_count) PySet_Add(block, node_id_items[to]);
                    if (from == parent && to == state) break;
                }
                PyObject *blocks = mode == REPORT_SIMPLE_PATH_BLOCKS ? PyTuple_GET_ITEM(result, 0) : result;
                int append_status = (!has_multi_terminal_edges || PySet_GET_SIZE(block) >= 2) ? PyList_Append(blocks, block) : 0;
                Py_DECREF(block);
                if (append_status < 0) goto failed;
            }
            stack_top--;
        }
    }
    free(scratch); free(link_stack); free(critical_edges);
    return 0;
failed:
    free(scratch); free(link_stack); free(critical_edges);
    return -1;
}

/*
 * Parse (capsule[, edge_mask, node_mask, terminal_mask]) and run tarjan_search in
 * the given mode, instantiated for the graph's kind of edges.
 */
static PyObject *tarjan_entry(PyObject *args, const int mode) {
    PyObject *capsule, *edge_mask_obj = Py_None, *node_mask_obj = Py_None, *terminal_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OOO", &capsule, &edge_mask_obj, &node_mask_obj, &terminal_mask_obj)) return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    if (mode == REPORT_BRIDGE_PAIRS && graph->terminals) {
        PyErr_SetString(PyExc_NotImplementedError,
                        "bridges names edges by node pair, which cannot name an edge with more than two terminals; "
                        "use critical_edges()");
        return NULL;
    }
    if (graph->directed) {
        PyErr_SetString(PyExc_TypeError, "critical edges, critical nodes and blocks are not defined for directed graphs");
        return NULL;
    }
    PyObject *result = mode == REPORT_CRITICAL_NODES ? PySet_New(NULL)
                     : mode == REPORT_SIMPLE_PATH_BLOCKS ? Py_BuildValue("(NN)", PyList_New(0), PyDict_New())
                     : PyList_New(0);
    if (!result || graph->node_ids.node_count == 0 || !graph->has_adjacency) return result;

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, graph->node_ids.node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }
    TerminalMask terminal_mask;
    if (parse_terminal_mask(graph, terminal_mask_obj, edge_mask, &terminal_mask) < 0) { release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(result); return NULL; }
    const uint8_t *traversal_edge_mask = two_terminal_edge_mask(&terminal_mask, edge_mask);

    int status;
    switch (mode) {
    case REPORT_BRIDGE_PAIRS:
        status = tarjan_search(graph, traversal_edge_mask, NULL, NULL, node_mask, result, REPORT_BRIDGE_PAIRS, 0);
        break;
    case REPORT_CRITICAL_EDGES:
        status = graph->terminals
            ? tarjan_search(graph, traversal_edge_mask, edge_mask, terminal_mask.terminals, node_mask, result, REPORT_CRITICAL_EDGES, 1)
            : tarjan_search(graph, traversal_edge_mask, NULL, NULL, node_mask, result, REPORT_CRITICAL_EDGES, 0);
        break;
    case REPORT_CRITICAL_NODES:
        status = graph->terminals
            ? tarjan_search(graph, traversal_edge_mask, edge_mask, terminal_mask.terminals, node_mask, result, REPORT_CRITICAL_NODES, 1)
            : tarjan_search(graph, traversal_edge_mask, NULL, NULL, node_mask, result, REPORT_CRITICAL_NODES, 0);
        break;
    case REPORT_SIMPLE_PATH_BLOCKS:
        status = graph->terminals
            ? tarjan_search(graph, traversal_edge_mask, edge_mask, terminal_mask.terminals, node_mask, result, REPORT_SIMPLE_PATH_BLOCKS, 1)
            : tarjan_search(graph, traversal_edge_mask, NULL, NULL, node_mask, result, REPORT_SIMPLE_PATH_BLOCKS, 0);
        break;
    default:
        status = graph->terminals
            ? tarjan_search(graph, traversal_edge_mask, edge_mask, terminal_mask.terminals, node_mask, result, REPORT_BLOCKS, 1)
            : tarjan_search(graph, traversal_edge_mask, NULL, NULL, node_mask, result, REPORT_BLOCKS, 0);
    }
    release_terminal_mask(&terminal_mask); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
    if (status < 0) { Py_DECREF(result); return NULL; }
    return result;
}

PyObject *py_bridges_ctx(PyObject *self, PyObject *args) { return tarjan_entry(args, REPORT_BRIDGE_PAIRS); }
PyObject *py_critical_edges_ctx(PyObject *self, PyObject *args) { return tarjan_entry(args, REPORT_CRITICAL_EDGES); }
PyObject *py_articulation_points_ctx(PyObject *self, PyObject *args) { return tarjan_entry(args, REPORT_CRITICAL_NODES); }
PyObject *py_biconnected_components_ctx(PyObject *self, PyObject *args) { return tarjan_entry(args, REPORT_BLOCKS); }
PyObject *py_simple_path_blocks_ctx(PyObject *self, PyObject *args) { return tarjan_entry(args, REPORT_SIMPLE_PATH_BLOCKS); }
