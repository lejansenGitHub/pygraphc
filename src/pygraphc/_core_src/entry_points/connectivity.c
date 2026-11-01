/* Connected components (index, node-ID and context tiers) and strongly connected components. */
#include "entry_points.h"
#include "../helpers/terminals.h"
#include "../helpers/parse.h"
#include "../helpers/components.h"

/* ── connected_components(n, edges) -> list[set[int]] ── */

PyObject *py_connected_components(PyObject *self, PyObject *args) {
    int node_count;
    PyObject *edges_obj;
    if (!PyArg_ParseTuple(args, "iO", &node_count, &edges_obj)) return NULL;

    EdgeList edges;
    if (parse_edges(edges_obj, &edges) < 0) return NULL;

    ComponentResult components;
    if (compute_components(node_count, &edges, &components) < 0) {
        free_edges(&edges);
        return NULL;
    }
    free_edges(&edges);

    int component_count = components.component_count;

    /* Build sets directly — no intermediate bucketing */
    PyObject *result = PyList_New(component_count);
    if (!result) { free(components.labels); return NULL; }

    PyObject **component_sets = (PyObject **)malloc((size_t)component_count * sizeof(PyObject *));
    if (!component_sets) { free(components.labels); Py_DECREF(result); PyErr_NoMemory(); return NULL; }
    for (int component = 0; component < component_count; component++) {
        component_sets[component] = PySet_New(NULL);
        if (!component_sets[component]) {
            for (int previous_component = 0; previous_component < component; previous_component++) Py_DECREF(component_sets[previous_component]);
            free(component_sets); free(components.labels); Py_DECREF(result);
            return NULL;
        }
    }

    for (int node = 0; node < node_count; node++) {
        PyObject *node_obj = PyLong_FromLong(node);
        PySet_Add(component_sets[components.labels[node]], node_obj);
        Py_DECREF(node_obj);
    }

    for (int component = 0; component < component_count; component++)
        PyList_SET_ITEM(result, component, component_sets[component]);

    free(component_sets);
    free(components.labels);
    return result;
}

/* ── connected_components_with_branches(n, edges, branch_ids) -> list[tuple[set[int], set[int]]] ── */

PyObject *py_connected_components_with_branches(PyObject *self, PyObject *args) {
    int node_count;
    PyObject *edges_obj, *branch_ids_obj;
    if (!PyArg_ParseTuple(args, "iOO", &node_count, &edges_obj, &branch_ids_obj)) return NULL;

    EdgeList edges;
    if (parse_edges(edges_obj, &edges) < 0) return NULL;

    /* Parse branch_ids */
    PyObject *branch_ids_fast = PySequence_Fast(branch_ids_obj, "branch_ids must be a sequence");
    if (!branch_ids_fast) { free_edges(&edges); return NULL; }
    Py_ssize_t branch_id_count = PySequence_Fast_GET_SIZE(branch_ids_fast);

    ComponentResult components;
    if (compute_components(node_count, &edges, &components) < 0) {
        Py_DECREF(branch_ids_fast);
        free_edges(&edges);
        return NULL;
    }

    int component_count = components.component_count;

    /* Bucket nodes */
    int *sizes  = (int *)calloc(component_count, sizeof(int));
    for (int node = 0; node < node_count; node++) sizes[components.labels[node]]++;
    int **buckets = (int **)malloc(component_count * sizeof(int *));
    int *offsets  = (int *)calloc(component_count, sizeof(int));
    for (int component = 0; component < component_count; component++)
        buckets[component] = (int *)malloc(sizes[component] * sizeof(int));
    for (int node = 0; node < node_count; node++) {
        int component = components.labels[node];
        buckets[component][offsets[component]++] = node;
    }

    /* Build branch sets per component */
    PyObject **branch_sets = (PyObject **)calloc(component_count, sizeof(PyObject *));
    for (int component = 0; component < component_count; component++)
        branch_sets[component] = PySet_New(NULL);

    Py_ssize_t edge_branch_count = edges.edge_count < branch_id_count ? edges.edge_count : branch_id_count;
    PyObject **branch_id_items = PySequence_Fast_ITEMS(branch_ids_fast);
    for (Py_ssize_t edge_index = 0; edge_index < edge_branch_count; edge_index++) {
        int component = components.labels[EDGE_FIRST_ENDPOINT(&edges, edge_index)];
        PySet_Add(branch_sets[component], branch_id_items[edge_index]);
    }
    Py_DECREF(branch_ids_fast);
    free_edges(&edges);

    /* Build result list */
    PyObject *result = PyList_New(component_count);
    if (!result) goto cleanup;

    for (int component = 0; component < component_count; component++) {
        PyObject *node_set = PySet_New(NULL);
        for (int member = 0; member < sizes[component]; member++) {
            PyObject *node_obj = PyLong_FromLong(buckets[component][member]);
            PySet_Add(node_set, node_obj);
            Py_DECREF(node_obj);
        }
        PyObject *pair = PyTuple_Pack(2, node_set, branch_sets[component]);
        Py_DECREF(node_set);
        PyList_SET_ITEM(result, component, pair);
    }

cleanup:
    free(components.labels); free(sizes); free(offsets);
    for (int component = 0; component < component_count; component++) {
        free(buckets[component]);
        Py_DECREF(branch_sets[component]);
    }
    free(buckets); free(branch_sets);
    return result;
}

/* ── connected_components_remapped(node_ids, edges) -> list[set[NodeId]] ──
 *
 * Like connected_components but takes the node_ids remapping list and returns
 * sets containing the original node IDs (node_ids[index]) instead of indices.
 * This moves the remapping loop from Python into C.
 */
PyObject *py_connected_components_remapped(PyObject *self, PyObject *args) {
    PyObject *node_ids_obj, *edges_obj;
    if (!PyArg_ParseTuple(args, "OO", &node_ids_obj, &edges_obj)) return NULL;

    PyObject *node_ids_fast = PySequence_Fast(node_ids_obj, "node_ids must be a sequence");
    if (!node_ids_fast) return NULL;
    int node_count = (int)PySequence_Fast_GET_SIZE(node_ids_fast);
    PyObject **node_id_items = PySequence_Fast_ITEMS(node_ids_fast);

    EdgeList edges;
    if (parse_edges(edges_obj, &edges) < 0) { Py_DECREF(node_ids_fast); return NULL; }

    ComponentResult components;
    if (compute_components(node_count, &edges, &components) < 0) {
        free_edges(&edges);
        Py_DECREF(node_ids_fast);
        return NULL;
    }
    free_edges(&edges);

    int component_count = components.component_count;

    /* Bucket nodes by component */
    int *sizes  = (int *)calloc(component_count, sizeof(int));
    for (int node = 0; node < node_count; node++) sizes[components.labels[node]]++;
    int **buckets = (int **)malloc(component_count * sizeof(int *));
    int *offsets  = (int *)calloc(component_count, sizeof(int));
    for (int component = 0; component < component_count; component++)
        buckets[component] = (int *)malloc(sizes[component] * sizeof(int));
    for (int node = 0; node < node_count; node++) {
        int component = components.labels[node];
        buckets[component][offsets[component]++] = node;
    }

    /* Build Python list of sets with remapped node IDs */
    PyObject *result = PyList_New(component_count);
    if (!result) goto cleanup;

    for (int component = 0; component < component_count; component++) {
        PyObject *component_set = PySet_New(NULL);
        if (!component_set) { Py_DECREF(result); result = NULL; goto cleanup; }
        for (int member = 0; member < sizes[component]; member++) {
            /* Use the original node ID from the remapping array */
            PySet_Add(component_set, node_id_items[buckets[component][member]]);
        }
        PyList_SET_ITEM(result, component, component_set);
    }

cleanup:
    free(components.labels); free(sizes); free(offsets);
    for (int component = 0; component < component_count; component++) free(buckets[component]);
    free(buckets);
    Py_DECREF(node_ids_fast);
    return result;
}

/* ── connected_components_with_branches_remapped(node_ids, edges, branch_ids)
 *    -> list[tuple[set[NodeId], set[BranchId]]] ──
 *
 * Like connected_components_with_branches but remaps node indices to original
 * node IDs using the node_ids array.
 */
PyObject *py_connected_components_with_branches_remapped(PyObject *self, PyObject *args) {
    PyObject *node_ids_obj, *edges_obj, *branch_ids_obj;
    if (!PyArg_ParseTuple(args, "OOO", &node_ids_obj, &edges_obj, &branch_ids_obj)) return NULL;

    PyObject *node_ids_fast = PySequence_Fast(node_ids_obj, "node_ids must be a sequence");
    if (!node_ids_fast) return NULL;
    int node_count = (int)PySequence_Fast_GET_SIZE(node_ids_fast);
    PyObject **node_id_items = PySequence_Fast_ITEMS(node_ids_fast);

    EdgeList edges;
    if (parse_edges(edges_obj, &edges) < 0) { Py_DECREF(node_ids_fast); return NULL; }

    PyObject *branch_ids_fast = PySequence_Fast(branch_ids_obj, "branch_ids must be a sequence");
    if (!branch_ids_fast) { free_edges(&edges); Py_DECREF(node_ids_fast); return NULL; }
    Py_ssize_t branch_id_count = PySequence_Fast_GET_SIZE(branch_ids_fast);

    ComponentResult components;
    if (compute_components(node_count, &edges, &components) < 0) {
        Py_DECREF(branch_ids_fast);
        free_edges(&edges);
        Py_DECREF(node_ids_fast);
        return NULL;
    }

    int component_count = components.component_count;

    /* Bucket nodes */
    int *sizes  = (int *)calloc(component_count, sizeof(int));
    for (int node = 0; node < node_count; node++) sizes[components.labels[node]]++;
    int **buckets = (int **)malloc(component_count * sizeof(int *));
    int *offsets  = (int *)calloc(component_count, sizeof(int));
    for (int component = 0; component < component_count; component++)
        buckets[component] = (int *)malloc(sizes[component] * sizeof(int));
    for (int node = 0; node < node_count; node++) {
        int component = components.labels[node];
        buckets[component][offsets[component]++] = node;
    }

    /* Build branch sets per component */
    PyObject **branch_sets = (PyObject **)calloc(component_count, sizeof(PyObject *));
    for (int component = 0; component < component_count; component++)
        branch_sets[component] = PySet_New(NULL);

    Py_ssize_t edge_branch_count = edges.edge_count < branch_id_count ? edges.edge_count : branch_id_count;
    PyObject **branch_id_items = PySequence_Fast_ITEMS(branch_ids_fast);
    for (Py_ssize_t edge_index = 0; edge_index < edge_branch_count; edge_index++) {
        int component = components.labels[EDGE_FIRST_ENDPOINT(&edges, edge_index)];
        PySet_Add(branch_sets[component], branch_id_items[edge_index]);
    }
    Py_DECREF(branch_ids_fast);
    free_edges(&edges);

    /* Build result list with remapped node IDs */
    PyObject *result = PyList_New(component_count);
    if (!result) goto cleanup;

    for (int component = 0; component < component_count; component++) {
        PyObject *node_set = PySet_New(NULL);
        for (int member = 0; member < sizes[component]; member++)
            PySet_Add(node_set, node_id_items[buckets[component][member]]);
        PyObject *pair = PyTuple_Pack(2, node_set, branch_sets[component]);
        Py_DECREF(node_set);
        PyList_SET_ITEM(result, component, pair);
    }

cleanup:
    free(components.labels); free(sizes); free(offsets);
    for (int component = 0; component < component_count; component++) {
        free(buckets[component]);
        Py_DECREF(branch_sets[component]);
    }
    free(buckets); free(branch_sets);
    Py_DECREF(node_ids_fast);
    return result;
}

/* ── connected_components (node-ID edges) ── */

PyObject *py_connected_components_by_node_id(PyObject *self, PyObject *args) {
    PyObject *node_ids_obj, *edges_obj;
    if (!PyArg_ParseTuple(args, "OO", &node_ids_obj, &edges_obj)) return NULL;

    NodeIdContext context;
    if (node_id_parse(node_ids_obj, edges_obj, &context) < 0) return NULL;
    int node_count = context.node_count;
    if (node_count == 0) { node_id_context_free(&context); return PyList_New(0); }

    ComponentResult components;
    if (compute_components(node_count, &context.edges, &components) < 0) { node_id_context_free(&context); return NULL; }

    int component_count = components.component_count;

    /* Build sets directly — no intermediate bucketing */
    PyObject *result = PyList_New(component_count);
    if (!result) { free(components.labels); node_id_context_free(&context); return NULL; }

    PyObject **component_sets = (PyObject **)malloc((size_t)component_count * sizeof(PyObject *));
    if (!component_sets) { free(components.labels); Py_DECREF(result); node_id_context_free(&context); PyErr_NoMemory(); return NULL; }
    for (int component = 0; component < component_count; component++) {
        component_sets[component] = PySet_New(NULL);
        if (!component_sets[component]) {
            for (int previous_component = 0; previous_component < component; previous_component++) Py_DECREF(component_sets[previous_component]);
            free(component_sets); free(components.labels); Py_DECREF(result); node_id_context_free(&context);
            return NULL;
        }
    }

    for (int node = 0; node < node_count; node++)
        PySet_Add(component_sets[components.labels[node]], context.node_id_items[node]);

    for (int component = 0; component < component_count; component++)
        PyList_SET_ITEM(result, component, component_sets[component]);

    free(component_sets);
    free(components.labels);
    node_id_context_free(&context);
    return result;
}

/* ── connected_components (node-ID, split endpoint sequences) ── */

PyObject *py_connected_components_by_node_id_split(PyObject *self, PyObject *args) {
    PyObject *node_ids_obj, *first_endpoints_obj, *second_endpoints_obj;
    if (!PyArg_ParseTuple(args, "OOO", &node_ids_obj, &first_endpoints_obj, &second_endpoints_obj)) return NULL;

    NodeIdContext context;
    if (node_id_parse_split(node_ids_obj, first_endpoints_obj, second_endpoints_obj, &context) < 0) return NULL;
    int node_count = context.node_count;
    if (node_count == 0) { node_id_context_free(&context); return PyList_New(0); }

    ComponentResult components;
    if (compute_components(node_count, &context.edges, &components) < 0) { node_id_context_free(&context); return NULL; }

    int component_count = components.component_count;

    /* Build sets directly — no intermediate bucketing */
    PyObject *result = PyList_New(component_count);
    if (!result) { free(components.labels); node_id_context_free(&context); return NULL; }

    /* Create all sets upfront */
    PyObject **component_sets = (PyObject **)malloc((size_t)component_count * sizeof(PyObject *));
    if (!component_sets) { free(components.labels); Py_DECREF(result); node_id_context_free(&context); PyErr_NoMemory(); return NULL; }
    for (int component = 0; component < component_count; component++) {
        component_sets[component] = PySet_New(NULL);
        if (!component_sets[component]) {
            for (int previous_component = 0; previous_component < component; previous_component++) Py_DECREF(component_sets[previous_component]);
            free(component_sets); free(components.labels); Py_DECREF(result); node_id_context_free(&context);
            return NULL;
        }
    }

    /* Single pass: add each node to its component set */
    for (int node = 0; node < node_count; node++)
        PySet_Add(component_sets[components.labels[node]], context.node_id_items[node]);

    for (int component = 0; component < component_count; component++)
        PyList_SET_ITEM(result, component, component_sets[component]);

    free(component_sets);
    free(components.labels);
    node_id_context_free(&context);
    return result;
}

PyObject *py_connected_components_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *edge_mask_obj = Py_None, *node_mask_obj = Py_None, *terminal_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OOO", &capsule, &edge_mask_obj, &node_mask_obj, &terminal_mask_obj)) return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    int node_count = graph->node_ids.node_count;
    if (node_count == 0) return PyList_New(0);

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) return NULL;
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); return NULL; }

    ComponentResult components;
    TerminalMask terminal_mask;
    if (parse_terminal_mask(graph, terminal_mask_obj, edge_mask, &terminal_mask) < 0) {
        release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); return NULL;
    }
    int status = compute_graph_components(graph, edge_mask, &terminal_mask, node_mask, &components);
    release_terminal_mask(&terminal_mask);
    if (status < 0) { release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); return NULL; }
    release_mask(&edge_mask_buffer);
    int component_count = components.component_count;

    /* Build sets, skipping masked nodes */
    PyObject **component_sets = (PyObject **)malloc((size_t)component_count * sizeof(PyObject *));
    if (!component_sets) { free(components.labels); release_mask(&node_mask_buffer); PyErr_NoMemory(); return NULL; }
    for (int component = 0; component < component_count; component++) {
        component_sets[component] = PySet_New(NULL);
        if (!component_sets[component]) {
            for (int previous_component = 0; previous_component < component; previous_component++) Py_DECREF(component_sets[previous_component]);
            free(component_sets); free(components.labels); release_mask(&node_mask_buffer); return NULL;
        }
    }
    for (int node = 0; node < node_count; node++) {
        if (node_mask && node_mask[node]) continue;
        PySet_Add(component_sets[components.labels[node]], graph->node_ids.node_id_items[node]);
    }

    /* Collect non-empty sets */
    PyObject *result = PyList_New(0);
    if (!result) {
        for (int component = 0; component < component_count; component++) Py_DECREF(component_sets[component]);
        free(component_sets); free(components.labels); release_mask(&node_mask_buffer); return NULL;
    }
    for (int component = 0; component < component_count; component++) {
        if (PySet_GET_SIZE(component_sets[component]) > 0)
            PyList_Append(result, component_sets[component]);
        Py_DECREF(component_sets[component]);
    }
    free(component_sets); free(components.labels);
    release_mask(&node_mask_buffer);
    return result;
}

/* ── connected_components_with_branches_ctx(capsule, branch_ids[, excluded_edges, excluded_nodes])
 *    -> list[tuple[set[NodeId], set[BranchId]]]
 *
 * Connected components with branch IDs from a cached graph context.
 *
 * excluded_edges: edges with excluded_edges[i] != 0 are removed from
 *     connectivity and their branch IDs are dropped.
 * excluded_nodes: removed from the output node sets but their edges
 *     still contribute to connectivity and branch ID sets.
 */
/*
 * Add each edge's branch id to the branch set of the component it belongs to
 * (first_active_terminal_node). Instantiated per presence of masks: without one,
 * every terminal is active and each edge sits in the edge list at the node of its
 * first terminal, so it belongs to the component of its first endpoint.
 */
static inline __attribute__((always_inline)) void add_branch_ids(
    const GraphContext *graph, const int *labels, PyObject **branch_sets, PyObject **branch_id_items,
    Py_ssize_t edge_branch_count, const uint8_t *edge_mask, const uint8_t *terminal_mask, const uint8_t *node_mask,
    const int has_masks) {
    const EdgeList *edges = &graph->node_ids.edges;
    for (Py_ssize_t edge_index = 0; edge_index < edge_branch_count; edge_index++) {
        int member_node = has_masks ? first_active_terminal_node(graph, (int)edge_index, edge_mask, terminal_mask, node_mask)
                                    : EDGE_FIRST_ENDPOINT(edges, edge_index);
        if (has_masks && member_node < 0) continue;
        PySet_Add(branch_sets[labels[member_node]], branch_id_items[edge_index]);
    }
}

PyObject *py_connected_components_with_branches_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *branch_ids_obj, *edge_mask_obj = Py_None, *node_mask_obj = Py_None, *terminal_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OO|OOO", &capsule, &branch_ids_obj, &edge_mask_obj, &node_mask_obj, &terminal_mask_obj))
        return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    int node_count = graph->node_ids.node_count;

    PyObject *branch_ids_fast = PySequence_Fast(branch_ids_obj, "branch_ids must be a sequence");
    if (!branch_ids_fast) return NULL;
    Py_ssize_t branch_id_count = PySequence_Fast_GET_SIZE(branch_ids_fast);

    /* Parse excluded-edges mask */
    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) {
        Py_DECREF(branch_ids_fast); return NULL;
    }

    /* Parse excluded-nodes mask (used only for output filtering) */
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) {
        release_mask(&edge_mask_buffer); Py_DECREF(branch_ids_fast); return NULL;
    }

    /* Run union-find with both edge and node exclusion — excluded nodes do NOT
       participate in connectivity (traversal stops at them). Their incident
       edges still contribute branch IDs to the CC of the non-excluded endpoint. */
    ComponentResult components;
    TerminalMask terminal_mask;
    if (parse_terminal_mask(graph, terminal_mask_obj, edge_mask, &terminal_mask) < 0) {
        release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(branch_ids_fast);
        return NULL;
    }
    int status = compute_graph_components(graph, edge_mask, &terminal_mask, node_mask, &components);
    if (status < 0) {
        release_terminal_mask(&terminal_mask);
        release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); Py_DECREF(branch_ids_fast);
        return NULL;
    }
    /* Keep edge_mask_buffer alive — reused for branch assignment below */

    int component_count = components.component_count;

    /* Bucket nodes by component */
    int *sizes  = (int *)calloc(component_count, sizeof(int));
    int **buckets = (int **)malloc(component_count * sizeof(int *));
    int *offsets  = (int *)calloc(component_count, sizeof(int));
    if (!sizes || !buckets || !offsets) {
        free(sizes); free(buckets); free(offsets);
        free(components.labels); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
        Py_DECREF(branch_ids_fast);
        PyErr_NoMemory(); return NULL;
    }
    for (int node = 0; node < node_count; node++) sizes[components.labels[node]]++;
    for (int component = 0; component < component_count; component++) {
        buckets[component] = (int *)malloc(sizes[component] * sizeof(int));
        if (!buckets[component]) {
            for (int previous_component = 0; previous_component < component; previous_component++) free(buckets[previous_component]);
            free(sizes); free(buckets); free(offsets);
            free(components.labels); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
            Py_DECREF(branch_ids_fast);
            PyErr_NoMemory(); return NULL;
        }
    }
    for (int node = 0; node < node_count; node++) {
        int component = components.labels[node];
        buckets[component][offsets[component]++] = node;
    }

    /* Build branch sets per component (skipping excluded edges).
     *
     * Performance note: at 1M nodes / 1.5M edges, the branch set construction
     * adds ~60ms on top of plain CC's ~30ms. This is inherent — 1.5M PySet_Add
     * calls for branch assignment plus 54K PySet_New for branch sets. The
     * CPython PySet_Add hash-insert is the floor; no algorithmic improvement
     * possible without bypassing Python set objects entirely.
     */
    PyObject **branch_sets = (PyObject **)calloc(component_count, sizeof(PyObject *));
    if (!branch_sets) {
        for (int component = 0; component < component_count; component++) free(buckets[component]);
        free(sizes); free(buckets); free(offsets);
        free(components.labels); release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer);
        Py_DECREF(branch_ids_fast);
        PyErr_NoMemory(); return NULL;
    }
    for (int component = 0; component < component_count; component++)
        branch_sets[component] = PySet_New(NULL);

    EdgeList *edges = &graph->node_ids.edges;
    Py_ssize_t edge_branch_count = edges->edge_count < branch_id_count ? edges->edge_count : branch_id_count;
    PyObject **branch_id_items = PySequence_Fast_ITEMS(branch_ids_fast);

    if (edge_mask || terminal_mask.terminals || node_mask)
        add_branch_ids(graph, components.labels, branch_sets, branch_id_items, edge_branch_count, edge_mask, terminal_mask.terminals, node_mask, 1);
    else
        add_branch_ids(graph, components.labels, branch_sets, branch_id_items, edge_branch_count, NULL, NULL, NULL, 0);
    release_terminal_mask(&terminal_mask);
    release_mask(&edge_mask_buffer);
    Py_DECREF(branch_ids_fast);

    /* Build result list with remapped node IDs, filtering excluded nodes */
    PyObject *result = PyList_New(0);
    if (!result) goto cleanup;

    for (int component = 0; component < component_count; component++) {
        PyObject *node_set = PySet_New(NULL);
        for (int member = 0; member < sizes[component]; member++) {
            int idx = buckets[component][member];
            if (node_mask && node_mask[idx]) continue;
            PySet_Add(node_set, graph->node_ids.node_id_items[idx]);
        }
        /* A component without an unmasked node holds no edge either: an edge belongs to the component of an active terminal. */
        if (PySet_GET_SIZE(node_set) > 0) {
            PyObject *pair = PyTuple_Pack(2, node_set, branch_sets[component]);
            PyList_Append(result, pair);
            Py_DECREF(pair);
        }
        Py_DECREF(node_set);
    }

cleanup:
    free(components.labels); free(sizes); free(offsets);
    for (int component = 0; component < component_count; component++) {
        free(buckets[component]);
        Py_DECREF(branch_sets[component]);
    }
    free(buckets); free(branch_sets);
    release_mask(&node_mask_buffer);
    return result;
}

PyObject *py_strongly_connected_components_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *edge_mask_obj = Py_None, *node_mask_obj = Py_None, *terminal_mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OOO", &capsule, &edge_mask_obj, &node_mask_obj, &terminal_mask_obj)) return NULL;
    GraphContext *graph = get_graph_context(capsule);
    if (!graph) return NULL;
    int node_count = graph->node_ids.node_count;
    if (node_count == 0) return PyList_New(0);

    if (!graph->directed) {
        PyErr_SetString(PyExc_TypeError,
            "strongly_connected_components requires a directed graph");
        return NULL;
    }

    /* No adjacency list => every node is its own SCC */
    if (!graph->has_adjacency) {
        const uint8_t *node_mask = NULL; Py_buffer node_mask_buffer;
        if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) return NULL;
        PyObject *result = PyList_New(0);
        if (!result) { release_mask(&node_mask_buffer); return NULL; }
        for (int node = 0; node < node_count; node++) {
            if (node_mask && node_mask[node]) continue;
            PyObject *component_set = PySet_New(NULL);
            PySet_Add(component_set, graph->node_ids.node_id_items[node]);
            PyList_Append(result, component_set);
            Py_DECREF(component_set);
        }
        release_mask(&node_mask_buffer);
        return result;
    }

    const uint8_t *edge_mask; Py_buffer edge_mask_buffer;
    if (parse_mask(edge_mask_obj, graph->node_ids.edges.edge_count, &edge_mask, &edge_mask_buffer) < 0) return NULL;
    const uint8_t *node_mask; Py_buffer node_mask_buffer;
    if (parse_mask(node_mask_obj, node_count, &node_mask, &node_mask_buffer) < 0) { release_mask(&edge_mask_buffer); return NULL; }
    TerminalMask terminal_mask;
    if (parse_terminal_mask(graph, terminal_mask_obj, edge_mask, &terminal_mask) < 0) {
        release_mask(&node_mask_buffer); release_mask(&edge_mask_buffer); return NULL;
    }

    ComponentResult components;
    int status = compute_strongly_connected_components(graph, edge_mask, &terminal_mask, node_mask, &components);
    release_terminal_mask(&terminal_mask);
    release_mask(&edge_mask_buffer);
    if (status < 0) { release_mask(&node_mask_buffer); return NULL; }
    int component_count = components.component_count;

    /* Build sets, skipping masked nodes */
    PyObject **component_sets = (PyObject **)malloc((size_t)component_count * sizeof(PyObject *));
    if (!component_sets) { free(components.labels); release_mask(&node_mask_buffer); PyErr_NoMemory(); return NULL; }
    for (int component = 0; component < component_count; component++) {
        component_sets[component] = PySet_New(NULL);
        if (!component_sets[component]) {
            for (int previous_component = 0; previous_component < component; previous_component++) Py_DECREF(component_sets[previous_component]);
            free(component_sets); free(components.labels); release_mask(&node_mask_buffer); return NULL;
        }
    }
    for (int node = 0; node < node_count; node++) {
        if (node_mask && node_mask[node]) continue;
        PySet_Add(component_sets[components.labels[node]], graph->node_ids.node_id_items[node]);
    }

    /* Collect non-empty sets */
    PyObject *result = PyList_New(0);
    if (!result) {
        for (int component = 0; component < component_count; component++) Py_DECREF(component_sets[component]);
        free(component_sets); free(components.labels); release_mask(&node_mask_buffer); return NULL;
    }
    for (int component = 0; component < component_count; component++) {
        if (PySet_GET_SIZE(component_sets[component]) > 0)
            PyList_Append(result, component_sets[component]);
        Py_DECREF(component_sets[component]);
    }
    free(component_sets); free(components.labels);
    release_mask(&node_mask_buffer);
    return result;
}
