/*
 * DAG structure learning via hill-climb search with K2 scoring.
 *
 * Implements the same algorithm as pgmpy's HillClimbSearch.estimate()
 * with K2 scoring, but replaces the pandas/numpy hot path with pure C.
 *
 * The algorithm:
 *   1. Start with an empty DAG (no edges)
 *   2. Each iteration, evaluate all legal single-edge modifications
 *      (add, remove, flip) and pick the one with the highest K2 score delta
 *   3. Stop when no modification improves the score by more than epsilon
 *
 * Complexity per iteration: O(variable_count^2 * sample_count)
 * Total: O(max_iterations * variable_count^2 * sample_count)
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── Data structures ── */

/* Flat dataset: values[sample * variable_count + variable] = value */
typedef struct {
    int *values;        /* sample_count * variable_count */
    int sample_count;
    int variable_count;
    int *cardinalities; /* cardinality per variable */
} Dataset;

/* DAG adjacency: parent lists per node */
typedef struct {
    int *parents;        /* parents[node * max_parents .. + parent_counts[node]) */
    int *parent_counts;  /* number of parents per node */
    int max_parents;     /* max_indegree */
    int node_count;
} DAG;

/* Tabu list (circular buffer of operations) */
typedef struct {
    int *operations;     /* operation, from_node, to_node per entry */
    int capacity;
    int head;
    int count;
} TabuList;

enum { OPERATION_ADD = 0, OPERATION_REMOVE = 1, OPERATION_FLIP = 2 };

/* ── State counts (the hot path) ── */

/*
 * Count co-occurrences of (parent_config, child_value) in the dataset.
 *
 * parent_config = mixed-radix encoding of parent values.
 * Returns a flat array of size parent_configuration_count * child_cardinality.
 * counts[parent_config * child_cardinality + child_value] = count
 */
static int *compute_state_counts(
    const Dataset *dataset,
    int child,
    const int *parent_indices,
    int parent_count,
    int *out_parent_configuration_count)
{
    /* Compute number of parent configurations */
    int parent_configuration_count = 1;
    for (int parent_position = 0; parent_position < parent_count; parent_position++) {
        parent_configuration_count *= dataset->cardinalities[parent_indices[parent_position]];
    }
    *out_parent_configuration_count = parent_configuration_count;

    int child_cardinality = dataset->cardinalities[child];
    int table_size = parent_configuration_count * child_cardinality;

    int *counts = (int *)calloc(table_size, sizeof(int));
    if (!counts) return NULL;

    /* Precompute mixed-radix multipliers for parent encoding */
    int *multipliers = NULL;
    if (parent_count > 0) {
        multipliers = (int *)malloc(parent_count * sizeof(int));
        if (!multipliers) { free(counts); return NULL; }
        multipliers[parent_count - 1] = 1;
        for (int parent_position = parent_count - 2; parent_position >= 0; parent_position--) {
            multipliers[parent_position] = multipliers[parent_position + 1]
                           * dataset->cardinalities[parent_indices[parent_position + 1]];
        }
    }

    /* Single pass over data */
    for (int sample = 0; sample < dataset->sample_count; sample++) {
        const int *row = dataset->values + sample * dataset->variable_count;
        int parent_config = 0;
        for (int parent_position = 0; parent_position < parent_count; parent_position++) {
            parent_config += row[parent_indices[parent_position]] * multipliers[parent_position];
        }
        int child_value = row[child];
        counts[parent_config * child_cardinality + child_value]++;
    }

    free(multipliers);
    return counts;
}

/* ── K2 score ── */

/*
 * K2 local score for variable `child` given `parent_indices`.
 *
 * score = parent_configuration_count * lgamma(r)
 *       + sum_j [ sum_k lgamma(N_jk + 1) - lgamma(N_j + r) ]
 *
 * where r = child cardinality, N_jk = state count, N_j = sum_k N_jk.
 */
static double k2_local_score(
    const Dataset *dataset,
    int child,
    const int *parent_indices,
    int parent_count)
{
    int parent_configuration_count;
    int *counts = compute_state_counts(
        dataset, child, parent_indices, parent_count, &parent_configuration_count
    );
    if (!counts) return -INFINITY;

    int r = dataset->cardinalities[child];
    double score = (double)parent_configuration_count * lgamma((double)r);

    for (int j = 0; j < parent_configuration_count; j++) {
        int N_j = 0;
        double sum_lgamma_counts = 0.0;
        for (int k = 0; k < r; k++) {
            int N_jk = counts[j * r + k];
            N_j += N_jk;
            sum_lgamma_counts += lgamma((double)(N_jk + 1));
        }
        score += sum_lgamma_counts - lgamma((double)(N_j + r));
    }

    free(counts);
    return score;
}

/* ── DAG operations ── */

static void dag_init(DAG *dag, int node_count, int max_parents) {
    dag->node_count = node_count;
    dag->max_parents = max_parents;
    dag->parents = (int *)calloc(node_count * max_parents, sizeof(int));
    dag->parent_counts = (int *)calloc(node_count, sizeof(int));
}

static void dag_free(DAG *dag) {
    free(dag->parents);
    free(dag->parent_counts);
}

static int dag_has_edge(const DAG *dag, int parent, int child) {
    for (int parent_position = 0; parent_position < dag->parent_counts[child]; parent_position++) {
        if (dag->parents[child * dag->max_parents + parent_position] == parent) return 1;
    }
    return 0;
}

static void dag_add_edge(DAG *dag, int parent, int child) {
    int idx = dag->parent_counts[child];
    dag->parents[child * dag->max_parents + idx] = parent;
    dag->parent_counts[child]++;
}

static void dag_remove_edge(DAG *dag, int parent, int child) {
    int *parent_list = dag->parents + child * dag->max_parents;
    int parent_count = dag->parent_counts[child];
    for (int parent_position = 0; parent_position < parent_count; parent_position++) {
        if (parent_list[parent_position] == parent) {
            parent_list[parent_position] = parent_list[parent_count - 1];
            dag->parent_counts[child]--;
            return;
        }
    }
}

/*
 * Check if there is a directed path from `from_node` to `to_node` in the DAG.
 * Breadth-first search with a queue, scanning every node's parent list for children.
 */
static int dag_has_path(const DAG *dag, int from_node, int to_node) {
    if (from_node == to_node) return 1;
    int node_count = dag->node_count;
    uint8_t *visited = (uint8_t *)calloc(node_count, sizeof(uint8_t));
    int *queue = (int *)malloc(node_count * sizeof(int));
    if (!visited || !queue) { free(visited); free(queue); return 0; }

    int head = 0, tail = 0;
    queue[tail++] = from_node;
    visited[from_node] = 1;

    while (head < tail) {
        int node = queue[head++];
        /* Iterate over the children of node: the nodes that list it as a parent */
        for (int child = 0; child < node_count; child++) {
            if (visited[child]) continue;
            for (int parent_position = 0; parent_position < dag->parent_counts[child]; parent_position++) {
                if (dag->parents[child * dag->max_parents + parent_position] == node) {
                    if (child == to_node) { free(visited); free(queue); return 1; }
                    visited[child] = 1;
                    queue[tail++] = child;
                    break;
                }
            }
        }
    }

    free(visited);
    free(queue);
    return 0;
}

/* Build children adjacency for faster path checks */
static void dag_build_children(const DAG *dag, int **out_child_offset, int **out_children) {
    int node_count = dag->node_count;
    int *degree = (int *)calloc(node_count, sizeof(int));
    int edge_count = 0;
    for (int child = 0; child < node_count; child++) {
        for (int parent_position = 0; parent_position < dag->parent_counts[child]; parent_position++) {
            int parent = dag->parents[child * dag->max_parents + parent_position];
            degree[parent]++;
            edge_count++;
        }
    }
    int *offset = (int *)malloc((node_count + 1) * sizeof(int));
    offset[0] = 0;
    for (int node = 0; node < node_count; node++) offset[node + 1] = offset[node] + degree[node];
    int *children = (int *)malloc((edge_count > 0 ? edge_count : 1) * sizeof(int));
    int *fill_position = (int *)calloc(node_count, sizeof(int));
    for (int child = 0; child < node_count; child++) {
        for (int parent_position = 0; parent_position < dag->parent_counts[child]; parent_position++) {
            int parent = dag->parents[child * dag->max_parents + parent_position];
            children[offset[parent] + fill_position[parent]] = child;
            fill_position[parent]++;
        }
    }
    free(degree);
    free(fill_position);
    *out_child_offset = offset;
    *out_children = children;
}

/* Fast has_path using children adjacency */
static int dag_has_path_fast(int node_count, const int *child_offset, const int *children,
                             int from_node, int to_node) {
    if (from_node == to_node) return 1;
    uint8_t *visited = (uint8_t *)calloc(node_count, sizeof(uint8_t));
    int *queue = (int *)malloc(node_count * sizeof(int));
    if (!visited || !queue) { free(visited); free(queue); return 0; }

    int head = 0, tail = 0;
    queue[tail++] = from_node;
    visited[from_node] = 1;

    while (head < tail) {
        int node = queue[head++];
        for (int slot = child_offset[node]; slot < child_offset[node + 1]; slot++) {
            int child = children[slot];
            if (child == to_node) { free(visited); free(queue); return 1; }
            if (!visited[child]) {
                visited[child] = 1;
                queue[tail++] = child;
            }
        }
    }

    free(visited);
    free(queue);
    return 0;
}

/* ── Tabu list ── */

static void tabu_init(TabuList *tabu_list, int capacity) {
    tabu_list->capacity = capacity;
    tabu_list->operations = (int *)malloc(capacity * 3 * sizeof(int));
    tabu_list->head = 0;
    tabu_list->count = 0;
}

static void tabu_free(TabuList *tabu_list) {
    free(tabu_list->operations);
}

static void tabu_push(TabuList *tabu_list, int operation, int from_node, int to_node) {
    int idx = (tabu_list->head + tabu_list->count) % tabu_list->capacity;
    tabu_list->operations[idx * 3 + 0] = operation;
    tabu_list->operations[idx * 3 + 1] = from_node;
    tabu_list->operations[idx * 3 + 2] = to_node;
    if (tabu_list->count < tabu_list->capacity) {
        tabu_list->count++;
    } else {
        tabu_list->head = (tabu_list->head + 1) % tabu_list->capacity;
    }
}

static int tabu_contains(const TabuList *tabu_list, int operation, int from_node, int to_node) {
    for (int position = 0; position < tabu_list->count; position++) {
        int idx = (tabu_list->head + position) % tabu_list->capacity;
        if (tabu_list->operations[idx * 3 + 0] == operation &&
            tabu_list->operations[idx * 3 + 1] == from_node &&
            tabu_list->operations[idx * 3 + 2] == to_node) {
            return 1;
        }
    }
    return 0;
}

/* ── Score cache ── */

/*
 * Cache K2 local scores to avoid recomputation.
 * Key: (child, frozen_parent_set) → score
 * Implementation: simple hash map with open addressing.
 *
 * For max_indegree=1, the parent set is at most 1 element,
 * so we encode the key as (child * (variable_count+1) + (parent+1)).
 * For parent set = {}, parent+1 = 0.
 * For parent set = {p}, parent+1 = p+1.
 *
 * For max_indegree>1, we use a more general encoding.
 */
typedef struct {
    long *keys;
    double *values;
    int capacity;
    int capacity_mask;
} ScoreCache;

static void score_cache_init(ScoreCache *score_cache, int minimum_capacity) {
    /* Round up to power of 2 */
    int capacity = 1;
    while (capacity < minimum_capacity) capacity <<= 1;
    score_cache->capacity = capacity;
    score_cache->capacity_mask = capacity - 1;
    score_cache->keys = (long *)malloc(capacity * sizeof(long));
    score_cache->values = (double *)malloc(capacity * sizeof(double));
    memset(score_cache->keys, -1, capacity * sizeof(long)); /* -1 = empty */
}

static void score_cache_free(ScoreCache *score_cache) {
    free(score_cache->keys);
    free(score_cache->values);
}

static int score_cache_get(const ScoreCache *score_cache, long key, double *out) {
    int idx = (int)(key & score_cache->capacity_mask);
    for (int probe = 0; probe < score_cache->capacity; probe++) {
        int slot = (idx + probe) & score_cache->capacity_mask;
        if (score_cache->keys[slot] == key) { *out = score_cache->values[slot]; return 1; }
        if (score_cache->keys[slot] == -1) return 0;
    }
    return 0;
}

static void score_cache_put(ScoreCache *score_cache, long key, double value) {
    int idx = (int)(key & score_cache->capacity_mask);
    for (int probe = 0; probe < score_cache->capacity; probe++) {
        int slot = (idx + probe) & score_cache->capacity_mask;
        if (score_cache->keys[slot] == -1 || score_cache->keys[slot] == key) {
            score_cache->keys[slot] = key;
            score_cache->values[slot] = value;
            return;
        }
    }
    /* Cache full — overwrite at original position (LRU-ish) */
    score_cache->keys[idx] = key;
    score_cache->values[idx] = value;
}

/*
 * Encode a parent set as a cache key.
 * For max_indegree <= 2, we pack (child, parent1, parent2) into a long.
 */
static long encode_cache_key(int child, const int *parents, int parent_count, int variable_count) {
    long key = child;
    /* Sort parents for canonical representation */
    if (parent_count == 0) {
        key = key * (variable_count + 1);
    } else if (parent_count == 1) {
        key = key * (variable_count + 1) + parents[0] + 1;
    } else {
        /* General case: pack sorted parents */
        key = key * (variable_count + 1) + parents[0] + 1;
        for (int parent_position = 1; parent_position < parent_count; parent_position++) {
            key = key * (variable_count + 1) + parents[parent_position] + 1;
        }
    }
    return key;
}

/* Cached K2 local score */
static double cached_k2_score(
    const Dataset *dataset,
    ScoreCache *cache,
    int child,
    const int *parents,
    int parent_count)
{
    long key = encode_cache_key(child, parents, parent_count, dataset->variable_count);
    double score;
    if (score_cache_get(cache, key, &score)) return score;
    score = k2_local_score(dataset, child, parents, parent_count);
    score_cache_put(cache, key, score);
    return score;
}

/* ── Hill climb search ── */

/*
 * Main hill climb function.
 *
 * Returns a list of (parent, child) edge tuples representing the learned DAG.
 */
static PyObject *py_hill_climb_k2(PyObject *self, PyObject *args) {
    PyObject *data_obj;
    PyObject *cardinalities_obj;
    int max_indegree = 1;
    int tabu_length = 100;
    double epsilon = 1e-4;
    int max_iterations = 1000000;

    if (!PyArg_ParseTuple(args, "OO|iidi",
            &data_obj, &cardinalities_obj,
            &max_indegree, &tabu_length, &epsilon, &max_iterations))
        return NULL;

    /* Parse data: list of lists or 2D sequence */
    if (!PySequence_Check(data_obj)) {
        PyErr_SetString(PyExc_TypeError, "data must be a sequence of sequences");
        return NULL;
    }
    Py_ssize_t sample_count = PySequence_Size(data_obj);
    if (sample_count <= 0) {
        return PyList_New(0); /* Empty data → empty DAG */
    }

    /* Parse cardinalities */
    if (!PySequence_Check(cardinalities_obj)) {
        PyErr_SetString(PyExc_TypeError, "cardinalities must be a sequence of ints");
        return NULL;
    }
    Py_ssize_t variable_count = PySequence_Size(cardinalities_obj);
    if (variable_count <= 1) {
        return PyList_New(0); /* 0 or 1 variable → no edges possible */
    }

    /* Build Dataset */
    Dataset dataset;
    dataset.sample_count = (int)sample_count;
    dataset.variable_count = (int)variable_count;
    dataset.cardinalities = (int *)malloc(variable_count * sizeof(int));
    dataset.values = (int *)malloc(sample_count * variable_count * sizeof(int));

    if (!dataset.cardinalities || !dataset.values) {
        free(dataset.cardinalities); free(dataset.values);
        return PyErr_NoMemory();
    }

    for (Py_ssize_t variable = 0; variable < variable_count; variable++) {
        PyObject *cardinality_obj = PySequence_GetItem(cardinalities_obj, variable);
        dataset.cardinalities[variable] = (int)PyLong_AsLong(cardinality_obj);
        Py_DECREF(cardinality_obj);
    }

    for (Py_ssize_t sample = 0; sample < sample_count; sample++) {
        PyObject *row = PySequence_GetItem(data_obj, sample);
        for (Py_ssize_t variable = 0; variable < variable_count; variable++) {
            PyObject *value_obj = PySequence_GetItem(row, variable);
            dataset.values[sample * variable_count + variable] = (int)PyLong_AsLong(value_obj);
            Py_DECREF(value_obj);
        }
        Py_DECREF(row);
    }

    /* Initialize DAG, tabu list, score cache */
    DAG dag;
    dag_init(&dag, (int)variable_count, max_indegree);

    TabuList tabu_list;
    tabu_init(&tabu_list, tabu_length);

    ScoreCache cache;
    score_cache_init(&cache, 4096);

    /* Precompute base scores (no parents) for all variables */
    double *base_scores = (double *)malloc(variable_count * sizeof(double));
    for (int node = 0; node < (int)variable_count; node++) {
        base_scores[node] = cached_k2_score(&dataset, &cache, node, NULL, 0);
    }

    /* Hill climb iterations */
    for (int iteration = 0; iteration < max_iterations; iteration++) {
        /* Build children adjacency for fast path checking */
        int *child_offset, *children;
        dag_build_children(&dag, &child_offset, &children);

        int best_operation = -1, best_from_node = -1, best_to_node = -1;
        double best_delta = -INFINITY;

        int node_count = (int)variable_count;

        /* Evaluate all possible ADD operations */
        for (int from_node = 0; from_node < node_count; from_node++) {
            for (int to_node = 0; to_node < node_count; to_node++) {
                if (from_node == to_node) continue;
                if (dag_has_edge(&dag, from_node, to_node)) continue;
                if (dag_has_edge(&dag, to_node, from_node)) continue;
                if (dag.parent_counts[to_node] >= max_indegree) continue;
                if (tabu_contains(&tabu_list, OPERATION_ADD, from_node, to_node)) continue;

                /* Check acyclicity: adding from_node→to_node creates a cycle iff a path to_node→from_node exists */
                if (dag_has_path_fast(node_count, child_offset, children, to_node, from_node)) continue;

                /* Score delta for adding from_node as parent of to_node */
                int new_parents[16]; /* max_indegree should be small */
                int parent_count = dag.parent_counts[to_node];
                memcpy(new_parents, dag.parents + to_node * dag.max_parents, parent_count * sizeof(int));
                new_parents[parent_count] = from_node;

                double old_score = cached_k2_score(
                    &dataset, &cache, to_node, dag.parents + to_node * dag.max_parents, parent_count);
                double new_score = cached_k2_score(
                    &dataset, &cache, to_node, new_parents, parent_count + 1);
                double delta = new_score - old_score;

                if (delta > best_delta) {
                    best_delta = delta;
                    best_operation = OPERATION_ADD;
                    best_from_node = from_node;
                    best_to_node = to_node;
                }
            }
        }

        /* Evaluate all possible REMOVE operations */
        for (int to_node = 0; to_node < node_count; to_node++) {
            for (int edge_position = 0; edge_position < dag.parent_counts[to_node]; edge_position++) {
                int from_node = dag.parents[to_node * dag.max_parents + edge_position];
                if (tabu_contains(&tabu_list, OPERATION_REMOVE, from_node, to_node)) continue;

                /* Score delta for removing from_node as parent of to_node */
                int new_parents[16];
                int parent_count = 0;
                for (int parent_position = 0; parent_position < dag.parent_counts[to_node]; parent_position++) {
                    int parent = dag.parents[to_node * dag.max_parents + parent_position];
                    if (parent != from_node) new_parents[parent_count++] = parent;
                }

                double old_score = cached_k2_score(
                    &dataset, &cache, to_node, dag.parents + to_node * dag.max_parents, dag.parent_counts[to_node]);
                double new_score = cached_k2_score(
                    &dataset, &cache, to_node, new_parents, parent_count);
                double delta = new_score - old_score;

                if (delta > best_delta) {
                    best_delta = delta;
                    best_operation = OPERATION_REMOVE;
                    best_from_node = from_node;
                    best_to_node = to_node;
                }
            }
        }

        /* Evaluate all possible FLIP operations */
        for (int to_node = 0; to_node < node_count; to_node++) {
            for (int edge_position = 0; edge_position < dag.parent_counts[to_node]; edge_position++) {
                int from_node = dag.parents[to_node * dag.max_parents + edge_position];
                if (tabu_contains(&tabu_list, OPERATION_FLIP, from_node, to_node)) continue;
                if (tabu_contains(&tabu_list, OPERATION_FLIP, to_node, from_node)) continue;
                if (dag.parent_counts[from_node] >= max_indegree) continue;

                /*
                 * Flip from_node→to_node to to_node→from_node.
                 * Acyclicity: After removing from_node→to_node and adding to_node→from_node,
                 * there is a cycle iff there is a path from from_node to to_node
                 * in the DAG *without* the edge from_node→to_node.
                 * Check: any path from_node→...→to_node of length > 1?
                 */
                /* Temporarily remove from_node→to_node */
                dag_remove_edge(&dag, from_node, to_node);
                int *child_offset_without_edge, *children_without_edge;
                dag_build_children(&dag, &child_offset_without_edge, &children_without_edge);
                int creates_cycle = dag_has_path_fast(node_count, child_offset_without_edge, children_without_edge, from_node, to_node);
                free(child_offset_without_edge); free(children_without_edge);
                dag_add_edge(&dag, from_node, to_node); /* restore */

                if (creates_cycle) continue;

                /* Score delta for flipping from_node→to_node to to_node→from_node */
                /* Remove from_node from to_node's parents */
                int to_node_new_parents[16];
                int to_node_parent_count = 0;
                for (int parent_position = 0; parent_position < dag.parent_counts[to_node]; parent_position++) {
                    int parent = dag.parents[to_node * dag.max_parents + parent_position];
                    if (parent != from_node) to_node_new_parents[to_node_parent_count++] = parent;
                }

                /* Add to_node to from_node's parents */
                int from_node_new_parents[16];
                int from_node_parent_count = dag.parent_counts[from_node];
                memcpy(from_node_new_parents, dag.parents + from_node * dag.max_parents, from_node_parent_count * sizeof(int));
                from_node_new_parents[from_node_parent_count] = to_node;

                double old_to_node_score = cached_k2_score(
                    &dataset, &cache, to_node, dag.parents + to_node * dag.max_parents, dag.parent_counts[to_node]);
                double new_to_node_score = cached_k2_score(
                    &dataset, &cache, to_node, to_node_new_parents, to_node_parent_count);
                double old_from_node_score = cached_k2_score(
                    &dataset, &cache, from_node, dag.parents + from_node * dag.max_parents, dag.parent_counts[from_node]);
                double new_from_node_score = cached_k2_score(
                    &dataset, &cache, from_node, from_node_new_parents, from_node_parent_count + 1);

                double delta = (new_to_node_score + new_from_node_score) - (old_to_node_score + old_from_node_score);

                if (delta > best_delta) {
                    best_delta = delta;
                    best_operation = OPERATION_FLIP;
                    best_from_node = from_node;
                    best_to_node = to_node;
                }
            }
        }

        free(child_offset);
        free(children);

        /* Apply best operation or stop */
        if (best_operation < 0 || best_delta < epsilon) break;

        if (best_operation == OPERATION_ADD) {
            dag_add_edge(&dag, best_from_node, best_to_node);
            tabu_push(&tabu_list, OPERATION_REMOVE, best_from_node, best_to_node);
        } else if (best_operation == OPERATION_REMOVE) {
            dag_remove_edge(&dag, best_from_node, best_to_node);
            tabu_push(&tabu_list, OPERATION_ADD, best_from_node, best_to_node);
        } else if (best_operation == OPERATION_FLIP) {
            dag_remove_edge(&dag, best_from_node, best_to_node);
            dag_add_edge(&dag, best_to_node, best_from_node);
            tabu_push(&tabu_list, OPERATION_FLIP, best_from_node, best_to_node);
        }
    }

    /* Build result: list of (parent, child) tuples */
    PyObject *result = PyList_New(0);
    for (int to_node = 0; to_node < (int)variable_count; to_node++) {
        for (int edge_position = 0; edge_position < dag.parent_counts[to_node]; edge_position++) {
            int from_node = dag.parents[to_node * dag.max_parents + edge_position];
            PyObject *edge_tuple = Py_BuildValue("(ii)", from_node, to_node);
            PyList_Append(result, edge_tuple);
            Py_DECREF(edge_tuple);
        }
    }

    /* Cleanup */
    free(base_scores);
    dag_free(&dag);
    tabu_free(&tabu_list);
    score_cache_free(&cache);
    free(dataset.values);
    free(dataset.cardinalities);

    return result;
}

/* Also expose K2 local score directly for testing */
static PyObject *py_k2_local_score(PyObject *self, PyObject *args) {
    PyObject *data_obj, *cardinalities_obj, *parents_obj;
    int child;

    if (!PyArg_ParseTuple(args, "OOiO", &data_obj, &cardinalities_obj, &child, &parents_obj))
        return NULL;

    Py_ssize_t sample_count = PySequence_Size(data_obj);
    Py_ssize_t variable_count = PySequence_Size(cardinalities_obj);

    Dataset dataset;
    dataset.sample_count = (int)sample_count;
    dataset.variable_count = (int)variable_count;
    dataset.cardinalities = (int *)malloc(variable_count * sizeof(int));
    dataset.values = (int *)malloc(sample_count * variable_count * sizeof(int));

    for (Py_ssize_t variable = 0; variable < variable_count; variable++) {
        PyObject *cardinality_obj = PySequence_GetItem(cardinalities_obj, variable);
        dataset.cardinalities[variable] = (int)PyLong_AsLong(cardinality_obj);
        Py_DECREF(cardinality_obj);
    }
    for (Py_ssize_t sample = 0; sample < sample_count; sample++) {
        PyObject *row = PySequence_GetItem(data_obj, sample);
        for (Py_ssize_t variable = 0; variable < variable_count; variable++) {
            PyObject *value_obj = PySequence_GetItem(row, variable);
            dataset.values[sample * variable_count + variable] = (int)PyLong_AsLong(value_obj);
            Py_DECREF(value_obj);
        }
        Py_DECREF(row);
    }

    Py_ssize_t parent_count = PySequence_Size(parents_obj);
    int *parents = (int *)malloc(parent_count * sizeof(int));
    for (Py_ssize_t parent_position = 0; parent_position < parent_count; parent_position++) {
        PyObject *parent_obj = PySequence_GetItem(parents_obj, parent_position);
        parents[parent_position] = (int)PyLong_AsLong(parent_obj);
        Py_DECREF(parent_obj);
    }

    double score = k2_local_score(&dataset, child, parents, (int)parent_count);

    free(parents);
    free(dataset.values);
    free(dataset.cardinalities);

    return PyFloat_FromDouble(score);
}

/*
 * Estimate CPDs (Conditional Probability Distributions) for all nodes
 * given a DAG structure and dataset. Uses Laplace (add-1) smoothing.
 *
 * Returns a dict: { child_var: list[list[float]] }
 * where the outer list is indexed by parent_config and the inner by child_value.
 * Each inner list sums to 1.0.
 *
 * This replaces pgmpy's BayesianEstimator.estimate_cpd which uses pandas.
 */
static PyObject *py_estimate_cpds(PyObject *self, PyObject *args) {
    PyObject *data_obj, *cardinalities_obj, *edges_obj;

    if (!PyArg_ParseTuple(args, "OOO", &data_obj, &cardinalities_obj, &edges_obj))
        return NULL;

    Py_ssize_t sample_count = PySequence_Size(data_obj);
    Py_ssize_t variable_count = PySequence_Size(cardinalities_obj);

    if (sample_count <= 0 || variable_count <= 0) {
        return PyDict_New();
    }

    /* Parse dataset */
    int *values = (int *)malloc(sample_count * variable_count * sizeof(int));
    int *cardinalities = (int *)malloc(variable_count * sizeof(int));
    if (!values || !cardinalities) {
        free(values); free(cardinalities);
        return PyErr_NoMemory();
    }

    for (Py_ssize_t variable = 0; variable < variable_count; variable++) {
        PyObject *cardinality_obj = PySequence_GetItem(cardinalities_obj, variable);
        cardinalities[variable] = (int)PyLong_AsLong(cardinality_obj);
        Py_DECREF(cardinality_obj);
    }
    for (Py_ssize_t sample = 0; sample < sample_count; sample++) {
        PyObject *row = PySequence_GetItem(data_obj, sample);
        for (Py_ssize_t variable = 0; variable < variable_count; variable++) {
            PyObject *value_obj = PySequence_GetItem(row, variable);
            values[sample * variable_count + variable] = (int)PyLong_AsLong(value_obj);
            Py_DECREF(value_obj);
        }
        Py_DECREF(row);
    }

    Dataset dataset;
    dataset.values = values;
    dataset.sample_count = (int)sample_count;
    dataset.variable_count = (int)variable_count;
    dataset.cardinalities = cardinalities;

    /* Parse edges into parent lists */
    Py_ssize_t edge_count = PySequence_Size(edges_obj);
    int *parent_counts = (int *)calloc(variable_count, sizeof(int));
    /* max_indegree is at most variable_count; allocate generously */
    int **parent_lists = (int **)calloc(variable_count, sizeof(int *));
    for (Py_ssize_t variable = 0; variable < variable_count; variable++) {
        parent_lists[variable] = (int *)malloc(variable_count * sizeof(int));
    }

    for (Py_ssize_t edge_index = 0; edge_index < edge_count; edge_index++) {
        PyObject *edge = PySequence_GetItem(edges_obj, edge_index);
        PyObject *parent_obj = PySequence_GetItem(edge, 0);
        PyObject *child_obj = PySequence_GetItem(edge, 1);
        int parent = (int)PyLong_AsLong(parent_obj);
        int child = (int)PyLong_AsLong(child_obj);
        Py_DECREF(parent_obj); Py_DECREF(child_obj); Py_DECREF(edge);
        parent_lists[child][parent_counts[child]++] = parent;
    }

    /* Build CPDs for all variables */
    PyObject *result = PyDict_New();

    for (int child = 0; child < (int)variable_count; child++) {
        int parent_configuration_count;
        int *counts = compute_state_counts(
            &dataset, child, parent_lists[child], parent_counts[child], &parent_configuration_count
        );
        if (!counts) continue;

        int child_cardinality = cardinalities[child];

        /* Build Python list of lists with Laplace smoothing */
        PyObject *distributions = PyList_New(parent_configuration_count);
        for (int parent_configuration = 0; parent_configuration < parent_configuration_count; parent_configuration++) {
            /* Compute total count + smoothing for this parent config */
            double total = (double)child_cardinality; /* Laplace: add 1 per state */
            for (int child_value = 0; child_value < child_cardinality; child_value++) {
                total += (double)counts[parent_configuration * child_cardinality + child_value];
            }

            PyObject *probabilities = PyList_New(child_cardinality);
            for (int child_value = 0; child_value < child_cardinality; child_value++) {
                double probability = ((double)counts[parent_configuration * child_cardinality + child_value] + 1.0) / total;
                PyList_SET_ITEM(probabilities, child_value, PyFloat_FromDouble(probability));
            }
            PyList_SET_ITEM(distributions, parent_configuration, probabilities);
        }

        free(counts);

        PyObject *key = PyLong_FromLong(child);
        PyDict_SetItem(result, key, distributions);
        Py_DECREF(key);
        Py_DECREF(distributions);
    }

    /* Cleanup */
    for (Py_ssize_t variable = 0; variable < variable_count; variable++) free(parent_lists[variable]);
    free(parent_lists);
    free(parent_counts);
    free(values);
    free(cardinalities);

    return result;
}

/* ── Module definition ── */

static PyMethodDef dag_learn_methods[] = {
    {"hill_climb_k2", py_hill_climb_k2, METH_VARARGS,
     "hill_climb_k2(data, cardinalities[, max_indegree, tabu_length, epsilon, max_iter])"
     " -> list[tuple[int, int]]\n\n"
     "Learn DAG structure via greedy hill-climb with K2 scoring.\n"
     "data: list of lists (n_samples x n_vars), values 0..card-1.\n"
     "cardinalities: list of int, one per variable.\n"
     "Returns list of (parent, child) edge tuples."},
    {"k2_local_score", py_k2_local_score, METH_VARARGS,
     "k2_local_score(data, cardinalities, child, parents) -> float\n\n"
     "Compute K2 local score for a variable given its parents."},
    {"estimate_cpds", py_estimate_cpds, METH_VARARGS,
     "estimate_cpds(data, cardinalities, edges) -> dict[int, list[list[float]]]\n\n"
     "Estimate CPDs for all variables given DAG edges and data.\n"
     "Uses Laplace (add-1) smoothing. Returns {var: [[prob_per_child_value] per parent_config]}."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef dag_learn_module = {
    PyModuleDef_HEAD_INIT, "_dag_learn",
    "DAG structure learning via hill-climb search with K2 scoring.",
    -1, dag_learn_methods,
};

PyMODINIT_FUNC PyInit__dag_learn(void) {
    return PyModule_Create(&dag_learn_module);
}
