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
 * Complexity per iteration: O(n^2 * n_samples) where n = number of variables
 * Total: O(max_iter * n^2 * n_samples)
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

/* ── Data structures ── */

/* Flat dataset: data[sample * n_vars + var] = value */
typedef struct {
    int *values;        /* n_samples * n_vars */
    int n_samples;
    int n_vars;
    int *cardinalities; /* cardinality per variable */
} Dataset;

/* DAG adjacency: parent lists per node */
typedef struct {
    int *parents;       /* parents[node * max_parents .. + n_parents[node]) */
    int *n_parents;     /* number of parents per node */
    int max_parents;    /* max_indegree */
    int n;              /* number of nodes */
} DAG;

/* Tabu list (circular buffer of operations) */
typedef struct {
    int *ops;           /* op_type, u, v per entry */
    int capacity;
    int head;
    int count;
} TabuList;

enum { OP_ADD = 0, OP_REMOVE = 1, OP_FLIP = 2 };

/* ── State counts (the hot path) ── */

/*
 * Count co-occurrences of (parent_config, child_value) in the dataset.
 *
 * parent_config = mixed-radix encoding of parent values.
 * Returns a flat array of size n_parent_configs * child_card.
 * counts[parent_config * child_card + child_value] = count
 */
static int *compute_state_counts(
    const Dataset *ds,
    int child,
    const int *parent_indices,
    int n_parents,
    int *out_n_parent_configs)
{
    /* Compute number of parent configurations */
    int n_parent_configs = 1;
    for (int i = 0; i < n_parents; i++) {
        n_parent_configs *= ds->cardinalities[parent_indices[i]];
    }
    *out_n_parent_configs = n_parent_configs;

    int child_card = ds->cardinalities[child];
    int table_size = n_parent_configs * child_card;

    int *counts = (int *)calloc(table_size, sizeof(int));
    if (!counts) return NULL;

    /* Precompute mixed-radix multipliers for parent encoding */
    int *multipliers = NULL;
    if (n_parents > 0) {
        multipliers = (int *)malloc(n_parents * sizeof(int));
        if (!multipliers) { free(counts); return NULL; }
        multipliers[n_parents - 1] = 1;
        for (int i = n_parents - 2; i >= 0; i--) {
            multipliers[i] = multipliers[i + 1]
                           * ds->cardinalities[parent_indices[i + 1]];
        }
    }

    /* Single pass over data */
    for (int s = 0; s < ds->n_samples; s++) {
        const int *row = ds->values + s * ds->n_vars;
        int parent_config = 0;
        for (int i = 0; i < n_parents; i++) {
            parent_config += row[parent_indices[i]] * multipliers[i];
        }
        int child_val = row[child];
        counts[parent_config * child_card + child_val]++;
    }

    free(multipliers);
    return counts;
}

/* ── K2 score ── */

/*
 * K2 local score for variable `child` given `parent_indices`.
 *
 * score = n_parent_configs * lgamma(r)
 *       + sum_j [ sum_k lgamma(N_jk + 1) - lgamma(N_j + r) ]
 *
 * where r = child cardinality, N_jk = state count, N_j = sum_k N_jk.
 */
static double k2_local_score(
    const Dataset *ds,
    int child,
    const int *parent_indices,
    int n_parents)
{
    int n_parent_configs;
    int *counts = compute_state_counts(
        ds, child, parent_indices, n_parents, &n_parent_configs
    );
    if (!counts) return -INFINITY;

    int r = ds->cardinalities[child];
    double score = (double)n_parent_configs * lgamma((double)r);

    for (int j = 0; j < n_parent_configs; j++) {
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

static void dag_init(DAG *dag, int n, int max_parents) {
    dag->n = n;
    dag->max_parents = max_parents;
    dag->parents = (int *)calloc(n * max_parents, sizeof(int));
    dag->n_parents = (int *)calloc(n, sizeof(int));
}

static void dag_free(DAG *dag) {
    free(dag->parents);
    free(dag->n_parents);
}

static int dag_has_edge(const DAG *dag, int u, int v) {
    for (int i = 0; i < dag->n_parents[v]; i++) {
        if (dag->parents[v * dag->max_parents + i] == u) return 1;
    }
    return 0;
}

static void dag_add_edge(DAG *dag, int u, int v) {
    int idx = dag->n_parents[v];
    dag->parents[v * dag->max_parents + idx] = u;
    dag->n_parents[v]++;
}

static void dag_remove_edge(DAG *dag, int u, int v) {
    int *plist = dag->parents + v * dag->max_parents;
    int np = dag->n_parents[v];
    for (int i = 0; i < np; i++) {
        if (plist[i] == u) {
            plist[i] = plist[np - 1];
            dag->n_parents[v]--;
            return;
        }
    }
}

/*
 * Check if there is a directed path from `src` to `dst` in the DAG.
 * Uses iterative BFS with a small stack.
 */
static int dag_has_path(const DAG *dag, int src, int dst) {
    if (src == dst) return 1;
    int n = dag->n;
    uint8_t *visited = (uint8_t *)calloc(n, sizeof(uint8_t));
    int *queue = (int *)malloc(n * sizeof(int));
    if (!visited || !queue) { free(visited); free(queue); return 0; }

    int head = 0, tail = 0;
    queue[tail++] = src;
    visited[src] = 1;

    while (head < tail) {
        int u = queue[head++];
        /* Iterate over children of u: nodes v where u is a parent */
        for (int v = 0; v < n; v++) {
            if (visited[v]) continue;
            for (int i = 0; i < dag->n_parents[v]; i++) {
                if (dag->parents[v * dag->max_parents + i] == u) {
                    if (v == dst) { free(visited); free(queue); return 1; }
                    visited[v] = 1;
                    queue[tail++] = v;
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
    int n = dag->n;
    int *degree = (int *)calloc(n, sizeof(int));
    int total_edges = 0;
    for (int v = 0; v < n; v++) {
        for (int i = 0; i < dag->n_parents[v]; i++) {
            int u = dag->parents[v * dag->max_parents + i];
            degree[u]++;
            total_edges++;
        }
    }
    int *offset = (int *)malloc((n + 1) * sizeof(int));
    offset[0] = 0;
    for (int i = 0; i < n; i++) offset[i + 1] = offset[i] + degree[i];
    int *children = (int *)malloc((total_edges > 0 ? total_edges : 1) * sizeof(int));
    int *pos = (int *)calloc(n, sizeof(int));
    for (int v = 0; v < n; v++) {
        for (int i = 0; i < dag->n_parents[v]; i++) {
            int u = dag->parents[v * dag->max_parents + i];
            children[offset[u] + pos[u]] = v;
            pos[u]++;
        }
    }
    free(degree);
    free(pos);
    *out_child_offset = offset;
    *out_children = children;
}

/* Fast has_path using children adjacency */
static int dag_has_path_fast(int n, const int *child_offset, const int *children,
                             int src, int dst) {
    if (src == dst) return 1;
    uint8_t *visited = (uint8_t *)calloc(n, sizeof(uint8_t));
    int *queue = (int *)malloc(n * sizeof(int));
    if (!visited || !queue) { free(visited); free(queue); return 0; }

    int head = 0, tail = 0;
    queue[tail++] = src;
    visited[src] = 1;

    while (head < tail) {
        int u = queue[head++];
        for (int i = child_offset[u]; i < child_offset[u + 1]; i++) {
            int v = children[i];
            if (v == dst) { free(visited); free(queue); return 1; }
            if (!visited[v]) {
                visited[v] = 1;
                queue[tail++] = v;
            }
        }
    }

    free(visited);
    free(queue);
    return 0;
}

/* ── Tabu list ── */

static void tabu_init(TabuList *tl, int capacity) {
    tl->capacity = capacity;
    tl->ops = (int *)malloc(capacity * 3 * sizeof(int));
    tl->head = 0;
    tl->count = 0;
}

static void tabu_free(TabuList *tl) {
    free(tl->ops);
}

static void tabu_push(TabuList *tl, int op, int u, int v) {
    int idx = (tl->head + tl->count) % tl->capacity;
    tl->ops[idx * 3 + 0] = op;
    tl->ops[idx * 3 + 1] = u;
    tl->ops[idx * 3 + 2] = v;
    if (tl->count < tl->capacity) {
        tl->count++;
    } else {
        tl->head = (tl->head + 1) % tl->capacity;
    }
}

static int tabu_contains(const TabuList *tl, int op, int u, int v) {
    for (int i = 0; i < tl->count; i++) {
        int idx = (tl->head + i) % tl->capacity;
        if (tl->ops[idx * 3 + 0] == op &&
            tl->ops[idx * 3 + 1] == u &&
            tl->ops[idx * 3 + 2] == v) {
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
 * so we encode the key as (child * (n+1) + (parent+1)).
 * For parent set = {}, parent+1 = 0.
 * For parent set = {p}, parent+1 = p+1.
 *
 * For max_indegree>1, we use a more general encoding.
 */
typedef struct {
    long *keys;
    double *values;
    int capacity;
    int mask;
} ScoreCache;

static void score_cache_init(ScoreCache *sc, int capacity) {
    /* Round up to power of 2 */
    int cap = 1;
    while (cap < capacity) cap <<= 1;
    sc->capacity = cap;
    sc->mask = cap - 1;
    sc->keys = (long *)malloc(cap * sizeof(long));
    sc->values = (double *)malloc(cap * sizeof(double));
    memset(sc->keys, -1, cap * sizeof(long)); /* -1 = empty */
}

static void score_cache_free(ScoreCache *sc) {
    free(sc->keys);
    free(sc->values);
}

static int score_cache_get(const ScoreCache *sc, long key, double *out) {
    int idx = (int)(key & sc->mask);
    for (int i = 0; i < sc->capacity; i++) {
        int pos = (idx + i) & sc->mask;
        if (sc->keys[pos] == key) { *out = sc->values[pos]; return 1; }
        if (sc->keys[pos] == -1) return 0;
    }
    return 0;
}

static void score_cache_put(ScoreCache *sc, long key, double value) {
    int idx = (int)(key & sc->mask);
    for (int i = 0; i < sc->capacity; i++) {
        int pos = (idx + i) & sc->mask;
        if (sc->keys[pos] == -1 || sc->keys[pos] == key) {
            sc->keys[pos] = key;
            sc->values[pos] = value;
            return;
        }
    }
    /* Cache full — overwrite at original position (LRU-ish) */
    sc->keys[idx] = key;
    sc->values[idx] = value;
}

/*
 * Encode a parent set as a cache key.
 * For max_indegree <= 2, we pack (child, parent1, parent2) into a long.
 */
static long encode_cache_key(int child, const int *parents, int n_parents, int n_vars) {
    long key = child;
    /* Sort parents for canonical representation */
    if (n_parents == 0) {
        key = key * (n_vars + 1);
    } else if (n_parents == 1) {
        key = key * (n_vars + 1) + parents[0] + 1;
    } else {
        /* General case: pack sorted parents */
        key = key * (n_vars + 1) + parents[0] + 1;
        for (int i = 1; i < n_parents; i++) {
            key = key * (n_vars + 1) + parents[i] + 1;
        }
    }
    return key;
}

/* Cached K2 local score */
static double cached_k2_score(
    const Dataset *ds,
    ScoreCache *cache,
    int child,
    const int *parents,
    int n_parents)
{
    long key = encode_cache_key(child, parents, n_parents, ds->n_vars);
    double score;
    if (score_cache_get(cache, key, &score)) return score;
    score = k2_local_score(ds, child, parents, n_parents);
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
    PyObject *card_obj;
    int max_indegree = 1;
    int tabu_length = 100;
    double epsilon = 1e-4;
    int max_iter = 1000000;

    if (!PyArg_ParseTuple(args, "OO|iidi",
            &data_obj, &card_obj,
            &max_indegree, &tabu_length, &epsilon, &max_iter))
        return NULL;

    /* Parse data: list of lists or 2D sequence */
    if (!PySequence_Check(data_obj)) {
        PyErr_SetString(PyExc_TypeError, "data must be a sequence of sequences");
        return NULL;
    }
    Py_ssize_t n_samples = PySequence_Size(data_obj);
    if (n_samples <= 0) {
        return PyList_New(0); /* Empty data → empty DAG */
    }

    /* Parse cardinalities */
    if (!PySequence_Check(card_obj)) {
        PyErr_SetString(PyExc_TypeError, "cardinalities must be a sequence of ints");
        return NULL;
    }
    Py_ssize_t n_vars = PySequence_Size(card_obj);
    if (n_vars <= 1) {
        return PyList_New(0); /* 0 or 1 variable → no edges possible */
    }

    /* Build Dataset */
    Dataset ds;
    ds.n_samples = (int)n_samples;
    ds.n_vars = (int)n_vars;
    ds.cardinalities = (int *)malloc(n_vars * sizeof(int));
    ds.values = (int *)malloc(n_samples * n_vars * sizeof(int));

    if (!ds.cardinalities || !ds.values) {
        free(ds.cardinalities); free(ds.values);
        return PyErr_NoMemory();
    }

    for (Py_ssize_t j = 0; j < n_vars; j++) {
        PyObject *c = PySequence_GetItem(card_obj, j);
        ds.cardinalities[j] = (int)PyLong_AsLong(c);
        Py_DECREF(c);
    }

    for (Py_ssize_t i = 0; i < n_samples; i++) {
        PyObject *row = PySequence_GetItem(data_obj, i);
        for (Py_ssize_t j = 0; j < n_vars; j++) {
            PyObject *val = PySequence_GetItem(row, j);
            ds.values[i * n_vars + j] = (int)PyLong_AsLong(val);
            Py_DECREF(val);
        }
        Py_DECREF(row);
    }

    /* Initialize DAG, tabu list, score cache */
    DAG dag;
    dag_init(&dag, (int)n_vars, max_indegree);

    TabuList tabu;
    tabu_init(&tabu, tabu_length);

    ScoreCache cache;
    score_cache_init(&cache, 4096);

    /* Precompute base scores (no parents) for all variables */
    double *base_scores = (double *)malloc(n_vars * sizeof(double));
    for (int v = 0; v < (int)n_vars; v++) {
        base_scores[v] = cached_k2_score(&ds, &cache, v, NULL, 0);
    }

    /* Hill climb iterations */
    for (int iter = 0; iter < max_iter; iter++) {
        /* Build children adjacency for fast path checking */
        int *child_offset, *children_arr;
        dag_build_children(&dag, &child_offset, &children_arr);

        int best_op = -1, best_u = -1, best_v = -1;
        double best_delta = -INFINITY;

        int n = (int)n_vars;

        /* Evaluate all possible ADD operations */
        for (int u = 0; u < n; u++) {
            for (int v = 0; v < n; v++) {
                if (u == v) continue;
                if (dag_has_edge(&dag, u, v)) continue;
                if (dag_has_edge(&dag, v, u)) continue;
                if (dag.n_parents[v] >= max_indegree) continue;
                if (tabu_contains(&tabu, OP_ADD, u, v)) continue;

                /* Check acyclicity: adding u→v creates cycle iff path v→u exists */
                if (dag_has_path_fast(n, child_offset, children_arr, v, u)) continue;

                /* Score delta for adding u as parent of v */
                int new_parents[16]; /* max_indegree should be small */
                int np = dag.n_parents[v];
                memcpy(new_parents, dag.parents + v * dag.max_parents, np * sizeof(int));
                new_parents[np] = u;

                double old_score = cached_k2_score(
                    &ds, &cache, v, dag.parents + v * dag.max_parents, np);
                double new_score = cached_k2_score(
                    &ds, &cache, v, new_parents, np + 1);
                double delta = new_score - old_score;

                if (delta > best_delta) {
                    best_delta = delta;
                    best_op = OP_ADD;
                    best_u = u;
                    best_v = v;
                }
            }
        }

        /* Evaluate all possible REMOVE operations */
        for (int v = 0; v < n; v++) {
            for (int i = 0; i < dag.n_parents[v]; i++) {
                int u = dag.parents[v * dag.max_parents + i];
                if (tabu_contains(&tabu, OP_REMOVE, u, v)) continue;

                /* Score delta for removing u as parent of v */
                int new_parents[16];
                int np = 0;
                for (int k = 0; k < dag.n_parents[v]; k++) {
                    int p = dag.parents[v * dag.max_parents + k];
                    if (p != u) new_parents[np++] = p;
                }

                double old_score = cached_k2_score(
                    &ds, &cache, v, dag.parents + v * dag.max_parents, dag.n_parents[v]);
                double new_score = cached_k2_score(
                    &ds, &cache, v, new_parents, np);
                double delta = new_score - old_score;

                if (delta > best_delta) {
                    best_delta = delta;
                    best_op = OP_REMOVE;
                    best_u = u;
                    best_v = v;
                }
            }
        }

        /* Evaluate all possible FLIP operations */
        for (int v = 0; v < n; v++) {
            for (int i = 0; i < dag.n_parents[v]; i++) {
                int u = dag.parents[v * dag.max_parents + i];
                if (tabu_contains(&tabu, OP_FLIP, u, v)) continue;
                if (tabu_contains(&tabu, OP_FLIP, v, u)) continue;
                if (dag.n_parents[u] >= max_indegree) continue;

                /*
                 * Flip u→v to v→u.
                 * Acyclicity: After removing u→v and adding v→u,
                 * there is a cycle iff there is a path from u to v
                 * in the DAG *without* the edge u→v.
                 * Check: any path u→...→v of length > 1?
                 */
                /* Temporarily remove u→v */
                dag_remove_edge(&dag, u, v);
                int *co2, *ch2;
                dag_build_children(&dag, &co2, &ch2);
                int creates_cycle = dag_has_path_fast(n, co2, ch2, u, v);
                free(co2); free(ch2);
                dag_add_edge(&dag, u, v); /* restore */

                if (creates_cycle) continue;

                /* Score delta for flipping u→v to v→u */
                /* Remove u from v's parents */
                int v_new_parents[16];
                int v_np = 0;
                for (int k = 0; k < dag.n_parents[v]; k++) {
                    int p = dag.parents[v * dag.max_parents + k];
                    if (p != u) v_new_parents[v_np++] = p;
                }

                /* Add v to u's parents */
                int u_new_parents[16];
                int u_np = dag.n_parents[u];
                memcpy(u_new_parents, dag.parents + u * dag.max_parents, u_np * sizeof(int));
                u_new_parents[u_np] = v;

                double old_v_score = cached_k2_score(
                    &ds, &cache, v, dag.parents + v * dag.max_parents, dag.n_parents[v]);
                double new_v_score = cached_k2_score(
                    &ds, &cache, v, v_new_parents, v_np);
                double old_u_score = cached_k2_score(
                    &ds, &cache, u, dag.parents + u * dag.max_parents, dag.n_parents[u]);
                double new_u_score = cached_k2_score(
                    &ds, &cache, u, u_new_parents, u_np + 1);

                double delta = (new_v_score + new_u_score) - (old_v_score + old_u_score);

                if (delta > best_delta) {
                    best_delta = delta;
                    best_op = OP_FLIP;
                    best_u = u;
                    best_v = v;
                }
            }
        }

        free(child_offset);
        free(children_arr);

        /* Apply best operation or stop */
        if (best_op < 0 || best_delta < epsilon) break;

        if (best_op == OP_ADD) {
            dag_add_edge(&dag, best_u, best_v);
            tabu_push(&tabu, OP_REMOVE, best_u, best_v);
        } else if (best_op == OP_REMOVE) {
            dag_remove_edge(&dag, best_u, best_v);
            tabu_push(&tabu, OP_ADD, best_u, best_v);
        } else if (best_op == OP_FLIP) {
            dag_remove_edge(&dag, best_u, best_v);
            dag_add_edge(&dag, best_v, best_u);
            tabu_push(&tabu, OP_FLIP, best_u, best_v);
        }
    }

    /* Build result: list of (parent, child) tuples */
    PyObject *result = PyList_New(0);
    for (int v = 0; v < (int)n_vars; v++) {
        for (int i = 0; i < dag.n_parents[v]; i++) {
            int u = dag.parents[v * dag.max_parents + i];
            PyObject *tup = Py_BuildValue("(ii)", u, v);
            PyList_Append(result, tup);
            Py_DECREF(tup);
        }
    }

    /* Cleanup */
    free(base_scores);
    dag_free(&dag);
    tabu_free(&tabu);
    score_cache_free(&cache);
    free(ds.values);
    free(ds.cardinalities);

    return result;
}

/* Also expose K2 local score directly for testing */
static PyObject *py_k2_local_score(PyObject *self, PyObject *args) {
    PyObject *data_obj, *card_obj, *parents_obj;
    int child;

    if (!PyArg_ParseTuple(args, "OOiO", &data_obj, &card_obj, &child, &parents_obj))
        return NULL;

    Py_ssize_t n_samples = PySequence_Size(data_obj);
    Py_ssize_t n_vars = PySequence_Size(card_obj);

    Dataset ds;
    ds.n_samples = (int)n_samples;
    ds.n_vars = (int)n_vars;
    ds.cardinalities = (int *)malloc(n_vars * sizeof(int));
    ds.values = (int *)malloc(n_samples * n_vars * sizeof(int));

    for (Py_ssize_t j = 0; j < n_vars; j++) {
        PyObject *c = PySequence_GetItem(card_obj, j);
        ds.cardinalities[j] = (int)PyLong_AsLong(c);
        Py_DECREF(c);
    }
    for (Py_ssize_t i = 0; i < n_samples; i++) {
        PyObject *row = PySequence_GetItem(data_obj, i);
        for (Py_ssize_t j = 0; j < n_vars; j++) {
            PyObject *val = PySequence_GetItem(row, j);
            ds.values[i * n_vars + j] = (int)PyLong_AsLong(val);
            Py_DECREF(val);
        }
        Py_DECREF(row);
    }

    Py_ssize_t n_parents = PySequence_Size(parents_obj);
    int *parents = (int *)malloc(n_parents * sizeof(int));
    for (Py_ssize_t i = 0; i < n_parents; i++) {
        PyObject *p = PySequence_GetItem(parents_obj, i);
        parents[i] = (int)PyLong_AsLong(p);
        Py_DECREF(p);
    }

    double score = k2_local_score(&ds, child, parents, (int)n_parents);

    free(parents);
    free(ds.values);
    free(ds.cardinalities);

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
    PyObject *data_obj, *card_obj, *edges_obj;

    if (!PyArg_ParseTuple(args, "OOO", &data_obj, &card_obj, &edges_obj))
        return NULL;

    Py_ssize_t n_samples = PySequence_Size(data_obj);
    Py_ssize_t n_vars = PySequence_Size(card_obj);

    if (n_samples <= 0 || n_vars <= 0) {
        return PyDict_New();
    }

    /* Parse dataset */
    int *values = (int *)malloc(n_samples * n_vars * sizeof(int));
    int *cardinalities = (int *)malloc(n_vars * sizeof(int));
    if (!values || !cardinalities) {
        free(values); free(cardinalities);
        return PyErr_NoMemory();
    }

    for (Py_ssize_t j = 0; j < n_vars; j++) {
        PyObject *c = PySequence_GetItem(card_obj, j);
        cardinalities[j] = (int)PyLong_AsLong(c);
        Py_DECREF(c);
    }
    for (Py_ssize_t i = 0; i < n_samples; i++) {
        PyObject *row = PySequence_GetItem(data_obj, i);
        for (Py_ssize_t j = 0; j < n_vars; j++) {
            PyObject *val = PySequence_GetItem(row, j);
            values[i * n_vars + j] = (int)PyLong_AsLong(val);
            Py_DECREF(val);
        }
        Py_DECREF(row);
    }

    Dataset ds;
    ds.values = values;
    ds.n_samples = (int)n_samples;
    ds.n_vars = (int)n_vars;
    ds.cardinalities = cardinalities;

    /* Parse edges into parent lists */
    Py_ssize_t n_edges = PySequence_Size(edges_obj);
    int *parent_count = (int *)calloc(n_vars, sizeof(int));
    /* max_indegree is at most n_vars; allocate generously */
    int **parent_lists = (int **)calloc(n_vars, sizeof(int *));
    for (Py_ssize_t j = 0; j < n_vars; j++) {
        parent_lists[j] = (int *)malloc(n_vars * sizeof(int));
    }

    for (Py_ssize_t e = 0; e < n_edges; e++) {
        PyObject *edge = PySequence_GetItem(edges_obj, e);
        PyObject *py_u = PySequence_GetItem(edge, 0);
        PyObject *py_v = PySequence_GetItem(edge, 1);
        int u = (int)PyLong_AsLong(py_u);
        int v = (int)PyLong_AsLong(py_v);
        Py_DECREF(py_u); Py_DECREF(py_v); Py_DECREF(edge);
        parent_lists[v][parent_count[v]++] = u;
    }

    /* Build CPDs for all variables */
    PyObject *result = PyDict_New();

    for (int v = 0; v < (int)n_vars; v++) {
        int n_parent_configs;
        int *counts = compute_state_counts(
            &ds, v, parent_lists[v], parent_count[v], &n_parent_configs
        );
        if (!counts) continue;

        int child_card = cardinalities[v];

        /* Build Python list of lists with Laplace smoothing */
        PyObject *cpd_list = PyList_New(n_parent_configs);
        for (int j = 0; j < n_parent_configs; j++) {
            /* Compute total count + smoothing for this parent config */
            double total = (double)child_card; /* Laplace: add 1 per state */
            for (int k = 0; k < child_card; k++) {
                total += (double)counts[j * child_card + k];
            }

            PyObject *probs = PyList_New(child_card);
            for (int k = 0; k < child_card; k++) {
                double prob = ((double)counts[j * child_card + k] + 1.0) / total;
                PyList_SET_ITEM(probs, k, PyFloat_FromDouble(prob));
            }
            PyList_SET_ITEM(cpd_list, j, probs);
        }

        free(counts);

        PyObject *key = PyLong_FromLong(v);
        PyDict_SetItem(result, key, cpd_list);
        Py_DECREF(key);
        Py_DECREF(cpd_list);
    }

    /* Cleanup */
    for (Py_ssize_t j = 0; j < n_vars; j++) free(parent_lists[j]);
    free(parent_lists);
    free(parent_count);
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
