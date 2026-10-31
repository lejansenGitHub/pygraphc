#include "entry_points.h"
#include "../helpers/parse.h"
#include "../helpers/components.h"

/* ── Label kernels: int32 byte arrays instead of Python containers ──
 *
 * These return raw int32 buffers indexed by node or edge index, so a result
 * over a million nodes is one allocation and no Python object per node.
 */

/* component_labels_ctx(capsule[, edge_mask, node_mask]) -> bytes of int32[n]
 *
 * labels[i] is the smallest node index of the connected component of node i
 * under the masks, -1 for an excluded node. Component ids of
 * compute_components_masked are numbered in order of first appearance while
 * scanning node indices upwards, so the first index carrying an id is the
 * smallest index of its component.
 */
PyObject *py_component_labels_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *mask_obj = Py_None, *nmask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OO", &capsule, &mask_obj, &nmask_obj)) return NULL;
    GraphCtx *g = get_graphctx(capsule);
    if (!g) return NULL;
    int n = g->nid.n;

    const uint8_t *mask; Py_buffer mbuf;
    if (parse_mask(mask_obj, g->nid.el.m, &mask, &mbuf) < 0) return NULL;
    const uint8_t *nmask; Py_buffer nmbuf;
    if (parse_mask(nmask_obj, n, &nmask, &nmbuf) < 0) { release_mask(&mbuf); return NULL; }

    PyObject *result = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)n * (Py_ssize_t)sizeof(int32_t));
    if (!result || n == 0) { release_mask(&nmbuf); release_mask(&mbuf); return result; }

    ComponentResult cr;
    if (compute_components_masked(n, &g->nid.el, mask, nmask, &cr) < 0) {
        Py_DECREF(result); release_mask(&nmbuf); release_mask(&mbuf); return NULL;
    }
    release_mask(&mbuf);

    int *smallest_index = (int *)malloc((size_t)cr.num_comp * sizeof(int));
    if (!smallest_index) {
        free(cr.labels); Py_DECREF(result); release_mask(&nmbuf); PyErr_NoMemory(); return NULL;
    }
    memset(smallest_index, -1, (size_t)cr.num_comp * sizeof(int));
    int32_t *labels = (int32_t *)PyBytes_AS_STRING(result);
    for (int i = 0; i < n; i++) {
        int component = cr.labels[i];
        if (smallest_index[component] == -1) smallest_index[component] = i;
        labels[i] = (nmask && nmask[i]) ? -1 : (int32_t)smallest_index[component];
    }
    free(smallest_index); free(cr.labels);
    release_mask(&nmbuf);
    return result;
}

/* quotient_edges_ctx(capsule, labels[, edge_mask])
 *     -> (src_labels, dst_labels, edge_indices, internal_edge_indices), each bytes of int32
 *
 * One pass over the edges in index order. An unmasked edge whose endpoints
 * carry different labels is a crossing edge and fills the three parallel
 * arrays; one whose endpoints carry the same label is internal. Edges at a
 * node labelled -1 (excluded) are neither. Every edge is reported by its
 * index; nothing is keyed by endpoint pair.
 */
PyObject *py_quotient_edges_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *labels_obj, *mask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "OO|O", &capsule, &labels_obj, &mask_obj)) return NULL;
    GraphCtx *g = get_graphctx(capsule);
    if (!g) return NULL;
    int n = g->nid.n;
    Py_ssize_t m = g->nid.el.m;

    const int32_t *labels; Py_buffer lbuf;
    if (parse_labels(labels_obj, n, &labels, &lbuf) < 0) return NULL;
    const uint8_t *mask; Py_buffer mbuf;
    if (parse_mask(mask_obj, m, &mask, &mbuf) < 0) { PyBuffer_Release(&lbuf); return NULL; }

    /* Four scratch arrays of m entries each; at least one entry so the pointers are never NULL. */
    size_t capacity = (size_t)(m > 0 ? m : 1);
    int32_t *scratch = (int32_t *)malloc(4 * capacity * sizeof(int32_t));
    if (!scratch) { release_mask(&mbuf); PyBuffer_Release(&lbuf); PyErr_NoMemory(); return NULL; }
    int32_t *src_labels = scratch, *dst_labels = scratch + capacity;
    int32_t *crossing = scratch + 2 * capacity, *internal = scratch + 3 * capacity;
    Py_ssize_t crossing_count = 0, internal_count = 0;

    for (Py_ssize_t i = 0; i < m; i++) {
        if (mask && mask[i]) continue;
        int32_t from_label = labels[EDGE_SRC(&g->nid.el, i)];
        int32_t to_label = labels[EDGE_DST(&g->nid.el, i)];
        if (from_label < 0 || to_label < 0) continue;
        if (from_label == to_label) { internal[internal_count++] = (int32_t)i; continue; }
        src_labels[crossing_count] = from_label;
        dst_labels[crossing_count] = to_label;
        crossing[crossing_count++] = (int32_t)i;
    }
    release_mask(&mbuf); PyBuffer_Release(&lbuf);

    Py_ssize_t crossing_bytes = crossing_count * (Py_ssize_t)sizeof(int32_t);
    Py_ssize_t internal_bytes = internal_count * (Py_ssize_t)sizeof(int32_t);
    PyObject *result = Py_BuildValue("(y#y#y#y#)",
                                     (const char *)src_labels, crossing_bytes,
                                     (const char *)dst_labels, crossing_bytes,
                                     (const char *)crossing, crossing_bytes,
                                     (const char *)internal, internal_bytes);
    free(scratch);
    return result;
}

/* degrees_ctx(capsule[, edge_mask, node_mask]) -> bytes of int32[n]
 *
 * Incidence degree of every node under the masks, a self-loop counting twice,
 * 0 for an excluded node and for edges at an excluded node. For a directed
 * graph the out-degree, like degree_ctx.
 */
PyObject *py_degrees_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *mask_obj = Py_None, *nmask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OO", &capsule, &mask_obj, &nmask_obj)) return NULL;
    GraphCtx *g = get_graphctx(capsule);
    if (!g) return NULL;
    int n = g->nid.n;
    Py_ssize_t m = g->nid.el.m;

    const uint8_t *mask; Py_buffer mbuf;
    if (parse_mask(mask_obj, m, &mask, &mbuf) < 0) return NULL;
    const uint8_t *nmask; Py_buffer nmbuf;
    if (parse_mask(nmask_obj, n, &nmask, &nmbuf) < 0) { release_mask(&mbuf); return NULL; }

    PyObject *result = PyBytes_FromStringAndSize(NULL, (Py_ssize_t)n * (Py_ssize_t)sizeof(int32_t));
    if (!result || n == 0) { release_mask(&nmbuf); release_mask(&mbuf); return result; }
    int32_t *degrees = (int32_t *)PyBytes_AS_STRING(result);
    memset(degrees, 0, (size_t)n * sizeof(int32_t));

    for (Py_ssize_t i = 0; i < m; i++) {
        if (mask && mask[i]) continue;
        int u = EDGE_SRC(&g->nid.el, i), v = EDGE_DST(&g->nid.el, i);
        if (nmask && (nmask[u] || nmask[v])) continue;
        degrees[u]++;
        if (!g->directed) degrees[v]++;
    }
    release_mask(&nmbuf); release_mask(&mbuf);
    return result;
}

/* bcc_edge_labels_ctx(capsule[, edge_mask, node_mask]) -> bytes of int32[m]
 *
 * Biconnected component id of every edge, numbered in the order the
 * components complete in the depth-first search. A bridge is a singleton
 * component. Masked edges, edges at an excluded node and self-loops get -1.
 * Same iterative Tarjan as bcc_ctx with an edge stack of edge indices.
 */
PyObject *py_bcc_edge_labels_ctx(PyObject *self, PyObject *args) {
    PyObject *capsule, *mask_obj = Py_None, *nmask_obj = Py_None;
    if (!PyArg_ParseTuple(args, "O|OO", &capsule, &mask_obj, &nmask_obj)) return NULL;
    GraphCtx *g = get_graphctx(capsule);
    if (!g) return NULL;
    int n = g->nid.n;
    Py_ssize_t m = g->nid.el.m;

    PyObject *result = PyBytes_FromStringAndSize(NULL, m * (Py_ssize_t)sizeof(int32_t));
    if (!result) return NULL;
    int32_t *labels = (int32_t *)PyBytes_AS_STRING(result);
    memset(labels, -1, (size_t)m * sizeof(int32_t));
    if (n == 0 || m == 0 || !g->has_adj) return result;

    const uint8_t *mask; Py_buffer mbuf;
    if (parse_mask(mask_obj, m, &mask, &mbuf) < 0) { Py_DECREF(result); return NULL; }
    const uint8_t *nmask; Py_buffer nmbuf;
    if (parse_mask(nmask_obj, n, &nmask, &nmbuf) < 0) { release_mask(&mbuf); Py_DECREF(result); return NULL; }

    AdjList *al = &g->al;
    int *buf5 = (int *)malloc(5 * (size_t)n * sizeof(int));
    int *edge_stack = (int *)malloc((size_t)m * sizeof(int));
    if (!buf5 || !edge_stack) {
        free(buf5); free(edge_stack); release_mask(&nmbuf); release_mask(&mbuf); Py_DECREF(result);
        PyErr_NoMemory(); return NULL;
    }
    int *disc = buf5, *low = buf5 + n, *stk = buf5 + 2*n;
    int *sidx = buf5 + 3*n, *peid = buf5 + 4*n;
    memset(disc, -1, (size_t)n * sizeof(int));
    int timer = 0, esp = 0, component = 0;

    for (int start = 0; start < n; start++) {
        if (disc[start] != -1) continue;
        if (nmask && nmask[start]) continue;
        int sp = 0;
        stk[0] = start; sidx[0] = al->offset[start];
        disc[start] = low[start] = timer++; peid[start] = -1;
        while (sp >= 0) {
            int u = stk[sp];
            if (sidx[sp] < al->offset[u + 1]) {
                int i = sidx[sp]++;
                int v = al->adj[i], eid = al->eid[i];
                if (mask && mask[eid]) continue;
                if (nmask && nmask[v]) continue;
                if (eid == peid[u]) continue;
                if (disc[v] == -1) {
                    edge_stack[esp++] = eid;
                    disc[v] = low[v] = timer++; peid[v] = eid;
                    stk[++sp] = v; sidx[sp] = al->offset[v];
                } else if (disc[v] < disc[u]) {
                    edge_stack[esp++] = eid;
                    if (low[u] > disc[v]) low[u] = disc[v];
                }
            } else {
                if (sp > 0) {
                    int p = stk[sp-1];
                    if (low[p] > low[u]) low[p] = low[u];
                    if (low[u] >= disc[p]) {
                        while (esp > 0) {
                            int eid = edge_stack[--esp];
                            labels[eid] = (int32_t)component;
                            if (eid == peid[u]) break;
                        }
                        component++;
                    }
                }
                sp--;
            }
        }
    }
    free(buf5); free(edge_stack);
    release_mask(&nmbuf); release_mask(&mbuf);
    return result;
}
