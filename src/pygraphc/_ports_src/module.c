/* The _ports extension: graphs whose edges have ports, kept apart from _core so that adding
 * multiport kernels cannot change how the line kernels of _core are compiled and inlined. */
#include "entry_points.h"

static PyMethodDef port_methods[] = {
    {"parse_port_graph", py_parse_port_graph, METH_VARARGS,
     "parse_port_graph(node_ids, edges[, directed]) -> capsule\n\nParse a graph whose edges are tuples of ports."},
    {"port_graph_shape", py_port_graph_shape, METH_VARARGS,
     "port_graph_shape(capsule) -> (node_count, edge_count, port_count) as stored in C."},
    {"port_offsets", py_port_offsets, METH_VARARGS,
     "port_offsets(capsule) -> bytes\n\nint32 offsets: the ports of edge e are offsets[e] .. offsets[e + 1] - 1."},
    {"port_component_labels", py_port_component_labels, METH_VARARGS,
     "port_component_labels(capsule[, edge_mask, port_mask, node_mask]) -> bytes\n\n"
     "int32 smallest node index of each node's component; masked nodes -1."},
    {"port_connected_components", py_port_connected_components, METH_VARARGS,
     "port_connected_components(capsule[, edge_mask, port_mask, node_mask]) -> list[set]"},
    {"port_connected_components_with_branch_ids", py_port_connected_components_with_branch_ids, METH_VARARGS,
     "port_connected_components_with_branch_ids(capsule, branch_ids[, edge_mask, port_mask, node_mask])\n\n"
     "-> list[(node set, branch id set)]; an edge belongs to the component of its active ports."},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef ports_module = {
    PyModuleDef_HEAD_INIT, "_ports", "Graphs whose edges have ports.", -1, port_methods,
};

PyMODINIT_FUNC PyInit__ports(void) { return PyModule_Create(&ports_module); }
