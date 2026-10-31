/* The Python entry points of the _ports extension, registered in module.c. */
#ifndef PYGRAPHC_PORTS_ENTRY_POINTS_H
#define PYGRAPHC_PORTS_ENTRY_POINTS_H

#include "../_core_src/models.h"

PYGRAPHC_INTERNAL PyObject *py_parse_port_graph(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_port_graph_shape(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_port_offsets(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_port_component_labels(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_port_connected_components(PyObject *self, PyObject *args);
PYGRAPHC_INTERNAL PyObject *py_port_connected_components_with_branch_ids(PyObject *self, PyObject *args);

#endif
