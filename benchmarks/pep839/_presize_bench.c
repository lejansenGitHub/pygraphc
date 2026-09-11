/* Measurement-only extension for the PEP 839 PyFrozenSetWriter estimate.
 *
 * Times set and frozenset construction over a fixed member list, comparing the
 * library's current incremental PySet_Add loop against the two argument types
 * that reach CPython's presize path (a set and an exact dict).  The source
 * container is built by the caller before the timed region, so the reported
 * time covers insertion into an already-sized table and nothing else.
 *
 * Not part of the library.  Uses only the stable C API.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <string.h>
#include <time.h>

/* Keeps the hash-only loop observable to the optimizer. */
static volatile Py_hash_t hash_sink;

static double
monotonic_seconds(void)
{
    struct timespec timestamp;
    clock_gettime(CLOCK_MONOTONIC, &timestamp);
    return (double)timestamp.tv_sec + 1e-9 * (double)timestamp.tv_nsec;
}

/* Build `repeats` containers and report the wall time of the build loop only.
 *
 * Built containers are held alive until after the clock stops, matching the
 * real workload where every component set stays live, and are released outside
 * the timed region.
 */
static PyObject *
time_build(PyObject *module, PyObject *args)
{
    const char *mode;
    PyObject *source;
    Py_ssize_t repeats;

    if (!PyArg_ParseTuple(args, "sOn", &mode, &source, &repeats)) {
        return NULL;
    }
    if (repeats <= 0) {
        PyErr_SetString(PyExc_ValueError, "repeats must be positive");
        return NULL;
    }

    if (strcmp(mode, "hash_only") == 0) {
        if (!PyList_CheckExact(source)) {
            PyErr_SetString(PyExc_TypeError, "hash_only needs a list");
            return NULL;
        }
        Py_ssize_t count = PyList_GET_SIZE(source);
        PyObject **items = count > 0 ? &PyList_GET_ITEM(source, 0) : NULL;
        Py_hash_t accumulator = 0;
        double hash_started = monotonic_seconds();
        for (Py_ssize_t repeat = 0; repeat < repeats; repeat++) {
            for (Py_ssize_t index = 0; index < count; index++) {
                Py_hash_t hash = PyObject_Hash(items[index]);
                if (hash == -1 && PyErr_Occurred()) {
                    return NULL;
                }
                accumulator += hash;
            }
        }
        double hash_elapsed = monotonic_seconds() - hash_started;
        hash_sink = accumulator;
        return Py_BuildValue("(dnn)", hash_elapsed, repeats, repeats * count);
    }

    int frozen = strncmp(mode, "frozen", 6) == 0;
    int incremental = strstr(mode, "incremental") != NULL;

    PyObject **members = NULL;
    Py_ssize_t member_count = 0;
    if (incremental) {
        if (!PyList_CheckExact(source)) {
            PyErr_SetString(PyExc_TypeError, "incremental modes need a list");
            return NULL;
        }
        member_count = PyList_GET_SIZE(source);
        if (member_count > 0) {
            members = &PyList_GET_ITEM(source, 0);
        }
    }

    PyObject **built = PyMem_Malloc((size_t)repeats * sizeof(PyObject *));
    if (built == NULL) {
        return PyErr_NoMemory();
    }
    Py_ssize_t done = 0;
    int failed = 0;

    double started = monotonic_seconds();
    if (incremental) {
        for (Py_ssize_t repeat = 0; repeat < repeats; repeat++) {
            PyObject *container = frozen ? PyFrozenSet_New(NULL) : PySet_New(NULL);
            if (container == NULL) {
                failed = 1;
                break;
            }
            built[done++] = container;
            for (Py_ssize_t index = 0; index < member_count; index++) {
                if (PySet_Add(container, members[index]) < 0) {
                    failed = 1;
                    break;
                }
            }
            if (failed) {
                break;
            }
        }
    }
    else {
        for (Py_ssize_t repeat = 0; repeat < repeats; repeat++) {
            PyObject *container = frozen ? PyFrozenSet_New(source) : PySet_New(source);
            if (container == NULL) {
                failed = 1;
                break;
            }
            built[done++] = container;
        }
    }
    double elapsed = monotonic_seconds() - started;

    Py_ssize_t total_members = 0;
    int shared = 0;
    for (Py_ssize_t index = 0; index < done; index++) {
        total_members += PySet_GET_SIZE(built[index]);
        if (Py_REFCNT(built[index]) != 1) {
            shared = 1;
        }
        Py_DECREF(built[index]);
    }
    PyMem_Free(built);

    if (failed) {
        return NULL;
    }
    if (shared) {
        PyErr_SetString(PyExc_RuntimeError,
                        "a built container was shared; timing is not attributable");
        return NULL;
    }
    return Py_BuildValue("(dnn)", elapsed, done, total_members);
}

static PyMethodDef methods[] = {
    {"time_build", time_build, METH_VARARGS,
     "time_build(mode, source, repeats) -> (seconds, containers, members)"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module_definition = {
    PyModuleDef_HEAD_INIT, "_presize_bench", NULL, -1, methods,
};

PyMODINIT_FUNC
PyInit__presize_bench(void)
{
    return PyModule_Create(&module_definition);
}
