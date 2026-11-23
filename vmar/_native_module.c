#define PY_SSIZE_T_CLEAN
#include <Python.h>

/* Loading this Python module links vMAR C23 sources into a native binary.
 * ctypes calls the exported vmar_* C symbols; no mathematical rewrite. */
static struct PyModuleDef vmar_native_module = {
    PyModuleDef_HEAD_INIT,
    "_native",
    "Native C23 vMAR differential core for ctypes bindings.",
    -1,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit__native(void) {
    return PyModule_Create(&vmar_native_module);
}
