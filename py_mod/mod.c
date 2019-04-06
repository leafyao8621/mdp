#include <Python.h>
#include <string.h>
#include "mdp.h"
#include "util.h"

#define GETSTATE(m) ((struct module_state*)PyModule_GetState(m))

struct module_state {
    PyObject* error;
};

static PyObject* evaluate_poisson(PyObject* self, PyObject* args) {
    long s, S;
    double lambda, price, penalty;
    if (!PyArg_ParseTuple(args, "dlldd", &lambda, &s, &S, &price,
        &penalty)) {
        Py_RETURN_NAN;
    }
    double val = evaluate_policy(pois_prob, &lambda, s, S, price, penalty);
    return PyFloat_FromDouble(val);
}

static PyObject* evaluate_normal(PyObject* self, PyObject* args) {
    long s, S;
    double mu, sigma, price, penalty;
    if (!PyArg_ParseTuple(args, "ddlldd", &mu, &sigma, &s, &S, &price,
        &penalty)) {
        Py_RETURN_NAN;
    }
    double par[2];
    *par = mu;
    par[1] = sigma;
    double val = evaluate_policy(norm_prob, par, s, S, price, penalty);
    return PyFloat_FromDouble(val);
}

static PyMethodDef mdp_methods[] = {
    {"evaluate_poisson", (PyCFunction)evaluate_poisson, METH_VARARGS, 0},
    {"evaluate_normal", (PyCFunction)evaluate_normal, METH_VARARGS, 0},
    {0}
};

static int mdp_traverse(PyObject *m, visitproc visit, void *arg) {
    Py_VISIT(GETSTATE(m)->error);
    return 0;
}

static int mdp_clear(PyObject *m) {
    Py_CLEAR(GETSTATE(m)->error);
    return 0;
}

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "mdp",
    NULL,
    sizeof(struct module_state),
    mdp_methods,
    NULL,
    mdp_traverse,
    mdp_clear,
    NULL
};

PyMODINIT_FUNC PyInit_mdp(void) {
    return PyModule_Create(&moduledef);
}
