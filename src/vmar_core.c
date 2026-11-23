#include "vmar/vmar.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static int vmar_all_finite(const double *values, size_t dimension) {
    for (size_t i = 0; i < dimension; ++i) {
        if (!isfinite(values[i])) return 0;
    }
    return 1;
}

vmar_status vmar_workspace_required(vmar_integrator method, size_t dimension,
                                    size_t *required_doubles) {
    if (required_doubles == NULL || dimension == 0) return VMAR_ERR_ARGUMENT;
    size_t factor;
    switch (method) {
        case VMAR_EULER: factor = 1; break;
        case VMAR_RK4: factor = 5; break;
        default: return VMAR_ERR_ARGUMENT;
    }
    if (dimension > SIZE_MAX / factor ||
        dimension * factor > SIZE_MAX / sizeof(double)) {
        return VMAR_ERR_DIMENSION;
    }
    *required_doubles = dimension * factor;
    return VMAR_OK;
}

static vmar_status vmar_evaluate(vmar_field_fn field, double time,
                                 const double *state, size_t dimension,
                                 double *derivative, void *context) {
    if (!isfinite(time) || !vmar_all_finite(state, dimension)) {
        return VMAR_ERR_NUMERIC;
    }
    vmar_status status = field(time, state, dimension, derivative, context);
    if (status != VMAR_OK) return VMAR_ERR_FIELD;
    if (!vmar_all_finite(derivative, dimension)) return VMAR_ERR_NUMERIC;
    return VMAR_OK;
}

vmar_status vmar_step(vmar_integrator method, double time, double step,
                      vmar_field_fn field, void *context,
                      const double *state, size_t dimension,
                      double *workspace, size_t workspace_doubles,
                      double *next_state) {
    if (field == NULL || state == NULL || workspace == NULL ||
        next_state == NULL || !isfinite(time) || !isfinite(step) || step <= 0) {
        return VMAR_ERR_ARGUMENT;
    }
    size_t required = 0;
    vmar_status status = vmar_workspace_required(method, dimension, &required);
    if (status != VMAR_OK) return status;
    if (workspace_doubles < required) return VMAR_ERR_WORKSPACE;
    if (!vmar_all_finite(state, dimension)) return VMAR_ERR_NUMERIC;

    if (method == VMAR_EULER) {
        status = vmar_evaluate(field, time, state, dimension, workspace, context);
        if (status != VMAR_OK) return status;
        for (size_t i = 0; i < dimension; ++i) {
            double value = state[i] + step * workspace[i];
            if (!isfinite(value)) return VMAR_ERR_NUMERIC;
            workspace[i] = value;
        }
        memcpy(next_state, workspace, dimension * sizeof(double));
        return VMAR_OK;
    }

    double *k1 = workspace;
    double *k2 = workspace + dimension;
    double *k3 = workspace + 2 * dimension;
    double *k4 = workspace + 3 * dimension;
    double *temp = workspace + 4 * dimension;

    status = vmar_evaluate(field, time, state, dimension, k1, context);
    if (status != VMAR_OK) return status;
    for (size_t i = 0; i < dimension; ++i) temp[i] = state[i] + 0.5 * step * k1[i];
    status = vmar_evaluate(field, time + 0.5 * step, temp, dimension, k2, context);
    if (status != VMAR_OK) return status;
    for (size_t i = 0; i < dimension; ++i) temp[i] = state[i] + 0.5 * step * k2[i];
    status = vmar_evaluate(field, time + 0.5 * step, temp, dimension, k3, context);
    if (status != VMAR_OK) return status;
    for (size_t i = 0; i < dimension; ++i) temp[i] = state[i] + step * k3[i];
    status = vmar_evaluate(field, time + step, temp, dimension, k4, context);
    if (status != VMAR_OK) return status;
    for (size_t i = 0; i < dimension; ++i) {
        temp[i] = state[i] + step * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]) / 6.0;
        if (!isfinite(temp[i])) return VMAR_ERR_NUMERIC;
    }
    memcpy(next_state, temp, dimension * sizeof(double));
    return VMAR_OK;
}

const char *vmar_status_string(vmar_status status) {
    switch (status) {
        case VMAR_OK: return "ok";
        case VMAR_ERR_ARGUMENT: return "invalid argument";
        case VMAR_ERR_DIMENSION: return "invalid dimension or size overflow";
        case VMAR_ERR_WORKSPACE: return "insufficient workspace";
        case VMAR_ERR_NUMERIC: return "non-finite numerical value";
        case VMAR_ERR_FIELD: return "field callback failed";
        default: return "unknown status";
    }
}
