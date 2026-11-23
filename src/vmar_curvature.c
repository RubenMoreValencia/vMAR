#include "vmar/curvature.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

#define VMAR_CURVATURE_MAX_PARAMETERS 16u

static int disjoint(const void *a, size_t a_bytes,
                    const void *b, size_t b_bytes) {
    /* uintptr_t is available on the supported macOS/Linux build targets.
     * Compare address intervals without relational pointer UB. */
    const uintptr_t aa = (uintptr_t)a;
    const uintptr_t bb = (uintptr_t)b;
    if (aa > UINTPTR_MAX - a_bytes || bb > UINTPTR_MAX - b_bytes)
        return 0;
    return (aa + a_bytes <= bb || bb + b_bytes <= aa);
}

vmar_status vmar_curvature_workspace_required(size_t p, size_t *required) {
    if (!required || p == 0 || p > VMAR_CURVATURE_MAX_PARAMETERS)
        return VMAR_ERR_DIMENSION;
    *required = p * p + p;
    return VMAR_OK;
}

static vmar_status evaluate(vmar_loss_fn loss, void *context,
                            double *shifted, const double *theta,
                            size_t p, size_t i, size_t j,
                            double si, double sj, double h,
                            double *out) {
    memcpy(shifted, theta, p * sizeof(*theta));
    shifted[i] += si * h;
    shifted[j] += sj * h;
    for (size_t k = 0; k < p; ++k)
        if (!isfinite(shifted[k])) return VMAR_ERR_NUMERIC;
    double value = 0.0;
    const vmar_status status = loss(shifted, p, context, &value);
    if (status != VMAR_OK) return VMAR_ERR_FIELD;
    if (!isfinite(value)) return VMAR_ERR_NUMERIC;
    *out = value;
    return VMAR_OK;
}

vmar_status vmar_curvature_hessian(const double *theta, size_t p,
                                  vmar_loss_fn loss, void *context,
                                  const vmar_curvature_options *options,
                                  double *workspace, size_t workspace_doubles,
                                  double *hessian,
                                  vmar_curvature_report *report) {
    if (!theta || !loss || !options || !workspace || !hessian || !report)
        return VMAR_ERR_ARGUMENT;
    size_t required = 0;
    if (vmar_curvature_workspace_required(p, &required) != VMAR_OK ||
        options->max_parameters == 0 || p > options->max_parameters)
        return VMAR_ERR_DIMENSION;
    if (!isfinite(options->step) || options->step <= 0)
        return VMAR_ERR_ARGUMENT;
    const size_t evaluations = 2u * p * (p + 1u);
    if (options->max_loss_calls < evaluations)
        return VMAR_ERR_ARGUMENT;
    if (workspace_doubles < required) return VMAR_ERR_WORKSPACE;

    const size_t state_bytes = p * sizeof(*theta);
    const size_t output_bytes = p * p * sizeof(*hessian);
    const size_t work_bytes = required * sizeof(*workspace);
    const size_t report_bytes = sizeof(*report);
    const void *buffers[] = {theta, workspace, hessian, report};
    const size_t sizes[] = {state_bytes, work_bytes, output_bytes, report_bytes};
    for (size_t a = 0; a < 4; ++a)
        for (size_t b = a + 1; b < 4; ++b)
            if (!disjoint(buffers[a], sizes[a], buffers[b], sizes[b]))
                return VMAR_ERR_ARGUMENT;
    for (size_t k = 0; k < p; ++k) {
        if (!isfinite(theta[k])) return VMAR_ERR_NUMERIC;
        /* If the requested perturbation rounds back to the same double,
         * the finite difference has no meaningful coordinate resolution. */
        if (!isfinite(theta[k] + options->step) ||
            !isfinite(theta[k] - options->step) ||
            theta[k] + options->step == theta[k] ||
            theta[k] - options->step == theta[k])
            return VMAR_ERR_NUMERIC;
    }

    double *const temporary = workspace;
    double *const shifted = workspace + p * p;
    const double h = options->step;
    const double denominator = 4.0 * h * h;
    if (!isfinite(denominator) || denominator == 0.0)
        return VMAR_ERR_NUMERIC;

    size_t calls = 0;
    for (size_t i = 0; i < p; ++i) {
        for (size_t j = i; j < p; ++j) {
            double vpp, vpm, vmp, vmm;
            vmar_status s;
            s = evaluate(loss, context, shifted, theta, p, i, j,
                         1.0, 1.0, h, &vpp);
            if (s != VMAR_OK) return s;
            ++calls;
            s = evaluate(loss, context, shifted, theta, p, i, j,
                         1.0, -1.0, h, &vpm);
            if (s != VMAR_OK) return s;
            ++calls;
            s = evaluate(loss, context, shifted, theta, p, i, j,
                         -1.0, 1.0, h, &vmp);
            if (s != VMAR_OK) return s;
            ++calls;
            s = evaluate(loss, context, shifted, theta, p, i, j,
                         -1.0, -1.0, h, &vmm);
            if (s != VMAR_OK) return s;
            ++calls;
            const double value = ((vpp - vpm) - (vmp - vmm)) / denominator;
            if (!isfinite(value)) return VMAR_ERR_NUMERIC;
            temporary[i * p + j] = value;
            temporary[j * p + i] = value;
        }
    }
    /* Transactional commit: neither output nor report changed earlier. */
    memcpy(hessian, temporary, output_bytes);
    *report = (vmar_curvature_report){p, calls, h, 0.0};
    return VMAR_OK;
}
