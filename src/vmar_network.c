#include "vmar/network.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

static int valid_activation(vmar_activation a) {
    return a >= VMAR_ACT_TANH && a <= VMAR_ACT_LEAKY_RELU;
}

static int add_ok(size_t a, size_t b) { return a <= SIZE_MAX - b; }
static int mul_ok(size_t a, size_t b) { return !a || b <= SIZE_MAX / a; }

vmar_status vmar_network_layout_make(const size_t *widths,
                                    size_t widths_count,
                                    const vmar_activation *activations,
                                    size_t *offsets, size_t offsets_capacity,
                                    vmar_network_layout *out) {
    if (!widths || !activations || !offsets || !out || widths_count < 2)
        return VMAR_ERR_ARGUMENT;
    const size_t layers = widths_count - 1;
    if (offsets_capacity < layers) return VMAR_ERR_WORKSPACE;
    if (widths[widths_count - 1] != 1) return VMAR_ERR_DIMENSION;
    size_t total = 0, sum = 0, max = 0;
    for (size_t i = 0; i < widths_count; ++i) {
        if (!widths[i]) return VMAR_ERR_DIMENSION;
        if (!add_ok(sum, widths[i])) return VMAR_ERR_DIMENSION;
        sum += widths[i];
        if (widths[i] > max) max = widths[i];
    }
    for (size_t i = 0; i < layers; ++i) {
        if (!valid_activation(activations[i]) ||
            !mul_ok(widths[i], widths[i+1])) return VMAR_ERR_DIMENSION;
        const size_t w = widths[i] * widths[i+1];
        if (!add_ok(w, widths[i+1]) ||
            !add_ok(total, w + widths[i+1])) return VMAR_ERR_DIMENSION;
        total += w + widths[i+1];
    }
    if (!mul_ok(total, sizeof(double)) ||
        !mul_ok(sum, sizeof(double))) return VMAR_ERR_DIMENSION;
    size_t current = 0;
    for (size_t i = 0; i < layers; ++i) {
        offsets[i] = current;
        current += widths[i] * widths[i+1] + widths[i+1];
    }
    *out = (vmar_network_layout){widths, activations, offsets, layers,
                                  total, sum, max};
    return VMAR_OK;
}

static vmar_status check_layout(const vmar_network_layout *layout) {
    if (!layout || !layout->widths || !layout->activations ||
        !layout->offsets || !layout->layer_count ||
        layout->widths[layout->layer_count] != 1)
        return VMAR_ERR_ARGUMENT;
    size_t total = 0, sum = 0, max = 0;
    for (size_t i = 0; i <= layout->layer_count; ++i) {
        size_t width = layout->widths[i];
        if (!width || !add_ok(sum, width)) return VMAR_ERR_DIMENSION;
        sum += width;
        if (width > max) max = width;
        if (i == layout->layer_count) break;
        if (!valid_activation(layout->activations[i]) ||
            layout->offsets[i] != total ||
            !mul_ok(width, layout->widths[i+1])) return VMAR_ERR_DIMENSION;
        size_t count = width * layout->widths[i+1];
        if (!add_ok(count, layout->widths[i+1]) ||
            !add_ok(total, count + layout->widths[i+1]))
            return VMAR_ERR_DIMENSION;
        total += count + layout->widths[i+1];
    }
    return layout->parameters == total && layout->sum_widths == sum &&
           layout->max_width == max ? VMAR_OK : VMAR_ERR_DIMENSION;
}

vmar_status vmar_network_workspace_required(const vmar_network_layout *layout,
                                            size_t *required) {
    if (!required) return VMAR_ERR_ARGUMENT;
    vmar_status valid = check_layout(layout);
    if (valid != VMAR_OK) return valid;
    if (!mul_ok(layout->sum_widths, 2) ||
        !mul_ok(layout->max_width, 2) ||
        !add_ok(2 * layout->sum_widths, 2 * layout->max_width))
        return VMAR_ERR_DIMENSION;
    size_t n = 2 * layout->sum_widths + 2 * layout->max_width;
    if (!mul_ok(n, sizeof(double))) return VMAR_ERR_DIMENSION;
    *required = n;
    return VMAR_OK;
}

static vmar_status validate_data(const vmar_network_layout *layout,
                                 const double *x, size_t samples,
                                 const double *theta, double *work,
                                 size_t work_len, size_t *required) {
    vmar_status valid = vmar_network_workspace_required(layout, required);
    if (valid != VMAR_OK) return valid;
    if (!x || !theta || !work || !samples) return VMAR_ERR_ARGUMENT;
    if (work_len < *required) return VMAR_ERR_WORKSPACE;
    if (!mul_ok(samples, layout->widths[0])) return VMAR_ERR_DIMENSION;
    for (size_t i = 0; i < samples * layout->widths[0]; ++i)
        if (!isfinite(x[i])) return VMAR_ERR_NUMERIC;
    for (size_t j = 0; j < layout->parameters; ++j)
        if (!isfinite(theta[j])) return VMAR_ERR_NUMERIC;
    return VMAR_OK;
}

/* Activations occupy work[0..sum_widths), derivatives the next block.
 * Input is copied for each sample. Each layer is evaluated exactly once.
 */
static vmar_status evaluate_sample(const vmar_network_layout *layout,
                                   const double *input, const double *theta,
                                   double *work) {
    double *a = work, *d = work + layout->sum_widths;
    size_t base = 0;
    memcpy(a, input, layout->widths[0] * sizeof(double));
    for (size_t l = 0; l < layout->layer_count; ++l) {
        size_t ni = layout->widths[l], no = layout->widths[l+1];
        size_t offset = layout->offsets[l], wcount = ni * no;
        for (size_t j = 0; j < no; ++j) {
            double z = theta[offset + wcount + j];
            for (size_t i = 0; i < ni; ++i)
                z += theta[offset + j*ni + i] * a[base + i];
            if (!isfinite(z)) return VMAR_ERR_NUMERIC;
            double value, derivative;
            vmar_status s = vmar_activation_evaluate(layout->activations[l],
                                                      z, &value, &derivative);
            if (s != VMAR_OK) return s;
            a[base + ni + j] = value;
            d[base + ni + j] = derivative;
        }
        base += ni;
    }
    return VMAR_OK;
}

vmar_status vmar_network_forward(const vmar_network_layout *layout,
                                 const double *x, size_t samples,
                                 const double *theta,
                                 double *workspace, size_t workspace_doubles,
                                 double *predictions) {
    size_t required = 0;
    vmar_status s = validate_data(layout, x, samples, theta, workspace,
                                   workspace_doubles, &required);
    if (s != VMAR_OK) return s;
    if (!predictions) return VMAR_ERR_ARGUMENT;
    for (size_t r = 0; r < samples; ++r) {
        s = evaluate_sample(layout, x + r*layout->widths[0], theta, workspace);
        if (s != VMAR_OK) return s;
        predictions[r] = workspace[layout->sum_widths - 1];
    }
    return VMAR_OK;
}

vmar_status vmar_network_loss_gradient(const vmar_network_layout *layout,
                                       const double *x, const double *target,
                                       size_t samples, const double *theta,
                                       double *workspace,
                                       size_t workspace_doubles,
                                       double *loss, double *gradient,
                                       double *predictions) {
    size_t required = 0;
    vmar_status s = validate_data(layout, x, samples, theta, workspace,
                                   workspace_doubles, &required);
    if (s != VMAR_OK) return s;
    if (!target || !loss || !gradient) return VMAR_ERR_ARGUMENT;
    for (size_t r = 0; r < samples; ++r)
        if (!isfinite(target[r])) return VMAR_ERR_NUMERIC;

    for (size_t i = 0; i < layout->parameters; ++i) gradient[i] = 0.0;
    double objective = 0;
    const size_t sum = layout->sum_widths;
    double *a = workspace, *d = workspace + sum;
    double *delta = workspace + 2 * sum;
    double *next_delta = delta + layout->max_width;
    for (size_t r = 0; r < samples; ++r) {
        s = evaluate_sample(layout, x + r*layout->widths[0], theta, workspace);
        if (s != VMAR_OK) return s;
        const double y = a[sum - 1];
        if (predictions) predictions[r] = y;
        const double error = y - target[r];
        objective += error * error / (double)samples;
        if (!isfinite(objective)) return VMAR_ERR_NUMERIC;
        delta[0] = (2.0 * error / (double)samples) * d[sum - 1];

        /* layer ends[k] = sum widths through output of layer k.
         * Iterate backward while preserving the old delta until complete.
         */
        size_t layer_end = sum;
        for (size_t rev = layout->layer_count; rev > 0; --rev) {
            size_t l = rev - 1;
            size_t ni = layout->widths[l], no = layout->widths[l+1];
            size_t previous_start = layer_end - no - ni;
            size_t offset = layout->offsets[l], wcount = ni * no;
            for (size_t j = 0; j < no; ++j) {
                for (size_t i = 0; i < ni; ++i)
                    gradient[offset + j*ni + i] +=
                        delta[j] * a[previous_start + i];
                gradient[offset + wcount + j] += delta[j];
            }
            if (l > 0) {
                for (size_t i = 0; i < ni; ++i) {
                    double value = 0.0;
                    for (size_t j = 0; j < no; ++j)
                        value += delta[j] * theta[offset + j*ni + i];
                    next_delta[i] = value * d[previous_start + i];
                }
                memcpy(delta, next_delta, ni * sizeof(double));
            }
            layer_end -= no;
        }
    }
    for (size_t i = 0; i < layout->parameters; ++i)
        if (!isfinite(gradient[i])) return VMAR_ERR_NUMERIC;
    *loss = objective;
    return VMAR_OK;
}

vmar_status vmar_network_field(double time, const double *theta,
                               size_t dimension, double *derivative,
                               void *context) {
    if (!isfinite(time) || !theta || !derivative || !context)
        return VMAR_ERR_ARGUMENT;
    const vmar_network_problem *p = context;
    if (!p->layout || dimension != p->layout->parameters ||
        !isfinite(p->eta) || p->eta <= 0) return VMAR_ERR_DIMENSION;
    double loss;
    vmar_status s = vmar_network_loss_gradient(p->layout, p->x, p->target,
        p->samples, theta, p->workspace, p->workspace_doubles,
        &loss, derivative, NULL);
    if (s != VMAR_OK) return s;
    for (size_t i = 0; i < dimension; ++i) {
        derivative[i] *= -p->eta;
        if (!isfinite(derivative[i])) return VMAR_ERR_NUMERIC;
    }
    return VMAR_OK;
}
