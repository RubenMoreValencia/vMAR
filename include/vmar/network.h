#ifndef VMAR_NETWORK_H
#define VMAR_NETWORK_H

#include "vmar/neuron.h"

#ifdef __cplusplus
extern "C" {
#endif

/* G0.7-B: arbitrary-depth scalar-output fully-connected network.
 * Caller owns descriptor arrays; all weights are row-major by layer,
 * immediately followed by biases. No hidden allocation.
 */
typedef struct vmar_network_layout {
    const size_t *widths;               /* L+1 widths, input first */
    const vmar_activation *activations; /* L activation kinds */
    const size_t *offsets;              /* L theta offsets, caller-owned */
    size_t layer_count;
    size_t parameters;
    size_t sum_widths;
    size_t max_width;
} vmar_network_layout;

/* widths_count>=2, widths last==1. offsets_capacity>=widths_count-1.
 * On an invalid descriptor, *out and offsets are not modified.
 */
vmar_status vmar_network_layout_make(const size_t *widths,
                                    size_t widths_count,
                                    const vmar_activation *activations,
                                    size_t *offsets, size_t offsets_capacity,
                                    vmar_network_layout *out);

/* Workspace in doubles for both forward and gradient operations.
 * The workspace is caller-owned, disjoint from dataset, theta and outputs.
 */
vmar_status vmar_network_workspace_required(const vmar_network_layout *layout,
                                            size_t *required);

/* Row-major input x[sample*input_width+feature]. Predictions length n.
 * Output buffers must be disjoint; output on error is unspecified.
 */
vmar_status vmar_network_forward(const vmar_network_layout *layout,
                                 const double *x, size_t samples,
                                 const double *theta,
                                 double *workspace, size_t workspace_doubles,
                                 double *predictions);

/* MSE is mean((prediction-target)^2); gradient length layout->parameters.
 * Predictions may be NULL. All math uses the same forward network field.
 */
vmar_status vmar_network_loss_gradient(const vmar_network_layout *layout,
                                       const double *x, const double *target,
                                       size_t samples, const double *theta,
                                       double *workspace,
                                       size_t workspace_doubles,
                                       double *loss, double *gradient,
                                       double *predictions);

typedef struct vmar_network_problem {
    const vmar_network_layout *layout;
    const double *x;
    const double *target;
    size_t samples;
    double eta; /* positive differential field scale, not numerical step */
    double *workspace;
    size_t workspace_doubles;
} vmar_network_problem;

/* vmar_field_fn-compatible theta'=-eta*grad(MSE).
 * No special optimizer, controller or memory model is embedded.
 */
vmar_status vmar_network_field(double time, const double *theta,
                               size_t dimension, double *derivative,
                               void *context);

#ifdef __cplusplus
}
#endif

#endif
