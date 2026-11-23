#ifndef VMAR_NEURON_H
#define VMAR_NEURON_H

#include "vmar/vmar.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef enum vmar_activation {
    VMAR_ACT_TANH = 1,
    VMAR_ACT_SIGMOID = 2,
    VMAR_ACT_IDENTITY = 3,
    VMAR_ACT_SOFTPLUS = 4,
    VMAR_ACT_RELU = 5,
    VMAR_ACT_LEAKY_RELU = 6
} vmar_activation;

/*
 * Evaluate a built-in activation and its reference derivative.
 * At z=0, ReLU uses derivative 0, leaky ReLU uses derivative 0.01.
 * Both are piecewise conventions, not classical derivatives at the kink.
 */
vmar_status vmar_activation_evaluate(vmar_activation kind, double z,
                                     double *value, double *derivative);

/*
 * Single neuron: z_i = X_i * weights + bias; prediction_i = phi(z_i).
 * X is an observation-major, contiguous row-major matrix: X[i*p + j].
 * theta has p weights followed by one bias, i.e., dimension = features + 1.
 * All pointers remain owned by the caller. The dataset must remain immutable
 * during an integration step. This contract specifies a constant learning rate.
 */
typedef struct vmar_neuron_problem {
    const double *x;
    const double *target;
    size_t samples;
    size_t features;
    vmar_activation activation;
    double eta;
} vmar_neuron_problem;

/*
 * Evaluate MSE and its exact analytic gradient; prediction may be NULL.
 * gradient has features+1 doubles; prediction, if present, has samples doubles.
 * Outputs must not overlap the dataset, theta, or each other. On success,
 * they contain finite results. Outputs on failure are unspecified.
 */
vmar_status vmar_neuron_loss_gradient(const vmar_neuron_problem *problem,
                                      const double *theta, double *loss,
                                      double *gradient, double *prediction);

/* vmar_field_fn-compatible field: dtheta/dt = -eta * gradient(MSE). */
vmar_status vmar_neuron_field(double time, const double *theta,
                              size_t dimension, double *derivative,
                              void *context);

#ifdef __cplusplus
}
#endif

#endif
