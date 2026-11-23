#ifndef VMAR_MEMORY_H
#define VMAR_MEMORY_H

#include "vmar/neuron.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Discrete EMA matching vMAR.Core::vmar_memory_update(kind="ema"). */
vmar_status vmar_memory_ema(const double *memory, const double *observation,
                            size_t dimension, double decay, double *next);

/* Exact relaxation for a piecewise-constant signal over one time step. */
vmar_status vmar_memory_relaxation(const double *memory, const double *signal,
                                   size_t dimension, double step, double tau,
                                   double *next);

/*
 * Augmented state s = [theta (features+1), memory (features+1)].
 * The memory signal is the current analytic MSE gradient.
 *   dtheta/dt = -eta * (grad L(theta) + coupling * memory)
 *   dmemory/dt = (grad L(theta) - memory) / tau
 * coupling=0 preserves the baseline neural trajectory; coupling>0 is an
 * experimental hypothesis, not a validated improvement in learning.
 * No independent allocation: the generic integrator owns stage buffers.
 */
typedef struct vmar_learning_system {
    vmar_neuron_problem neuron;
    double tau;
    double coupling;
} vmar_learning_system;

vmar_status vmar_learning_field(double time, const double *state,
                                size_t dimension, double *derivative,
                                void *context);

#ifdef __cplusplus
}
#endif

#endif
