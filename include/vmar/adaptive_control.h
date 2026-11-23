#ifndef VMAR_ADAPTIVE_CONTROL_H
#define VMAR_ADAPTIVE_CONTROL_H

#include "vmar/control.h"
#include "vmar/custom_activation.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Augmented continuous state [theta(d), memory(d)]; theta includes alpha
 * for trainable activations. A single gradient observation drives memory.
 * Coupling is experimental and is zero in the observational baseline. */
typedef struct vmar_adaptive_system {
    vmar_custom_neuron *neuron;
    double tau;
    double coupling;
} vmar_adaptive_system;

vmar_status vmar_adaptive_dimension(const vmar_adaptive_system *system,
                                    size_t *dimension);
vmar_status vmar_adaptive_field(double time, const double *state,
                                size_t dimension, double *derivative,
                                void *context);

/* Reuses vmar_control_policy and vmar_control_report. A rejected proposal
 * returns VMAR_OK and leaves state, time and step_id unmodified. */
vmar_status vmar_adaptive_control_step(vmar_integrator method,
    vmar_adaptive_system *system, const vmar_control_policy *policy,
    double *state, size_t dimension, double *time, size_t *step_id,
    double *workspace, size_t workspace_doubles,
    vmar_control_report *report);

#ifdef __cplusplus
}
#endif
#endif
