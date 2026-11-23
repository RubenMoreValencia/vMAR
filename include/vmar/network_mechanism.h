#ifndef VMAR_NETWORK_MECHANISM_H
#define VMAR_NETWORK_MECHANISM_H

#include "vmar/network_system.h"

#ifdef __cplusplus
extern "C" {
#endif

/* G0.7-C: one canonical state [theta, memory], regardless of mode.
 * M0/M1 form the stable baseline; M2/M3 remain opt-in experiments.
 */
typedef enum vmar_network_mode {
    VMAR_NETWORK_BASELINE = 1,    /* M0: theta only; memory unchanged */
    VMAR_NETWORK_OBSERVE = 2,     /* M1: memory evolves, coupling zero */
    VMAR_NETWORK_COUPLED = 3,     /* M2: fixed coupling, fixed step */
    VMAR_NETWORK_CONTROLLED = 4   /* M3: coupling and backtracking */
} vmar_network_mode;

typedef struct vmar_network_mechanism {
    vmar_network_mode mode;
    vmar_network_system system;
    vmar_control_policy policy; /* used only by M3 */
} vmar_network_mechanism;

typedef struct vmar_network_decision {
    vmar_network_mode mode;
    int accepted;
    size_t attempts;
    size_t rejected;
    size_t field_evaluations;
    double loss_before;
    double loss_after;
    double gradient_norm_before;
    double memory_norm_before;
    double coupling_term_norm_before;
    double requested_step;
    double effective_step;
    double time_before;
    double time_after;
} vmar_network_decision;

/* Select the stable observational mode by default. This is a descriptor,
 * not a mutable process-wide configuration or an optimization policy. */
vmar_network_mode vmar_network_default_mode(void);

/* Supports all modes with a single 2p state and explicit, caller-owned
 * workspace. M3 uses policy.initial_step; all other modes use `step`.
 * The neural problem has an additional, separate scratch workspace.
 * Rejections leave state/time/step_id unchanged and return VMAR_OK with
 * decision.accepted=0. Decision is updated on successful API return.
 */
vmar_status vmar_network_mechanism_step(
    vmar_integrator method, const vmar_network_mechanism *mechanism,
    double step, double *state, size_t dimension, double *time,
    size_t *step_id, double *workspace, size_t workspace_doubles,
    vmar_network_decision *decision);

#ifdef __cplusplus
}
#endif

#endif
