#ifndef VMAR_CONTROL_H
#define VMAR_CONTROL_H

#include "vmar/memory.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Deterministic experimental feedback controller for augmented neuron states.
 * Loss acceptance uses only theta; memory changes only on committed steps.
 * The optional memory gate observes the CONFIRMED pre-step memory norm,
 * never the tentative memory, and reduces the initial proposed step.
 * Backtracking retries from the same confirmed state and time.
 */
typedef struct vmar_control_policy {
    double initial_step;
    double reduction;       /* 0 < reduction < 1 */
    size_t max_attempts;    /* >=1 */
    double loss_tolerance;  /* >=0; candidate_loss <= old_loss + tolerance */
    double memory_threshold;/* >=0; gate enabled when >0 */
} vmar_control_policy;

typedef struct vmar_control_report {
    int accepted;
    size_t attempts;
    size_t rejected;
    size_t field_evaluations;
    double old_loss;
    double candidate_loss;
    double effective_step;
    double final_time;
} vmar_control_report;

/*
 * The caller supplies workspace with >= vmar_control_workspace_required()
 * doubles. On failure or a rejected proposal, state, time and step_id remain
 * unchanged. On acceptance, all are committed atomically after evaluation.
 * Error reports are unspecified; ordinary rejection returns VMAR_OK and
 * report.accepted=0. The output report must not overlap state/workspace.
 */
vmar_status vmar_control_workspace_required(vmar_integrator method,
                                            size_t dimension, size_t *required);

vmar_status vmar_control_step(vmar_integrator method,
                              const vmar_learning_system *system,
                              const vmar_control_policy *policy,
                              double *state, size_t dimension,
                              double *time, size_t *step_id,
                              double *workspace, size_t workspace_doubles,
                              vmar_control_report *report);

#ifdef __cplusplus
}
#endif

#endif
