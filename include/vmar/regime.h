#ifndef VMAR_REGIME_H
#define VMAR_REGIME_H

#include "vmar/adaptive_control.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Experimental diagnostic on confirmed [theta,m] only. All quantities are
 * descriptive; the regime is not a Lyapunov or generalization certificate. */
typedef enum vmar_regime {
    VMAR_REGIME_STATIONARY=0,
    VMAR_REGIME_UNOBSERVED=1,
    VMAR_REGIME_ALIGNED=2,
    VMAR_REGIME_OPPOSED=3,
    VMAR_REGIME_ORTHOGONAL=4
} vmar_regime;

typedef struct vmar_regime_report {
    vmar_regime regime;
    double gradient_norm;
    double memory_norm;
    double alignment;
    double risk_ratio;
    double coupling;
} vmar_regime_report;

vmar_status vmar_regime_diagnose(vmar_adaptive_system *system,
    const double *state,size_t dimension,double max_coupling,double epsilon,
    vmar_regime_report *result);

/* Coupling is diagnosed once per proposal and held fixed across all
 * Runge-Kutta stages and backtracking retries. `system` is not mutated. */
vmar_status vmar_regime_control_step(vmar_integrator method,
    vmar_adaptive_system *system,const vmar_control_policy *policy,
    double max_coupling,double *state,size_t dimension,double *time,
    size_t *step_id,double *workspace,size_t workspace_doubles,
    vmar_regime_report *diagnostic,vmar_control_report *control);

#ifdef __cplusplus
}
#endif
#endif
