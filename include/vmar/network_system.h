#ifndef VMAR_NETWORK_SYSTEM_H
#define VMAR_NETWORK_SYSTEM_H

#include "vmar/network.h"
#include "vmar/control.h"

#ifdef __cplusplus
extern "C" {
#endif

/* G0.7-B.1: state [theta(p), first-memory(p)].
 * theta'=-eta * (gradient(MSE) + coupling * memory)
 * memory'=(gradient(MSE)-memory)/tau.
 * coupling=0 recovers G0.7-B theta dynamics exactly.
 * Memory evolves even at coupling=0: observation does not imply intervention.
 */
typedef struct vmar_network_system {
    vmar_network_problem network;
    double tau;
    double coupling;
} vmar_network_system;

vmar_status vmar_network_system_field(double time, const double *state,
                                      size_t dimension, double *derivative,
                                      void *context);

/* Required workspace for transactional controlled integration; supplied in
 * doubles. Includes candidate, integrator stages, and analytic gradient.
 * The network problem's own workspace remains a separate caller-owned buffer.
 */
vmar_status vmar_network_control_workspace_required(vmar_integrator method,
                                                    size_t parameters,
                                                    size_t *required);

/* Policy is the same observable deterministic contract used by the original
 * single-neuron controller. The gate observes confirmed pre-step memory.
 * Backtracking tries from the same confirmed state/time, evaluating loss
 * only on theta. Accepted proposals commit theta and memory together.
 * report.field_evaluations counts field calls including rejected proposals.
 */
vmar_status vmar_network_control_step(vmar_integrator method,
                                      const vmar_network_system *system,
                                      const vmar_control_policy *policy,
                                      double *state, size_t dimension,
                                      double *time, size_t *step_id,
                                      double *workspace, size_t workspace_doubles,
                                      vmar_control_report *report);

#ifdef __cplusplus
}
#endif

#endif
