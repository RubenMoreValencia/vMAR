#include "vmar/network_system.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

static vmar_status system_dimensions(const vmar_network_system *system,
                                     size_t dimension, size_t *parameters) {
    if (!system || !system->network.layout || !parameters) return VMAR_ERR_ARGUMENT;
    *parameters = system->network.layout->parameters;
    if (!*parameters || *parameters > SIZE_MAX/2 || dimension != 2*(*parameters))
        return VMAR_ERR_DIMENSION;
    if (!isfinite(system->network.eta) || system->network.eta <= 0 ||
        !isfinite(system->coupling) || system->coupling < 0 ||
        !isfinite(system->tau) || system->tau <= 0) return VMAR_ERR_ARGUMENT;
    return VMAR_OK;
}

vmar_status vmar_network_system_field(double time, const double *state,
                                      size_t dimension, double *derivative,
                                      void *context) {
    if (!context || !state || !derivative || !isfinite(time))
        return VMAR_ERR_ARGUMENT;
    const vmar_network_system *system = context;
    size_t p;
    vmar_status s = system_dimensions(system, dimension, &p);
    if (s != VMAR_OK) return s;
    const vmar_network_problem *n = &system->network;
    double loss = 0;
    s = vmar_network_loss_gradient(n->layout,n->x,n->target,n->samples,
                                   state,n->workspace,n->workspace_doubles,
                                   &loss,derivative,NULL);
    if (s != VMAR_OK) return s;
    for (size_t i=0; i<p; ++i) {
        double g = derivative[i], m = state[p+i];
        if (!isfinite(m)) return VMAR_ERR_NUMERIC;
        derivative[p+i] = (g-m)/system->tau;
        derivative[i] = -n->eta*(g+system->coupling*m);
        if (!isfinite(derivative[i]) || !isfinite(derivative[p+i]))
            return VMAR_ERR_NUMERIC;
    }
    return VMAR_OK;
}

vmar_status vmar_network_control_workspace_required(vmar_integrator method,
                                                    size_t parameters,
                                                    size_t *required) {
    if (!required || !parameters) return VMAR_ERR_ARGUMENT;
    if (parameters > SIZE_MAX/2) return VMAR_ERR_DIMENSION;
    size_t dimension = parameters*2, base = 0;
    vmar_status s = vmar_workspace_required(method, dimension, &base);
    if (s != VMAR_OK) return s;
    if (base > SIZE_MAX - dimension ||
        base + dimension > SIZE_MAX - parameters ||
        base + dimension + parameters > SIZE_MAX/sizeof(double))
        return VMAR_ERR_DIMENSION;
    *required = base + dimension + parameters;
    return VMAR_OK;
}

vmar_status vmar_network_control_step(vmar_integrator method,
                                      const vmar_network_system *system,
                                      const vmar_control_policy *policy,
                                      double *state, size_t dimension,
                                      double *time, size_t *step_id,
                                      double *workspace, size_t workspace_doubles,
                                      vmar_control_report *report) {
    if (!system || !policy || !state || !time || !step_id || !workspace ||
        !report || !isfinite(*time) || !isfinite(policy->initial_step) ||
        policy->initial_step <= 0 || !isfinite(policy->reduction) ||
        policy->reduction <= 0 || policy->reduction >= 1 ||
        policy->max_attempts == 0 || !isfinite(policy->loss_tolerance) ||
        policy->loss_tolerance < 0 || !isfinite(policy->memory_threshold) ||
        policy->memory_threshold < 0 || *step_id == SIZE_MAX)
        return VMAR_ERR_ARGUMENT;
    size_t p=0, required=0, integration_required=0;
    vmar_status s = system_dimensions(system, dimension, &p);
    if (s != VMAR_OK) return s;
    s = vmar_network_control_workspace_required(method,p,&required);
    if (s != VMAR_OK) return s;
    if (workspace_doubles < required) return VMAR_ERR_WORKSPACE;
    s = vmar_workspace_required(method,dimension,&integration_required);
    if (s != VMAR_OK) return s;

    double *candidate = workspace + integration_required;
    double *gradient = candidate + dimension;
    const vmar_network_problem *n = &system->network;
    double old_loss = 0;
    s = vmar_network_loss_gradient(n->layout,n->x,n->target,n->samples,
                                   state,n->workspace,n->workspace_doubles,
                                   &old_loss,gradient,NULL);
    if (s != VMAR_OK) return s;
    if (!isfinite(old_loss) || old_loss < 0) return VMAR_ERR_NUMERIC;
    double step = policy->initial_step;
    if (policy->memory_threshold > 0) {
        double maximum=0;
        for (size_t i=0;i<p;++i) {
            if (!isfinite(state[p+i])) return VMAR_ERR_NUMERIC;
            if (fabs(state[p+i]) > maximum) maximum=fabs(state[p+i]);
        }
        if (maximum > policy->memory_threshold) step *= policy->reduction;
    }
    vmar_control_report local={0};
    local.old_loss=old_loss;
    local.candidate_loss=old_loss;
    local.final_time=*time;
    for (size_t k=0;k<policy->max_attempts;++k) {
        if (!isfinite(step) || step <= 0 || !isfinite(*time+step))
            return VMAR_ERR_NUMERIC;
        s = vmar_step(method,*time,step,vmar_network_system_field,(void *)system,
                      state,dimension,workspace,integration_required,candidate);
        if (s != VMAR_OK) return s;
        double candidate_loss=0;
        s = vmar_network_loss_gradient(n->layout,n->x,n->target,n->samples,
                                       candidate,n->workspace,n->workspace_doubles,
                                       &candidate_loss,gradient,NULL);
        if (s != VMAR_OK) return s;
        local.attempts=k+1;
        local.field_evaluations += method == VMAR_EULER ? 1 : 4;
        local.candidate_loss=candidate_loss;
        local.effective_step=step;
        if (candidate_loss <= old_loss+policy->loss_tolerance) {
            memcpy(state,candidate,dimension*sizeof(double));
            *time += step;
            ++*step_id;
            local.accepted=1;
            local.final_time=*time;
            *report=local;
            return VMAR_OK;
        }
        ++local.rejected;
        step *= policy->reduction;
    }
    *report=local;
    return VMAR_OK;
}
