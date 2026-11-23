#include "vmar/control.h"

#include <math.h>
#include <stdint.h>
#include <string.h>

vmar_status vmar_control_workspace_required(vmar_integrator method,
                                            size_t dimension, size_t *required) {
    if (!required) return VMAR_ERR_ARGUMENT;
    size_t base = 0;
    vmar_status status = vmar_workspace_required(method, dimension, &base);
    if (status != VMAR_OK) return status;
    if (dimension > SIZE_MAX - base ||
        base + dimension > SIZE_MAX / sizeof(double)) return VMAR_ERR_DIMENSION;
    *required = base + dimension; /* extra candidate state */
    return VMAR_OK;
}

static vmar_status neuron_loss(const vmar_learning_system *system,
                               const double *state, double *loss,
                               double *gradient) {
    return vmar_neuron_loss_gradient(&system->neuron,state,loss,gradient,NULL);
}

vmar_status vmar_control_step(vmar_integrator method,
                              const vmar_learning_system *system,
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
    size_t required = 0;
    vmar_status status = vmar_control_workspace_required(method,dimension,&required);
    if (status != VMAR_OK) return status;
    if (workspace_doubles < required) return VMAR_ERR_WORKSPACE;
    size_t d = system->neuron.features + 1;
    if (system->neuron.features == SIZE_MAX || d > SIZE_MAX/2 ||
        dimension != 2*d) return VMAR_ERR_DIMENSION;
    double *next = workspace + (required-dimension);
    /* The base workspace is disjoint from candidate. */
    size_t base = required - dimension;
    double old_loss = 0, grad[d];
    status = neuron_loss(system,state,&old_loss,grad);
    if (status != VMAR_OK) return status;
    if (!isfinite(old_loss) || old_loss < 0) return VMAR_ERR_NUMERIC;
    double step = policy->initial_step;
    if (policy->memory_threshold > 0) {
        double maximum = 0;
        for (size_t j=0;j<d;++j) {
            if (!isfinite(state[d+j])) return VMAR_ERR_NUMERIC;
            double v = fabs(state[d+j]);
            if (v > maximum) maximum = v;
        }
        if (maximum > policy->memory_threshold) step *= policy->reduction;
    }
    vmar_control_report local = {0};
    local.old_loss = old_loss;
    local.candidate_loss = old_loss;
    local.final_time = *time;
    for (size_t k=0;k<policy->max_attempts;++k) {
        if (!isfinite(step) || step <= 0 || !isfinite(*time+step))
            return VMAR_ERR_NUMERIC;
        status = vmar_step(method,*time,step,vmar_learning_field,(void *)system,
                           state,dimension,workspace,base,next);
        if (status != VMAR_OK) return status;
        double candidate_loss = 0;
        status = neuron_loss(system,next,&candidate_loss,grad);
        if (status != VMAR_OK) return status;
        local.attempts = k+1;
        local.field_evaluations += method == VMAR_EULER ? 1 : 4;
        local.candidate_loss = candidate_loss;
        local.effective_step = step;
        if (candidate_loss <= old_loss + policy->loss_tolerance) {
            memcpy(state,next,dimension*sizeof(double));
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
