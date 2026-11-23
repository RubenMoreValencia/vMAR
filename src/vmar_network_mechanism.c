#include "vmar/network_mechanism.h"
#include <math.h>
#include <stdint.h>

vmar_network_mode vmar_network_default_mode(void) {
    return VMAR_NETWORK_OBSERVE;
}

vmar_status vmar_network_mechanism_step(
    vmar_integrator method, const vmar_network_mechanism *mechanism,
    double step, double *state, size_t dimension, double *time,
    size_t *step_id, double *workspace, size_t workspace_doubles,
    vmar_network_decision *decision) {
    if (!mechanism || !state || !time || !step_id || !workspace || !decision ||
        !isfinite(*time) || !isfinite(step) || step <= 0 ||
        *step_id == SIZE_MAX ||
        mechanism->mode < VMAR_NETWORK_BASELINE ||
        mechanism->mode > VMAR_NETWORK_CONTROLLED)
        return VMAR_ERR_ARGUMENT;
    const vmar_network_problem *n = &mechanism->system.network;
    if (!n->layout || !n->layout->parameters ||
        n->layout->parameters > SIZE_MAX/2 ||
        dimension != 2*n->layout->parameters ||
        !isfinite(mechanism->system.coupling) ||
        mechanism->system.coupling < 0 ||
        !isfinite(mechanism->system.tau) ||
        mechanism->system.tau <= 0 ||
        !isfinite(n->eta) || n->eta <= 0)
        return VMAR_ERR_DIMENSION;
    size_t p = n->layout->parameters, required = 0, stage = 0;
    vmar_status s = vmar_network_control_workspace_required(method,p,&required);
    if (s != VMAR_OK) return s;
    if (workspace_doubles < required) return VMAR_ERR_WORKSPACE;
    s = vmar_workspace_required(method,dimension,&stage);
    if (s != VMAR_OK) return s;
    double *gradient = workspace + required - p;
    double old_loss=0;
    s=vmar_network_loss_gradient(n->layout,n->x,n->target,n->samples,state,
                                 n->workspace,n->workspace_doubles,
                                 &old_loss,gradient,NULL);
    if(s!=VMAR_OK)return s;
    double gradnorm=0,memnorm=0;
    for(size_t i=0;i<p;++i){
        if(!isfinite(state[p+i]))return VMAR_ERR_NUMERIC;
        gradnorm=hypot(gradnorm,gradient[i]);
        memnorm=hypot(memnorm,state[p+i]);
    }
    vmar_network_decision report={0};
    report.mode=mechanism->mode;
    report.loss_before=old_loss;
    report.gradient_norm_before=gradnorm;
    report.memory_norm_before=memnorm;
    report.coupling_term_norm_before =
        mechanism->mode >= VMAR_NETWORK_COUPLED ?
        n->eta * mechanism->system.coupling * memnorm : 0;
    report.requested_step=step;
    report.time_before=*time;

    if(mechanism->mode==VMAR_NETWORK_CONTROLLED){
        vmar_control_policy policy=mechanism->policy;
        policy.initial_step=step;
        vmar_control_report control;
        s=vmar_network_control_step(method,&mechanism->system,&policy,state,
                                     dimension,time,step_id,workspace,
                                     workspace_doubles,&control);
        if(s!=VMAR_OK)return s;
        report.accepted=control.accepted;
        report.attempts=control.attempts;
        report.rejected=control.rejected;
        report.field_evaluations=control.field_evaluations;
        report.effective_step=control.effective_step;
    } else {
        size_t integrated=mechanism->mode==VMAR_NETWORK_BASELINE?p:dimension;
        vmar_network_system configured=mechanism->system;
        if(mechanism->mode==VMAR_NETWORK_OBSERVE)configured.coupling=0;
        vmar_field_fn field=mechanism->mode==VMAR_NETWORK_BASELINE?
            vmar_network_field:vmar_network_system_field;
        void *context=mechanism->mode==VMAR_NETWORK_BASELINE?
            (void *)n:(void *)&configured;
        s=vmar_step(method,*time,step,field,context,state,integrated,
                    workspace,stage,state);
        if(s!=VMAR_OK)return s;
        *time+=step;
        ++*step_id;
        report.accepted=1;
        report.attempts=1;
        report.field_evaluations=method==VMAR_EULER?1:4;
        report.effective_step=step;
    }
    s=vmar_network_loss_gradient(n->layout,n->x,n->target,n->samples,state,
                                 n->workspace,n->workspace_doubles,
                                 &report.loss_after,gradient,NULL);
    if(s!=VMAR_OK)return s;
    report.time_after=*time;
    *decision=report;
    return VMAR_OK;
}
