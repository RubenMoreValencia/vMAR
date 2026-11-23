#include "vmar/adaptive_control.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

vmar_status vmar_adaptive_dimension(const vmar_adaptive_system *s,
                                    size_t *dimension) {
    if (!s || !s->neuron || !dimension || !isfinite(s->tau) || s->tau<=0 ||
        !isfinite(s->coupling) || s->coupling<0) return VMAR_ERR_ARGUMENT;
    size_t d=0;
    vmar_status st=vmar_custom_neuron_dimension(s->neuron,&d);
    if (st!=VMAR_OK) return st;
    if (d>SIZE_MAX/2 || 2*d>SIZE_MAX/sizeof(double)) return VMAR_ERR_DIMENSION;
    *dimension=2*d;
    return VMAR_OK;
}

vmar_status vmar_adaptive_field(double time,const double *state,
                                size_t dimension,double *derivative,
                                void *context) {
    if (!state || !derivative || !context || !isfinite(time)) return VMAR_ERR_ARGUMENT;
    const vmar_adaptive_system *s=context;
    size_t expected=0;
    vmar_status st=vmar_adaptive_dimension(s,&expected);
    if (st!=VMAR_OK) return st;
    if (dimension!=expected) return VMAR_ERR_DIMENSION;
    size_t d=dimension/2;
    for(size_t i=0;i<d;++i) if (!isfinite(state[d+i])) return VMAR_ERR_NUMERIC;
    double loss=0;
    st=vmar_custom_neuron_loss_gradient(s->neuron,state,d,&loss,derivative,NULL);
    if (st!=VMAR_OK) return st;
    for(size_t i=0;i<d;++i) {
        double g=derivative[i];
        derivative[d+i]=(g-state[d+i])/s->tau;
        derivative[i]=-s->neuron->eta*(g+s->coupling*state[d+i]);
        if (!isfinite(derivative[d+i]) || !isfinite(derivative[i])) return VMAR_ERR_NUMERIC;
    }
    return VMAR_OK;
}

vmar_status vmar_adaptive_control_step(vmar_integrator method,
    vmar_adaptive_system *system,const vmar_control_policy *policy,
    double *state,size_t dimension,double *time,size_t *step_id,
    double *workspace,size_t workspace_doubles,vmar_control_report *report) {
    if (!system || !policy || !state || !time || !step_id || !workspace || !report ||
        !isfinite(*time) || *step_id==SIZE_MAX ||
        !isfinite(policy->initial_step) || policy->initial_step<=0 ||
        !isfinite(policy->reduction) || policy->reduction<=0 || policy->reduction>=1 ||
        !policy->max_attempts || !isfinite(policy->loss_tolerance) ||
        policy->loss_tolerance<0 || !isfinite(policy->memory_threshold) ||
        policy->memory_threshold<0) return VMAR_ERR_ARGUMENT;
    size_t expected=0,required=0;
    vmar_status st=vmar_adaptive_dimension(system,&expected);
    if (st!=VMAR_OK) return st;
    if (dimension!=expected) return VMAR_ERR_DIMENSION;
    st=vmar_control_workspace_required(method,dimension,&required);
    if (st!=VMAR_OK) return st;
    if (workspace_doubles<required) return VMAR_ERR_WORKSPACE;
    size_t d=dimension/2,base=required-dimension;
    double *candidate=workspace+base;
    double old_loss=0,new_loss=0;
    double gradient[d];
    st=vmar_custom_neuron_loss_gradient(system->neuron,state,d,&old_loss,gradient,NULL);
    if (st!=VMAR_OK) return st;
    double h=policy->initial_step;
    if (policy->memory_threshold>0) {
        double peak=0;
        for(size_t j=0;j<d;++j) {
            if (!isfinite(state[d+j])) return VMAR_ERR_NUMERIC;
            if (fabs(state[d+j])>peak) peak=fabs(state[d+j]);
        }
        if (peak>policy->memory_threshold) h*=policy->reduction;
    }
    vmar_control_report local={0};
    local.old_loss=old_loss;local.candidate_loss=old_loss;local.final_time=*time;
    for(size_t i=0;i<policy->max_attempts;++i) {
        if (!isfinite(h) || h<=0 || !isfinite(*time+h)) return VMAR_ERR_NUMERIC;
        st=vmar_step(method,*time,h,vmar_adaptive_field,system,
                     state,dimension,workspace,base,candidate);
        if (st!=VMAR_OK) return st;
        st=vmar_custom_neuron_loss_gradient(system->neuron,candidate,d,
                                             &new_loss,gradient,NULL);
        if (st!=VMAR_OK) return st;
        local.attempts=i+1;
        local.field_evaluations+=(method==VMAR_EULER?1:4);
        local.candidate_loss=new_loss;local.effective_step=h;
        if (new_loss<=old_loss+policy->loss_tolerance) {
            memcpy(state,candidate,dimension*sizeof(double));
            *time+=h;++*step_id;
            local.accepted=1;local.final_time=*time;
            *report=local;return VMAR_OK;
        }
        ++local.rejected;h*=policy->reduction;
    }
    *report=local;
    return VMAR_OK;
}
