#include "vmar/regime.h"
#include <math.h>

vmar_status vmar_regime_diagnose(vmar_adaptive_system *sys,
    const double *state,size_t dimension,double max_coupling,double epsilon,
    vmar_regime_report *result) {
    if(!sys || !state || !result || !isfinite(max_coupling) ||
       max_coupling<0 || !isfinite(epsilon) || epsilon<=0)
        return VMAR_ERR_ARGUMENT;
    size_t expected=0;
    vmar_status status=vmar_adaptive_dimension(sys,&expected);
    if(status!=VMAR_OK) return status;
    if(dimension!=expected) return VMAR_ERR_DIMENSION;
    size_t d=dimension/2;
    double g[d],loss=0,gn=0,mn=0,dot=0;
    status=vmar_custom_neuron_loss_gradient(sys->neuron,state,d,&loss,g,NULL);
    if(status!=VMAR_OK) return status;
    for(size_t j=0;j<d;++j) {
        if(!isfinite(state[d+j])) return VMAR_ERR_NUMERIC;
        gn+=g[j]*g[j];mn+=state[d+j]*state[d+j];
        dot+=g[j]*state[d+j];
    }
    gn=sqrt(gn);mn=sqrt(mn);
    if(!isfinite(gn) || !isfinite(mn) || !isfinite(dot)) return VMAR_ERR_NUMERIC;
    double cosine=0;
    if(gn>epsilon && mn>epsilon) {
        cosine=dot/(gn*mn);
        if(cosine>1)cosine=1;
        if(cosine< -1)cosine=-1;
    }
    vmar_regime regime=gn<=epsilon?VMAR_REGIME_STATIONARY:
        mn<=epsilon?VMAR_REGIME_UNOBSERVED:
        cosine>0?VMAR_REGIME_ALIGNED:
        cosine<0?VMAR_REGIME_OPPOSED:VMAR_REGIME_ORTHOGONAL;
    double risk=gn<=epsilon?0:max_coupling*mn/gn;
    double coupling=0;
    if(gn>epsilon && mn>epsilon && cosine>0) {
        double scale=gn/mn;
        coupling=max_coupling*cosine*(scale<1?scale:1);
    }
    if(!isfinite(risk) || !isfinite(coupling)) return VMAR_ERR_NUMERIC;
    *result=(vmar_regime_report){regime,gn,mn,cosine,risk,coupling};
    return VMAR_OK;
}

vmar_status vmar_regime_control_step(vmar_integrator method,
    vmar_adaptive_system *sys,const vmar_control_policy *policy,
    double max_coupling,double *state,size_t dimension,double *time,
    size_t *step_id,double *workspace,size_t workspace_doubles,
    vmar_regime_report *diagnostic,vmar_control_report *control) {
    if(!diagnostic || !control) return VMAR_ERR_ARGUMENT;
    vmar_regime_report local;
    vmar_status status=vmar_regime_diagnose(sys,state,dimension,
                                           max_coupling,1e-12,&local);
    if(status!=VMAR_OK) return status;
    vmar_adaptive_system frozen=*sys;
    frozen.coupling=local.coupling;
    status=vmar_adaptive_control_step(method,&frozen,policy,state,dimension,
                                     time,step_id,workspace,workspace_doubles,control);
    if(status==VMAR_OK) *diagnostic=local;
    return status;
}
