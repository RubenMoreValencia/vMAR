#include "second_moment_reference.h"
#include <math.h>
#include <stdint.h>

vmar_status vmar_second_field(double time,const double *state,
    size_t dimension,double *derivative,void *context) {
    if(!isfinite(time)||!state||!derivative||!context)return VMAR_ERR_ARGUMENT;
    const vmar_second_system *sys=context;
    if(!sys->model||!isfinite(sys->tau1)||sys->tau1<=0||
       !isfinite(sys->tau2)||sys->tau2<=0||
       !isfinite(sys->coupling)||sys->coupling<0||
       !isfinite(sys->scale)||sys->scale<=0||
       sys->mode<VMAR_SECOND_M1||sys->mode>VMAR_SECOND_M2_CONTROL)
       return VMAR_ERR_ARGUMENT;
    size_t d=0;
    vmar_status status=vmar_custom_neuron_dimension(sys->model,&d);
    if(status!=VMAR_OK)return status;
    size_t factor=sys->mode==VMAR_SECOND_M1?2:3;
    if(d>SIZE_MAX/factor||factor*d!=dimension)return VMAR_ERR_DIMENSION;
    double mean_v=0;
    for(size_t j=0;j<d;++j) {
        if(!isfinite(state[d+j]))return VMAR_ERR_NUMERIC;
        if(factor==3) {
            if(!isfinite(state[2*d+j])||state[2*d+j]<0)return VMAR_ERR_NUMERIC;
            mean_v+=state[2*d+j]/(double)d;
        }
    }
    double c=sys->coupling;
    if(sys->mode==VMAR_SECOND_M2_CONTROL)
        c=c/(1+sqrt(mean_v)/sys->scale);
    double loss=0;
    status=vmar_custom_neuron_loss_gradient(sys->model,state,d,&loss,
                                             derivative,NULL);
    if(status!=VMAR_OK)return status;
    for(size_t j=0;j<d;++j) {
        double g=derivative[j];
        derivative[d+j]=(g-state[d+j])/sys->tau1;
        if(factor==3)derivative[2*d+j]=(g*g-state[2*d+j])/sys->tau2;
        derivative[j]=-sys->model->eta*(g+c*state[d+j]);
        if(!isfinite(derivative[j])||!isfinite(derivative[d+j])||
           (factor==3&&!isfinite(derivative[2*d+j])))return VMAR_ERR_NUMERIC;
    }
    return VMAR_OK;
}
