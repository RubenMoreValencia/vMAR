#include "vmar/custom_activation.h"

#include <math.h>
#include <stdint.h>

static int finite_values(const double *x, size_t n) {
    for (size_t i=0;i<n;++i) if (!isfinite(x[i])) return 0;
    return 1;
}

vmar_status vmar_custom_neuron_dimension(const vmar_custom_neuron *m,
                                         size_t *dimension) {
    if (!m || !dimension || !m->activation || !m->x || !m->target ||
        !m->features || !m->samples ||
        (m->trainable!=0 && m->trainable!=1) ||
        !isfinite(m->eta) || m->eta<=0) return VMAR_ERR_ARGUMENT;
    if (m->features==SIZE_MAX ||
        m->features>SIZE_MAX/m->samples ||
        m->features*m->samples>SIZE_MAX/sizeof(double) ||
        m->features+1>SIZE_MAX/sizeof(double) ||
        m->samples>SIZE_MAX/sizeof(double)) return VMAR_ERR_DIMENSION;
    size_t d=m->features+1;
    if (m->trainable) {
        if (!m->nparams || !m->scratch ||
            m->scratch_doubles<m->nparams) return VMAR_ERR_WORKSPACE;
        if (m->nparams>SIZE_MAX-d ||
            d+m->nparams>SIZE_MAX/sizeof(double)) return VMAR_ERR_DIMENSION;
        d+=m->nparams;
    }
    if (m->nparams && !m->params) return VMAR_ERR_ARGUMENT;
    *dimension=d;
    return VMAR_OK;
}

vmar_status vmar_custom_neuron_loss_gradient(
    const vmar_custom_neuron *m, const double *theta,
    size_t dimension, double *loss, double *gradient, double *prediction) {
    if (!theta || !loss || !gradient) return VMAR_ERR_ARGUMENT;
    size_t d=0;
    vmar_status status=vmar_custom_neuron_dimension(m,&d);
    if (status!=VMAR_OK) return status;
    if (dimension!=d) return VMAR_ERR_DIMENSION;
    if (!finite_values(theta,d) ||
        !finite_values(m->x,m->samples*m->features) ||
        !finite_values(m->target,m->samples) ||
        !finite_values(m->params,m->nparams)) return VMAR_ERR_NUMERIC;
    const size_t n=m->samples,p=m->features,q=m->nparams;
    const double *alpha=m->trainable?theta+p+1:m->params;
    for(size_t j=0;j<d;++j) gradient[j]=0;
    double total=0;
    for(size_t i=0;i<n;++i) {
        double z=theta[p];
        for(size_t j=0;j<p;++j) z+=m->x[i*p+j]*theta[j];
        if (!isfinite(z)) return VMAR_ERR_NUMERIC;
        double y=0,dz=0;
        double *dp=m->trainable?m->scratch:NULL;
        status=m->activation(z,alpha,q,&y,&dz,dp,m->user);
        if (status!=VMAR_OK) return VMAR_ERR_FIELD;
        if (!isfinite(y) || !isfinite(dz) ||
            (dp && !finite_values(dp,q))) return VMAR_ERR_NUMERIC;
        double r=y-m->target[i];
        double delta=(2.0/(double)n)*r;
        total+=r*r/(double)n;
        for(size_t j=0;j<p;++j) gradient[j]+=m->x[i*p+j]*delta*dz;
        gradient[p]+=delta*dz;
        if (m->trainable) for(size_t j=0;j<q;++j)
            gradient[p+1+j]+=delta*dp[j];
        if (prediction) prediction[i]=y;
    }
    if (!isfinite(total) || !finite_values(gradient,d)) return VMAR_ERR_NUMERIC;
    *loss=total;
    return VMAR_OK;
}

vmar_status vmar_custom_neuron_field(double time,const double *theta,
                                     size_t dimension,double *derivative,
                                     void *context) {
    if (!isfinite(time) || !context || !derivative) return VMAR_ERR_ARGUMENT;
    const vmar_custom_neuron *m=context;
    double loss=0;
    vmar_status status=vmar_custom_neuron_loss_gradient(
        m,theta,dimension,&loss,derivative,NULL);
    if (status!=VMAR_OK) return status;
    for(size_t j=0;j<dimension;++j) {
        derivative[j]*=-m->eta;
        if (!isfinite(derivative[j])) return VMAR_ERR_NUMERIC;
    }
    return VMAR_OK;
}
