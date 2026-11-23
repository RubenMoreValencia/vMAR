#include "adam_reference.h"
#include <math.h>
#include <stdint.h>
#include <stdlib.h>

vmar_status vmar_adam_reference_step(
    vmar_adam_reference *config, const double *gradient, size_t dimension,
    double *theta, double *first_moment, double *second_moment) {
    if (!config || !gradient || !theta || !first_moment || !second_moment ||
        !dimension || !isfinite(config->learning_rate) ||
        !(config->learning_rate>0) || !isfinite(config->beta1) ||
        config->beta1<0 || config->beta1>=1 || !isfinite(config->beta2) ||
        config->beta2<0 || config->beta2>=1 || !isfinite(config->epsilon) ||
        !(config->epsilon>0) || config->iteration==SIZE_MAX ||
        config->gradient_evaluations==SIZE_MAX ||
        dimension>SIZE_MAX/(3*sizeof(double))) return VMAR_ERR_ARGUMENT;
    /* Transactional reference update: rejected proposals leave all states
     * and accounting unchanged. */
    double *proposal=malloc(3*dimension*sizeof(double));
    if (!proposal) return VMAR_ERR_WORKSPACE;
    double *tm=proposal+dimension,*tv=tm+dimension;
    const double correction1=-expm1(((double)config->iteration+1)*log(config->beta1));
    const double correction2=-expm1(((double)config->iteration+1)*log(config->beta2));
    /* beta==0 is a valid no-memory limit; log(0)=-inf, expm1(-inf)=-1. */
    vmar_status status=VMAR_OK;
    for(size_t j=0;j<dimension;++j) {
        double g=gradient[j];
        if (!isfinite(g) || !isfinite(theta[j]) ||
            !isfinite(first_moment[j]) || !isfinite(second_moment[j]) ||
            second_moment[j]<0) {status=VMAR_ERR_NUMERIC;break;}
        tm[j]=config->beta1*first_moment[j]+(1-config->beta1)*g;
        tv[j]=config->beta2*second_moment[j]+(1-config->beta2)*g*g;
        double mh=tm[j]/correction1,vh=tv[j]/correction2;
        proposal[j]=theta[j]-config->learning_rate*mh/(sqrt(vh)+config->epsilon);
        if (!isfinite(tm[j]) || !isfinite(tv[j]) || !isfinite(proposal[j])) {
            status=VMAR_ERR_NUMERIC;break;
        }
    }
    if(status==VMAR_OK) {
        for(size_t j=0;j<dimension;++j) {
            theta[j]=proposal[j];first_moment[j]=tm[j];second_moment[j]=tv[j];
        }
        ++config->iteration;++config->gradient_evaluations;
    }
    free(proposal);
    return status;
}
