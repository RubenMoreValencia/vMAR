#include "vmar/regime.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int n_checks,n_failures;
#define CHECK(x) do {++n_checks;if(!(x)){++n_failures;fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);}}while(0)
static vmar_status prelu(double z,const double *p,size_t q,
    double *v,double *dz,double *dp,void *unused) {
    (void)unused;
    if(!p||q!=1)return VMAR_ERR_ARGUMENT;
    *v=z>0?z:p[0]*z;*dz=z>0?1:p[0];
    if(dp)dp[0]=z<0?z:0;
    return VMAR_OK;
}
static vmar_status simulate(vmar_integrator method,int adaptive,
                             double state[8],size_t *gradient_count) {
    const double x[]={1,0.5,-0.5,1.5,0.7,-1};
    const double target[]={0.25,-0.3,0.7};
    double alpha=0.1,scratch[1];
    vmar_custom_neuron neuron={x,target,3,2,&alpha,1,1,0.3,prelu,NULL,scratch,1};
    vmar_adaptive_system system={&neuron,0.8,0};
    size_t steps=method==VMAR_EULER?20u:8u;
    double h=1.0/(double)steps,workspace[40],next[8];
    *gradient_count=0;
    for(size_t k=0;k<steps;++k) {
        vmar_regime_report diagnostic;
        vmar_status s=vmar_regime_diagnose(&system,state,8,0,1e-12,&diagnostic);
        if(s!=VMAR_OK)return s;
        ++*gradient_count;
        system.coupling=adaptive?diagnostic.coupling:0;
        s=vmar_step(method,(double)k*h,h,vmar_adaptive_field,&system,
            state,8,workspace,method==VMAR_EULER?8u:40u,next);
        if(s!=VMAR_OK)return s;
        *gradient_count+=method==VMAR_EULER?1u:4u;
        memcpy(state,next,sizeof(next));
    }
    return VMAR_OK;
}
int main(void) {
    for(int m=0;m<2;++m) {
        vmar_integrator method=m?VMAR_RK4:VMAR_EULER;
        double baseline[8]={0.1,-0.4,0.15,0.1,0.8,0.8,0.8,0.8};
        double adaptive[8];memcpy(adaptive,baseline,sizeof(baseline));
        size_t cost0=0,cost1=0;
        CHECK(simulate(method,0,baseline,&cost0)==VMAR_OK);
        CHECK(simulate(method,1,adaptive,&cost1)==VMAR_OK);
        CHECK(cost0==40 && cost1==40);
        for(size_t j=0;j<8;++j)CHECK(fabs(adaptive[j]-baseline[j])<1e-12);
    }
    printf("vMAR intervention invariants: %d assertions, %d failures\n",
           n_checks,n_failures);
    return n_failures!=0;
}
