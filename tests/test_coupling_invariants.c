#include "vmar/adaptive_control.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int checks,failures;
#define CHECK(x) do {++checks;if(!(x)){++failures;fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);}}while(0)
static vmar_status prelu(double z,const double *p,size_t q,
                          double *v,double *dz,double *dp,void *u) {
    (void)u;
    if(!p||q!=1) return VMAR_ERR_ARGUMENT;
    *v=z>0?z:p[0]*z;*dz=z>0?1:p[0];
    if(dp)dp[0]=z<0?z:0;
    return VMAR_OK;
}

static vmar_status evolve(vmar_integrator method,double coupling,
                           double memory0,double state[8]) {
    const double x[]={1,0.5,-0.5,1.5,0.7,-1};
    const double target[]={0.25,-0.3,0.7};
    double alpha=0.1,scratch[1];
    vmar_custom_neuron neuron={x,target,3,2,&alpha,1,1,0.3,
                               prelu,NULL,scratch,1};
    vmar_adaptive_system sys={&neuron,0.8,coupling};
    double current[8]={0.2,-0.4,0.15,0.1,
                       memory0,memory0,memory0,memory0};
    double work[40],next[8];
    for(size_t k=0;k<10;++k) {
        vmar_status status=vmar_step(method,(double)k*0.1,0.1,
          vmar_adaptive_field,&sys,current,8,work,
          method==VMAR_EULER?8u:40u,next);
        if(status!=VMAR_OK) return status;
        memcpy(current,next,sizeof(current));
    }
    memcpy(state,current,sizeof(current));
    return VMAR_OK;
}
int main(void) {
    for(int integrator=0;integrator<2;++integrator) {
        vmar_integrator m=integrator?VMAR_RK4:VMAR_EULER;
        double a[8],b[8],c[8];
        CHECK(evolve(m,0,0,a)==VMAR_OK);
        CHECK(evolve(m,0,0.8,b)==VMAR_OK);
        CHECK(evolve(m,0.2,0.8,c)==VMAR_OK);
        for(size_t j=0;j<4;++j) CHECK(fabs(a[j]-b[j])<1e-12);
        int memory_changed=0,coupling_changed=0;
        for(size_t j=0;j<4;++j) {
            if(fabs(a[4+j]-b[4+j])>1e-5) memory_changed=1;
            if(fabs(b[j]-c[j])>1e-5) coupling_changed=1;
        }
        CHECK(memory_changed);
        CHECK(coupling_changed);
    }
    printf("vMAR coupling invariants: %d assertions, %d failures\n",checks,failures);
    return failures!=0;
}
