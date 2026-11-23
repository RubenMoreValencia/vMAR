#include "vmar/regime.h"
#include <math.h>
#include <stdio.h>

static int checks,failures;
#define CHECK(x) do{++checks;if(!(x)){++failures;fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);}}while(0)
static vmar_status activation(double z,const double *p,size_t q,
    double *y,double *dz,double *dp,void *context) {
    int kind=*(int*)context;
    if(kind==0) {
        if(q)return VMAR_ERR_ARGUMENT;
        *y=tanh(z);*dz=1-(*y)*(*y);return VMAR_OK;
    }
    if(!p||q!=1)return VMAR_ERR_ARGUMENT;
    if(kind==1) {
        *y=z>0?z:p[0]*z;*dz=z>0?1:p[0];
        if(dp)dp[0]=z<0?z:0;
        return VMAR_OK;
    }
    double s=1/(1+exp(-p[0]*z));
    *y=z*s;*dz=s+p[0]*z*s*(1-s);
    if(dp)dp[0]=z*z*s*(1-s);
    return VMAR_OK;
}
int main(void) {
    const double x[]={1,0.5,-0.5,1.5,0.7,-1};
    const double target[]={0.25,-0.3,0.7};
    for(int kind=0;kind<3;++kind) {
        double alpha=kind==1?0.1:0.8,scratch[1];
        vmar_custom_neuron neuron={x,target,3,2,kind?&alpha:NULL,
            kind?1u:0u,kind?1:0,0.3,activation,&kind,
            kind?scratch:NULL,kind?1u:0u};
        vmar_adaptive_system sys={&neuron,0.8,0};
        size_t d=0;
        CHECK(vmar_custom_neuron_dimension(&neuron,&d)==VMAR_OK);
        CHECK(d==(kind?4u:3u));
        double state[8]={0.2,-0.4,0.15,0,0,0,0,0};
        if(kind)state[3]=alpha;
        double g[4],loss=0;
        CHECK(vmar_custom_neuron_loss_gradient(&neuron,state,d,&loss,g,NULL)==VMAR_OK);
        vmar_regime_report diagnostic;
        CHECK(vmar_regime_diagnose(&sys,state,2*d,0.2,1e-12,
                                   &diagnostic)==VMAR_OK);
        CHECK(diagnostic.coupling==0);
        for(size_t j=0;j<d;++j)state[d+j]=-g[j];
        CHECK(vmar_regime_diagnose(&sys,state,2*d,0.2,1e-12,
                                   &diagnostic)==VMAR_OK);
        CHECK(diagnostic.coupling==0);
        CHECK(diagnostic.regime==VMAR_REGIME_OPPOSED);
        for(size_t j=0;j<d;++j)state[d+j]=g[j];
        CHECK(vmar_regime_diagnose(&sys,state,2*d,0.2,1e-12,
                                   &diagnostic)==VMAR_OK);
        CHECK(diagnostic.coupling>0 && diagnostic.coupling<=0.2);
    }
    printf("vMAR transfer invariants: %d assertions, %d failures\n",checks,failures);
    return failures!=0;
}
