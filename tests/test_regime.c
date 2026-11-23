#include "vmar/regime.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int checks,failures;
#define CHECK(v) do {++checks;if(!(v)){++failures;fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#v);}}while(0)
static vmar_status prelu(double z,const double *p,size_t q,
    double *y,double *dz,double *dp,void *u) {
    (void)u;if(!p||q!=1)return VMAR_ERR_ARGUMENT;
    *y=z>0?z:p[0]*z;*dz=z>0?1:p[0];
    if(dp)dp[0]=z<0?z:0;
    return VMAR_OK;
}
int main(void) {
    const double x[]={1,0.5,-0.5,1.5,0.7,-1};
    const double target[]={0.25,-0.3,0.7};
    double alpha=0.1,scratch[1];
    vmar_custom_neuron neuron={x,target,3,2,&alpha,1,1,0.3,prelu,NULL,scratch,1};
    vmar_adaptive_system system={&neuron,0.8,0};
    double state[8]={0.2,-0.4,0.15,0.1,0,0,0,0};
    double gradient[4],loss=0;
    CHECK(vmar_custom_neuron_loss_gradient(&neuron,state,4,
                                            &loss,gradient,NULL)==VMAR_OK);
    vmar_regime_report d;
    CHECK(vmar_regime_diagnose(&system,state,8,0.2,1e-12,&d)==VMAR_OK);
    CHECK(d.regime==VMAR_REGIME_UNOBSERVED);
    CHECK(d.coupling==0);
    for(size_t j=0;j<4;++j)state[4+j]=gradient[j];
    CHECK(vmar_regime_diagnose(&system,state,8,0.2,1e-12,&d)==VMAR_OK);
    CHECK(d.regime==VMAR_REGIME_ALIGNED);
    CHECK(fabs(d.alignment-1)<1e-12);
    CHECK(fabs(d.coupling-0.2)<1e-12);
    double aligned=d.coupling;
    for(size_t j=0;j<4;++j)state[4+j]=-gradient[j];
    CHECK(vmar_regime_diagnose(&system,state,8,0.2,1e-12,&d)==VMAR_OK);
    CHECK(d.regime==VMAR_REGIME_OPPOSED);
    CHECK(d.coupling==0);
    CHECK(vmar_regime_diagnose(&system,state,8,-0.2,1e-12,&d)==VMAR_ERR_ARGUMENT);
    CHECK(vmar_regime_diagnose(&system,state,7,0.2,1e-12,&d)==VMAR_ERR_DIMENSION);
    vmar_control_policy policy={0.1,0.5,4,0,0.5};
    double workspace[48],time=0;size_t step_id=0;
    vmar_control_report control={0};
    double before[8];memcpy(before,state,sizeof(before));
    CHECK(vmar_regime_control_step(VMAR_EULER,&system,&policy,0.2,
      state,8,&time,&step_id,workspace,48,&d,&control)==VMAR_OK);
    CHECK(d.coupling==0);
    CHECK(system.coupling==0);
    CHECK(control.accepted==1);
    CHECK(step_id==1 && time>0);
    CHECK(aligned>0);
    CHECK(memcmp(state,before,sizeof(state))!=0);
    printf("vMAR regime contracts: %d assertions, %d failures\n",checks,failures);
    return failures!=0;
}
