#include "vmar/adaptive_control.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int checked,failed;
#define CHECK(x) do {++checked;if(!(x)){++failed;fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);}}while(0)
static vmar_status prelu(double z,const double *p,size_t q,
                         double *y,double *dz,double *dp,void *user) {
    (void)user;
    if(q!=1 || !p) return VMAR_ERR_ARGUMENT;
    *y=z>0?z:p[0]*z;*dz=z>0?1:p[0];
    if(dp) dp[0]=z<0?z:0;
    return VMAR_OK;
}

int main(void) {
    const double x[]={1,0.5,-0.5,1.5,0.7,-1};
    const double target[]={0.25,-0.3,0.7};
    double alpha=0.1,scratch[1];
    vmar_custom_neuron n={x,target,3,2,&alpha,1,1,0.3,prelu,NULL,scratch,1};
    vmar_adaptive_system s={&n,0.8,0};
    size_t dim=0;
    CHECK(vmar_adaptive_dimension(&s,&dim)==VMAR_OK);
    CHECK(dim==8);
    double initial[8]={0.2,-0.4,0.15,0.1,0,0,0,0};
    double deriv[8];
    CHECK(vmar_adaptive_field(0,initial,dim,deriv,&s)==VMAR_OK);
    double grad[4],loss=0;
    CHECK(vmar_custom_neuron_loss_gradient(&n,initial,4,&loss,grad,NULL)==VMAR_OK);
    for(size_t j=0;j<4;++j) {
        CHECK(fabs(deriv[j]+0.3*grad[j])<1e-13);
        CHECK(fabs(deriv[4+j]-grad[j]/0.8)<1e-13);
    }
    for(int m=0;m<2;++m) {
        vmar_integrator method=m?VMAR_RK4:VMAR_EULER;
        size_t required=0;
        CHECK(vmar_control_workspace_required(method,8,&required)==VMAR_OK);
        double workspace[48],a[8],b[8];
        memcpy(a,initial,sizeof(a));memcpy(b,initial,sizeof(b));
        double t=0;size_t stepid=0;
        vmar_control_policy p={0.1,0.5,3,0,0};
        vmar_control_report r;
        CHECK(vmar_adaptive_control_step(method,&s,&p,a,8,&t,&stepid,
               workspace,required,&r)==VMAR_OK);
        CHECK(r.accepted==1);
        CHECK(r.attempts==1 && r.rejected==0);
        CHECK(stepid==1 && fabs(t-0.1)<1e-14);
        CHECK(r.field_evaluations==(m?4u:1u));
        for(size_t j=0;j<8;++j) CHECK(isfinite(a[j]));
        CHECK(fabs(a[3]-initial[3])>1e-9);
        vmar_control_policy reject={9,0.5,1,0,0};
        double t0=t;size_t id0=stepid;
        CHECK(vmar_adaptive_control_step(method,&s,&reject,a,8,&t,&stepid,
               workspace,required,&r)==VMAR_OK);
        /* A long step can be rejected but is not guaranteed to be for RK4;
           check transactional invariants conditionally. */
        if(!r.accepted) {
            CHECK(memcmp(a,b,sizeof(a))!=0);
            CHECK(t==t0 && stepid==id0);
        }
        s.coupling=0.2;
        CHECK(vmar_adaptive_field(0,a,8,deriv,&s)==VMAR_OK);
        s.coupling=0;
    }
    s.tau=0;
    CHECK(vmar_adaptive_dimension(&s,&dim)==VMAR_ERR_ARGUMENT);
    printf("vMAR adaptive control: %d assertions, %d failures\n",checked,failed);
    return failed!=0;
}
