#include "vmar/vmar.h"
#include "vmar/state_layout.h"
#include "vmar/memory.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int checks,failures,calls;
#define CHECK(x) do{++checks;if(!(x)){++failures;fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x);}}while(0)

static vmar_status smooth(double t,const double *state,size_t n,
                          double *derivative,void *ctx) {
    (void)t;(void)ctx;++calls;
    for(size_t j=0;j<n;++j)derivative[j]=-state[j];
    return VMAR_OK;
}
static vmar_status fail(double t,const double *state,size_t n,
                        double *derivative,void *ctx) {
    (void)t;(void)state;(void)n;(void)derivative;(void)ctx;
    ++calls;return VMAR_ERR_FIELD;
}
static vmar_status nonfinite(double t,const double *state,size_t n,
                             double *derivative,void *ctx) {
    (void)t;(void)state;(void)ctx;++calls;
    for(size_t j=0;j<n;++j)derivative[j]=NAN;
    return VMAR_OK;
}
static int close_to(double a,double b) {return fabs(a-b)<1e-12;}

int main(void) {
    const double initial[2]={1,-2};
    for(int method=VMAR_EULER;method<=VMAR_RK4;++method) {
        size_t needed=0;
        CHECK(vmar_workspace_required((vmar_integrator)method,2,&needed)==VMAR_OK);
        CHECK(needed==(size_t)(method==VMAR_EULER?2:10));
        double workspace[10],out[2]={88,99};calls=0;
        CHECK(vmar_step((vmar_integrator)method,0,0.1,smooth,NULL,
                        initial,2,workspace,needed,out)==VMAR_OK);
        CHECK(calls==(method==VMAR_EULER?1:4));
        CHECK(initial[0]==1&&initial[1]==-2);
        CHECK(isfinite(out[0])&&isfinite(out[1]));
        /* Linear homogeneous field implies exact sign/scale equivariance
         * for both explicit methods, not a universal nonlinear invariant. */
        CHECK(close_to(out[1],-2*out[0]));
        out[0]=88;out[1]=99;
        CHECK(vmar_step((vmar_integrator)method,0,0.1,fail,NULL,
                        initial,2,workspace,needed,out)==VMAR_ERR_FIELD);
        CHECK(out[0]==88&&out[1]==99);
        CHECK(vmar_step((vmar_integrator)method,0,0.1,nonfinite,NULL,
                        initial,2,workspace,needed,out)==VMAR_ERR_NUMERIC);
        CHECK(out[0]==88&&out[1]==99);
        CHECK(vmar_step((vmar_integrator)method,0,-0.1,smooth,NULL,
                        initial,2,workspace,needed,out)==VMAR_ERR_ARGUMENT);
        CHECK(out[0]==88&&out[1]==99);
        CHECK(vmar_step((vmar_integrator)method,0,0.1,smooth,NULL,
                        initial,2,workspace,needed-1,out)==VMAR_ERR_WORKSPACE);
        CHECK(out[0]==88&&out[1]==99);
    }
    vmar_state_layout layout;
    CHECK(vmar_state_layout_make(3,VMAR_STATE_SECOND_MOMENT,&layout)==VMAR_OK);
    double s[9]={1,2,3,0,0,0,0,0,0};
    CHECK(vmar_state_layout_validate(&layout,s,9)==VMAR_OK);
    s[8]=-1;
    CHECK(vmar_state_layout_validate(&layout,s,9)==VMAR_ERR_NUMERIC);
    s[8]=0;
    CHECK(vmar_state_layout_validate(&layout,s,8)==VMAR_ERR_DIMENSION);
    double m[]={0,0},g[]={2,-4},next[]={91,92};
    CHECK(vmar_memory_relaxation(m,g,2,0.1,0.5,next)==VMAR_OK);
    CHECK(close_to(next[0],2*(-expm1(-0.2))));
    CHECK(close_to(next[1],-4*(-expm1(-0.2))));
    CHECK(m[0]==0&&m[1]==0&&g[0]==2&&g[1]==-4);
    printf("vMAR FormalSpec alignment properties: %d assertions, %d failures\n",
           checks,failures);
    return failures!=0;
}
