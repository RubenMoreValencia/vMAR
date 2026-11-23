#include "../reference/second_moment_reference.h"
#include <math.h>
#include <stdio.h>

static int checks,failures;
#define CHECK(x) do{++checks;if(!(x)){++failures;fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);}}while(0)
static vmar_status identity(double z,const double *p,size_t q,double *v,
    double *dz,double *dp,void *user) {
    (void)p;(void)dp;(void)user;
    if(q)return VMAR_ERR_ARGUMENT;
    *v=z;*dz=1;return VMAR_OK;
}
int main(void) {
    const double x[]={1,2},target[]={0,1};
    vmar_custom_neuron model={x,target,2,1,NULL,0,0,0.3,
                               identity,NULL,NULL,0};
    double state[6]={0.2,0.1,0.3,0.4,0,0},derivative[6];
    vmar_second_system sys={&model,0.8,0.4,0.2,0.1,VMAR_SECOND_M2_OBSERVE};
    CHECK(vmar_second_field(0,state,6,derivative,&sys)==VMAR_OK);
    CHECK(derivative[4]>=0&&derivative[5]>=0);
    double plain[4],m1[]={0.2,0.1,0.3,0.4};
    sys.mode=VMAR_SECOND_M1;
    CHECK(vmar_second_field(0,m1,4,plain,&sys)==VMAR_OK);
    for(int j=0;j<4;++j)CHECK(fabs(plain[j]-derivative[j])<1e-14);
    sys.mode=VMAR_SECOND_M2_CONTROL;
    CHECK(vmar_second_field(0,state,6,derivative,&sys)==VMAR_OK);
    for(int j=0;j<4;++j)CHECK(fabs(plain[j]-derivative[j])<1e-14);
    state[4]=1;
    CHECK(vmar_second_field(0,state,6,derivative,&sys)==VMAR_OK);
    CHECK(fabs(plain[0]-derivative[0])>1e-6);
    state[4]=-1;
    CHECK(vmar_second_field(0,state,6,derivative,&sys)==VMAR_ERR_NUMERIC);
    state[4]=0;sys.tau2=0;
    CHECK(vmar_second_field(0,state,6,derivative,&sys)==VMAR_ERR_ARGUMENT);
    printf("vMAR second moment research contracts: %d assertions, %d failures\n",checks,failures);
    return failures!=0;
}
