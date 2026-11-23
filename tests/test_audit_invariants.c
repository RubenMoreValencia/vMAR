#include "vmar/control.h"

#include <math.h>
#include <stdio.h>

static int tests, failures;
#define CHECK(v) do { ++tests; if(!(v)) {++failures; \
    fprintf(stderr,"FAIL %d: %s\n",__LINE__,#v);} }while(0)
static int near(double a,double b) {return fabs(a-b)<1e-12;}

int main(void) {
    const double x[]={1,0.5,-0.5,1.5,0.7,-1};
    const double target[]={0.25,-0.3,0.7};
    for(int act=VMAR_ACT_TANH; act<=VMAR_ACT_SIGMOID; ++act) {
        for(int method=VMAR_EULER;method<=VMAR_RK4;++method) {
            vmar_learning_system a={{x,target,3,2,(vmar_activation)act,0.3},0.6,0};
            double baseline[6]={0.2,-0.4,0.15,0,0,0};
            double observed[6]={0.2,-0.4,0.15,0.8,0,0};
            double work[30];size_t required=method==VMAR_EULER?6:30;
            for(size_t k=0;k<20;++k) {
                CHECK(vmar_step((vmar_integrator)method,k*0.05,0.05,
                    vmar_learning_field,&a,baseline,6,work,required,baseline)==VMAR_OK);
                CHECK(vmar_step((vmar_integrator)method,k*0.05,0.05,
                    vmar_learning_field,&a,observed,6,work,required,observed)==VMAR_OK);
            }
            for(size_t j=0;j<3;++j) CHECK(near(baseline[j],observed[j]));
            CHECK(isfinite(observed[3]));
        }
    }
    printf("vMAR stability audit invariants: %d assertions, %d failures\n",
           tests,failures);
    return failures!=0;
}
