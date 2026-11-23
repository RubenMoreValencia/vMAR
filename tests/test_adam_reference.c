#include "../reference/adam_reference.h"
#include <math.h>
#include <stdio.h>

static int checks, failures;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x); } } while (0)
static int near(double x,double y) {return fabs(x-y)<1e-12;}
int main(void) {
    vmar_adam_reference a={0.01,0.9,0.999,1e-8,0,0};
    double theta[]={1,2},m[]={0,0},v[]={0,0},g[]={2,-4};
    CHECK(vmar_adam_reference_step(&a,g,2,theta,m,v)==VMAR_OK);
    CHECK(near(theta[0],1-0.01*2/(2+1e-8)));
    CHECK(near(theta[1],2+0.01*4/(4+1e-8)));
    CHECK(near(m[0],0.2)&&near(m[1],-0.4));
    CHECK(near(v[0],0.004)&&near(v[1],0.016));
    CHECK(a.iteration==1&&a.gradient_evaluations==1);
    CHECK(vmar_adam_reference_step(&a,g,2,theta,m,v)==VMAR_OK);
    CHECK(near(theta[0],1-2*0.01*2/(2+1e-8)));
    CHECK(a.gradient_evaluations==2);
    double before=theta[0],bm=m[0],bv=v[0];
    g[1]=NAN;
    CHECK(vmar_adam_reference_step(&a,g,2,theta,m,v)==VMAR_ERR_NUMERIC);
    CHECK(theta[0]==before&&m[0]==bm&&v[0]==bv&&a.iteration==2);
    a.beta2=1;
    CHECK(vmar_adam_reference_step(&a,g,2,theta,m,v)==VMAR_ERR_ARGUMENT);
    a.beta2=0.999;
    /* Exact exponential relaxation correspondence for constant signal. */
    double h=0.1,tau=-h/log(0.9),beta=exp(-h/tau),memory=0;
    for(int i=0;i<20;++i) memory=beta*memory+(1-beta)*2;
    CHECK(near(memory,2*(1-pow(0.9,20))));
    printf("vMAR Adam research contracts: %d assertions, %d failures\n",checks,failures);
    return failures!=0;
}
