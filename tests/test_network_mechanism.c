#include "vmar/network_mechanism.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int checks=0,failures=0;
#define CHECK(x) do {++checks;if(!(x)){++failures;fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);}}while(0)
static int eq(double a,double b){return isfinite(a)&&isfinite(b)&&fabs(a-b)<=1e-10*(1+fabs(b));}

int main(void){
    size_t widths[]={2,3,2,1},offsets[3];
    vmar_activation acts[]={VMAR_ACT_TANH,VMAR_ACT_SIGMOID,VMAR_ACT_IDENTITY};
    vmar_network_layout layout;
    CHECK(vmar_network_layout_make(widths,4,acts,offsets,3,&layout)==VMAR_OK);
    double x[]={-0.3,0.7,0.2,-0.5,0.9,0.1},y[]={0.2,-0.1,0.4};
    double nwork[128],work[260],initial[40]={0},states[4][40];
    size_t needed=0;
    CHECK(vmar_network_workspace_required(&layout,&needed)==VMAR_OK);
    for(size_t i=0;i<20;++i)initial[i]=0.12*sin((double)(i+1));
    initial[20]=0.5;initial[21]=-0.1;
    vmar_network_problem neural={&layout,x,y,3,0.2,nwork,needed};
    vmar_network_system system={neural,0.7,0.2};
    vmar_control_policy policy={0.05,0.5,4,0,0.2};
    CHECK(vmar_network_default_mode()==VMAR_NETWORK_OBSERVE);
    for(int method=VMAR_EULER;method<=VMAR_RK4;++method){
        vmar_network_decision decisions[4];
        for(int i=0;i<4;++i){
            memcpy(states[i],initial,sizeof(initial));
            vmar_network_mechanism m={(vmar_network_mode)(i+1),system,policy};
            double t=0;
            size_t id=0;
            CHECK(vmar_network_mechanism_step((vmar_integrator)method,&m,
                  0.05,states[i],40,&t,&id,work,260,&decisions[i])==VMAR_OK);
            CHECK(decisions[i].accepted==1);
            CHECK(id==1);
            CHECK(eq(decisions[i].time_after,t));
            CHECK(eq(decisions[i].time_before,0));
            CHECK(eq(decisions[i].requested_step,0.05));
            CHECK(isfinite(decisions[i].loss_after));
            CHECK(decisions[i].gradient_norm_before>0);
            CHECK(decisions[i].memory_norm_before>0);
            CHECK(decisions[i].field_evaluations>=
                  ((vmar_integrator)method==VMAR_EULER?1u:4u));
            if(i<2) CHECK(eq(decisions[i].coupling_term_norm_before,0));
            else CHECK(decisions[i].coupling_term_norm_before>0);
        }
        for(size_t j=0;j<20;++j)CHECK(eq(states[0][j],states[1][j]));
        CHECK(eq(states[0][20],initial[20]));
        CHECK(!eq(states[1][20],initial[20]));
        CHECK(!eq(states[2][0],states[1][0]));
        CHECK(decisions[3].effective_step<decisions[2].effective_step);

        vmar_network_mechanism invalid={VMAR_NETWORK_CONTROLLED,system,policy};
        double unchanged[40];memcpy(unchanged,initial,sizeof(unchanged));
        double t=0;size_t id=0;
        CHECK(vmar_network_mechanism_step((vmar_integrator)method,&invalid,
              0.05,unchanged,40,&t,&id,work,1,&decisions[0])==VMAR_ERR_WORKSPACE);
        CHECK(eq(t,0));CHECK(id==0);
        for(size_t j=0;j<40;++j)CHECK(eq(unchanged[j],initial[j]));
    }
    printf("vMAR selected multilayer mechanisms: %d assertions, %d failures\n",
           checks,failures);
    return failures?1:0;
}
