#include "vmar/network_system.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

static int checks=0, failures=0;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); } } while(0)
static int near(double a, double b) { return isfinite(a) && isfinite(b) && fabs(a-b)<1e-10*(1+fabs(b)); }

int main(void) {
    size_t widths[]={2,3,2,1}, offsets[3]={0};
    vmar_activation activations[]={VMAR_ACT_TANH,VMAR_ACT_SIGMOID,VMAR_ACT_IDENTITY};
    vmar_network_layout layout;
    CHECK(vmar_network_layout_make(widths,4,activations,offsets,3,&layout)==VMAR_OK);
    CHECK(layout.parameters==20);
    double x[]={-0.3,0.7,0.2,-0.5,0.9,0.1},y[]={0.2,-0.1,0.4};
    double scratch[128], deriv[40], state[40]={0}, baseline[20], stage[200];
    size_t work=0;
    CHECK(vmar_network_workspace_required(&layout,&work)==VMAR_OK);
    for(size_t i=0;i<20;++i) state[i]=0.12*sin((double)(i+1));
    state[20]=0.5; state[21]=-0.1;
    vmar_network_problem network={&layout,x,y,3,0.2,scratch,work};
    vmar_network_system system={network,0.7,0};
    CHECK(vmar_network_system_field(0,state,40,deriv,&system)==VMAR_OK);
    CHECK(vmar_network_field(0,state,20,baseline,&network)==VMAR_OK);
    for(size_t i=0;i<20;++i) CHECK(near(deriv[i],baseline[i]));
    CHECK(near(deriv[20],(-baseline[0]/0.2-0.5)/0.7));
    for(int method=1;method<=2;++method) {
        double s[40], b[20];
        memcpy(s,state,sizeof(s)); memcpy(b,state,sizeof(b));
        CHECK(vmar_step((vmar_integrator)method,0,.05,vmar_network_system_field,
                        &system,s,40,stage,200,s)==VMAR_OK);
        CHECK(vmar_step((vmar_integrator)method,0,.05,vmar_network_field,
                        &network,b,20,stage,200,b)==VMAR_OK);
        for(size_t i=0;i<20;++i) CHECK(near(s[i],b[i]));
    }
    system.coupling=0.2;
    CHECK(vmar_network_system_field(0,state,40,deriv,&system)==VMAR_OK);
    CHECK(near(deriv[0],baseline[0]-0.2*0.2*state[20]));
    CHECK(vmar_network_system_field(0,state,39,deriv,&system)==VMAR_ERR_DIMENSION);
    system.tau=0;
    CHECK(vmar_network_system_field(0,state,40,deriv,&system)==VMAR_ERR_ARGUMENT);
    system.tau=0.7;
    size_t required=0;
    CHECK(vmar_network_control_workspace_required(VMAR_RK4,20,&required)==VMAR_OK);
    CHECK(required==260);
    double control_work[260],state_before[40];
    memcpy(state_before,state,sizeof(state));
    vmar_control_policy policy={0.05,0.5,3,0,0.2};
    vmar_control_report report;
    double time=0;
    size_t step_id=0;
    CHECK(vmar_network_control_step(VMAR_EULER,&system,&policy,state,40,&time,
                                     &step_id,control_work,260,&report)==VMAR_OK);
    CHECK(report.accepted==1);
    CHECK(report.attempts>=1);
    CHECK(near(report.effective_step,0.025));
    CHECK(near(time,report.final_time));
    CHECK(step_id==1);
    CHECK(report.field_evaluations==report.attempts);
    CHECK(!near(state[0],state_before[0]));
    CHECK(!near(state[20],state_before[20]));
    CHECK(report.candidate_loss<=report.old_loss);

    memcpy(state,state_before,sizeof(state)); time=0;step_id=0;
    policy.initial_step=10; policy.max_attempts=1;policy.memory_threshold=0;
    CHECK(vmar_network_control_step(VMAR_EULER,&system,&policy,state,40,&time,
                                     &step_id,control_work,260,&report)==VMAR_OK);
    CHECK(report.accepted==0);
    CHECK(report.attempts==1);
    CHECK(report.rejected==1);
    CHECK(step_id==0);
    CHECK(time==0);
    for(size_t i=0;i<40;++i) CHECK(near(state[i],state_before[i]));
    CHECK(vmar_network_control_step(VMAR_RK4,&system,&policy,state,40,&time,
                                     &step_id,control_work,10,&report)==VMAR_ERR_WORKSPACE);
    printf("vMAR coupled multilayer contracts: %d assertions, %d failures\n",checks,failures);
    return failures?1:0;
}
