#include "vmar/control.h"

#include <math.h>
#include <stdio.h>

static int count, fail;
#define CHECK(c) do { ++count; if (!(c)) {++fail; \
    fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c);} } while(0)
static int close_to(double a,double b) {return fabs(a-b)<=1e-13;}

int main(void) {
    const double x[]={1,0.5,-0.5,1.5,0.7,-1};
    const double target[]={0.25,-0.3,0.7};
    vmar_learning_system sys={{x,target,3,2,VMAR_ACT_TANH,0.3},0.6,0.2};
    vmar_control_policy policy={0.1,0.5,5,0.0,0.0};
    for (int method=VMAR_EULER;method<=VMAR_RK4;++method) {
        double state[]={0.2,-0.4,0.15,0,0,0};
        double time=0, workspace[36];size_t step_id=0,required=0;
        vmar_control_report report;
        CHECK(vmar_control_workspace_required((vmar_integrator)method,6,
                                               &required)==VMAR_OK);
        CHECK(required==(method==VMAR_EULER?12:36));
        CHECK(vmar_control_step((vmar_integrator)method,&sys,&policy,
              state,6,&time,&step_id,workspace,required,&report)==VMAR_OK);
        CHECK(report.accepted==1);
        CHECK(report.attempts>=1 && report.attempts<=5);
        CHECK(report.old_loss>=report.candidate_loss);
        CHECK(report.field_evaluations==report.attempts*(method==VMAR_EULER?1u:4u));
        CHECK(close_to(time,report.effective_step));
        CHECK(step_id==1);
        CHECK(isfinite(state[0])&&isfinite(state[3]));
    }
    /* A gradient-flow proposal with sufficiently extreme step should be
     * rejected; verify complete rollback of augmented state and metadata. */
    double state[]={0.2,-0.4,0.15,0,0,0};
    double original[]={0.2,-0.4,0.15,0,0,0};
    double workspace[36];size_t step_id=3;
    double time=1;
    vmar_control_report report;
    vmar_control_policy reject={1000,0.5,1,0,0};
    CHECK(vmar_control_step(VMAR_EULER,&sys,&reject,state,6,&time,
           &step_id,workspace,12,&report)==VMAR_OK);
    CHECK(report.accepted==0 && report.attempts==1);
    CHECK(report.rejected==1);
    CHECK(time==1 && step_id==3);
    for (size_t j=0;j<6;++j) CHECK(state[j]==original[j]);

    /* Feedback step gates observe only the confirmed memory. */
    policy.memory_threshold=0.5;
    state[3]=0.8;
    time=0;step_id=0;
    CHECK(vmar_control_step(VMAR_EULER,&sys,&policy,state,6,&time,
          &step_id,workspace,12,&report)==VMAR_OK);
    CHECK(report.accepted==1);
    CHECK(close_to(report.effective_step,0.05));
    CHECK(close_to(time,0.05));

    policy.reduction=1;
    CHECK(vmar_control_step(VMAR_EULER,&sys,&policy,state,6,&time,
          &step_id,workspace,12,&report)==VMAR_ERR_ARGUMENT);
    printf("vMAR control C23: %d assertions, %d failures\n",count,fail);
    return fail!=0;
}
