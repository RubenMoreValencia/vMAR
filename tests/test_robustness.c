#include "vmar/control.h"

#include <math.h>
#include <stdio.h>

int main(void) {
    const double x[]={1,0.5,-0.5,1.5,0.7,-1};
    const double target[]={0.25,-0.3,0.7};
    const double perturbs[]={-0.5,0,0.5};
    const double taus[]={0.1,0.6,2};
    const double couplings[]={0,0.2,0.8};
    const double steps[]={0.1,1,10};
    const size_t limits[]={1,6};
    size_t cases=0,fail=0,accepted=0,exhausted=0,recovered=0,rejected=0;
    for(int a=VMAR_ACT_TANH;a<=VMAR_ACT_SIGMOID;++a)
    for(int method=VMAR_EULER;method<=VMAR_RK4;++method)
    for(size_t i=0;i<3;++i)
    for(size_t j=0;j<3;++j)
    for(size_t k=0;k<3;++k)
    for(size_t h=0;h<3;++h)
    for(size_t l=0;l<2;++l) {
        ++cases;
        vmar_learning_system sys={
            {x,target,3,2,(vmar_activation)a,0.3},taus[j],couplings[k]};
        vmar_control_policy p={steps[h],0.5,limits[l],0,0.5};
        double state[]={0.2+perturbs[i],-0.4,0.15,
                        0.8+perturbs[i],0,0};
        double original[6];for(size_t z=0;z<6;++z) original[z]=state[z];
        double workspace[36],time=0;size_t step_id=0;
        vmar_control_report r;
        vmar_status status=vmar_control_step((vmar_integrator)method,
            &sys,&p,state,6,&time,&step_id,workspace,
            method==VMAR_EULER?12:36,&r);
        if(status!=VMAR_OK) {
            fprintf(stderr,"STATUS FAIL case %zu: %s\n",cases,
                    vmar_status_string(status));++fail;continue;
        }
        if(r.attempts<1||r.attempts>limits[l]||
           r.field_evaluations!=r.attempts*(method==VMAR_EULER?1u:4u)||
           r.rejected+r.accepted!=r.attempts||
           !isfinite(r.old_loss)||!isfinite(r.candidate_loss)) ++fail;
        rejected+=r.rejected;
        if(r.accepted) {
            ++accepted;
            if(r.rejected) ++recovered;
            if(step_id!=1||time<=0||!isfinite(time)||
               r.candidate_loss>r.old_loss) ++fail;
        } else {
            ++exhausted;
            if(step_id!=0||time!=0) ++fail;
            for(size_t z=0;z<6;++z) if(state[z]!=original[z]) ++fail;
        }
    }
    if(cases!=648||accepted==0||exhausted==0||recovered==0||rejected==0) ++fail;
    printf("vMAR robustness: %zu cases, %zu accepted, %zu exhausted, "
           "%zu recovered, %zu rejected attempts, %zu failures\n",
           cases,accepted,exhausted,recovered,rejected,fail);
    return fail!=0;
}
