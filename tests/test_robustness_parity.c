#include "vmar/control.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int near(double a,double b) {
    return isfinite(a)&&isfinite(b)&&fabs(a-b)<=5e-10+5e-10*fabs(b);
}
static int check(const char *line) {
    char id[128],activation[20],method[20];
    double perturb,tau,coupling,initial_step,effective_step,old_loss,
           candidate_loss,final_loss,final_time,expected_state[6];
    size_t max_attempts,attempts,rejected,calls,step_id;
    int accepted;
    int parsed=sscanf(line,
      "%127[^,],%19[^,],%19[^,],%lf,%lf,%lf,%lf,%zu,%d,%zu,%zu,%zu,%lf,%lf,%lf,%lf,%lf,%zu,%lf,%lf,%lf,%lf,%lf,%lf",
      id,activation,method,&perturb,&tau,&coupling,&initial_step,
      &max_attempts,&accepted,&attempts,&rejected,&calls,
      &effective_step,&old_loss,&candidate_loss,&final_loss,&final_time,
      &step_id,&expected_state[0],&expected_state[1],&expected_state[2],
      &expected_state[3],&expected_state[4],&expected_state[5]);
    if(parsed!=24||(max_attempts!=1&&max_attempts!=6)) return 1;
    vmar_activation a;
    if(!strcmp(activation,"tanh")) a=VMAR_ACT_TANH;
    else if(!strcmp(activation,"sigmoid")) a=VMAR_ACT_SIGMOID;
    else return 1;
    vmar_integrator integrator;
    if(!strcmp(method,"euler")) integrator=VMAR_EULER;
    else if(!strcmp(method,"rk4")) integrator=VMAR_RK4;
    else return 1;
    const double x[]={1,0.5,-0.5,1.5,0.7,-1};
    const double target[]={0.25,-0.3,0.7};
    vmar_learning_system sys={{x,target,3,2,a,0.3},tau,coupling};
    vmar_control_policy policy={initial_step,0.5,max_attempts,0,0.5};
    double state[]={0.2+perturb,-0.4,0.15,0.8+perturb,0,0};
    double work[36],time=0;
    size_t confirmed_id=0,required=integrator==VMAR_EULER?12:36;
    vmar_control_report report;
    vmar_status status=vmar_control_step(integrator,&sys,&policy,
          state,6,&time,&confirmed_id,work,required,&report);
    int ok=status==VMAR_OK;
    if(ok) {
        double loss=0,gradient[3];
        ok &= vmar_neuron_loss_gradient(&sys.neuron,state,&loss,
                                        gradient,NULL)==VMAR_OK;
        ok &= report.accepted==accepted;
        ok &= report.attempts==attempts && report.rejected==rejected;
        ok &= report.field_evaluations==calls;
        ok &= near(report.effective_step,effective_step);
        ok &= near(report.old_loss,old_loss);
        ok &= near(report.candidate_loss,candidate_loss);
        ok &= near(report.final_time,final_time);
        ok &= near(loss,final_loss) && near(time,final_time);
        ok &= confirmed_id==step_id;
        for(size_t j=0;j<6;++j) ok &= near(state[j],expected_state[j]);
    }
    if(!ok) fprintf(stderr,"ROBUSTNESS PARITY FAIL %s\n",id);
    return !ok;
}

int main(int argc,char **argv) {
    if(argc!=2) return 1;
    FILE *file=fopen(argv[1],"r");
    if(!file) {perror(argv[1]);return 1;}
    char line[4096];size_t cases=0;int failures=0;
    if(!fgets(line,sizeof line,file)) {fclose(file);return 1;}
    while(fgets(line,sizeof line,file)) {
        if(line[0]=='\n'||line[0]=='\r') continue;
        ++cases;failures+=check(line);
    }
    if(ferror(file)) ++failures;
    fclose(file);
    if(cases!=648) ++failures;
    printf("vMAR R-C23 robustness parity: %zu cases, %d failures\n",
           cases,failures);
    return failures!=0;
}
