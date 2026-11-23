#include "vmar/control.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int near(double a,double b) {
    return isfinite(a) && isfinite(b) &&
           fabs(a-b) <= 5e-11 + 5e-11*fabs(b);
}
static int check(const char *line) {
    char id[96],activation[20],method[20];
    double coupling,initial_step,old_loss,candidate_loss,effective_step,
           final_time,expected[6];
    int accepted;
    size_t attempts,rejected,step_id;
    int read=sscanf(line,
       "%95[^,],%19[^,],%19[^,],%lf,%lf,%d,%zu,%zu,%lf,%lf,%lf,%lf,%zu,%lf,%lf,%lf,%lf,%lf,%lf",
       id,activation,method,&coupling,&initial_step,&accepted,&attempts,
       &rejected,&old_loss,&candidate_loss,&effective_step,&final_time,
       &step_id,&expected[0],&expected[1],&expected[2],&expected[3],
       &expected[4],&expected[5]);
    if (read!=19) return 1;
    vmar_activation a;
    if (!strcmp(activation,"tanh")) a=VMAR_ACT_TANH;
    else if (!strcmp(activation,"sigmoid")) a=VMAR_ACT_SIGMOID;
    else return 1;
    vmar_integrator integrator;
    if (!strcmp(method,"euler")) integrator=VMAR_EULER;
    else if (!strcmp(method,"rk4")) integrator=VMAR_RK4;
    else return 1;
    const double x[]={1,0.5,-0.5,1.5,0.7,-1};
    const double target[]={0.25,-0.3,0.7};
    vmar_learning_system sys={{x,target,3,2,a,0.3},0.6,coupling};
    vmar_control_policy policy={initial_step,0.5,5,0,0.5};
    double state[]={0.2,-0.4,0.15,0.8,0,0};
    double workspace[36],time=0;
    size_t current_id=0,required=integrator==VMAR_EULER?12:36;
    vmar_control_report report;
    vmar_status status=vmar_control_step(integrator,&sys,&policy,
          state,6,&time,&current_id,workspace,required,&report);
    int ok=status==VMAR_OK;
    if (ok) {
        ok &= report.accepted==accepted;
        ok &= report.attempts==attempts && report.rejected==rejected;
        ok &= near(report.old_loss,old_loss);
        ok &= near(report.candidate_loss,candidate_loss);
        ok &= near(report.effective_step,effective_step);
        ok &= near(report.final_time,final_time);
        ok &= near(time,final_time) && current_id==step_id;
        for (size_t j=0;j<6;++j) ok &= near(state[j],expected[j]);
    }
    printf("CONTROL PARITY %s %s\n",ok?"PASS":"FAIL",id);
    return !ok;
}
int main(int argc,char **argv) {
    if (argc!=2) return 1;
    FILE *f=fopen(argv[1],"r");
    if (!f) {perror(argv[1]);return 1;}
    char line[2048];size_t cases=0;int failures=0;
    if (!fgets(line,sizeof line,f)) {fclose(f);return 1;}
    while (fgets(line,sizeof line,f)) {
        if (line[0]=='\n'||line[0]=='\r') continue;
        ++cases;failures+=check(line);
    }
    if (ferror(f)) ++failures;
    fclose(f);
    if (cases!=16) ++failures;
    printf("vMAR R-C23 control parity: %zu cases, %d failures\n",
           cases,failures);
    return failures!=0;
}
