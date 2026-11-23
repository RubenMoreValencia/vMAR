#include "vmar/network_system.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int near(double a,double b) {
    return isfinite(a)&&isfinite(b)&&fabs(a-b)<=3e-10*(1+fabs(b));
}

static int check_row(const char *method,double coupling,double proposed_step,
                     size_t coordinate,double initial,double final,
                     double final_time,int accepted,size_t attempts,
                     size_t rejected,double effective_step,double old_loss,
                     double candidate_loss) {
    size_t widths[]={2,3,2,1}, offsets[3];
    vmar_activation acts[]={VMAR_ACT_TANH,VMAR_ACT_SIGMOID,VMAR_ACT_IDENTITY};
    vmar_network_layout layout;
    if(vmar_network_layout_make(widths,4,acts,offsets,3,&layout)!=VMAR_OK ||
       coordinate<1 || coordinate>40) return 0;
    double x[]={-0.3,0.7,0.2,-0.5,0.9,0.1}, y[]={0.2,-0.1,0.4};
    double state[40]={0},work[128],control_work[260];
    for(size_t i=0;i<20;++i)state[i]=0.12*sin((double)(i+1));
    state[20]=0.5;state[21]=-0.1;
    if(!near(state[coordinate-1],initial))return 0;
    size_t network_required=0;
    if(vmar_network_workspace_required(&layout,&network_required)!=VMAR_OK)return 0;
    vmar_network_problem net={&layout,x,y,3,0.2,work,network_required};
    vmar_network_system system={net,0.7,coupling};
    vmar_control_policy policy={proposed_step,0.5,4,0,0.2};
    vmar_control_report report;
    size_t step_id=0;
    double time=0;
    vmar_integrator integrator;
    if(strcmp(method,"euler")==0)integrator=VMAR_EULER;
    else if(strcmp(method,"rk4")==0)integrator=VMAR_RK4;
    else return 0;
    if(vmar_network_control_step(integrator,&system,&policy,state,40,&time,
                                  &step_id,control_work,260,&report)!=VMAR_OK)
        return 0;
    return near(state[coordinate-1],final) && near(time,final_time) &&
           report.accepted==accepted && report.attempts==attempts &&
           report.rejected==rejected && near(report.effective_step,effective_step) &&
           near(report.old_loss,old_loss) &&
           near(report.candidate_loss,candidate_loss) &&
           step_id==(size_t)accepted;
}

int main(int argc,char **argv) {
    if(argc!=2){fprintf(stderr,"usage: %s CSV\n",argv[0]);return 2;}
    FILE *f=fopen(argv[1],"r");
    if(!f){perror(argv[1]);return 2;}
    char line[2048];
    if(!fgets(line,sizeof(line),f)){fclose(f);return 2;}
    size_t rows=0, failures=0;
    while(fgets(line,sizeof(line),f)) {
        char name[128],method[32];
        double coupling,h,initial,final,time,eff,old_loss,new_loss;
        size_t coordinate,attempts,rejected;
        int accepted;
        int fields=sscanf(line,
          "%127[^,],%31[^,],%lf,%lf,%zu,%lf,%lf,%lf,%d,%zu,%zu,%lf,%lf,%lf",
          name,method,&coupling,&h,&coordinate,&initial,&final,&time,&accepted,
          &attempts,&rejected,&eff,&old_loss,&new_loss);
        ++rows;
        if(fields!=14 || !check_row(method,coupling,h,coordinate,initial,final,
                                    time,accepted,attempts,rejected,eff,
                                    old_loss,new_loss)){
            ++failures;
            fprintf(stderr,"MULTILAYER SYSTEM PARITY FAIL row %zu\n",rows);
        }
    }
    fclose(f);
    if(rows!=320){++failures;fprintf(stderr,"expected 320 rows, got %zu\n",rows);}
    printf("vMAR R-C23 coupled multilayer parity: %zu rows, %zu failures\n",
           rows,failures);
    return failures?EXIT_FAILURE:EXIT_SUCCESS;
}
