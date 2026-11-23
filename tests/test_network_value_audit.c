#include "vmar/network_system.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct result {
    double initial_loss, final_loss, final_time, theta1, memory1;
    size_t accepted, attempts, rejected, field_evaluations, interventions;
    int reached_target;
} result;

static int near(double a,double b){return isfinite(a)&&isfinite(b)&&fabs(a-b)<=4e-10*(1+fabs(b));}

static int run_case(vmar_integrator method,const char *policy,
                    double perturb,double memory_sign,result *out) {
    size_t widths[]={2,3,2,1},offsets[3];
    vmar_activation acts[]={VMAR_ACT_TANH,VMAR_ACT_SIGMOID,VMAR_ACT_IDENTITY};
    vmar_network_layout layout;
    if(vmar_network_layout_make(widths,4,acts,offsets,3,&layout)!=VMAR_OK)return 0;
    double x[]={-0.3,0.7,0.2,-0.5,0.9,0.1}, target[]={0.2,-0.1,0.4};
    double state[40]={0},work[128],integration[200],control_work[260],gradient[20];
    for(size_t j=0;j<20;++j)state[j]=0.12*sin((double)(j+1))+perturb;
    state[20]=memory_sign*0.5;
    size_t needed=0;
    if(vmar_network_workspace_required(&layout,&needed)!=VMAR_OK)return 0;
    vmar_network_problem n={&layout,x,target,3,0.2,work,needed};
    int m0=strcmp(policy,"M0")==0, m3=strcmp(policy,"M3")==0;
    double coupling=(strcmp(policy,"M2")==0||m3)?0.2:0;
    if(!m0 && !m3 && strcmp(policy,"M1")!=0 && strcmp(policy,"M2")!=0)
        return 0;
    vmar_network_system system={n,0.7,coupling};
    memset(out,0,sizeof(*out));
    if(vmar_network_loss_gradient(&layout,x,target,3,state,work,needed,
                                   &out->initial_loss,gradient,NULL)!=VMAR_OK)return 0;
    const double T=0.5, h=0.05;
    double time=0;
    while(time<T-1e-12) {
        double step=fmin(h,T-time);
        if(m3) {
            vmar_control_policy control={step,0.5,4,0,0.2};
            vmar_control_report report;
            size_t count=out->accepted;
            if(vmar_network_control_step(method,&system,&control,state,40,&time,
                 &count,control_work,260,&report)!=VMAR_OK)return 0;
            out->attempts+=report.attempts;
            out->rejected+=report.rejected;
            out->field_evaluations+=report.field_evaluations;
            out->interventions+=(size_t)(report.effective_step<step);
            if(!report.accepted)break;
        } else {
            vmar_status status;
            if(m0) status=vmar_step(method,time,step,vmar_network_field,&n,
                                   state,20,integration,200,state);
            else status=vmar_step(method,time,step,vmar_network_system_field,
                                  &system,state,40,integration,200,state);
            if(status!=VMAR_OK)return 0;
            out->field_evaluations+=method==VMAR_EULER?1:4;
            ++out->attempts;
            time+=step;
        }
        if(++out->accepted>500)return 0;
    }
    out->final_time=time;
    out->reached_target=fabs(T-time)<=1e-10;
    out->theta1=state[0];
    out->memory1=state[20];
    return vmar_network_loss_gradient(&layout,x,target,3,state,work,needed,
                                      &out->final_loss,gradient,NULL)==VMAR_OK;
}

int main(int argc,char **argv){
    if(argc==2 && strcmp(argv[1],"--smoke")==0){
        const char *policies[]={"M0","M1","M2","M3"};
        size_t checked=0,failures=0;
        for(int method=1;method<=2;++method){
            for(size_t perturb=0;perturb<2;++perturb){
                for(size_t sign=0;sign<2;++sign){
                    result observed[4];
                    for(size_t p=0;p<4;++p){
                        if(!run_case((vmar_integrator)method,policies[p],
                                     perturb==0?-0.05:0.05,sign==0?-1:1,
                                     &observed[p]))++failures;
                        else {
                            ++checked;
                            if(!observed[p].reached_target ||
                               !isfinite(observed[p].final_loss) ||
                               observed[p].field_evaluations==0)++failures;
                        }
                    }
                    if(!near(observed[0].final_loss,observed[1].final_loss)||
                       !near(observed[0].theta1,observed[1].theta1))++failures;
                }
            }
        }
        printf("vMAR G0.7-B.2 fixed-time policies: %zu runs, %zu failures\n",
               checked,failures);
        return failures?1:0;
    }
    if(argc!=2){fprintf(stderr,"usage: %s CSV|--smoke\n",argv[0]);return 2;}
    FILE *f=fopen(argv[1],"r");if(!f){perror(argv[1]);return 2;}
    char line[2048];if(!fgets(line,sizeof(line),f)){fclose(f);return 2;}
    size_t rows=0,failed=0;
    while(fgets(line,sizeof(line),f)) {
        char method_name[24],policy[8],reached[12];
        double perturb,sign,initial,final,time,theta1,memory1;
        size_t accepted,attempts,rejected,evaluations,interventions;
        int n=sscanf(line,
            "%23[^,],%7[^,],%lf,%lf,%lf,%lf,%lf,%zu,%zu,%zu,%zu,%zu,%lf,%lf,%11s",
            method_name,policy,&perturb,&sign,&initial,&final,&time,
            &accepted,&attempts,&rejected,&evaluations,&interventions,
            &theta1,&memory1,reached);
        vmar_integrator method=strcmp(method_name,"rk4")==0?VMAR_RK4:VMAR_EULER;
        result r;
        int expected_reached=strcmp(reached,"TRUE")==0;
        ++rows;
        if(n!=15 || !run_case(method,policy,perturb,sign,&r) ||
           !near(r.initial_loss,initial)||!near(r.final_loss,final)||
           !near(r.final_time,time)||!near(r.theta1,theta1)||
           !near(r.memory1,memory1)||r.accepted!=accepted||
           r.attempts!=attempts||r.rejected!=rejected||
           r.field_evaluations!=evaluations||r.interventions!=interventions||
           r.reached_target!=expected_reached){
            fprintf(stderr,"VALUE AUDIT PARITY FAIL row %zu\n",rows);++failed;
        }
    }
    fclose(f);
    if(rows!=32){++failed;fprintf(stderr,"expected 32 rows, got %zu\n",rows);}
    printf("vMAR R-C23 effective-contribution audit: %zu rows, %zu failures\n",rows,failed);
    return failed?EXIT_FAILURE:EXIT_SUCCESS;
}
