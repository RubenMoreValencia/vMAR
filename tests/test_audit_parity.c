#include "vmar/control.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct audit_result {
    double final_time,loss,max_theta,max_memory,error;
    size_t accepted,rejected,calls;
    double state[6];
} audit_result;

static double max_abs(const double *data,size_t n) {
    double m=0;
    for(size_t i=0;i<n;++i) if(fabs(data[i])>m) m=fabs(data[i]);
    return m;
}
static int close_to(double a,double b) {
    return isfinite(a)&&isfinite(b)&&fabs(a-b)<=5e-10+5e-10*fabs(b);
}
static int run(const char *phase,const char *scenario,vmar_integrator method,
               vmar_activation activation,double perturb,audit_result *r) {
    const double x[]={1,0.5,-0.5,1.5,0.7,-1};
    const double target[]={0.25,-0.3,0.7};
    const int control=strcmp(phase,"control")==0;
    const int budget=strcmp(phase,"equal_budget")==0;
    const int baseline=strcmp(scenario,"baseline")==0;
    const double coupling=(strcmp(scenario,"coupled")==0 || control)?0.2:0;
    if (!baseline && !control && strcmp(scenario,"observe") &&
        strcmp(scenario,"coupled")) return 1;
    vmar_learning_system system={{x,target,3,2,activation,0.3},0.6,coupling};
    double s[6]={0.2+perturb,-0.4,0.15,baseline?0:0.8,0,0};
    double initial[6];memcpy(initial,s,sizeof s);
    double workspace[36];double time=0;
    size_t step_id=0;
    *r=(audit_result){0};
    r->max_theta=max_abs(s,3);
    r->max_memory=max_abs(s+3,3);
    if (!control) {
        size_t required=method==VMAR_EULER?6:30;
        size_t steps=(budget && method==VMAR_RK4)?5:20;
        double h=(budget && method==VMAR_RK4)?0.2:0.05;
        for(size_t k=0;k<steps;++k) {
            vmar_status status=vmar_step(method,(double)k*h,h,
                 vmar_learning_field,&system,s,6,workspace,required,s);
            if(status!=VMAR_OK) return 1;
            r->calls+=method==VMAR_EULER?1:4;
            time=(k+1)*h;
            ++r->accepted;
            double v=max_abs(s,3),m=max_abs(s+3,3);
            if(v>r->max_theta) r->max_theta=v;
            if(m>r->max_memory) r->max_memory=m;
        }
    } else {
        vmar_control_policy policy={0.2,0.5,6,0,0.5};
        for(size_t k=0;k<20;++k) {
            vmar_control_report report;
            vmar_status status=vmar_control_step(method,&system,&policy,s,6,
                 &time,&step_id,workspace,method==VMAR_EULER?12:36,&report);
            if(status!=VMAR_OK) return 1;
            r->accepted+=report.accepted?1u:0u;
            r->rejected+=report.rejected;
            r->calls+=report.field_evaluations;
            double v=max_abs(s,3),m=max_abs(s+3,3);
            if(v>r->max_theta) r->max_theta=v;
            if(m>r->max_memory) r->max_memory=m;
        }
    }
    double gradient[3];
    if(vmar_neuron_loss_gradient(&system.neuron,s,&r->loss,
                                 gradient,NULL)!=VMAR_OK) return 1;
    r->final_time=time;
    memcpy(r->state,s,sizeof s);
    r->error=-1;
    if(!control) {
        double ref[6];memcpy(ref,initial,sizeof ref);
        for(size_t k=0;k<200;++k) {
            double w[30];
            if(vmar_step(VMAR_RK4,(double)k*0.005,0.005,
                vmar_learning_field,&system,ref,6,w,30,ref)!=VMAR_OK) return 1;
        }
        double sq=0;
        for(size_t j=0;j<6;++j) sq+=(s[j]-ref[j])*(s[j]-ref[j]);
        r->error=sqrt(sq);
    }
    return 0;
}

static int check(const char *line) {
    char id[96],phase[24],activation[24],method[24],scenario[24];
    double perturb,expected_time,expected_loss,expected_max_theta,
           expected_max_memory,expected_error,expected_state[6];
    size_t accepted,rejected,calls;
    int n=sscanf(line,
      "%95[^,],%23[^,],%23[^,],%23[^,],%23[^,],%lf,%lf,%lf,%lf,%lf,%lf,%zu,%zu,%zu,%lf,%lf,%lf,%lf,%lf,%lf",
      id,phase,activation,method,scenario,&perturb,&expected_time,
      &expected_loss,&expected_max_theta,&expected_max_memory,&expected_error,
      &accepted,&rejected,&calls,&expected_state[0],&expected_state[1],
      &expected_state[2],&expected_state[3],&expected_state[4],&expected_state[5]);
    if(n!=20) return 1;
    vmar_activation a;
    if(!strcmp(activation,"tanh")) a=VMAR_ACT_TANH;
    else if(!strcmp(activation,"sigmoid")) a=VMAR_ACT_SIGMOID;
    else return 1;
    vmar_integrator integrator;
    if(!strcmp(method,"euler")) integrator=VMAR_EULER;
    else if(!strcmp(method,"rk4")) integrator=VMAR_RK4;
    else return 1;
    if((strcmp(phase,"control")==0)!=(strcmp(scenario,"control")==0))
        return 1;
    audit_result got;
    if(run(phase,scenario,integrator,a,perturb,&got)) return 1;
    int ok=close_to(got.final_time,expected_time) &&
        close_to(got.loss,expected_loss) &&
        close_to(got.max_theta,expected_max_theta) &&
        close_to(got.max_memory,expected_max_memory) &&
        close_to(got.error,expected_error) &&
        got.accepted==accepted && got.rejected==rejected && got.calls==calls;
    for(size_t j=0;j<6;++j) ok &= close_to(got.state[j],expected_state[j]);
    printf("AUDIT PARITY %s %s\n",ok?"PASS":"FAIL",id);
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
    if(cases!=84) ++failures;
    printf("vMAR R-C23 stability audit: %zu cases, %d failures\n",
           cases,failures);
    return failures!=0;
}
