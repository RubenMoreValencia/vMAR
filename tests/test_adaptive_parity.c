#include "vmar/adaptive_control.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static vmar_status prelu(double z,const double *p,size_t q,
    double *y,double *dz,double *dp,void *u) {
    (void)u;if(q!=1||!p)return VMAR_ERR_ARGUMENT;
    *y=z>0?z:p[0]*z;*dz=z>0?1:p[0];
    if(dp)dp[0]=z<0?z:0;
    return VMAR_OK;
}
static int close_to(double a,double b) {
    return isfinite(a)&&isfinite(b)&&fabs(a-b)<=5e-11*(1+fabs(b));
}
int main(int argc,char **argv) {
    if(argc!=2) {fprintf(stderr,"Usage: %s FILE.csv\n",argv[0]);return 2;}
    FILE *f=fopen(argv[1],"r");
    if(!f) {perror(argv[1]);return 2;}
    char line[2048];
    if(!fgets(line,sizeof(line),f)){fclose(f);return 2;}
    int count=0,fail=0;
    const double x[]={1,0.5,-0.5,1.5,0.7,-1};
    const double target[]={0.25,-0.3,0.7};
    while(fgets(line,sizeof(line),f)) {
        char *fields[20];int n=0;
        for(char *token=strtok(line,",\r\n");token && n<20;
            token=strtok(NULL,",\r\n")) fields[n++]=token;
        if(n!=20){fprintf(stderr,"Malformed fixture %d (%d fields)\n",count+1,n);++fail;continue;}
        const char *id=fields[0];
        vmar_integrator method=strcmp(fields[1],"euler")==0?VMAR_EULER:VMAR_RK4;
        double coupling=strtod(fields[2],NULL),mem=strtod(fields[3],NULL);
        double step=strtod(fields[4],NULL);
        double alpha=0.1,scratch[1];
        vmar_custom_neuron neuron={x,target,3,2,&alpha,1,1,0.3,prelu,NULL,scratch,1};
        vmar_adaptive_system system={&neuron,0.8,coupling};
        double state[8]={0.2,-0.4,0.15,0.1,mem,mem,mem,mem};
        size_t required=0;
        if(vmar_control_workspace_required(method,8,&required)!=VMAR_OK) return 2;
        double workspace[48],time=0;size_t step_id=0;
        vmar_control_policy policy={step,0.5,4,0,0.5};
        vmar_control_report report={0};
        vmar_status status=vmar_adaptive_control_step(method,&system,&policy,
            state,8,&time,&step_id,workspace,required,&report);
        int ok=status==VMAR_OK &&
            report.accepted==atoi(fields[5]) &&
            report.attempts==(size_t)strtoul(fields[6],NULL,10) &&
            report.rejected==(size_t)strtoul(fields[7],NULL,10) &&
            report.field_evaluations==(size_t)strtoul(fields[8],NULL,10) &&
            close_to(report.effective_step,strtod(fields[9],NULL)) &&
            close_to(time,strtod(fields[10],NULL)) &&
            close_to(report.candidate_loss,strtod(fields[11],NULL));
        for(int j=0;j<8;++j) ok=ok && close_to(state[j],strtod(fields[12+j],NULL));
        if(!ok){++fail;fprintf(stderr,"ADAPTIVE PARITY FAIL %s\n",id);}
        else printf("ADAPTIVE PARITY PASS %s\n",id);
        ++count;
    }
    fclose(f);
    if(count!=16){++fail;fprintf(stderr,"Expected 16 cases, got %d\n",count);}
    printf("vMAR R-C23 adaptive parity: %d cases, %d failures\n",count,fail);
    return fail!=0;
}
