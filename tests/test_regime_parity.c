#include "vmar/regime.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static vmar_status prelu(double z,const double *p,size_t q,
    double *v,double *dz,double *dp,void *u) {
    (void)u;if(!p||q!=1)return VMAR_ERR_ARGUMENT;
    *v=z>0?z:p[0]*z;*dz=z>0?1:p[0];
    if(dp)dp[0]=z<0?z:0;
    return VMAR_OK;
}
static int near(double a,double b) {
    return isfinite(a)&&isfinite(b)&&fabs(a-b)<=5e-11*(1+fabs(b));
}
static const char *regime_name(vmar_regime r) {
    switch(r) {
        case VMAR_REGIME_STATIONARY:return "stationary";
        case VMAR_REGIME_UNOBSERVED:return "unobserved";
        case VMAR_REGIME_ALIGNED:return "aligned";
        case VMAR_REGIME_OPPOSED:return "opposed";
        default:return "orthogonal";
    }
}
int main(int argc,char **argv) {
    if(argc!=2){fprintf(stderr,"Usage: %s fixtures.csv\n",argv[0]);return 2;}
    FILE *f=fopen(argv[1],"r");
    if(!f){perror(argv[1]);return 2;}
    char line[3072];
    if(!fgets(line,sizeof(line),f)){fclose(f);return 2;}
    const double x[]={1,0.5,-0.5,1.5,0.7,-1};
    const double target[]={0.25,-0.3,0.7};
    int count=0,fail=0;
    while(fgets(line,sizeof(line),f)) {
        char *cols[25];int nf=0;
        for(char *token=strtok(line,",\r\n");token && nf<25;
            token=strtok(NULL,",\r\n")) cols[nf++]=token;
        if(nf!=25){fprintf(stderr,"Malformed fixture %d fields=%d\n",count+1,nf);++fail;continue;}
        vmar_integrator method=strcmp(cols[0],"euler")==0?VMAR_EULER:VMAR_RK4;
        double cap=strtod(cols[1],NULL),perturb=strtod(cols[2],NULL);
        double memory0=strtod(cols[3],NULL),h=strtod(cols[4],NULL);
        double alpha=0.1,scratch[1];
        vmar_custom_neuron neuron={x,target,3,2,&alpha,1,1,0.3,prelu,NULL,scratch,1};
        vmar_adaptive_system sys={&neuron,0.8,0};
        double state[8]={0.2+perturb,-0.4,0.15,0.1,
                         memory0,memory0,memory0,memory0};
        vmar_control_policy policy={h,0.5,4,0,0.5};
        double workspace[48],time=0;size_t id=0;
        vmar_regime_report diagnostic;
        vmar_control_report response={0};
        vmar_status status=vmar_regime_control_step(method,&sys,&policy,cap,
            state,8,&time,&id,workspace,48,&diagnostic,&response);
        int ok=status==VMAR_OK;
        if(ok) {
            ok=strcmp(cols[5],regime_name(diagnostic.regime))==0 &&
                near(diagnostic.gradient_norm,strtod(cols[6],NULL)) &&
                near(diagnostic.memory_norm,strtod(cols[7],NULL)) &&
                near(diagnostic.alignment,strtod(cols[8],NULL)) &&
                near(diagnostic.risk_ratio,strtod(cols[9],NULL)) &&
                near(diagnostic.coupling,strtod(cols[10],NULL)) &&
                response.accepted==atoi(cols[11]) &&
                response.attempts==(size_t)strtoul(cols[12],NULL,10) &&
                response.rejected==(size_t)strtoul(cols[13],NULL,10) &&
                response.field_evaluations==(size_t)strtoul(cols[14],NULL,10) &&
                near(time,strtod(cols[15],NULL)) &&
                near(response.candidate_loss,strtod(cols[16],NULL));
            for(int j=0;j<8;++j)
                ok=ok && near(state[j],strtod(cols[17+j],NULL));
        }
        if(!ok){++fail;fprintf(stderr,"REGIME PARITY FAIL case %d\n",count+1);}
        ++count;
    }
    fclose(f);
    if(count!=32){++fail;fprintf(stderr,"Expected 32 cases, got %d\n",count);}
    printf("vMAR R-C23 regime parity: %d cases, %d failures\n",count,fail);
    return fail!=0;
}
