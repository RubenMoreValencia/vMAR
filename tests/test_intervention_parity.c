#include "vmar/regime.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static vmar_status prelu(double z,const double *p,size_t q,
    double *v,double *dz,double *dp,void *unused) {
    (void)unused;if(!p||q!=1)return VMAR_ERR_ARGUMENT;
    *v=z>0?z:p[0]*z;*dz=z>0?1:p[0];
    if(dp)dp[0]=z<0?z:0;
    return VMAR_OK;
}
static int close_enough(double x,double y) {
    return isfinite(x)&&isfinite(y)&&fabs(x-y)<=1e-10*(1+fabs(y));
}
int main(int argc,char **argv) {
    if(argc!=2){fprintf(stderr,"Usage: %s parity.csv\n",argv[0]);return 2;}
    FILE *file=fopen(argv[1],"r");
    if(!file){perror(argv[1]);return 2;}
    char line[4096];
    if(!fgets(line,sizeof(line),file)){fclose(file);return 2;}
    const double x[]={1,0.5,-0.5,1.5,0.7,-1};
    const double target[]={0.25,-0.3,0.7};
    int count=0,fail=0;
    while(fgets(line,sizeof(line),file)) {
        char *columns[16];int nf=0;
        for(char *tok=strtok(line,",\r\n");tok&&nf<16;
            tok=strtok(NULL,",\r\n")) columns[nf++]=tok;
        if(nf!=16){fprintf(stderr,"Malformed row %d\n",count+1);++fail;continue;}
        vmar_integrator method=strcmp(columns[0],"euler")==0?VMAR_EULER:VMAR_RK4;
        int policy=strcmp(columns[1],"observe")==0?0:
                   strcmp(columns[1],"fixed")==0?1:2;
        double perturb=strtod(columns[2],NULL),m0=strtod(columns[3],NULL);
        double alpha=0.1,scratch[1];
        vmar_custom_neuron neuron={x,target,3,2,&alpha,1,1,0.3,prelu,NULL,scratch,1};
        vmar_adaptive_system system={&neuron,0.8,0};
        double state[8]={0.2+perturb,-0.4,0.15,0.1,m0,m0,m0,m0};
        double workspace[40],candidate[8],sum_c=0,sum_contribution=0;
        size_t intervention_count=0,n=method==VMAR_EULER?20u:8u;
        double h=1.0/(double)n;vmar_status status=VMAR_OK;
        for(size_t k=0;k<n;++k) {
            vmar_regime_report diag;
            status=vmar_regime_diagnose(&system,state,8,0.2,1e-12,&diag);
            if(status!=VMAR_OK)break;
            system.coupling=policy==0?0:(policy==1?0.2:diag.coupling);
            if(system.coupling>1e-14)++intervention_count;
            sum_c+=system.coupling;
            if(diag.gradient_norm>1e-12)
                sum_contribution+=system.coupling*diag.memory_norm/diag.gradient_norm;
            status=vmar_step(method,(double)k*h,h,vmar_adaptive_field,
              &system,state,8,workspace,method==VMAR_EULER?8u:40u,candidate);
            if(status!=VMAR_OK)break;
            memcpy(state,candidate,sizeof(state));
        }
        double loss=0,g[4];
        if(status==VMAR_OK)status=vmar_custom_neuron_loss_gradient(
                                  &neuron,state,4,&loss,g,NULL);
        int ok=status==VMAR_OK && close_enough(loss,strtod(columns[4],NULL)) &&
          close_enough((double)intervention_count/(double)n,strtod(columns[5],NULL)) &&
          close_enough(sum_c/(double)n,strtod(columns[6],NULL)) &&
          close_enough(sum_contribution/(double)n,strtod(columns[7],NULL));
        for(int j=0;j<8;++j)
            ok=ok&&close_enough(state[j],strtod(columns[8+j],NULL));
        if(!ok){++fail;fprintf(stderr,"INTERVENTION PARITY FAIL row %d\n",count+1);}
        ++count;
    }
    fclose(file);
    if(count!=24){++fail;fprintf(stderr,"Expected 24 cases, got %d\n",count);}
    printf("vMAR R-C23 intervention parity: %d cases, %d failures\n",count,fail);
    return fail!=0;
}
