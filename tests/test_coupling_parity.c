#include "vmar/adaptive_control.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static vmar_status prelu(double z,const double *p,size_t q,
    double *v,double *dz,double *dp,void *context) {
    (void)context;
    if(q!=1 || !p) return VMAR_ERR_ARGUMENT;
    *v=z>0?z:p[0]*z; *dz=z>0?1:p[0];
    if(dp) dp[0]=z<0?z:0;
    return VMAR_OK;
}
static int close_enough(double a,double b) {
    return isfinite(a)&&isfinite(b)&&fabs(a-b)<=1e-10*(1+fabs(b));
}
int main(int argc,char **argv) {
    if(argc!=2) {fprintf(stderr,"Usage: %s fixture.csv\n",argv[0]);return 2;}
    FILE *fp=fopen(argv[1],"r");
    if(!fp) {perror(argv[1]);return 2;}
    char line[2048];
    if(!fgets(line,sizeof(line),fp)){fclose(fp);return 2;}
    const double x[]={1,0.5,-0.5,1.5,0.7,-1};
    const double target[]={0.25,-0.3,0.7};
    int n=0,fail=0;
    while(fgets(line,sizeof(line),fp)) {
        char *cols[15];int count=0;
        for(char *item=strtok(line,",\r\n");item && count<15;
            item=strtok(NULL,",\r\n")) cols[count++]=item;
        if(count!=15){fprintf(stderr,"Bad fixture row %d\n",n+1);++fail;continue;}
        vmar_integrator method=strcmp(cols[0],"euler")==0?VMAR_EULER:VMAR_RK4;
        double coupling=strtod(cols[1],NULL),perturb=strtod(cols[2],NULL);
        double mem=strtod(cols[3],NULL),h=strtod(cols[4],NULL);
        size_t steps=(size_t)strtoul(cols[5],NULL,10);
        double alpha=0.1,scratch[1];
        vmar_custom_neuron neuron={x,target,3,2,&alpha,1,1,0.3,
                                   prelu,NULL,scratch,1};
        vmar_adaptive_system system={&neuron,0.8,coupling};
        double state[8]={0.2+perturb,-0.4,0.15,0.1,mem,mem,mem,mem};
        double workspace[40],next[8];
        vmar_status status=VMAR_OK;
        for(size_t k=0;k<steps;++k) {
            status=vmar_step(method,(double)k*h,h,vmar_adaptive_field,
                 &system,state,8,workspace,method==VMAR_EULER?8u:40u,next);
            if(status!=VMAR_OK) break;
            memcpy(state,next,sizeof(state));
        }
        double loss=0,g[4];
        if(status==VMAR_OK) status=vmar_custom_neuron_loss_gradient(
                                  &neuron,state,4,&loss,g,NULL);
        int ok=status==VMAR_OK && close_enough(loss,strtod(cols[6],NULL));
        for(int j=0;j<8;++j)
            ok=ok && close_enough(state[j],strtod(cols[7+j],NULL));
        if(!ok) {++fail;fprintf(stderr,"COUPLING PARITY FAIL case %d\n",n+1);}
        ++n;
    }
    fclose(fp);
    if(n!=48) {++fail;fprintf(stderr,"Expected 48 cases; read %d\n",n);}
    printf("vMAR R-C23 coupling parity: %d cases, %d failures\n",n,fail);
    return fail!=0;
}
