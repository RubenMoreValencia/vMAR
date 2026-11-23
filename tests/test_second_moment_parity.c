#include "../reference/second_moment_reference.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static vmar_status activation(double z,const double *p,size_t q,
    double *y,double *dz,double *dp,void *context) {
    int kind=*(const int*)context;
    if(kind==0) {
        if(q)return VMAR_ERR_ARGUMENT;
        *y=tanh(z);*dz=1-(*y)*(*y);return VMAR_OK;
    }
    if(!p||q!=1)return VMAR_ERR_ARGUMENT;
    if(kind==1) {
        *y=z>0?z:p[0]*z;*dz=z>0?1:p[0];
        if(dp)dp[0]=z<0?z:0;
        return VMAR_OK;
    }
    double s=1/(1+exp(-p[0]*z));
    *y=z*s;*dz=s+p[0]*z*s*(1-s);
    if(dp)dp[0]=z*z*s*(1-s);
    return VMAR_OK;
}
static int close_enough(double a,double b) {
    return isfinite(a)&&isfinite(b)&&fabs(a-b)<=1e-10*(1+fabs(b));
}
int main(int argc,char **argv) {
    if(argc!=2){fprintf(stderr,"Usage: %s reference.csv\n",argv[0]);return 2;}
    FILE *file=fopen(argv[1],"r");
    if(!file){perror(argv[1]);return 2;}
    char line[4096];if(!fgets(line,sizeof(line),file)){fclose(file);return 2;}
    const double x[]={1,0.5,-0.5,1.5,0.7,-1};
    const double target[]={0.25,-0.3,0.7};
    int ncases=0,failures=0;
    while(fgets(line,sizeof(line),file)) {
        char *col[24];int fields=0;
        for(char *part=strtok(line,",\r\n");part&&fields<24;
            part=strtok(NULL,",\r\n"))col[fields++]=part;
        ++ncases;
        if(fields!=23){++failures;fprintf(stderr,"Bad row %d: %d cols\n",ncases,fields);continue;}
        int kind=strcmp(col[0],"tanh")==0?0:
                 strcmp(col[0],"prelu")==0?1:2;
        vmar_integrator method=strcmp(col[1],"euler")==0?VMAR_EULER:VMAR_RK4;
        double perturb=strtod(col[2],NULL),alpha=kind==1?0.1:0.8,scratch[1];
        vmar_custom_neuron model={x,target,3,2,kind?&alpha:NULL,
             kind?1u:0u,kind?1:0,0.3,activation,&kind,
             kind?scratch:NULL,kind?1u:0u};
        vmar_second_system sys={&model,0.8,0.4,0.2,0.1,VMAR_SECOND_M2_CONTROL};
        size_t d=kind?4u:3u,dimension=3*d,nsteps=method==VMAR_EULER?40u:10u;
        double state[12]={0},next[12],workspace[60],h=1.0/(double)nsteps;
        state[0]=0.2+perturb;state[1]=-0.4;state[2]=0.15;
        if(kind)state[3]=alpha;
        double initial_loss=0,loss=0,gradient[4];
        vmar_status status=vmar_custom_neuron_loss_gradient(&model,state,d,
                                                 &initial_loss,gradient,NULL);
        for(size_t k=0;k<nsteps&&status==VMAR_OK;++k) {
            status=vmar_step(method,(double)k*h,h,vmar_second_field,
                &sys,state,dimension,workspace,
                method==VMAR_EULER?dimension:5*dimension,next);
            if(status==VMAR_OK)memcpy(state,next,dimension*sizeof(double));
        }
        if(status==VMAR_OK)status=vmar_custom_neuron_loss_gradient(
            &model,state,d,&loss,gradient,NULL);
        double mean_v=0;
        for(size_t j=0;j<d;++j)mean_v+=state[2*d+j]/(double)d;
        int ok=status==VMAR_OK&&strcmp(col[3],"m2c")==0&&
            d==(size_t)strtoul(col[4],NULL,10)&&
            close_enough(strtod(col[5],NULL),1)&&
            strtoul(col[6],NULL,10)==40u&&
            strtoul(col[7],NULL,10)==nsteps&&
            close_enough(initial_loss,strtod(col[8],NULL))&&
            close_enough(loss,strtod(col[9],NULL))&&
            close_enough(mean_v,strtod(col[10],NULL));
        for(int j=0;j<4;++j) {
            double theta=j<(int)d?state[j]:0;
            double memory=j<(int)d?state[d+j]:0;
            double second=j<(int)d?state[2*d+j]:0;
            ok=ok&&close_enough(theta,strtod(col[11+j],NULL))&&
                close_enough(memory,strtod(col[15+j],NULL))&&
                close_enough(second,strtod(col[19+j],NULL));
        }
        if(!ok){++failures;fprintf(stderr,"SECOND PARITY FAIL case %d\n",ncases);}
    }
    fclose(file);
    if(ncases!=12){++failures;fprintf(stderr,"Expected 12, got %d\n",ncases);}
    printf("vMAR R-C23 second moment parity: %d cases, %d failures\n",ncases,failures);
    return failures!=0;
}
