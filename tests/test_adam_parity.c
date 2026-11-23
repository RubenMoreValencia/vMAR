#include "../reference/adam_reference.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static vmar_status activation(double z,const double *p,size_t q,
    double *y,double *dz,double *dp,void *context) {
    int kind=*(const int*)context;
    if(kind==0) {
        if(q) return VMAR_ERR_ARGUMENT;
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
static int near(double a,double b) {
    return isfinite(a)&&isfinite(b)&&fabs(a-b)<=1e-10*(1+fabs(b));
}
int main(int argc,char **argv) {
    if(argc!=2){fprintf(stderr,"Usage: %s fixtures.csv\n",argv[0]);return 2;}
    FILE *file=fopen(argv[1],"r");
    if(!file){perror(argv[1]);return 2;}
    char line[4096];if(!fgets(line,sizeof(line),file)){fclose(file);return 2;}
    int count=0,failures=0;
    const double x[]={1,0.5,-0.5,1.5,0.7,-1};
    const double target[]={0.25,-0.3,0.7};
    while(fgets(line,sizeof(line),file)) {
        char *col[23];int columns=0;
        for(char *part=strtok(line,",\r\n");part&&columns<23;
            part=strtok(NULL,",\r\n")) col[columns++]=part;
        ++count;
        if(columns!=23){++failures;fprintf(stderr,"Bad row %d\n",count);continue;}
        int kind=strcmp(col[0],"tanh")==0?0:
                 strcmp(col[0],"prelu")==0?1:2;
        double perturb=strtod(col[1],NULL),alpha=kind==1?0.1:0.8,scratch[1];
        vmar_adam_reference cfg={strtod(col[2],NULL),strtod(col[3],NULL),
            strtod(col[4],NULL),strtod(col[5],NULL),0,0};
        vmar_custom_neuron model={x,target,3,2,kind?&alpha:NULL,
            kind?1u:0u,kind?1:0,0.3,activation,&kind,
            kind?scratch:NULL,kind?1u:0u};
        size_t dimension=kind?4u:3u;
        double theta[4]={0.2+perturb,-0.4,0.15,kind?alpha:0},
               m[4]={0},v[4]={0},g[4]={0},loss=0,initial_loss=0;
        vmar_status status=vmar_custom_neuron_loss_gradient(
            &model,theta,dimension,&initial_loss,g,NULL);
        for(int k=0;k<40&&status==VMAR_OK;++k) {
            status=vmar_custom_neuron_loss_gradient(
                &model,theta,dimension,&loss,g,NULL);
            if(status==VMAR_OK)status=vmar_adam_reference_step(
                &cfg,g,dimension,theta,m,v);
        }
        if(status==VMAR_OK)status=vmar_custom_neuron_loss_gradient(
            &model,theta,dimension,&loss,g,NULL);
        int ok=status==VMAR_OK && dimension==(size_t)strtoul(col[6],NULL,10)&&
            cfg.gradient_evaluations==(size_t)strtoul(col[7],NULL,10)&&
            cfg.iteration==(size_t)strtoul(col[8],NULL,10)&&
            near(initial_loss,strtod(col[9],NULL))&&near(loss,strtod(col[10],NULL));
        for(int j=0;j<4;++j) {
            ok=ok&&near(theta[j],strtod(col[11+j],NULL))&&
                near(m[j],strtod(col[15+j],NULL))&&
                near(v[j],strtod(col[19+j],NULL));
        }
        if(!ok){++failures;fprintf(stderr,"ADAM PARITY FAIL row %d\n",count);}
    }
    fclose(file);
    if(count!=24){++failures;fprintf(stderr,"Expected 24; got %d\n",count);}
    printf("vMAR R-C23 Adam reference parity: %d cases, %d failures\n",count,failures);
    return failures!=0;
}
