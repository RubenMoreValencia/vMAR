#include "vmar/custom_activation.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static vmar_status prelu(double z,const double *p,size_t q,
                         double *y,double *dz,double *dp,void *user) {
    (void)user;
    if(q!=1||!p) return VMAR_ERR_ARGUMENT;
    *y=z>0?z:p[0]*z;*dz=z>0?1:p[0];
    if(dp) dp[0]=z<0?z:0;
    return VMAR_OK;
}
static vmar_status swish(double z,const double *p,size_t q,
                         double *y,double *dz,double *dp,void *user) {
    (void)user;
    if(q!=1||!p) return VMAR_ERR_ARGUMENT;
    double s=1/(1+exp(-p[0]*z));
    *y=z*s;*dz=s+p[0]*z*s*(1-s);
    if(dp) dp[0]=z*z*s*(1-s);
    return VMAR_OK;
}
static vmar_status sine(double z,const double *p,size_t q,
                        double *y,double *dz,double *dp,void *user) {
    (void)p;(void)dp;(void)user;
    if(q!=0) return VMAR_ERR_ARGUMENT;
    *y=sin(z);*dz=cos(z);return VMAR_OK;
}
static int near(double a,double b) {
    return isfinite(a)&&isfinite(b)&&fabs(a-b)<=5e-11+5e-11*fabs(b);
}
static int verify(const char *line) {
    char id[128],model_name[24],method_name[24];
    int trainable;
    size_t steps;
    double eta,step,parameter0,old_loss,grad_ref[4],pred_ref[3],
           final_ref[4],final_loss;
    int parsed=sscanf(line,
       "%127[^,],%23[^,],%d,%23[^,],%lf,%lf,%zu,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
       id,model_name,&trainable,method_name,&eta,&step,&steps,&parameter0,
       &old_loss,&grad_ref[0],&grad_ref[1],&grad_ref[2],&grad_ref[3],
       &pred_ref[0],&pred_ref[1],&pred_ref[2],
       &final_ref[0],&final_ref[1],&final_ref[2],&final_ref[3],&final_loss);
    if(parsed!=21 || steps==0 || steps>1000 || (trainable!=0&&trainable!=1))
        return 1;
    vmar_activation_rule callback;
    size_t q;
    if(!strcmp(model_name,"prelu")) {callback=prelu;q=1;}
    else if(!strcmp(model_name,"swish")) {callback=swish;q=1;}
    else if(!strcmp(model_name,"sine")) {callback=sine;q=0;}
    else return 1;
    vmar_integrator integrator;
    if(!strcmp(method_name,"euler")) integrator=VMAR_EULER;
    else if(!strcmp(method_name,"rk4")) integrator=VMAR_RK4;
    else return 1;
    const double x[]={1,0.5,-0.5,1.5,0.7,-1};
    const double target[]={0.25,-0.3,0.7};
    double scratch[1];
    vmar_custom_neuron m={x,target,3,2,q?&parameter0:NULL,q,trainable,
        eta,callback,NULL,trainable?scratch:NULL,trainable?1u:0u};
    size_t d=3+(trainable?1u:0u);
    double theta[]={0.2,-0.4,0.15,parameter0},grad[4],pred[3],loss=0;
    if(vmar_custom_neuron_loss_gradient(&m,theta,d,&loss,grad,pred)!=VMAR_OK)
        return 1;
    int ok=near(loss,old_loss);
    for(size_t j=0;j<d;++j) ok &= near(grad[j],grad_ref[j]);
    if(!trainable) ok &= near(grad_ref[3],0);
    for(size_t j=0;j<3;++j) ok &= near(pred[j],pred_ref[j]);
    double workspace[20];size_t required=integrator==VMAR_EULER?d:5*d;
    for(size_t k=0;k<steps;++k) {
        if(vmar_step(integrator,(double)k*step,step,vmar_custom_neuron_field,
                     &m,theta,d,workspace,required,theta)!=VMAR_OK) return 1;
    }
    if(vmar_custom_neuron_loss_gradient(&m,theta,d,&loss,grad,NULL)!=VMAR_OK)
        return 1;
    ok &= near(loss,final_loss);
    for(size_t j=0;j<d;++j) ok &= near(theta[j],final_ref[j]);
    if(!trainable) ok &= near(final_ref[3],0);
    printf("CUSTOM PARITY %s %s\n",ok?"PASS":"FAIL",id);
    return !ok;
}
int main(int argc,char **argv) {
    if(argc!=2) return 1;
    FILE *input=fopen(argv[1],"r");
    if(!input) {perror(argv[1]);return 1;}
    char line[4096];size_t cases=0;int failures=0;
    if(!fgets(line,sizeof line,input)) {fclose(input);return 1;}
    while(fgets(line,sizeof line,input)) {
        if(line[0]=='\n'||line[0]=='\r') continue;
        ++cases;failures+=verify(line);
    }
    if(ferror(input)) ++failures;
    fclose(input);
    if(cases!=40) ++failures;
    printf("vMAR R-C23 custom activation parity: %zu cases, %d failures\n",
           cases,failures);
    return failures!=0;
}
