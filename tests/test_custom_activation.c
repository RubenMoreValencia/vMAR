#include "vmar/custom_activation.h"

#include <math.h>
#include <stdio.h>

static int count,failures;
#define CHECK(c) do {++count;if(!(c)){++failures; \
    fprintf(stderr,"FAIL %d: %s\n",__LINE__,#c);}}while(0)
static int near(double a,double b,double tol) {
    return isfinite(a)&&fabs(a-b)<=tol;
}

static vmar_status prelu(double z,const double *p,size_t q,
                         double *y,double *dz,double *dp,void *user) {
    (void)user;
    if(q!=1||!p) return VMAR_ERR_ARGUMENT;
    *y=z>0?z:p[0]*z;
    *dz=z>0?1:p[0];
    if(dp) dp[0]=z<0?z:0;
    return VMAR_OK;
}
static vmar_status swish(double z,const double *p,size_t q,
                         double *y,double *dz,double *dp,void *user) {
    (void)user;
    if(q!=1||!p) return VMAR_ERR_ARGUMENT;
    double s=1/(1+exp(-p[0]*z));
    *y=z*s;
    *dz=s+p[0]*z*s*(1-s);
    if(dp) dp[0]=z*z*s*(1-s);
    return VMAR_OK;
}
static vmar_status sine(double z,const double *p,size_t q,
                        double *y,double *dz,double *dp,void *user) {
    (void)p;(void)dp;(void)user;
    if(q!=0) return VMAR_ERR_ARGUMENT;
    *y=sin(z);*dz=cos(z);return VMAR_OK;
}

int main(void) {
    const double x[]={1,0.5,-0.5,1.5,0.7,-1};
    const double target[]={0.25,-0.3,0.7};
    for(int model=0;model<5;++model) {
        double parameter=model<2?0.1:0.8;
        const int trainable=model==1||model==3;
        const size_t q=model==4?0:1;
        double scratch[1];
        vmar_custom_neuron m={
            x,target,3,2,q?&parameter:NULL,q,trainable,0.3,
            model<2?prelu:(model<4?swish:sine),NULL,
            trainable?scratch:NULL,trainable?1u:0u};
        double theta[]={0.2,-0.4,0.15,parameter};
        size_t d=0;
        CHECK(vmar_custom_neuron_dimension(&m,&d)==VMAR_OK);
        CHECK(d==(size_t)(trainable?4:3));
        double loss,gradient[4],prediction[3];
        CHECK(vmar_custom_neuron_loss_gradient(&m,theta,d,&loss,
                                                gradient,prediction)==VMAR_OK);
        CHECK(isfinite(loss) && loss>=0);
        for(size_t j=0;j<d;++j) {
            double plus[4]={theta[0],theta[1],theta[2],theta[3]};
            double minus[4]={theta[0],theta[1],theta[2],theta[3]};
            double temp_grad[4],lplus,lminus;
            plus[j]+=1e-6;minus[j]-=1e-6;
            CHECK(vmar_custom_neuron_loss_gradient(&m,plus,d,&lplus,
                                                    temp_grad,NULL)==VMAR_OK);
            CHECK(vmar_custom_neuron_loss_gradient(&m,minus,d,&lminus,
                                                    temp_grad,NULL)==VMAR_OK);
            CHECK(near(gradient[j],(lplus-lminus)/2e-6,1e-8));
        }
        double field[4];
        CHECK(vmar_custom_neuron_field(0,theta,d,field,&m)==VMAR_OK);
        for(size_t j=0;j<d;++j)
            CHECK(near(field[j],-0.3*gradient[j],1e-14));
        double w[20],next[4];
        CHECK(vmar_step(VMAR_EULER,0,0.1,vmar_custom_neuron_field,&m,
                        theta,d,w,d,next)==VMAR_OK);
        CHECK(vmar_step(VMAR_RK4,0,0.1,vmar_custom_neuron_field,&m,
                        theta,d,w,5*d,next)==VMAR_OK);
    }
    printf("vMAR custom activation: %d assertions, %d failures\n",count,failures);
    return failures!=0;
}
