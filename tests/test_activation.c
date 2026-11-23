#include "vmar/neuron.h"

#include <math.h>
#include <stdio.h>

static int count,fail;
#define CHECK(c) do {++count;if(!(c)){++fail; \
  fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#c);}}while(0)
static int near(double a,double b,double t){return isfinite(a)&&fabs(a-b)<t;}

int main(void) {
    for(int a=VMAR_ACT_TANH;a<=VMAR_ACT_LEAKY_RELU;++a) {
        for(size_t j=0;j<4;++j) {
            const double points[]={-1.3,-0.2,0.25,1.7};
            double z=points[j],y,dy,plus,minus,tmp;
            CHECK(vmar_activation_evaluate((vmar_activation)a,z,&y,&dy)==VMAR_OK);
            CHECK(vmar_activation_evaluate((vmar_activation)a,z+1e-6,
                                            &plus,&tmp)==VMAR_OK);
            CHECK(vmar_activation_evaluate((vmar_activation)a,z-1e-6,
                                            &minus,&tmp)==VMAR_OK);
            CHECK(near(dy,(plus-minus)/(2e-6),1e-8));
        }
    }
    double value,deriv;
    CHECK(vmar_activation_evaluate(VMAR_ACT_SOFTPLUS,1000,&value,&deriv)==VMAR_OK);
    CHECK(value==1000 && deriv==1);
    CHECK(vmar_activation_evaluate(VMAR_ACT_SOFTPLUS,-1000,&value,&deriv)==VMAR_OK);
    CHECK(value==0 && deriv==0);
    CHECK(vmar_activation_evaluate(VMAR_ACT_RELU,0,&value,&deriv)==VMAR_OK);
    CHECK(value==0 && deriv==0);
    CHECK(vmar_activation_evaluate(VMAR_ACT_LEAKY_RELU,0,&value,&deriv)==VMAR_OK);
    CHECK(value==0 && deriv==0.01);
    CHECK(vmar_activation_evaluate((vmar_activation)99,1,&value,&deriv)==VMAR_ERR_ARGUMENT);
    CHECK(vmar_activation_evaluate(VMAR_ACT_TANH,INFINITY,&value,&deriv)==VMAR_ERR_NUMERIC);

    const double x[]={1,0.5,-0.5,1.5,0.7,-1};
    const double target[]={0.25,-0.3,0.7};
    const double theta[]={0.2,-0.4,0.15};
    for(int a=VMAR_ACT_IDENTITY;a<=VMAR_ACT_LEAKY_RELU;++a) {
        vmar_neuron_problem p={x,target,3,2,(vmar_activation)a,0.3};
        double loss,grad[3],pred[3];
        CHECK(vmar_neuron_loss_gradient(&p,theta,&loss,grad,pred)==VMAR_OK);
        CHECK(loss>=0 && isfinite(loss));
        for(size_t j=0;j<3;++j) CHECK(isfinite(grad[j]));
        double state[3]={0.2,-0.4,0.15},work[15];
        CHECK(vmar_step(VMAR_EULER,0,0.1,vmar_neuron_field,&p,
                        state,3,work,3,state)==VMAR_OK);
        for(size_t j=0;j<3;++j)
            CHECK(near(state[j],theta[j]-0.1*0.3*grad[j],1e-14));
    }
    printf("vMAR activation contracts: %d assertions, %d failures\n",count,fail);
    return fail!=0;
}
