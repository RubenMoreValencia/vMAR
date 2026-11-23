#include "vmar/network.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>

static int checks = 0, failures = 0;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; \
  fprintf(stderr, "FAIL %d: %s\n", __LINE__, #x); } } while (0)

static int close_enough(double a, double b, double tol) {
    return isfinite(a) && isfinite(b) &&
           fabs(a-b) <= tol * fmax(1.0, fmax(fabs(a),fabs(b)));
}

int main(void) {
    size_t w[] = {2,1,1}, off[2] = {99,99};
    vmar_activation acts[] = {VMAR_ACT_IDENTITY, VMAR_ACT_IDENTITY};
    vmar_network_layout layout = {0};
    CHECK(vmar_network_layout_make(w,3,acts,off,2,&layout)==VMAR_OK);
    CHECK(layout.layer_count==2 && layout.parameters==5);
    CHECK(off[0]==0 && off[1]==3);

    size_t request=0;
    CHECK(vmar_network_workspace_required(&layout,&request)==VMAR_OK);
    double work[100]={0};
    CHECK(request<=100);
    /* Two affine identity layers: forward(1,2)=17. */
    double theta[] = {2,3,1,2,-1};
    /* The 2->1->1 descriptor has (2+1)+(1+1)=5 parameters. */
    CHECK(layout.parameters==5);
    double input[]={1,2}, prediction=0, gradient[6]={0}, loss=0;
    CHECK(vmar_network_forward(&layout,input,1,theta,work,request,
                               &prediction)==VMAR_OK);
    CHECK(close_enough(prediction,17,1e-13));
    double target[]={16};
    CHECK(vmar_network_loss_gradient(&layout,input,target,1,theta,work,request,
                                     &loss,gradient,NULL)==VMAR_OK);
    CHECK(close_enough(loss,1,1e-13));
    double expected[]={4,8,4,18,2};
    for(size_t i=0;i<5;++i)CHECK(close_enough(gradient[i],expected[i],1e-12));

    size_t widths[]={2,3,2,1}, offsets[3]={0};
    vmar_activation activation[]={VMAR_ACT_TANH,VMAR_ACT_SIGMOID,
                                   VMAR_ACT_IDENTITY};
    CHECK(vmar_network_layout_make(widths,4,activation,offsets,3,&layout)==VMAR_OK);
    CHECK(layout.parameters==20 && layout.layer_count==3);
    CHECK(offsets[0]==0 && offsets[1]==9 && offsets[2]==17);
    CHECK(vmar_network_workspace_required(&layout,&request)==VMAR_OK);
    CHECK(request<=100);
    double x[]={-0.3,0.7,0.2,-0.5,0.9,0.1};
    double y[]={0.2,-0.1,0.4};
    double param[20]={0}, g[20]={0};
    for(size_t i=0;i<20;++i)param[i]=0.12*sin((double)(i+1));
    CHECK(vmar_network_loss_gradient(&layout,x,y,3,param,work,request,
                                     &loss,g,NULL)==VMAR_OK);
    CHECK(isfinite(loss) && loss>=0);
    for(size_t i=0;i<20;++i){
        double up_loss=0,down_loss=0, h=1e-6;
        param[i]+=h;
        CHECK(vmar_network_loss_gradient(&layout,x,y,3,param,work,request,
                                         &up_loss,g,NULL)==VMAR_OK);
        param[i]-=2*h;
        CHECK(vmar_network_loss_gradient(&layout,x,y,3,param,work,request,
                                         &down_loss,g,NULL)==VMAR_OK);
        param[i]+=h;
        /* Reset gradient after each finite difference because calls above
         * intentionally reuse the same output buffer. */
        double analytical[20]={0}, baseline=0;
        CHECK(vmar_network_loss_gradient(&layout,x,y,3,param,work,request,
                                         &baseline,analytical,NULL)==VMAR_OK);
        CHECK(close_enough(analytical[i],(up_loss-down_loss)/(2*h),1e-6));
    }

    vmar_network_problem problem={&layout,x,y,3,0.2,work,request};
    double field[20]={0};
    CHECK(vmar_network_field(0,param,20,field,&problem)==VMAR_OK);
    double numerical[20]={0}, baseline_gradient[20]={0};
    CHECK(vmar_network_loss_gradient(&layout,x,y,3,param,work,request,
                                     &loss,baseline_gradient,NULL)==VMAR_OK);
    for(size_t i=0;i<20;++i)
        CHECK(close_enough(field[i],-0.2*baseline_gradient[i],1e-12));
    double integrate_work[100]={0};
    CHECK(vmar_step(VMAR_EULER,0,0.05,vmar_network_field,&problem,
                    param,20,integrate_work,20,numerical)==VMAR_OK);
    for(size_t i=0;i<20;++i)CHECK(isfinite(numerical[i]));
    CHECK(vmar_step(VMAR_RK4,0,0.05,vmar_network_field,&problem,
                    param,20,integrate_work,100,numerical)==VMAR_OK);
    for(size_t i=0;i<20;++i)CHECK(isfinite(numerical[i]));

    size_t badw[]={2,0,1};
    size_t guard[]={456,789};
    vmar_network_layout invalid={0};
    CHECK(vmar_network_layout_make(badw,3,acts,guard,2,&invalid)==VMAR_ERR_DIMENSION);
    CHECK(guard[0]==456 && guard[1]==789);
    CHECK(vmar_network_layout_make(widths,4,activation,offsets,2,&invalid)==VMAR_ERR_WORKSPACE);
    CHECK(vmar_network_forward(&layout,x,3,param,work,1,numerical)==VMAR_ERR_WORKSPACE);
    param[0]=NAN;
    CHECK(vmar_network_forward(&layout,x,3,param,work,request,numerical)==VMAR_ERR_NUMERIC);

    printf("vMAR C23 multilayer contracts: %d assertions, %d failures\n",checks,failures);
    return failures!=0;
}
