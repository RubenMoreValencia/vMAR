#include "vmar/network_mechanism.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int near(double a,double b){return isfinite(a)&&isfinite(b)&&fabs(a-b)<=3e-10*(1+fabs(b));}

static int match_case(const char *method_name,const char *mode_name,
                      double memory_sign,double theta1,double memory1,
                      double loss0,double loss1,double gnorm,double mnorm,
                      double cnorm,double requested,double effective,
                      double after_time,int accepted,size_t attempts,
                      size_t rejected,size_t calls){
    size_t widths[]={2,3,2,1},offsets[3];
    vmar_activation acts[]={VMAR_ACT_TANH,VMAR_ACT_SIGMOID,VMAR_ACT_IDENTITY};
    vmar_network_layout layout;
    if(vmar_network_layout_make(widths,4,acts,offsets,3,&layout)!=VMAR_OK)return 0;
    double x[]={-0.3,0.7,0.2,-0.5,0.9,0.1},y[]={0.2,-0.1,0.4};
    double state[40]={0},nwork[128],work[260];
    for(size_t i=0;i<20;++i)state[i]=0.12*sin((double)(i+1));
    state[20]=memory_sign*0.5;state[21]=-0.1;
    size_t needed=0;
    if(vmar_network_workspace_required(&layout,&needed)!=VMAR_OK)return 0;
    vmar_network_problem neural={&layout,x,y,3,0.2,nwork,needed};
    vmar_network_system system={neural,0.7,0.2};
    vmar_control_policy policy={0.05,0.5,4,0,0.2};
    vmar_network_mode mode=VMAR_NETWORK_BASELINE;
    if(strcmp(mode_name,"M1")==0)mode=VMAR_NETWORK_OBSERVE;
    else if(strcmp(mode_name,"M2")==0)mode=VMAR_NETWORK_COUPLED;
    else if(strcmp(mode_name,"M3")==0)mode=VMAR_NETWORK_CONTROLLED;
    else if(strcmp(mode_name,"M0")!=0)return 0;
    vmar_integrator integrator;
    if(strcmp(method_name,"euler")==0)integrator=VMAR_EULER;
    else if(strcmp(method_name,"rk4")==0)integrator=VMAR_RK4;
    else return 0;
    vmar_network_mechanism mechanism={mode,system,policy};
    vmar_network_decision decision;
    double time=0;size_t id=0;
    if(vmar_network_mechanism_step(integrator,&mechanism,0.05,state,40,
                                   &time,&id,work,260,&decision)!=VMAR_OK)return 0;
    return near(state[0],theta1)&&near(state[20],memory1)&&
           near(decision.loss_before,loss0)&&near(decision.loss_after,loss1)&&
           near(decision.gradient_norm_before,gnorm)&&
           near(decision.memory_norm_before,mnorm)&&
           near(decision.coupling_term_norm_before,cnorm)&&
           near(decision.requested_step,requested)&&
           near(decision.effective_step,effective)&&near(time,after_time)&&
           decision.accepted==accepted&&decision.attempts==attempts&&
           decision.rejected==rejected&&decision.field_evaluations==calls;
}

int main(int argc,char **argv){
    if(argc!=2){fprintf(stderr,"usage: %s CSV\n",argv[0]);return 2;}
    FILE *f=fopen(argv[1],"r");if(!f){perror(argv[1]);return 2;}
    char line[2048];if(!fgets(line,sizeof(line),f)){fclose(f);return 2;}
    size_t rows=0,failures=0;
    while(fgets(line,sizeof(line),f)){
        char method[32]={0},mode[16]={0};
        double sign,theta,memory,loss0,loss1,g,m,c,request,effective,time;
        int accepted;size_t attempts,rejected,calls;
        int n=sscanf(line,
          "%31[^,],%15[^,],%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%d,%zu,%zu,%zu",
          method,mode,&sign,&theta,&memory,&loss0,&loss1,&g,&m,&c,
          &request,&effective,&time,&accepted,&attempts,&rejected,&calls);
        ++rows;
        if(n!=17||!match_case(method,mode,sign,theta,memory,loss0,loss1,g,m,c,
                               request,effective,time,accepted,attempts,
                               rejected,calls)){
            ++failures;fprintf(stderr,"MECHANISM PARITY FAIL row %zu\n",rows);
        }
    }
    fclose(f);
    if(rows!=16){++failures;fprintf(stderr,"expected 16 rows, got %zu\n",rows);}
    printf("vMAR R-C23 selected mechanism parity: %zu rows, %zu failures\n",
           rows,failures);
    return failures?EXIT_FAILURE:EXIT_SUCCESS;
}
