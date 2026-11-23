#include "vmar/memory.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int near(double a, double b) {
    return isfinite(a) && isfinite(b) &&
           fabs(a-b) <= 5e-11 + 5e-11*fabs(b);
}
static int verify(const char *line) {
    char id[96], activation[20], method[20];
    double coupling,eta,tau,step,expected[6];
    size_t steps;
    int n = sscanf(line,
        "%95[^,],%19[^,],%19[^,],%lf,%lf,%lf,%lf,%zu,%lf,%lf,%lf,%lf,%lf,%lf",
        id,activation,method,&coupling,&eta,&tau,&step,&steps,
        &expected[0],&expected[1],&expected[2],&expected[3],
        &expected[4],&expected[5]);
    if (n != 14 || steps == 0 || steps > 1000) return 1;
    vmar_activation a;
    if (!strcmp(activation,"tanh")) a=VMAR_ACT_TANH;
    else if (!strcmp(activation,"sigmoid")) a=VMAR_ACT_SIGMOID;
    else return 1;
    vmar_integrator integrator;
    if (!strcmp(method,"euler")) integrator=VMAR_EULER;
    else if (!strcmp(method,"rk4")) integrator=VMAR_RK4;
    else return 1;
    const double x[] = {1,0.5,-0.5,1.5,0.7,-1};
    const double target[] = {0.25,-0.3,0.7};
    vmar_learning_system sys = {{x,target,3,2,a,eta},tau,coupling};
    double s[] = {0.2,-0.4,0.15,0,0,0};
    double work[30];
    size_t required = integrator == VMAR_EULER ? 6 : 30;
    for (size_t k=0; k<steps; ++k) {
        vmar_status status = vmar_step(integrator,k*step,step,
                 vmar_learning_field,&sys,s,6,work,required,s);
        if (status != VMAR_OK) {
            fprintf(stderr,"MEMORY FIELD ERROR %s: %s\n",id,
                    vmar_status_string(status));
            return 1;
        }
    }
    int ok=1;
    for (size_t j=0;j<6;++j) ok &= near(s[j],expected[j]);
    printf("MEMORY PARITY %s %s\n",ok ? "PASS":"FAIL",id);
    return !ok;
}
int main(int argc,char **argv) {
    if (argc != 2) return 1;
    FILE *f=fopen(argv[1],"r");
    if (!f) {perror(argv[1]);return 1;}
    char line[2048];
    size_t count=0;int fails=0;
    if (!fgets(line,sizeof line,f)) {fclose(f);return 1;}
    while (fgets(line,sizeof line,f)) {
        if (line[0]=='\n'||line[0]=='\r') continue;
        ++count;fails+=verify(line);
    }
    if (ferror(f)) ++fails;
    fclose(f);
    if (count!=16) ++fails;
    printf("vMAR R-C23 memory parity: %zu cases, %d failures\n",count,fails);
    return fails!=0;
}
