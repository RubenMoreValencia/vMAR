#include "vmar/memory.h"

#include <math.h>
#include <stdio.h>

static int total, failures;
#define CHECK(c) do { ++total; if (!(c)) { ++failures; \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #c); } } while (0)
static int near(double a, double b, double tol) {
    return isfinite(a) && fabs(a-b) <= tol;
}

int main(void) {
    double zero[2] = {0,0}, obs[2] = {2,4}, m[2] = {0,0};
    CHECK(vmar_memory_ema(zero, obs, 2, 0.5, m) == VMAR_OK);
    CHECK(near(m[0],1,1e-15)); CHECK(near(m[1],2,1e-15));
    CHECK(vmar_memory_ema(m, obs, 2, 0.5, m) == VMAR_OK);
    CHECK(near(m[0],1.5,1e-15)); CHECK(near(m[1],3,1e-15));
    double signal[2] = {4,8}, result[2];
    CHECK(vmar_memory_relaxation(zero, signal, 2, 0.2, 0.1, result) == VMAR_OK);
    CHECK(near(result[0],4*(1-exp(-2)),1e-14));
    CHECK(near(result[1],8*(1-exp(-2)),1e-14));
    CHECK(vmar_memory_relaxation(zero, signal, 2, 0.2, 0.1, zero) == VMAR_OK);
    CHECK(near(zero[0],result[0],1e-14));
    double unchanged[] = {7,9};
    CHECK(vmar_memory_ema(m, obs, 2, 1, unchanged) == VMAR_ERR_ARGUMENT);
    CHECK(unchanged[0] == 7 && unchanged[1] == 9);
    CHECK(vmar_memory_relaxation(m, obs, 2, -1, 1, unchanged) == VMAR_ERR_ARGUMENT);
    CHECK(unchanged[0] == 7 && unchanged[1] == 9);

    const double x[] = {1, 0.5, -0.5, 1.5, 0.7, -1};
    const double target[] = {0.25, -0.3, 0.7};
    vmar_learning_system system = {
        {x, target, 3, 2, VMAR_ACT_TANH, 0.3}, 0.6, 0.0
    };
    double augmented[] = {0.2, -0.4, 0.15, 0, 0, 0};
    double field[6], neural[3];
    CHECK(vmar_learning_field(0, augmented, 6, field, &system) == VMAR_OK);
    CHECK(vmar_neuron_field(0, augmented, 3, neural, &system.neuron) == VMAR_OK);
    for (size_t j=0; j<3; ++j) {
        CHECK(near(field[j],neural[j],1e-14));
        CHECK(near(field[3+j],-neural[j]/(0.3*0.6),1e-13));
    }
    double scratch_aug[30], scratch_neural[15];
    for (int method=VMAR_EULER; method<=VMAR_RK4; ++method) {
        double a[6] = {0.2,-0.4,0.15,0,0,0};
        double b[3] = {0.2,-0.4,0.15};
        size_t n1 = method == VMAR_EULER ? 6 : 30;
        size_t n2 = method == VMAR_EULER ? 3 : 15;
        CHECK(vmar_step((vmar_integrator)method, 0, 0.1,
              vmar_learning_field, &system, a, 6, scratch_aug, n1, a) == VMAR_OK);
        CHECK(vmar_step((vmar_integrator)method, 0, 0.1,
              vmar_neuron_field, &system.neuron, b, 3,
              scratch_neural, n2, b) == VMAR_OK);
        for (size_t j=0; j<3; ++j) CHECK(near(a[j],b[j],1e-14));
    }
    system.coupling = 0.2;
    augmented[3] = 1;
    CHECK(vmar_learning_field(0, augmented, 6, field, &system) == VMAR_OK);
    CHECK(near(field[0],neural[0]-0.3*0.2,1e-14));
    system.tau = 0;
    CHECK(vmar_learning_field(0, augmented, 6, field, &system) == VMAR_ERR_ARGUMENT);
    CHECK(vmar_learning_field(0, augmented, 5, field, &system) == VMAR_ERR_DIMENSION);
    printf("vMAR memory C23: %d assertions, %d failures\n", total, failures);
    return failures != 0;
}
