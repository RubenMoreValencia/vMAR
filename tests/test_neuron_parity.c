#include "vmar/neuron.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int near(double a, double r) {
    return isfinite(a) && isfinite(r) &&
           fabs(a-r) <= 5e-11 + 5e-11 * fabs(r);
}

#ifndef VMAR_EXPECT_CASES
#define VMAR_EXPECT_CASES 16
#endif

static int verify(const char *line, size_t line_num) {
    char id[96], act[24], method[24];
    double eta, step, loss_ref, grad_ref[3], pred_ref[3], final_ref[3], final_loss;
    size_t steps;
    int read = sscanf(line,
        "%95[^,],%23[^,],%23[^,],%lf,%lf,%zu,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
        id, act, method, &eta, &step, &steps, &loss_ref,
        &grad_ref[0], &grad_ref[1], &grad_ref[2],
        &pred_ref[0], &pred_ref[1], &pred_ref[2],
        &final_ref[0], &final_ref[1], &final_ref[2], &final_loss);
    if (read != 17 || steps == 0 || steps > 1000) {
        fprintf(stderr, "Invalid neuron fixture line %zu\n", line_num);
        return 1;
    }
    vmar_activation activation;
    if (strcmp(act, "tanh") == 0) activation = VMAR_ACT_TANH;
    else if (strcmp(act, "sigmoid") == 0) activation = VMAR_ACT_SIGMOID;
    else if (strcmp(act, "identity") == 0) activation = VMAR_ACT_IDENTITY;
    else if (strcmp(act, "softplus") == 0) activation = VMAR_ACT_SOFTPLUS;
    else if (strcmp(act, "relu") == 0) activation = VMAR_ACT_RELU;
    else if (strcmp(act, "leaky_relu") == 0) activation = VMAR_ACT_LEAKY_RELU;
    else return 1;
    vmar_integrator integrator;
    if (strcmp(method, "euler") == 0) integrator = VMAR_EULER;
    else if (strcmp(method, "rk4") == 0) integrator = VMAR_RK4;
    else return 1;
    const double x[] = {1, 0.5, -0.5, 1.5, 0.7, -1};
    const double target[] = {0.25, -0.3, 0.7};
    double theta[] = {0.2, -0.4, 0.15};
    vmar_neuron_problem problem = {x, target, 3, 2, activation, eta};
    double loss, gradient[3], prediction[3];
    if (vmar_neuron_loss_gradient(&problem, theta, &loss, gradient,
                                  prediction) != VMAR_OK) return 1;
    int ok = near(loss, loss_ref);
    for (size_t j = 0; j < 3; ++j)
        ok &= near(gradient[j], grad_ref[j]) && near(prediction[j], pred_ref[j]);
    double workspace[15];
    size_t required = integrator == VMAR_EULER ? 3 : 15;
    for (size_t k = 0; k < steps; ++k) {
        vmar_status status = vmar_step(integrator, (double)k*step, step,
            vmar_neuron_field, &problem, theta, 3, workspace, required, theta);
        if (status != VMAR_OK) {
            fprintf(stderr, "Field integration failed in %s: %s\n",
                    id, vmar_status_string(status));
            return 1;
        }
    }
    if (vmar_neuron_loss_gradient(&problem, theta, &loss, gradient,
                                  NULL) != VMAR_OK) return 1;
    ok &= near(loss, final_loss);
    for (size_t j = 0; j < 3; ++j) ok &= near(theta[j], final_ref[j]);
    printf("NEURON PARITY %s %s\n", ok ? "PASS" : "FAIL", id);
    return !ok;
}

int main(int argc, char **argv) {
    if (argc != 2) return EXIT_FAILURE;
    FILE *file = fopen(argv[1], "r");
    if (!file) { perror(argv[1]); return EXIT_FAILURE; }
    char line[2048];
    size_t lines = 0, cases = 0;
    int failures = 0;
    while (fgets(line, sizeof(line), file)) {
        ++lines;
        if (lines == 1 || line[0] == '\n' || line[0] == '\r') continue;
        ++cases;
        failures += verify(line, lines);
    }
    if (ferror(file)) ++failures;
    fclose(file);
    if (cases != VMAR_EXPECT_CASES) ++failures;
    printf("vMAR R-C23 neuron parity: %zu cases, %d failures\n",
           cases, failures);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
