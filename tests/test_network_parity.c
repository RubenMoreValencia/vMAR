#include "vmar/network.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int close_value(double actual, double expected) {
    return isfinite(actual) && isfinite(expected) &&
           fabs(actual - expected) <= 2e-10 * (1.0 + fabs(expected));
}

static int check_row(const char *name, const char *method, size_t steps,
                     double h, size_t coordinate, double initial_expected,
                     double final_expected, double gradient_expected,
                     double loss_initial_expected, double loss_final_expected) {
    const size_t legacy_widths[] = {2, 1, 1};
    const vmar_activation legacy_activations[] = {VMAR_ACT_IDENTITY,
                                                   VMAR_ACT_IDENTITY};
    const double legacy_x[] = {1, 2};
    const double legacy_y[] = {16};
    const double legacy_theta[] = {2, 3, 1, 2, -1};
    const size_t deep_widths[] = {2, 3, 2, 1};
    const vmar_activation deep_activations[] = {VMAR_ACT_TANH,
                                                VMAR_ACT_SIGMOID,
                                                VMAR_ACT_IDENTITY};
    const double deep_x[] = {-0.3, 0.7, 0.2, -0.5, 0.9, 0.1};
    const double deep_y[] = {0.2, -0.1, 0.4};

    int legacy = strcmp(name, "legacy") == 0;
    if (!legacy && strcmp(name, "deep") != 0) return 0;
    const size_t *widths = legacy ? legacy_widths : deep_widths;
    const vmar_activation *activations = legacy ? legacy_activations : deep_activations;
    const double *x = legacy ? legacy_x : deep_x;
    const double *target = legacy ? legacy_y : deep_y;
    size_t sample_count = legacy ? 1 : 3;
    size_t p = legacy ? 5 : 20;
    size_t offsets[3] = {0};
    vmar_network_layout layout;
    if (vmar_network_layout_make(widths, legacy ? 3 : 4, activations,
                                 offsets, 3, &layout) != VMAR_OK) return 0;

    double theta[20] = {0}, gradient[20], scratch[128], integrator_scratch[128];
    for (size_t i = 0; i < p; ++i) {
        theta[i] = legacy ? legacy_theta[i] : 0.12 * sin((double)(i + 1));
    }
    size_t workspace_count = 0;
    if (vmar_network_workspace_required(&layout, &workspace_count) != VMAR_OK ||
        workspace_count > 128 || coordinate == 0 || coordinate > p ||
        steps != 4 || !close_value(h, 0.05)) return 0;
    vmar_network_problem problem = {&layout, x, target, sample_count, 0.2,
                                    scratch, workspace_count};

    double loss_initial = 0, loss_final = 0;
    if (vmar_network_loss_gradient(&layout, x, target, sample_count, theta,
                                  scratch, workspace_count, &loss_initial,
                                  gradient, NULL) != VMAR_OK) return 0;
    size_t index = coordinate - 1;
    if (!close_value(theta[index], initial_expected) ||
        !close_value(gradient[index], gradient_expected) ||
        !close_value(loss_initial, loss_initial_expected)) return 0;

    vmar_integrator integrator;
    if (strcmp(method, "euler") == 0) integrator = VMAR_EULER;
    else if (strcmp(method, "rk4") == 0) integrator = VMAR_RK4;
    else return 0;
    for (size_t i = 0; i < steps; ++i) {
        if (vmar_step(integrator, (double)i * h, h, vmar_network_field,
                      &problem, theta, p, integrator_scratch, 128,
                      theta) != VMAR_OK) return 0;
    }
    if (vmar_network_loss_gradient(&layout, x, target, sample_count, theta,
                                  scratch, workspace_count, &loss_final,
                                  gradient, NULL) != VMAR_OK) return 0;
    return close_value(theta[index], final_expected) &&
           close_value(loss_final, loss_final_expected);
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: %s reference/vmar_multilayer_parity.csv\n", argv[0]);
        return 2;
    }
    FILE *file = fopen(argv[1], "r");
    if (!file) { perror(argv[1]); return 2; }
    char line[1024];
    size_t tested = 0, failed = 0;
    if (!fgets(line, sizeof(line), file)) { fclose(file); return 2; }
    while (fgets(line, sizeof(line), file)) {
        char name[64], method[16];
        size_t steps, coordinate;
        double h, initial, final, gradient, initial_loss, final_loss;
        int fields = sscanf(line,
                            "%63[^,],%15[^,],%zu,%lf,%zu,%lf,%lf,%lf,%lf,%lf",
                            name, method, &steps, &h, &coordinate,
                            &initial, &final, &gradient, &initial_loss, &final_loss);
        ++tested;
        if (fields != 10 || !check_row(name, method, steps, h, coordinate,
                                        initial, final, gradient, initial_loss,
                                        final_loss)) {
            ++failed;
            fprintf(stderr, "MULTILAYER PARITY FAIL row %zu\n", tested);
        }
    }
    fclose(file);
    if (tested != 50) { ++failed; fprintf(stderr, "expected 50 rows, got %zu\n", tested); }
    printf("vMAR R-C23 multilayer parity: %zu coordinate rows, %zu failures\n",
           tested, failed);
    return failed ? EXIT_FAILURE : EXIT_SUCCESS;
}
