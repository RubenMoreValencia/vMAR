#include "vmar/curvature.h"
#include "vmar/network.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

typedef struct neural_loss_context {
    const vmar_network_layout *layout;
    const double *x;
    const double *target;
    double *network_workspace;
    size_t network_workspace_size;
    size_t callback_calls;
} neural_loss_context;

static vmar_status neural_loss(const double *theta, size_t dimension,
                               void *opaque, double *loss) {
    neural_loss_context *c = opaque;
    double gradient[9];
    if (dimension != 9) return VMAR_ERR_DIMENSION;
    ++c->callback_calls;
    return vmar_network_loss_gradient(c->layout, c->x, c->target, 6,
                                      theta, c->network_workspace,
                                      c->network_workspace_size,
                                      loss, gradient, NULL);
}

static int compute_states(double states[2][3][9],
                          neural_loss_context *c) {
    const double theta0[9] = {
        .3, -.2, -.4, .5, .05, -.1, .6, -.7, .03
    };
    vmar_network_problem problem = {c->layout, c->x, c->target, 6,
                                    1., c->network_workspace,
                                    c->network_workspace_size};
    double work[45], next[9];
    for (size_t k = 0; k < 2; ++k) {
        double current[9];
        memcpy(current, theta0, sizeof(current));
        memcpy(states[k][0], current, sizeof(current));
        const vmar_integrator method = k ? VMAR_RK4 : VMAR_EULER;
        for (size_t i = 0; i < 20; ++i) {
            if (vmar_step(method, (double)i * .05, .05,
                          vmar_network_field, &problem,
                          current, 9, work, 45, next) != VMAR_OK)
                return 0;
            memcpy(current, next, sizeof(current));
            if (i == 9) memcpy(states[k][1], current, sizeof(current));
            if (i == 19) memcpy(states[k][2], current, sizeof(current));
        }
    }
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: vmar_test_curvature_parity reference.csv\n");
        return 2;
    }
    FILE *fp = fopen(argv[1], "r");
    if (!fp) {
        perror(argv[1]);
        return 2;
    }
    size_t widths[] = {2, 2, 1}, offsets[2];
    const vmar_activation activations[] = {VMAR_ACT_TANH, VMAR_ACT_TANH};
    vmar_network_layout layout;
    if (vmar_network_layout_make(widths, 3, activations, offsets, 2,
                                 &layout) != VMAR_OK || layout.parameters != 9) {
        fclose(fp);
        return 2;
    }
    size_t required = 0;
    if (vmar_network_workspace_required(&layout, &required) != VMAR_OK ||
        required > 128) {
        fclose(fp);
        return 2;
    }
    double network_workspace[128];
    const double x[] = {-1., -.5, -.6, .2, -.2, .8,
                          .3, -.3, .7, .6, 1., -.8};
    const double target[] = {-.6, -.3, .1, .2, .5, .65};
    neural_loss_context context = {&layout, x, target,
                                   network_workspace, required, 0};
    double states[2][3][9];
    if (!compute_states(states, &context)) {
        fclose(fp);
        return 2;
    }

    char line[256];
    if (!fgets(line, sizeof(line), fp)) {
        fclose(fp);
        return 2;
    }
    unsigned rows = 0, cases = 0, failures = 0;
    int prior_method = -1, prior_step = -1;
    double prior_h = -1;
    double H[81], curvature_workspace[90];
    while (fgets(line, sizeof(line), fp)) {
        char method_name[16];
        int step, row, col;
        double h, expected;
        if (sscanf(line, "%15[^,],%d,%lf,%d,%d,%lf",
                   method_name, &step, &h, &row, &col, &expected) != 6) {
            ++failures;
            break;
        }
        const int method = !strcmp(method_name, "euler") ? 0 :
                           !strcmp(method_name, "rk4") ? 1 : -1;
        if (method < 0 || (step != 0 && step != 10 && step != 20) ||
            (h != 1e-3 && h != 1e-4) || row < 0 || row >= 9 ||
            col < 0 || col >= 9 || !isfinite(expected)) {
            ++failures;
            break;
        }
        if (method != prior_method || step != prior_step || h != prior_h) {
            prior_method = method;
            prior_step = step;
            prior_h = h;
            const vmar_curvature_options opt = {h, 16, 180};
            vmar_curvature_report report;
            const size_t before = context.callback_calls;
            const vmar_status s = vmar_curvature_hessian(
                states[method][step / 10], 9, neural_loss, &context, &opt,
                curvature_workspace, 90, H, &report);
            if (s != VMAR_OK || report.loss_calls != 180 ||
                context.callback_calls - before != 180) {
                ++failures;
                break;
            }
            ++cases;
        }
        const double actual = H[(size_t)row * 9u + (size_t)col];
        if (!isfinite(actual) || fabs(actual - expected) > 1e-5) {
            if (failures < 8)
                fprintf(stderr, "Mismatch %s step=%d h=%.5g [%d,%d]: %.17g vs %.17g\n",
                        method_name, step, h, row, col, actual, expected);
            ++failures;
        }
        ++rows;
    }
    fclose(fp);
    if (rows != 972 || cases != 12) {
        fprintf(stderr, "Unexpected fixture coverage: %u rows, %u cases\n",
                rows, cases);
        ++failures;
    }
    printf("vMAR R-C23 C01 Hessian parity: %u coordinate rows, %u cases, %u failures\n",
           rows, cases, failures);
    return failures ? 1 : 0;
}
