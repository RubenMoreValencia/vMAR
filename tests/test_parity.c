#include "vmar/vmar.h"

#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    double lambda;
    double equilibrium;
    double forcing;
    size_t calls;
} parity_context;

static vmar_status reference_field(double time, const double *state,
                                    size_t dimension, double *derivative,
                                    void *user) {
    parity_context *context = user;
    ++context->calls;
    for (size_t i = 0; i < dimension; ++i) {
        derivative[i] = -context->lambda * (state[i] - context->equilibrium)
                        + context->forcing * time;
    }
    return VMAR_OK;
}

static int close_to(double actual, double reference) {
    const double abs_tolerance = 5e-11;
    const double rel_tolerance = 5e-11;
    return isfinite(actual) && isfinite(reference) &&
           fabs(actual - reference) <= abs_tolerance +
           rel_tolerance * fabs(reference);
}

static int check_case(const char *line, size_t line_number) {
    char id[80], method[24], field[24];
    size_t dimension, n_steps;
    double t, h, lambda, equilibrium, forcing;
    double s1, s2, e1, e2;
    int parsed = sscanf(line,
        "%79[^,],%23[^,],%23[^,],%zu,%lf,%lf,%zu,%lf,%lf,%lf,%lf,%lf,%lf,%lf",
        id, method, field, &dimension, &t, &h, &n_steps,
        &lambda, &equilibrium, &forcing, &s1, &s2, &e1, &e2);
    if (parsed != 14 || (dimension != 1 && dimension != 2) ||
        n_steps == 0 || n_steps > 100000 || strcmp(field, "relaxation") != 0 ||
        !isfinite(lambda) || !isfinite(equilibrium) || !isfinite(forcing)) {
        fprintf(stderr, "Invalid fixture at line %zu\n", line_number);
        return 1;
    }
    vmar_integrator integrator;
    if (strcmp(method, "euler") == 0) integrator = VMAR_EULER;
    else if (strcmp(method, "rk4") == 0) integrator = VMAR_RK4;
    else {
        fprintf(stderr, "Unsupported integrator in %s\n", id);
        return 1;
    }
    double state[2] = { s1, s2 };
    double workspace[10];
    size_t required = 0;
    parity_context context = { lambda, equilibrium, forcing, 0 };
    if (vmar_workspace_required(integrator, dimension, &required) != VMAR_OK) {
        return 1;
    }
    for (size_t k = 0; k < n_steps; ++k) {
        vmar_status status = vmar_step(integrator, t + (double)k * h, h,
                                       reference_field, &context,
                                       state, dimension, workspace, required,
                                       state);
        if (status != VMAR_OK) {
            fprintf(stderr, "vMAR error in %s: %s\n", id, vmar_status_string(status));
            return 1;
        }
    }
    const size_t expected_calls = n_steps * (integrator == VMAR_EULER ? 1 : 4);
    if (context.calls != expected_calls || !close_to(state[0], e1) ||
        (dimension == 2 && !close_to(state[1], e2))) {
        fprintf(stderr,
                "PARITY FAIL %s: got %.17g / %.17g; expected %.17g / %.17g; calls %zu\n",
                id, state[0], state[1], e1, e2, context.calls);
        return 1;
    }
    printf("PARITY PASS %s (%s, dimension %zu, calls %zu)\n",
           id, method, dimension, context.calls);
    return 0;
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: vmar_test_parity <fixture.csv>\n");
        return EXIT_FAILURE;
    }
    FILE *input = fopen(argv[1], "r");
    if (input == NULL) {
        fprintf(stderr, "Cannot open fixture: %s (errno %d)\n", argv[1], errno);
        return EXIT_FAILURE;
    }
    char line[1024];
    size_t line_number = 0, cases = 0;
    int failures = 0;
    while (fgets(line, sizeof line, input) != NULL) {
        ++line_number;
        if (line_number == 1) continue; /* CSV header */
        if (line[0] == '\n' || line[0] == '\r') continue;
        ++cases;
        failures += check_case(line, line_number);
    }
    if (ferror(input)) ++failures;
    fclose(input);
    if (cases == 0) {
        fprintf(stderr, "No fixture cases were read\n");
        return EXIT_FAILURE;
    }
    printf("vMAR R-C23 parity: %zu cases, %d failures\n", cases, failures);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
