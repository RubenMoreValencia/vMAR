#include "vmar/vmar.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    double lambda;
    size_t calls;
    int produce_nan;
} decay_context;

static int failures = 0;
static int assertions = 0;

#define CHECK(condition, label) do { \
    ++assertions; \
    if (!(condition)) { \
        ++failures; \
        fprintf(stderr, "FAIL %s at line %d\n", label, __LINE__); \
    } \
} while (0)

static int close_to(double actual, double expected, double atol, double rtol) {
    return isfinite(actual) && fabs(actual - expected) <= atol + rtol * fabs(expected);
}

static vmar_status decay(double time, const double *state, size_t n,
                         double *derivative, void *user) {
    (void)time;
    decay_context *ctx = user;
    ++ctx->calls;
    for (size_t i = 0; i < n; ++i) {
        derivative[i] = ctx->produce_nan ? NAN : -ctx->lambda * state[i];
    }
    return VMAR_OK;
}

static vmar_status callback_fails(double time, const double *state, size_t n,
                                  double *derivative, void *user) {
    (void)time; (void)state; (void)n; (void)derivative; (void)user;
    return VMAR_ERR_ARGUMENT;
}

static void test_workspace_contracts(void) {
    size_t required = 0;
    CHECK(vmar_workspace_required(VMAR_EULER, 3, &required) == VMAR_OK && required == 3,
          "Euler workspace dimension");
    CHECK(vmar_workspace_required(VMAR_RK4, 3, &required) == VMAR_OK && required == 15,
          "RK4 workspace dimension");
    CHECK(vmar_workspace_required(VMAR_RK4, 0, &required) == VMAR_ERR_ARGUMENT,
          "zero dimension rejected");
    CHECK(vmar_workspace_required((vmar_integrator) 99, 3, &required) == VMAR_ERR_ARGUMENT,
          "unknown method rejected");
}

static void test_analytic_scalar(void) {
    /* Independent analytic reference: y'=-y, y(0)=1, y(1)=exp(-1). */
    decay_context ctx = { .lambda = 1, .calls = 0, .produce_nan = 0 };
    double state = 1.0;
    double workspace[5];
    for (int i = 0; i < 10; ++i) {
        CHECK(vmar_step(VMAR_RK4, i * 0.1, 0.1, decay, &ctx,
                        &state, 1, workspace, 5, &state) == VMAR_OK,
              "RK4 in-place scalar update");
    }
    CHECK(close_to(state, exp(-1.0), 4.0e-7, 0), "RK4 agrees with exact solution");
    CHECK(close_to(state, 0.36787977441250, 5.0e-13, 0),
          "RK4 agrees with R G0.5-D.1 fixture");
    CHECK(ctx.calls == 40, "RK4 uses exactly four field calls per step");

    state = 1.0;
    ctx.calls = 0;
    for (int i = 0; i < 10; ++i) {
        CHECK(vmar_step(VMAR_EULER, i * 0.1, 0.1, decay, &ctx,
                        &state, 1, workspace, 1, &state) == VMAR_OK,
              "Euler in-place scalar update");
    }
    CHECK(close_to(state, 0.34867844010000, 1e-13, 0),
          "Euler agrees with R G0.5-D.1 fixture");
    CHECK(ctx.calls == 10, "Euler uses one field call per step");
}

static void test_vector_field(void) {
    decay_context ctx = { .lambda = 2, .calls = 0, .produce_nan = 0 };
    double state[] = { 1.0, -2.0 };
    double result[] = { -99.0, -99.0 };
    double workspace[10];
    CHECK(vmar_step(VMAR_RK4, 0, 0.1, decay, &ctx, state, 2,
                    workspace, 10, result) == VMAR_OK,
          "vector field update accepted");
    CHECK(close_to(result[0], exp(-0.2), 6e-6, 0),
          "vector first component analytic reference");
    CHECK(close_to(result[1], -2.0 * exp(-0.2), 12e-6, 0),
          "vector second component analytic reference");
    CHECK(ctx.calls == 4, "vector RK4 four field evaluations");
}

static void test_stability_polynomials(void) {
    decay_context ctx = { .lambda = 1, .calls = 0, .produce_nan = 0 };
    double workspace[5];
    double initial = 1.0;
    double output = 0;
    CHECK(vmar_step(VMAR_EULER, 0, 2.2, decay, &ctx, &initial,
                    1, workspace, 1, &output) == VMAR_OK,
          "Euler large-step numerical update");
    CHECK(close_to(output, -1.2, 1e-13, 0), "Euler unstable amplification");
    CHECK(vmar_step(VMAR_RK4, 0, 2.2, decay, &ctx, &initial,
                    1, workspace, 5, &output) == VMAR_OK,
          "RK4 large-step numerical update");
    CHECK(fabs(output) < 1, "RK4 stable at h=2.2 for scalar decay");
    CHECK(vmar_step(VMAR_RK4, 0, 3, decay, &ctx, &initial,
                    1, workspace, 5, &output) == VMAR_OK,
          "RK4 very-large-step numerical update");
    CHECK(fabs(output) > 1, "RK4 unstable at h=3 for scalar decay");
}

static void test_transactional_failure(void) {
    decay_context ctx = { .lambda = 1, .calls = 0, .produce_nan = 1 };
    double state = 1.0;
    double output = 123.0;
    double workspace[5];
    CHECK(vmar_step(VMAR_RK4, 0, 0.1, decay, &ctx,
                    &state, 1, workspace, 5, &output) == VMAR_ERR_NUMERIC,
          "NaN callback rejected");
    CHECK(output == 123.0 && state == 1.0, "failure leaves caller buffers unchanged");
    CHECK(vmar_step(VMAR_RK4, 0, 0.1, callback_fails, NULL,
                    &state, 1, workspace, 5, &output) == VMAR_ERR_FIELD,
          "field failure surfaced");
    CHECK(output == 123.0, "field failure leaves output unchanged");
    ctx.produce_nan = 0;
    CHECK(vmar_step(VMAR_RK4, 0, 0.1, decay, &ctx,
                    &state, 1, workspace, 4, &output) == VMAR_ERR_WORKSPACE,
          "undersized workspace rejected");
    CHECK(output == 123.0, "workspace failure leaves output unchanged");
    CHECK(vmar_step(VMAR_EULER, 0, -0.1, decay, &ctx,
                    &state, 1, workspace, 5, &output) == VMAR_ERR_ARGUMENT,
          "nonpositive step rejected");
    CHECK(vmar_step(VMAR_RK4, 0, 0.1, decay, &ctx,
                    &state, 0, workspace, 5, &output) == VMAR_ERR_ARGUMENT,
          "invalid dimension rejected");
}

int main(void) {
#if !defined(__STDC_VERSION__) || __STDC_VERSION__ < 202000L
    fprintf(stderr, "A C23-capable language mode is required\n");
    return EXIT_FAILURE;
#elif __STDC_VERSION__ < 202311L
    /* GCC 14 advertises a draft-mode 202000L in -std=c23. This local
       test does not certify final C23 conformance. Apple Clang 17 must
       separately report 202311L for the vMAR environment acceptance gate. */
    fprintf(stderr, "NOTE: compiler advertises draft C23 (%ld)\n",
            (long)__STDC_VERSION__);
#endif
    test_workspace_contracts();
    test_analytic_scalar();
    test_vector_field();
    test_stability_polynomials();
    test_transactional_failure();
    printf("vMAR-Core C23: %d assertions, %d failures\n", assertions, failures);
    return failures ? EXIT_FAILURE : EXIT_SUCCESS;
}
