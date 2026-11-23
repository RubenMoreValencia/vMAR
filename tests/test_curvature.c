#include "vmar/curvature.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static unsigned checks = 0, failures = 0;
#define CHECK(expr) do { ++checks; if (!(expr)) { ++failures; \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #expr); } } while (0)

typedef struct quadratic_ctx {
    double a[4];
    size_t calls;
    size_t fail_after;
    int nonfinite;
} quadratic_ctx;

static vmar_status quadratic(const double *theta, size_t p, void *opaque,
                             double *out_loss) {
    quadratic_ctx *ctx = opaque;
    ++ctx->calls;
    if (ctx->fail_after && ctx->calls >= ctx->fail_after)
        return VMAR_ERR_FIELD;
    if (ctx->nonfinite) {
        *out_loss = NAN;
        return VMAR_OK;
    }
    if (p != 2) return VMAR_ERR_DIMENSION;
    *out_loss = .5 * (ctx->a[0] * theta[0] * theta[0] +
        (ctx->a[1] + ctx->a[2]) * theta[0] * theta[1] +
        ctx->a[3] * theta[1] * theta[1]);
    return VMAR_OK;
}

static void quadratic_cases(void) {
    const double matrices[][4] = {
        {2., 0., 0., 5.},
        {-2., 0., 0., -5.},
        {-2., 0., 0., 5.},
        {2., 0., 0., 0.},
        {2., .75, .75, 5.}
    };
    double theta[2] = {.3, -.7};
    const double original[2] = {.3, -.7};
    double workspace[6], H[4];
    vmar_curvature_report report;
    const vmar_curvature_options options = {1e-3, 16, 12};
    size_t required = 0;
    CHECK(vmar_curvature_workspace_required(2, &required) == VMAR_OK);
    CHECK(required == 6);
    CHECK(vmar_curvature_workspace_required(0, &required) == VMAR_ERR_DIMENSION);
    CHECK(vmar_curvature_workspace_required(17, &required) == VMAR_ERR_DIMENSION);
    for (size_t c = 0; c < sizeof(matrices) / sizeof(matrices[0]); ++c) {
        quadratic_ctx ctx = {{0}, 0, 0, 0};
        memcpy(ctx.a, matrices[c], sizeof(ctx.a));
        memset(H, 0, sizeof(H));
        CHECK(vmar_curvature_hessian(theta, 2, quadratic, &ctx, &options,
                                    workspace, 6, H, &report) == VMAR_OK);
        for (size_t i = 0; i < 4; ++i)
            CHECK(fabs(H[i] - matrices[c][i]) < 1e-7);
        CHECK(report.loss_calls == 12);
        CHECK(ctx.calls == 12);
        CHECK(report.parameters == 2);
        CHECK(report.step == options.step);
        CHECK(report.symmetry_error == 0);
        CHECK(!memcmp(theta, original, sizeof(theta)));
    }
}

static void negative_contracts(void) {
    double theta[2] = {.3, -.7}, workspace[6], H[4] = {7, 7, 7, 7};
    double old_h[4];
    vmar_curvature_report report = {123, 456, 789, 321}, old_report = report;
    vmar_curvature_options opts = {1e-3, 16, 12};
    quadratic_ctx ctx = {{2, 0, 0, 5}, 0, 0, 0};
#define UNCHANGED() do { CHECK(!memcmp(old_h, H, sizeof(H))); \
    CHECK(!memcmp(&report, &old_report, sizeof(report))); } while (0)
    memcpy(old_h, H, sizeof(H));
    CHECK(vmar_curvature_hessian(NULL, 2, quadratic, &ctx, &opts,
                                workspace, 6, H, &report) == VMAR_ERR_ARGUMENT);
    UNCHANGED();
    CHECK(vmar_curvature_hessian(theta, 2, NULL, &ctx, &opts,
                                workspace, 6, H, &report) == VMAR_ERR_ARGUMENT);
    UNCHANGED();
    CHECK(vmar_curvature_hessian(theta, 2, quadratic, &ctx, &opts,
                                workspace, 5, H, &report) == VMAR_ERR_WORKSPACE);
    UNCHANGED();
    CHECK(vmar_curvature_hessian(theta, 17, quadratic, &ctx, &opts,
                                workspace, 6, H, &report) == VMAR_ERR_DIMENSION);
    UNCHANGED();
    opts.max_loss_calls = 11;
    CHECK(vmar_curvature_hessian(theta, 2, quadratic, &ctx, &opts,
                                workspace, 6, H, &report) == VMAR_ERR_ARGUMENT);
    CHECK(ctx.calls == 0);
    UNCHANGED();
    opts.max_loss_calls = 12;
    opts.step = 0;
    CHECK(vmar_curvature_hessian(theta, 2, quadratic, &ctx, &opts,
                                workspace, 6, H, &report) == VMAR_ERR_ARGUMENT);
    UNCHANGED();
    opts.step = 1e-300;
    CHECK(vmar_curvature_hessian(theta, 2, quadratic, &ctx, &opts,
                                workspace, 6, H, &report) == VMAR_ERR_NUMERIC);
    UNCHANGED();
    opts.step = 1e-3;
    theta[0] = NAN;
    CHECK(vmar_curvature_hessian(theta, 2, quadratic, &ctx, &opts,
                                workspace, 6, H, &report) == VMAR_ERR_NUMERIC);
    UNCHANGED();
    theta[0] = .3;
    theta[0] = 1e20;
    CHECK(vmar_curvature_hessian(theta, 2, quadratic, &ctx, &opts,
                                workspace, 6, H, &report) == VMAR_ERR_NUMERIC);
    UNCHANGED();
    theta[0] = .3;
    CHECK(vmar_curvature_hessian(theta, 2, quadratic, &ctx, &opts,
                                workspace, 6, workspace, &report) == VMAR_ERR_ARGUMENT);
    UNCHANGED();
    ctx.fail_after = 3;
    CHECK(vmar_curvature_hessian(theta, 2, quadratic, &ctx, &opts,
                                workspace, 6, H, &report) == VMAR_ERR_FIELD);
    UNCHANGED();
    ctx.fail_after = 0;
    ctx.calls = 0;
    ctx.nonfinite = 1;
    CHECK(vmar_curvature_hessian(theta, 2, quadratic, &ctx, &opts,
                                workspace, 6, H, &report) == VMAR_ERR_NUMERIC);
    UNCHANGED();
    CHECK(vmar_curvature_hessian(theta, 2, quadratic, &ctx, &opts,
                                workspace, 6, H, (vmar_curvature_report *)H) ==
          VMAR_ERR_ARGUMENT);
    UNCHANGED();
}

int main(void) {
    quadratic_cases();
    negative_contracts();
    printf("vMAR G0.8-C.1 curvature contracts: %u assertions, %u failures\n",
           checks, failures);
    return failures ? 1 : 0;
}
