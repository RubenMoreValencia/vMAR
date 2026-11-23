#ifndef VMAR_CURVATURE_H
#define VMAR_CURVATURE_H

#include "vmar/vmar.h"

#ifdef __cplusplus
extern "C" {
#endif

/* G0.8-C.1 EXPERIMENTAL, opt-in, non-interventional diagnostic.
 * Not part of the installed stable API. The caller owns all buffers.
 * A loss callback must not mutate external learning state or parameters.
 */
typedef vmar_status (*vmar_loss_fn)(const double *theta, size_t parameters,
                                   void *context, double *out_loss);

typedef struct vmar_curvature_options {
    double step;              /* positive finite perturbation */
    size_t max_parameters;    /* explicit limit, always capped at 16 */
    size_t max_loss_calls;    /* at least 2*p*(p+1) */
} vmar_curvature_options;

typedef struct vmar_curvature_report {
    size_t parameters;
    size_t loss_calls;
    double step;
    double symmetry_error; /* exactly zero after mirrored assignment */
} vmar_curvature_report;

/* Workspace contains p*p temporary Hessian cells followed by p coordinates.
 * All buffers, including report, must be disjoint. No allocations occur.
 * On ANY failure hessian and report are unchanged; callback call counts on
 * failure must be recorded by a caller-owned observer if needed.
 */
vmar_status vmar_curvature_workspace_required(size_t parameters,
                                             size_t *required_doubles);

vmar_status vmar_curvature_hessian(const double *theta, size_t parameters,
                                  vmar_loss_fn loss, void *context,
                                  const vmar_curvature_options *options,
                                  double *workspace, size_t workspace_doubles,
                                  double *hessian_row_major,
                                  vmar_curvature_report *report);

#ifdef __cplusplus
}
#endif
#endif
