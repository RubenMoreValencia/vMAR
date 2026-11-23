#ifndef VMAR_VMAR_H
#define VMAR_VMAR_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define VMAR_API_VERSION_MAJOR 0
#define VMAR_API_VERSION_MINOR 1
#define VMAR_API_VERSION_PATCH 16

typedef enum vmar_status {
    VMAR_OK = 0,
    VMAR_ERR_ARGUMENT = 1,
    VMAR_ERR_DIMENSION = 2,
    VMAR_ERR_WORKSPACE = 3,
    VMAR_ERR_NUMERIC = 4,
    VMAR_ERR_FIELD = 5
} vmar_status;

typedef enum vmar_integrator {
    VMAR_EULER = 1,
    VMAR_RK4 = 2
} vmar_integrator;

/*
 * Evaluate d(state)/dt at time. The callback must fill every element of
 * derivative and return VMAR_OK; non-finite outputs are rejected by the core.
 * state and derivative refer to disjoint buffers during evaluation.
 */
typedef vmar_status (*vmar_field_fn)(double time, const double *state,
                                    size_t dimension, double *derivative,
                                    void *context);

/* Returns 1*n doubles for Euler and 5*n for RK4. */
vmar_status vmar_workspace_required(vmar_integrator method, size_t dimension,
                                    size_t *required_doubles);

/*
 * Propose one explicit integration step: field -> method -> next_state.
 * - time, step and all state values must be finite; step must be positive.
 * - workspace must have at least vmar_workspace_required() doubles.
 * - workspace must not overlap state or next_state.
 * - next_state may alias state exactly (in-place transition).
 * - if any validation, callback or numeric error occurs, next_state is not
 *   modified by the core (field callbacks must not alter caller buffers).
 * - all arithmetic is IEEE double where supported; bitwise identity across
 *   compilers is not guaranteed.
 *
 * The function is an integrator, not a learning algorithm or a controller.
 */
vmar_status vmar_step(vmar_integrator method, double time, double step,
                      vmar_field_fn field, void *context,
                      const double *state, size_t dimension,
                      double *workspace, size_t workspace_doubles,
                      double *next_state);

const char *vmar_status_string(vmar_status status);

#ifdef __cplusplus
}
#endif

#endif
