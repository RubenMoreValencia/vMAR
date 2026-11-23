#ifndef VMAR_ADAM_REFERENCE_H
#define VMAR_ADAM_REFERENCE_H

#include "vmar/custom_activation.h"

/* Research comparator only: never linked into vmar_core. */
typedef struct vmar_adam_reference {
    double learning_rate, beta1, beta2, epsilon;
    size_t iteration, gradient_evaluations;
} vmar_adam_reference;

/* theta, first_moment, second_moment are mutable dimension-sized arrays.
 * No update is committed on an invalid gradient, overflow, or invalid config.
 */
vmar_status vmar_adam_reference_step(
    vmar_adam_reference *config, const double *gradient, size_t dimension,
    double *theta, double *first_moment, double *second_moment);

#endif
