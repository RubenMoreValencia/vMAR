#ifndef VMAR_SECOND_MOMENT_REFERENCE_H
#define VMAR_SECOND_MOMENT_REFERENCE_H
#include "vmar/custom_activation.h"

/* Experimental only: no additions to vmar_core public ABI. */
typedef enum vmar_second_mode {
    VMAR_SECOND_M1=0, VMAR_SECOND_M2_OBSERVE=1,
    VMAR_SECOND_M2_CONTROL=2
} vmar_second_mode;

typedef struct vmar_second_system {
    const vmar_custom_neuron *model;
    double tau1, tau2, coupling, scale;
    vmar_second_mode mode;
} vmar_second_system;

/* State [theta(d), m(d)] for M1, [theta(d),m(d),v(d)] otherwise.
 * Control mode uses c_eff = coupling/(1+sqrt(mean(v))/scale).
 * v must be nonnegative; stage evaluations outside the domain are rejected.
 */
vmar_status vmar_second_field(double time,const double *state,
    size_t dimension,double *derivative,void *context);
#endif
