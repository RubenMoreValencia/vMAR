#ifndef VMAR_CUSTOM_ACTIVATION_H
#define VMAR_CUSTOM_ACTIVATION_H

#include "vmar/vmar.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * User-defined activation phi(z; alpha). Implementations fill value and dz.
 * If dparams is non-NULL, fill all nparams entries dphi/dalpha_j.
 * Callback must not mutate z/params or other caller-owned input buffers.
 */
typedef vmar_status (*vmar_activation_rule)(
    double z, const double *params, size_t nparams,
    double *value, double *dz, double *dparams, void *user);

/*
 * An independent, ABI-separated neuron model. The dataset is row-major.
 * State theta = [weights (features), bias, optionally trainable params].
 * With trainable=0, params points to fixed values and theta has p+1 entries.
 * With trainable=1, params contains the initial values for reference only:
 * actual params are read from theta[p+1...]. The declared parameter count
 * defines their dimension. `scratch` of >= nparams doubles is mandatory
 * for trainable parameters; no hidden heap allocation is performed.
 */
typedef struct vmar_custom_neuron {
    const double *x;
    const double *target;
    size_t samples;
    size_t features;
    const double *params;
    size_t nparams;
    int trainable;
    double eta;
    vmar_activation_rule activation;
    void *user;
    double *scratch;
    size_t scratch_doubles;
} vmar_custom_neuron;

vmar_status vmar_custom_neuron_dimension(const vmar_custom_neuron *model,
                                         size_t *dimension);

vmar_status vmar_custom_neuron_loss_gradient(
    const vmar_custom_neuron *model, const double *theta,
    size_t dimension, double *loss, double *gradient, double *prediction);

/* Suitable for the existing vmar_step() Euler/RK4 integrators. */
vmar_status vmar_custom_neuron_field(double time,const double *theta,
                                     size_t dimension,double *derivative,
                                     void *context);

#ifdef __cplusplus
}
#endif

#endif
