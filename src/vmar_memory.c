#include "vmar/memory.h"

#include <math.h>
#include <stdint.h>

static vmar_status size_check(size_t dimension) {
    if (!dimension) return VMAR_ERR_ARGUMENT;
    if (dimension > SIZE_MAX / sizeof(double)) return VMAR_ERR_DIMENSION;
    return VMAR_OK;
}

static int all_finite(const double *x, size_t n) {
    for (size_t i = 0; i < n; ++i) if (!isfinite(x[i])) return 0;
    return 1;
}

vmar_status vmar_memory_ema(const double *memory, const double *observation,
                            size_t dimension, double decay, double *next) {
    if (!memory || !observation || !next || !isfinite(decay) ||
        decay < 0 || decay >= 1) return VMAR_ERR_ARGUMENT;
    vmar_status s = size_check(dimension);
    if (s != VMAR_OK) return s;
    if (!all_finite(memory, dimension) || !all_finite(observation, dimension))
        return VMAR_ERR_NUMERIC;
    /* Preflight ensures next is unchanged on an arithmetic error. */
    for (size_t i = 0; i < dimension; ++i)
        if (!isfinite(decay*memory[i] + (1-decay)*observation[i]))
            return VMAR_ERR_NUMERIC;
    for (size_t i = 0; i < dimension; ++i)
        next[i] = decay*memory[i] + (1-decay)*observation[i];
    return VMAR_OK;
}

vmar_status vmar_memory_relaxation(const double *memory, const double *signal,
                                   size_t dimension, double step, double tau,
                                   double *next) {
    if (!memory || !signal || !next || !isfinite(step) || step <= 0 ||
        !isfinite(tau) || tau <= 0) return VMAR_ERR_ARGUMENT;
    vmar_status s = size_check(dimension);
    if (s != VMAR_OK) return s;
    if (!all_finite(memory, dimension) || !all_finite(signal, dimension))
        return VMAR_ERR_NUMERIC;
    double gain = -expm1(-step / tau);
    for (size_t i = 0; i < dimension; ++i)
        if (!isfinite(memory[i] + gain*(signal[i]-memory[i])))
            return VMAR_ERR_NUMERIC;
    for (size_t i = 0; i < dimension; ++i)
        next[i] = memory[i] + gain*(signal[i]-memory[i]);
    return VMAR_OK;
}

vmar_status vmar_learning_field(double time, const double *state,
                                size_t dimension, double *derivative,
                                void *context) {
    if (!context || !state || !derivative || !isfinite(time))
        return VMAR_ERR_ARGUMENT;
    const vmar_learning_system *system = context;
    size_t p = system->neuron.features;
    if (p == SIZE_MAX || p > (SIZE_MAX/2)-1 || dimension != 2*(p+1))
        return VMAR_ERR_DIMENSION;
    if (!isfinite(system->tau) || system->tau <= 0 ||
        !isfinite(system->coupling) || system->coupling < 0)
        return VMAR_ERR_ARGUMENT;
    const size_t d = p+1;
    if (!all_finite(state+d, d)) return VMAR_ERR_NUMERIC;
    double loss = 0;
    vmar_status s = vmar_neuron_loss_gradient(&system->neuron, state,
                                              &loss, derivative, NULL);
    if (s != VMAR_OK) return s;
    for (size_t j = 0; j < d; ++j) {
        double g = derivative[j];
        derivative[d+j] = (g - state[d+j]) / system->tau;
        derivative[j] = -system->neuron.eta *
                        (g + system->coupling * state[d+j]);
        if (!isfinite(derivative[j]) || !isfinite(derivative[d+j]))
            return VMAR_ERR_NUMERIC;
    }
    return VMAR_OK;
}
