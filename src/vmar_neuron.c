#include "vmar/neuron.h"

#include <math.h>
#include <stdint.h>

static int finite_vector(const double *data, size_t length) {
    for (size_t i = 0; i < length; ++i) {
        if (!isfinite(data[i])) return 0;
    }
    return 1;
}

static int activation_known(vmar_activation a) {
    return a == VMAR_ACT_TANH || a == VMAR_ACT_SIGMOID ||
           a == VMAR_ACT_IDENTITY || a == VMAR_ACT_SOFTPLUS ||
           a == VMAR_ACT_RELU || a == VMAR_ACT_LEAKY_RELU;
}

static vmar_status validate(const vmar_neuron_problem *p,
                             const double *theta) {
    if (!p || !theta || !p->x || !p->target || p->samples == 0 ||
        p->features == 0 || !isfinite(p->eta) || p->eta <= 0 ||
        !activation_known(p->activation)) {
        return VMAR_ERR_ARGUMENT;
    }
    if (p->features == SIZE_MAX || p->features > SIZE_MAX / p->samples ||
        p->features * p->samples > SIZE_MAX / sizeof(double) ||
        p->features + 1 > SIZE_MAX / sizeof(double) ||
        p->samples > SIZE_MAX / sizeof(double)) return VMAR_ERR_DIMENSION;
    if (!finite_vector(theta, p->features + 1) ||
        !finite_vector(p->x, p->samples * p->features) ||
        !finite_vector(p->target, p->samples)) return VMAR_ERR_NUMERIC;
    return VMAR_OK;
}

static double sigmoid(double z) {
    if (z >= 0) return 1.0 / (1.0 + exp(-z));
    double e = exp(z);
    return e / (1.0 + e);
}

vmar_status vmar_activation_evaluate(vmar_activation kind, double z,
                                     double *value, double *derivative) {
    if (!value || !derivative || !activation_known(kind)) return VMAR_ERR_ARGUMENT;
    if (!isfinite(z)) return VMAR_ERR_NUMERIC;
    double y, dy;
    switch (kind) {
        case VMAR_ACT_TANH:
            y=tanh(z);dy=1-y*y;break;
        case VMAR_ACT_SIGMOID:
            y=sigmoid(z);dy=y*(1-y);break;
        case VMAR_ACT_IDENTITY:
            y=z;dy=1;break;
        case VMAR_ACT_SOFTPLUS:
            y=fmax(z,0)+log1p(exp(-fabs(z)));
            dy=sigmoid(z);break;
        case VMAR_ACT_RELU:
            y=fmax(z,0);dy=z>0?1:0;break;
        case VMAR_ACT_LEAKY_RELU:
            y=z>0?z:0.01*z;dy=z>0?1:0.01;break;
        default: return VMAR_ERR_ARGUMENT;
    }
    if (!isfinite(y) || !isfinite(dy)) return VMAR_ERR_NUMERIC;
    *value=y;*derivative=dy;
    return VMAR_OK;
}

vmar_status vmar_neuron_loss_gradient(const vmar_neuron_problem *problem,
                                      const double *theta, double *loss,
                                      double *gradient, double *prediction) {
    if (!loss || !gradient) return VMAR_ERR_ARGUMENT;
    vmar_status status = validate(problem, theta);
    if (status != VMAR_OK) return status;
    const size_t n = problem->samples, p = problem->features;
    for (size_t j = 0; j <= p; ++j) gradient[j] = 0;
    double total = 0;
    for (size_t i = 0; i < n; ++i) {
        double z = theta[p];
        for (size_t j = 0; j < p; ++j) z += problem->x[i * p + j] * theta[j];
        if (!isfinite(z)) return VMAR_ERR_NUMERIC;
        double y, derivative;
        status=vmar_activation_evaluate(problem->activation,z,&y,&derivative);
        if (status != VMAR_OK) return status;
        double residual = y - problem->target[i];
        double delta = (2.0 / (double)n) * residual * derivative;
        total += residual * residual / (double)n;
        for (size_t j = 0; j < p; ++j)
            gradient[j] += problem->x[i * p + j] * delta;
        gradient[p] += delta;
        if (prediction) prediction[i] = y;
    }
    if (!isfinite(total) || !finite_vector(gradient, p+1))
        return VMAR_ERR_NUMERIC;
    *loss = total;
    return VMAR_OK;
}

vmar_status vmar_neuron_field(double time, const double *theta,
                              size_t dimension, double *derivative,
                              void *context) {
    if (!isfinite(time) || !context || !derivative) return VMAR_ERR_ARGUMENT;
    const vmar_neuron_problem *p = context;
    if (p->features == SIZE_MAX || dimension != p->features + 1)
        return VMAR_ERR_DIMENSION;
    double loss = 0;
    vmar_status status = vmar_neuron_loss_gradient(p, theta, &loss,
                                                    derivative, NULL);
    if (status != VMAR_OK) return status;
    for (size_t j = 0; j < dimension; ++j) {
        derivative[j] *= -p->eta;
        if (!isfinite(derivative[j])) return VMAR_ERR_NUMERIC;
    }
    return VMAR_OK;
}
