#include "vmar/neuron.h"

#include <math.h>
#include <stdio.h>

static int count = 0, failures = 0;
#define CHECK(test) do { ++count; if (!(test)) { ++failures; \
    fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #test); } } while (0)

static int near(double a, double b, double tolerance) {
    return isfinite(a) && fabs(a-b) <= tolerance;
}

int main(void) {
    /* 3 observations, 2 features, row-major layout. */
    const double x[] = { 1, 0.5, -0.5, 1.5, 0.7, -1 };
    const double target[] = { 0.25, -0.3, 0.7 };
    double theta[] = { 0.2, -0.4, 0.15 };
    vmar_neuron_problem p = { x, target, 3, 2, VMAR_ACT_TANH, 0.3 };
    for (int activation = VMAR_ACT_TANH; activation <= VMAR_ACT_SIGMOID;
         ++activation) {
        p.activation = (vmar_activation)activation;
        double loss = NAN, gradient[3], predictions[3];
        CHECK(vmar_neuron_loss_gradient(&p, theta, &loss, gradient,
                                         predictions) == VMAR_OK);
        CHECK(loss >= 0 && isfinite(loss));
        for (size_t i = 0; i < 3; ++i) {
            double z = theta[2] + x[i*2]*theta[0] + x[i*2+1]*theta[1];
            double expected = activation == VMAR_ACT_TANH ? tanh(z) :
                              1.0 / (1.0 + exp(-z));
            CHECK(near(predictions[i], expected, 1e-14));
        }
        double fd = 1e-6;
        for (size_t j = 0; j < 3; ++j) {
            double plus[3] = {theta[0], theta[1], theta[2]};
            double minus[3] = {theta[0], theta[1], theta[2]};
            double grad_tmp[3], l_plus, l_minus;
            plus[j] += fd;
            minus[j] -= fd;
            CHECK(vmar_neuron_loss_gradient(&p, plus, &l_plus,
                                             grad_tmp, NULL) == VMAR_OK);
            CHECK(vmar_neuron_loss_gradient(&p, minus, &l_minus,
                                             grad_tmp, NULL) == VMAR_OK);
            CHECK(near(gradient[j], (l_plus-l_minus)/(2*fd), 2e-9));
        }
        double field[3];
        CHECK(vmar_neuron_field(0, theta, 3, field, &p) == VMAR_OK);
        for (size_t j = 0; j < 3; ++j)
            CHECK(near(field[j], -p.eta * gradient[j], 1e-14));

        double workspace[15], euler[3], rk4[3];
        CHECK(vmar_step(VMAR_EULER, 0, 0.1, vmar_neuron_field, &p,
                        theta, 3, workspace, 3, euler) == VMAR_OK);
        for (size_t j = 0; j < 3; ++j)
            CHECK(near(euler[j], theta[j] - 0.1*p.eta*gradient[j], 1e-14));
        CHECK(vmar_step(VMAR_RK4, 0, 0.1, vmar_neuron_field, &p,
                        theta, 3, workspace, 15, rk4) == VMAR_OK);
        for (size_t j = 0; j < 3; ++j) CHECK(isfinite(rk4[j]));
    }

    p.activation = VMAR_ACT_TANH;
    double loss = 0, gradient[3];
    vmar_neuron_problem invalid = p;
    invalid.samples = 0;
    CHECK(vmar_neuron_loss_gradient(&invalid, theta, &loss,
                                    gradient, NULL) == VMAR_ERR_ARGUMENT);
    invalid = p;
    invalid.eta = -1;
    CHECK(vmar_neuron_loss_gradient(&invalid, theta, &loss,
                                    gradient, NULL) == VMAR_ERR_ARGUMENT);
    invalid = p;
    invalid.activation = (vmar_activation)99;
    CHECK(vmar_neuron_loss_gradient(&invalid, theta, &loss,
                                    gradient, NULL) == VMAR_ERR_ARGUMENT);
    double bad_theta[] = { NAN, -0.4, 0.15 };
    CHECK(vmar_neuron_loss_gradient(&p, bad_theta, &loss,
                                    gradient, NULL) == VMAR_ERR_NUMERIC);
    CHECK(vmar_neuron_field(0, theta, 2, gradient, &p) == VMAR_ERR_DIMENSION);
    CHECK(vmar_neuron_field(NAN, theta, 3, gradient, &p) == VMAR_ERR_ARGUMENT);
    printf("vMAR neuron C23: %d assertions, %d failures\n", count, failures);
    return failures ? 1 : 0;
}
