#include <vmar/vmar.h>
#include <math.h>
#include <stdio.h>

static vmar_status decay(double time, const double *state,
                         size_t dimension, double *derivative, void *context) {
    (void) time;
    (void) context;
    if (dimension != 1) return VMAR_ERR_DIMENSION;
    derivative[0] = -state[0];
    return VMAR_OK;
}

int main(void) {
    double state[1] = {1.0};
    double next[1] = {0.0};
    double workspace[5] = {0.0};
    const vmar_status status = vmar_step(VMAR_RK4, 0.0, 0.1,
                                        decay, NULL, state, 1,
                                        workspace, 5, next);
    if (status != VMAR_OK || fabs(next[0] - exp(-0.1)) > 1e-6) {
        fprintf(stderr, "vMAR external consumer FAILED (%s)\n",
                vmar_status_string(status));
        return 1;
    }
    printf("vMAR external consumer PASS: %.12f\n", next[0]);
    return 0;
}
