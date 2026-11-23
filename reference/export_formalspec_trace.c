#include "vmar/trace.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

/* G0.6-C.3: deterministic differential learning trace exporter.
 * CSV is a versioned interchange artifact, not FormalSpec semantics.
 * Usage: vmar_export_trace <prefix>
 * Generates <prefix>_<euler|rk4>_{trace,attempts,metadata}.csv.
 */

static vmar_status quadratic(double t, const double *theta, size_t dimension,
                             double *out, void *ctx) {
    (void)t; (void)ctx;
    if (dimension != 2) return VMAR_ERR_DIMENSION;
    out[0] = -theta[0];
    out[1] = -4.0 * theta[1];
    return VMAR_OK;
}

static double loss(const double *theta) {
    return 0.5 * (theta[0] * theta[0] + 4.0 * theta[1] * theta[1]);
}

static int output_path(char *buffer, size_t capacity, const char *prefix,
                       const char *method, const char *section) {
    const int n = snprintf(buffer, capacity, "%s_%s_%s.csv",
                           prefix, method, section);
    return n > 0 && (size_t)n < capacity;
}

static int write_trace(const char *prefix, const char *name,
                       const vmar_trace *trace) {
    char path[4096];
    if (!output_path(path, sizeof(path), prefix, name, "trace")) return 0;
    FILE *file = fopen(path, "w");
    if (!file) return 0;
    int ok = fprintf(file, "iter,time,loss,step,theta1,theta2\n") > 0;
    for (size_t i = 0; i < trace->length && ok; ++i) {
        const vmar_trace_record *r = trace->records + i;
        const double *theta = trace->states + i * trace->dimension;
        ok = fprintf(file, "%zu,%.17g,%.17g,%.17g,%.17g,%.17g\n",
                     r->iter, r->time, r->loss, r->step,
                     theta[0], theta[1]) > 0;
    }
    if (fclose(file) != 0) ok = 0;
    return ok;
}

static int write_attempts(const char *prefix, const char *name,
                          const vmar_attempt_log *log) {
    char path[4096];
    if (!output_path(path, sizeof(path), prefix, name, "attempts")) return 0;
    FILE *file = fopen(path, "w");
    if (!file) return 0;
    int ok = fprintf(file, "attempt_id,base_iter,accepted,proposed_step,candidate_loss\n") > 0;
    for (size_t i = 0; i < log->length && ok; ++i) {
        const vmar_attempt_record *r = log->records + i;
        ok = fprintf(file, "%zu,%zu,%d,%.17g,%.17g\n",
                     r->attempt_id, r->base_iter, r->accepted,
                     r->proposed_step, r->candidate_loss) > 0;
    }
    if (fclose(file) != 0) ok = 0;
    return ok;
}

static int write_metadata(const char *prefix, const char *name) {
    char path[4096];
    if (!output_path(path, sizeof(path), prefix, name, "metadata")) return 0;
    FILE *file = fopen(path, "w");
    if (!file) return 0;
    const int ok = fprintf(file,
        "schema_version,formal_spec_version,experiment_id,engine,engine_version,seed,dataset_id,model_id,optimizer,integrator\n"
        "vmar_trace_v1,1.0.0,vmar-g06-c3-%s,vMAR,0.1.12,,quadratic_reference_case_v1,diag-1-4,differential_gradient_flow,%s\n",
        name, name) > 0;
    return fclose(file) == 0 && ok;
}

static int run(const char *prefix, vmar_integrator method, const char *name) {
    vmar_trace_record records[19] = {{0}};
    double state_storage[38] = {0};
    vmar_attempt_record attempts_storage[19] = {{0}};
    vmar_trace trace = {0};
    vmar_attempt_log attempts = {0};
    if (vmar_trace_init(&trace, 2, 19, records, state_storage) != VMAR_OK ||
        vmar_attempt_log_init(&attempts, 19, attempts_storage) != VMAR_OK)
        return 0;

    double theta[2] = {1, 1};
    double next[2] = {0, 0};
    double work[10] = {0};
    double time = 0;
    if (vmar_trace_append(&trace, 0, time, loss(theta), 0, theta) != VMAR_OK)
        return 0;

    for (size_t k = 0; k < 18; ++k) {
        const double step = k < 8 ? 0.05 : 0.60;
        const double before = loss(theta);
        /* A genuinely evaluated but uncommitted proposal at k=8.
         * Rejection is due to a local loss gate, independent of the later
         * accepted solver step. This is an illustrative proposal audit.
         */
        if (k == 8) {
            double rejected[2] = {0};
            if (vmar_step(method, time, 1.2, quadratic, NULL, theta, 2,
                          work, method == VMAR_EULER ? 2 : 10,
                          rejected) != VMAR_OK)
                return 0;
            double rejected_loss = loss(rejected);
            if (!(rejected_loss > before)) return 0;
            if (vmar_attempt_log_append(&attempts,
                    (vmar_attempt_record){attempts.length, k, 0,
                                          1.2, rejected_loss}) != VMAR_OK)
                return 0;
        }
        if (vmar_step(method, time, step, quadratic, NULL, theta, 2,
                      work, method == VMAR_EULER ? 2 : 10, next) != VMAR_OK)
            return 0;
        const double candidate_loss = loss(next);
        if (vmar_attempt_log_append(&attempts,
                (vmar_attempt_record){attempts.length, k, 1,
                                      step, candidate_loss}) != VMAR_OK)
            return 0;
        time += step;
        if (vmar_trace_append(&trace, k + 1, time, candidate_loss,
                              step, next) != VMAR_OK)
            return 0;
        memcpy(theta, next, sizeof theta);
    }
    if (trace.length != 19 || attempts.length != 19) return 0;
    if (!write_trace(prefix, name, &trace) ||
        !write_attempts(prefix, name, &attempts) ||
        !write_metadata(prefix, name)) return 0;
    printf("TRACE EXPORT PASS %s: 19 confirmed, 19 attempts (1 rejected)\n",
           name);
    return 1;
}

int main(int argc, char **argv) {
    if (argc != 2 || !argv[1][0]) {
        fprintf(stderr, "Usage: vmar_export_trace <prefix>\n");
        return 2;
    }
    if (!run(argv[1], VMAR_EULER, "euler") ||
        !run(argv[1], VMAR_RK4, "rk4")) {
        fprintf(stderr, "TRACE EXPORT FAILED\n");
        return 1;
    }
    return 0;
}
