#include "vmar/trace.h"
#include <math.h>
#include <stdint.h>
#include <string.h>

vmar_status vmar_trace_init(vmar_trace *trace, size_t dimension,
                            size_t capacity, vmar_trace_record *records,
                            double *state_storage) {
    if (!trace || !records || !state_storage || !dimension || !capacity)
        return VMAR_ERR_ARGUMENT;
    if (dimension > SIZE_MAX / capacity ||
        dimension * capacity > SIZE_MAX / sizeof(double))
        return VMAR_ERR_DIMENSION;
    *trace = (vmar_trace){dimension, capacity, 0, records, state_storage};
    return VMAR_OK;
}

vmar_status vmar_trace_append(vmar_trace *trace, size_t iter, double time,
                              double loss, double step,
                              const double *source_state) {
    if (!trace || !source_state || !trace->records || !trace->states ||
        !trace->dimension || !trace->capacity)
        return VMAR_ERR_ARGUMENT;
    if (trace->length >= trace->capacity) return VMAR_ERR_WORKSPACE;
    if (!isfinite(time) || !isfinite(loss) || !isfinite(step) ||
        step < 0.0) return VMAR_ERR_NUMERIC;

    const size_t k = trace->length;
    if (!k) {
        if (iter != 0 || step != 0.0) return VMAR_ERR_ARGUMENT;
    } else {
        const vmar_trace_record previous = trace->records[k - 1];
        if (previous.iter == SIZE_MAX || iter != previous.iter + 1 ||
            time <= previous.time || step <= 0.0)
            return VMAR_ERR_ARGUMENT;
        const double delta = time - previous.time;
        const double tolerance = 32.0 * 2.2204460492503131e-16 *
              fmax(1.0, fmax(fabs(time), fabs(previous.time)));
        if (!isfinite(delta) || fabs(delta - step) > tolerance)
            return VMAR_ERR_ARGUMENT;
    }
    for (size_t i = 0; i < trace->dimension; ++i)
        if (!isfinite(source_state[i])) return VMAR_ERR_NUMERIC;

    memcpy(trace->states + k * trace->dimension, source_state,
           trace->dimension * sizeof(double));
    trace->records[k] = (vmar_trace_record){iter, time, loss, step};
    trace->length = k + 1;
    return VMAR_OK;
}

vmar_status vmar_attempt_log_init(vmar_attempt_log *log, size_t capacity,
                                  vmar_attempt_record *records) {
    if (!log || !records || !capacity) return VMAR_ERR_ARGUMENT;
    *log = (vmar_attempt_log){capacity, 0, records};
    return VMAR_OK;
}

vmar_status vmar_attempt_log_append(vmar_attempt_log *log,
                                    vmar_attempt_record attempt) {
    if (!log || !log->records || !log->capacity) return VMAR_ERR_ARGUMENT;
    if (log->length >= log->capacity) return VMAR_ERR_WORKSPACE;
    if ((attempt.accepted != 0 && attempt.accepted != 1) ||
        !isfinite(attempt.proposed_step) || attempt.proposed_step <= 0.0 ||
        (attempt.accepted && !isfinite(attempt.candidate_loss)))
        return VMAR_ERR_ARGUMENT;
    if (log->length) {
        size_t previous_id = log->records[log->length - 1].attempt_id;
        if (previous_id == SIZE_MAX || attempt.attempt_id != previous_id + 1)
            return VMAR_ERR_ARGUMENT;
    } else if (attempt.attempt_id != 0) {
        return VMAR_ERR_ARGUMENT;
    }
    log->records[log->length++] = attempt;
    return VMAR_OK;
}
