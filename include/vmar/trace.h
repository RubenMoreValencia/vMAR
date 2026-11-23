#ifndef VMAR_TRACE_H
#define VMAR_TRACE_H

#include "vmar/vmar.h"

#ifdef __cplusplus
extern "C" {
#endif

/* G0.6-C.2: confirmed-state trace; the caller owns all storage.
 * One record refers to exactly dimension doubles in state_storage.
 * iter is a discrete confirmed-update index; time is differential time.
 * Attempts, including rejected proposals, never belong to this sequence.
 */
typedef struct vmar_trace_record {
    size_t iter;
    double time;
    double loss;
    double step; /* 0 for the initial state; positive for later states. */
} vmar_trace_record;

typedef struct vmar_trace {
    size_t dimension;
    size_t capacity;
    size_t length;
    vmar_trace_record *records;
    double *states;
} vmar_trace;

/* No allocation, no mutation of caller storage on rejected initialization. */
vmar_status vmar_trace_init(vmar_trace *trace, size_t dimension,
                            size_t capacity, vmar_trace_record *records,
                            double *state_storage);

/* The first confirmed state must be iter=0, step=0. Subsequent iter values
 * must increase by exactly one, time must increase strictly, step must equal
 * time - previous time within rounding tolerance. A failed append is atomic.
 * Caller must not alias source_state with record/storage memory.
 */
vmar_status vmar_trace_append(vmar_trace *trace, size_t iter, double time,
                              double loss, double step,
                              const double *source_state);

/* Separate attempt log: never alters the confirmed trajectory. The caller
 * should log every proposal in attempt order, including accepted proposals.
 * A rejected attempt may have nonfinite candidate_loss (numeric rejection).
 * No provenance metadata is inferred by this low-level recording API.
 */
typedef struct vmar_attempt_record {
    size_t attempt_id;
    size_t base_iter;
    int accepted;
    double proposed_step;
    double candidate_loss;
} vmar_attempt_record;

typedef struct vmar_attempt_log {
    size_t capacity;
    size_t length;
    vmar_attempt_record *records;
} vmar_attempt_log;

vmar_status vmar_attempt_log_init(vmar_attempt_log *log, size_t capacity,
                                  vmar_attempt_record *records);
vmar_status vmar_attempt_log_append(vmar_attempt_log *log,
                                    vmar_attempt_record attempt);

#ifdef __cplusplus
}
#endif

#endif
