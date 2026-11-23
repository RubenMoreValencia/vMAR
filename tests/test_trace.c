#include "vmar/trace.h"
#include <math.h>
#include <stdio.h>

static int checks = 0, failures = 0;
#define CHECK(x) do { ++checks; if (!(x)) { ++failures; \
  fprintf(stderr, "FAIL line %d: %s\n", __LINE__, #x); } } while (0)

int main(void) {
    vmar_trace_record records[3] = {{0}};
    double storage[6] = {0};
    vmar_trace trace = {0};
    const double init[] = {1.0, 1.0};
    const double next[] = {0.95, 0.8};
    const double bad[] = {NAN, 1.0};
    CHECK(vmar_trace_init(&trace, 2, 3, records, storage) == VMAR_OK);
    CHECK(vmar_trace_append(&trace, 0, 0, 2.5, 0, init) == VMAR_OK);
    CHECK(trace.length == 1);
    CHECK(vmar_trace_append(&trace, 1, 0.05, 2.0, 0.05, bad) == VMAR_ERR_NUMERIC);
    CHECK(trace.length == 1 && storage[2] == 0);
    CHECK(vmar_trace_append(&trace, 1, 0.05, 2.0, 0.1, next) == VMAR_ERR_ARGUMENT);
    CHECK(vmar_trace_append(&trace, 2, 0.05, 2.0, 0.05, next) == VMAR_ERR_ARGUMENT);
    CHECK(vmar_trace_append(&trace, 1, 0.05, 2.0, 0.05, next) == VMAR_OK);
    CHECK(trace.length == 2 && records[1].iter == 1);
    CHECK(storage[2] == 0.95 && storage[3] == 0.8);
    CHECK(vmar_trace_append(&trace, 2, 0.05, 2.0, 0.05, next) == VMAR_ERR_ARGUMENT);
    CHECK(trace.length == 2);
    CHECK(vmar_trace_append(&trace, 2, 0.65, 2.0, 0.60, next) == VMAR_OK);
    CHECK(vmar_trace_append(&trace, 3, 0.70, 1.0, 0.05, next) == VMAR_ERR_WORKSPACE);

    vmar_attempt_record events[3] = {{0}};
    vmar_attempt_log log = {0};
    CHECK(vmar_attempt_log_init(&log, 3, events) == VMAR_OK);
    CHECK(vmar_attempt_log_append(&log, (vmar_attempt_record){0, 0, 0, 0.1, NAN}) == VMAR_OK);
    CHECK(trace.length == 3 && log.length == 1);
    CHECK(vmar_attempt_log_append(&log, (vmar_attempt_record){1, 0, 1, 0.05, 2.0}) == VMAR_OK);
    CHECK(vmar_attempt_log_append(&log, (vmar_attempt_record){3, 1, 0, 0.05, 2.0}) == VMAR_ERR_ARGUMENT);
    CHECK(log.length == 2);
    CHECK(vmar_attempt_log_append(&log, (vmar_attempt_record){2, 1, 0, 0.05, 3.0}) == VMAR_OK);
    CHECK(vmar_attempt_log_append(&log, (vmar_attempt_record){3, 1, 0, 0.05, 3.0}) == VMAR_ERR_WORKSPACE);
    printf("vMAR trace interoperability contracts: %d assertions, %d failures\n", checks, failures);
    return failures != 0;
}
