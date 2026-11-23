# G0.6-C.3 — cross-language parity for versioned C23 trace exports.
# Run from vMAR/: first execute CTest or ./cmake-build-debug/vmar_export_trace
# cmake-build-debug/vmar_c3. Neither this script nor the adapter rewrites data.

source("reference/vmar_formalspec_adapter.R")

if (!requireNamespace("DSNeuralRNAS.FormalSpec", quietly = TRUE) ||
    !requireNamespace("vMAR.Core", quietly = TRUE)) {
  stop("Install/load DSNeuralRNAS.FormalSpec and vMAR.Core.", call. = FALSE)
}
fs <- asNamespace("DSNeuralRNAS.FormalSpec")
prefix <- "cmake-build-debug/vmar_c3"

read_exact <- function(path) {
  if (!file.exists(path)) stop("Missing C23 fixture: ", path, call. = FALSE)
  utils::read.csv(path, check.names = FALSE, stringsAsFactors = FALSE)
}

assert_close <- function(actual, expected, label, tolerance = 1e-12) {
  if (!isTRUE(all.equal(as.numeric(actual), as.numeric(expected),
                        tolerance = tolerance, check.attributes = FALSE))) {
    stop("Parity failure: ", label, call. = FALSE)
  }
}

verify_one <- function(method) {
  confirmed <- read_exact(paste0(prefix, "_", method, "_trace.csv"))
  attempts <- read_exact(paste0(prefix, "_", method, "_attempts.csv"))
  provenance <- read_exact(paste0(prefix, "_", method, "_metadata.csv"))
  stopifnot(nrow(provenance) == 1L,
            identical(provenance$schema_version, "vmar_trace_v1"),
            identical(provenance$formal_spec_version, "1.0.0"),
            identical(provenance$integrator, method),
            identical(provenance$engine, "vMAR"),
            identical(provenance$engine_version, "0.1.12"))

  metadata <- fs$dsfs_metadata(
    experiment_id = provenance$experiment_id,
    engine = provenance$engine,
    engine_version = provenance$engine_version,
    seed = if (is.na(provenance$seed)) NA_integer_ else
      as.integer(provenance$seed),
    dataset_id = provenance$dataset_id,
    model_id = provenance$model_id,
    optimizer = provenance$optimizer
  )
  bridge <- vmar_as_formalspec_trace(confirmed, metadata, attempts)
  stopifnot(fs$dsfs_validation_is_valid(fs$dsfs_validate_trace(bridge$trace)))
  stopifnot(nrow(confirmed) == 19L, nrow(attempts) == 19L,
            sum(attempts$accepted == 0L) == 1L,
            sum(attempts$accepted == 1L) == 18L,
            all(attempts$base_iter >= 0L & attempts$base_iter <= 17L))
  stopifnot(identical(as.integer(attempts$attempt_id), 0:18))
  stopifnot(all(attempts$base_iter[attempts$accepted == 1] == 0:17))
  stopifnot(attempts$base_iter[attempts$accepted == 0] == 8L)
  stopifnot(attempts$candidate_loss[attempts$accepted == 0] >
              confirmed$loss[9L])
  assert_close(attempts$candidate_loss[attempts$accepted == 1],
               confirmed$loss[-1L], paste0(method, " committed candidate loss"))
  assert_close(attempts$proposed_step[attempts$accepted == 1],
               confirmed$step[-1L], paste0(method, " committed steps"))

  A <- diag(c(1, 4))
  eta <- c(rep(0.05, 8L), rep(0.60, 10L))
  field <- function(t, theta) -as.numeric(A %*% theta)
  advance <- if (method == "euler") vMAR.Core::vmar_euler_step else
    vMAR.Core::vmar_rk4_step
  states <- matrix(NA_real_, nrow = 19L, ncol = 2L)
  states[1L, ] <- c(1, 1)
  for (k in seq_along(eta)) {
    states[k + 1L, ] <- advance(field, sum(eta[seq_len(k - 1L)]),
                                states[k, ], eta[k])
  }
  assert_close(confirmed$iter, 0:18, paste0(method, " indices"))
  assert_close(confirmed$time, c(0, cumsum(eta)), paste0(method, " time"))
  assert_close(confirmed$step, c(0, eta), paste0(method, " step"))
  assert_close(confirmed$theta1, states[, 1L], paste0(method, " theta1"))
  assert_close(confirmed$theta2, states[, 2L], paste0(method, " theta2"))
  assert_close(confirmed$loss,
               0.5 * (states[, 1L]^2 + 4 * states[, 2L]^2),
               paste0(method, " loss"))
  stopifnot(identical(fs$dsfs_trace_metadata(bridge$trace)$formal_spec_version,
                      "1.0.0"))

  if (method == "euler") {
    # Only Euler has the exact discrete-GD algebraic reference.
    gd <- fs$dsfs_quadratic_trace(A, c(1, 1), eta, metadata)
    gd_data <- fs$dsfs_trace_data(gd)
    assert_close(confirmed$loss, gd_data$loss, "Euler FormalSpec loss")
    assert_close(confirmed$theta1, gd_data$theta1, "Euler FormalSpec theta1")
    assert_close(confirmed$theta2, gd_data$theta2, "Euler FormalSpec theta2")
  }

  # Negative structural and semantic checks; never repair corrupted exports.
  bad <- confirmed
  bad$iter[3L] <- bad$iter[2L]
  stopifnot(inherits(tryCatch(vmar_as_formalspec_trace(bad, metadata, attempts),
                              error = identity), "error"))
  bad <- confirmed
  bad$time[3L] <- bad$time[2L]
  stopifnot(inherits(tryCatch(vmar_as_formalspec_trace(bad, metadata, attempts),
                              error = identity), "error"))
  bad <- confirmed
  bad$loss[3L] <- Inf
  stopifnot(inherits(tryCatch(vmar_as_formalspec_trace(bad, metadata, attempts),
                              error = identity), "error"))

  cat("G0.6-C.3 PARITY PASS ", method,
      ": 19 confirmed states, 19 proposals, 1 rejection; ",
      "times/theta/loss/metadata and negative gates.\n", sep = "")
}

verify_one("euler")
verify_one("rk4")
cat("G0.6-C.3 R-C23 TRACE PARITY PASS: two integrators; ",
    "FormalSpec Ψ/Γ/Φ not reimplemented.\n", sep = "")
