# G0.6-C.2 reference audit. Execute from vMAR/ with both R packages loaded.
source("reference/vmar_formalspec_adapter.R")

if (!requireNamespace("DSNeuralRNAS.FormalSpec", quietly = TRUE) ||
    !requireNamespace("vMAR.Core", quietly = TRUE)) {
  stop("Both DSNeuralRNAS.FormalSpec and vMAR.Core must be installed.",
       call. = FALSE)
}
fs <- asNamespace("DSNeuralRNAS.FormalSpec")
metadata <- fs$dsfs_metadata(
  experiment_id = "vmar-g06-c2-trace",
  engine = "vMAR.Core",
  engine_version = as.character(utils::packageVersion("vMAR.Core")),
  seed = NA_integer_,
  dataset_id = "quadratic_reference_case_v1",
  model_id = "diag-1-4",
  optimizer = "euler_gradient_flow"
)
A <- diag(c(1, 4))
eta <- c(rep(0.05, 8), rep(0.60, 10))
field <- function(t, y) -as.numeric(A %*% y)
theta <- c(1, 1)
states <- matrix(NA_real_, nrow = length(eta) + 1L, ncol = 2L)
states[1L, ] <- theta
for (k in seq_along(eta)) {
  theta <- vMAR.Core::vmar_euler_step(field, 0, theta, eta[k])
  states[k + 1L, ] <- theta
}
confirmed <- data.frame(
  iter = 0:18,
  time = c(0, cumsum(eta)),
  loss = 0.5 * (states[, 1]^2 + 4 * states[, 2]^2),
  step = c(0, eta),
  theta1 = states[, 1], theta2 = states[, 2]
)
# One rejected proposal: proves it is never inserted into the confirmed trace.
attempts <- data.frame(
  attempt_id = 0:1, base_iter = c(1L, 1L),
  accepted = c(0L, 1L), proposed_step = c(0.1, 0.05),
  candidate_loss = c(NA_real_, confirmed$loss[2L])
)
bridge <- vmar_as_formalspec_trace(confirmed, metadata, attempts)
stopifnot(fs$dsfs_validation_is_valid(fs$dsfs_validate_trace(bridge$trace)))
stopifnot(nrow(fs$dsfs_trace_data(bridge$trace)) == 19L)
stopifnot(nrow(bridge$attempts) == 2L)
stopifnot(identical(fs$dsfs_trace_metadata(bridge$trace)$formal_spec_version,
                    "1.0.0"))
ref <- fs$dsfs_quadratic_trace(A, c(1, 1), eta, metadata)
actual <- fs$dsfs_trace_data(bridge$trace)
expected <- fs$dsfs_trace_data(ref)
stopifnot(isTRUE(all.equal(actual$loss, expected$loss, tolerance = 1e-13)))
stopifnot(isTRUE(all.equal(actual$theta1, expected$theta1, tolerance = 1e-13)))
stopifnot(isTRUE(all.equal(actual$theta2, expected$theta2, tolerance = 1e-13)))
stopifnot(isTRUE(all.equal(bridge$integrator_steps[-1L], eta)))

expect_failure <- function(x) inherits(tryCatch(force(x), error = identity), "error")
bad_order <- confirmed
bad_order$iter[3] <- 1
stopifnot(expect_failure(vmar_as_formalspec_trace(bad_order, metadata)))
bad_time <- confirmed
bad_time$time[3] <- bad_time$time[2]
stopifnot(expect_failure(vmar_as_formalspec_trace(bad_time, metadata)))
bad_loss <- confirmed
bad_loss$loss[3] <- NA_real_
stopifnot(expect_failure(vmar_as_formalspec_trace(bad_loss, metadata)))
bad_attempts <- attempts
bad_attempts$accepted[1] <- 1L
stopifnot(expect_failure(vmar_as_formalspec_trace(confirmed, metadata,
                                                 bad_attempts)))

cat("G0.6-C.2 PASS: 19 confirmed states, 18 updates, theta/loss parity, ",
    "separate attempts, metadata version, four negative contracts.\n", sep = "")
cat("FormalSpec Psi/Gamma/Phi remain delegated to FormalSpec.\n")
