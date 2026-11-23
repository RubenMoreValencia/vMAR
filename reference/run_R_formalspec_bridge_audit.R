# G0.6-C.1: source-based bridge to DSNeuralRNAS.FormalSpec 1.0.0.
# Run from the vMAR C23 project root, with vMAR.Core and FormalSpec loaded.
# This script does not modify FormalSpec or calibrate its frozen protocol.

if (!requireNamespace("vMAR.Core", quietly = TRUE)) {
  stop("Install/load vMAR.Core before running this bridge.", call. = FALSE)
}
if (!requireNamespace("DSNeuralRNAS.FormalSpec", quietly = TRUE)) {
  stop("Install DSNeuralRNAS.FormalSpec 1.0.1 before running this bridge.",
       call. = FALSE)
}

fs <- asNamespace("DSNeuralRNAS.FormalSpec")
spec <- fs$dsfs_spec(version = "1.0.0")
metadata <- fs$dsfs_metadata(
  experiment_id = "vmar-g06-c1-quadratic",
  engine = "vMAR.Core",
  engine_version = as.character(utils::packageVersion("vMAR.Core")),
  seed = NA_integer_,
  dataset_id = "quadratic_reference_case_v1",
  model_id = "diag-1-4",
  optimizer = "euler_discrete_gradient_flow"
)

A <- diag(c(1, 4))
eta <- c(rep(0.05, 8L), rep(0.60, 10L))
reference <- fs$dsfs_quadratic_stability_reference(A, eta)
stopifnot(isTRUE(all.equal(reference$eta_star, 0.5, tolerance = 1e-14)))
stopifnot(identical(as.integer(reference$first_crossing_iter), 8L))

formal_trace <- fs$dsfs_quadratic_trace(
  A = A, theta0 = c(1, 1), eta = eta,
  metadata = metadata, spec = spec
)
formal_data <- fs$dsfs_trace_data(formal_trace)
stopifnot(nrow(formal_data) == 19L)
stopifnot(identical(as.integer(formal_data$iter), 0:18))

field <- function(t, theta) -as.numeric(A %*% theta)
theta <- c(1, 1)
history <- matrix(NA_real_, nrow = length(eta) + 1L, ncol = 2L)
history[1L, ] <- theta
for (k in seq_along(eta)) {
  # Euler with h=eta is identical to the source's *discrete* GD update.
  # Never multiply eta into the field again.
  theta <- vMAR.Core::vmar_euler_step(field, k - 1, theta, eta[k])
  history[k + 1L, ] <- theta
}

loss <- 0.5 * (history[, 1L]^2 + 4 * history[, 2L]^2)
stopifnot(isTRUE(all.equal(history[, 1L], formal_data$theta1,
                           tolerance = 1e-13)))
stopifnot(isTRUE(all.equal(history[, 2L], formal_data$theta2,
                           tolerance = 1e-13)))
stopifnot(isTRUE(all.equal(loss, formal_data$loss,
                           tolerance = 1e-13)))

cat("G0.6-C.1 FormalSpec bridge PASS: 18 Euler/GD updates, 19 states, ",
    "theta/loss parity, eta_star=0.5, first crossing=8\n", sep = "")
cat("Boundary: this does not establish Ψ/Γ/Φ parity or RK4 stability parity.\n")
