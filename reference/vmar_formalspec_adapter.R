# G0.6-C.2 reference adapter, deliberately outside the vMAR C23 core.
# The FormalSpec package remains the authority for trace interpretation.

vmar_as_formalspec_trace <- function(confirmed, metadata, attempts = NULL,
                                    spec = NULL) {
  if (!requireNamespace("DSNeuralRNAS.FormalSpec", quietly = TRUE)) {
    stop("DSNeuralRNAS.FormalSpec is required", call. = FALSE)
  }
  fs <- asNamespace("DSNeuralRNAS.FormalSpec")
  if (is.null(spec)) spec <- fs$dsfs_spec("1.0.0")
  if (!identical(spec$version, "1.0.0")) {
    stop("G0.6-C.2 is contracted for FormalSpec 1.0.0", call. = FALSE)
  }
  if (!is.data.frame(confirmed) || nrow(confirmed) < 1L ||
      !all(c("iter", "time", "loss", "step") %in% names(confirmed))) {
    stop("confirmed needs iter, time, loss, step", call. = FALSE)
  }
  n <- nrow(confirmed)
  numeric_ok <- function(x) is.numeric(x) && length(x) == n &&
    all(is.finite(x))
  if (!all(vapply(confirmed[c("iter", "time", "loss", "step")],
                  numeric_ok, logical(1)))) {
    stop("confirmed core columns must be finite numeric", call. = FALSE)
  }
  if (!identical(as.numeric(confirmed$iter), as.numeric(0:(n - 1L))) ||
      confirmed$step[1L] != 0 || any(confirmed$step < 0)) {
    stop("confirmed iterations must start at 0 and have no gaps", call. = FALSE)
  }
  if (n > 1L && (any(diff(confirmed$time) <= 0) ||
      any(confirmed$step[-1L] <= 0) ||
      any(abs(diff(confirmed$time) - confirmed$step[-1L]) >
          32 * .Machine$double.eps * pmax(1, abs(confirmed$time[-1L]),
                                            abs(confirmed$time[-n]))))) {
    stop("integration time/steps are inconsistent", call. = FALSE)
  }
  theta_cols <- grep("^theta[1-9][0-9]*$", names(confirmed), value = TRUE)
  if (!length(theta_cols) || !all(vapply(confirmed[theta_cols],
                                      numeric_ok, logical(1)))) {
    stop("finite parameter columns theta1..thetaN are required", call. = FALSE)
  }
  theta_ids <- as.integer(sub("theta", "", theta_cols))
  if (!identical(sort(theta_ids), seq_along(theta_ids))) {
    stop("theta columns must use contiguous indices", call. = FALSE)
  }
  if (!is.null(attempts)) {
    required <- c("attempt_id", "base_iter", "accepted",
                  "proposed_step", "candidate_loss")
    if (!is.data.frame(attempts) || !all(required %in% names(attempts))) {
      stop("attempts must carry the separate attempt-log contract", call. = FALSE)
    }
    if (nrow(attempts) && (!identical(as.numeric(attempts$attempt_id),
                               as.numeric(0:(nrow(attempts) - 1L))) ||
        anyNA(attempts$accepted) ||
        !all(attempts$accepted %in% c(0L, 1L)) ||
        any(!is.finite(attempts$proposed_step) |
            attempts$proposed_step <= 0) ||
        any(attempts$accepted == 1 & !is.finite(attempts$candidate_loss)))) {
      stop("invalid attempt log", call. = FALSE)
    }
  }

  # Map only semantics actually shared with FormalSpec. In particular, 'step'
  # is a differential integration step, NOT necessarily learning rate 'eta'.
  formal_data <- confirmed[, setdiff(names(confirmed), "step"), drop = FALSE]
  formal_trace <- fs$dsfs_trace(formal_data, metadata, spec)
  list(trace = formal_trace, attempts = attempts,
       integrator_steps = confirmed$step, specification = spec$version)
}
