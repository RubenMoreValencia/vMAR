# G0.8-B / C01 — Observational Hessian pilot. Run from the vMAR root.
# This script does not modify vMAR.Core or the C23 API.

.vmar_b_loss <- function(loss, theta) {
  result <- loss(theta)
  if (!is.numeric(result) || length(result) != 1L ||
      !is.finite(result)) {
    stop("loss must return one finite numeric scalar", call. = FALSE)
  }
  as.numeric(result)
}

vmar_pilot_hessian <- function(loss, theta, h = 1e-4) {
  if (!is.function(loss) || !is.numeric(theta) ||
      !is.null(dim(theta)) || length(theta) < 1L ||
      !all(is.finite(theta)) || !is.numeric(h) ||
      length(h) != 1L || !is.finite(h) || h <= 0) {
    stop("invalid loss, theta or finite-difference step", call. = FALSE)
  }
  p <- length(theta)
  H <- matrix(0, p, p)
  calls <- 0L
  evaluated <- function(x) {
    calls <<- calls + 1L
    .vmar_b_loss(loss, x)
  }
  # Source-equivalent four-corner difference; avoid redundant mirrored pairs.
  for (i in seq_len(p)) {
    for (j in i:p) {
      ei <- ej <- numeric(p)
      ei[i] <- h
      ej[j] <- h
      value <- (evaluated(theta + ei + ej) -
        evaluated(theta + ei - ej) -
        evaluated(theta - ei + ej) +
        evaluated(theta - ei - ej)) / (4 * h^2)
      H[i, j] <- H[j, i] <- value
    }
  }
  if (!all(is.finite(H))) {
    stop("non-finite Hessian: step or loss scale is unsuitable", call. = FALSE)
  }
  list(hessian = H, loss_calls = calls,
       nominal_full_grid_calls = 4L * p * p,
       dimension = p, step = h)
}

vmar_pilot_hessian_report <- function(H, tol = 1e-8) {
  if (!is.matrix(H) || !is.numeric(H) || !nrow(H) ||
      nrow(H) != ncol(H) || !all(is.finite(H)) ||
      !is.numeric(tol) || length(tol) != 1L ||
      !is.finite(tol) || tol <= 0) {
    stop("invalid Hessian or tolerance", call. = FALSE)
  }
  if (max(abs(H - t(H))) > tol) {
    stop("Hessian must be symmetric within tolerance", call. = FALSE)
  }
  eigenvalues <- eigen((H + t(H))/2, symmetric = TRUE,
                       only.values = TRUE)$values
  positives <- sum(eigenvalues > tol)
  negatives <- sum(eigenvalues < -tol)
  type <- if (positives == length(eigenvalues)) "positive" else
    if (negatives == length(eigenvalues)) "negative" else
      if (positives > 0L && negatives > 0L) "saddle" else "semidefinite_or_flat"
  list(eigenvalues = eigenvalues, classification = type,
       symmetry_error = max(abs(H-t(H))))
}

# Secondary finite-difference method based on an independently implemented
# analytical gradient. Its calls must be counted separately from loss calls.
vmar_pilot_hessian_from_gradient <- function(gradient, theta, h = 1e-4) {
  if (!is.function(gradient) || !is.numeric(theta) ||
      !is.null(dim(theta)) || !length(theta) ||
      !all(is.finite(theta)) || !is.numeric(h) || length(h) != 1L ||
      !is.finite(h) || h <= 0) {
    stop("invalid gradient, state or step", call. = FALSE)
  }
  p <- length(theta)
  H <- matrix(0, p, p)
  for (j in seq_len(p)) {
    e <- numeric(p)
    e[j] <- h
    gp <- gradient(theta + e)
    gm <- gradient(theta - e)
    if (!is.numeric(gp) || !is.numeric(gm) || length(gp) != p ||
        length(gm) != p || !all(is.finite(gp)) || !all(is.finite(gm))) {
      stop("gradient must return a finite vector with state dimension",
           call. = FALSE)
    }
    H[, j] <- (gp - gm) / (2 * h)
  }
  list(hessian = (H + t(H))/2,
       unsymmetrized_error = max(abs(H - t(H))),
       gradient_calls = 2L * p)
}

# Run after devtools::load_all("/path/to/vMAR.Core") or library(vMAR.Core).
# Tests a smooth neural loss without claiming a general stability theorem.
vmar_pilot_neural_curvature <- function(output_dir = "reference") {
  if (!"vMAR.Core" %in% loadedNamespaces()) {
    stop("load vMAR.Core before the neural curvature case", call. = FALSE)
  }
  spec <- vMAR.Core::vmar_network_spec(c(2, 1), "tanh")
  x <- matrix(c(-1, 0.2, 0.3, 0.6, 0.8, -0.1, 1.1, 0.4),
              nrow = 4, ncol = 2, byrow = TRUE)
  target <- c(-0.3, 0.2, 0.6, 0.7)
  theta <- c(0.2, -0.4, 0.1)
  original <- theta
  evaluator <- function(th) vMAR.Core::vmar_network_loss_gradient(
    x, target, spec, th)
  loss <- function(th) evaluator(th)$loss
  gradient <- function(th) evaluator(th)$gradient
  comparison <- lapply(c(1e-2, 1e-3, 1e-4, 1e-5), function(h) {
    from_loss <- vmar_pilot_hessian(loss, theta, h)
    from_gradient <- vmar_pilot_hessian_from_gradient(gradient, theta, h)
    data.frame(h = h,
      method_disagreement = max(abs(from_loss$hessian - from_gradient$hessian)),
      loss_calls = from_loss$loss_calls,
      gradient_calls = from_gradient$gradient_calls,
      gradient_asymmetry = from_gradient$unsymmetrized_error)
  })
  result <- do.call(rbind, comparison)
  stopifnot(identical(theta, original), all(is.finite(result$method_disagreement)),
            all(result$loss_calls == 24L),
            all(result$gradient_calls == 6L),
            result$method_disagreement[result$h == 1e-3] < 1e-4)
  file <- file.path(output_dir, "vmar_g08b_neural_curvature.csv")
  utils::write.csv(result, file, row.names = FALSE)
  cat("G0.8-B C01 neural cross-method gate: PASS; results:",file,"\n")
  invisible(result)
}

vmar_run_curvature_pilot <- function(output_dir = "reference") {
  if (!is.character(output_dir) || length(output_dir) != 1L ||
      is.na(output_dir) || !dir.exists(output_dir)) {
    stop("output_dir must be an existing directory", call. = FALSE)
  }
  cases <- list(
    positive = diag(c(2, 5)),
    negative = diag(c(-2, -5)),
    saddle = diag(c(-2, 5)),
    rank_deficient = diag(c(2, 0))
  )
  rows <- list()
  k <- 0L
  for (name in names(cases)) {
    A <- cases[[name]]
    objective <- function(theta) {
      as.numeric(0.5 * crossprod(theta, A %*% theta))
    }
    for (h in c(1e-2, 1e-3, 1e-4, 1e-6, 1e-8)) {
      estimate <- vmar_pilot_hessian(objective, c(0.3, -0.7), h)
      audit <- vmar_pilot_hessian_report(estimate$hessian, tol = 1e-5)
      k <- k + 1L
      rows[[k]] <- data.frame(
        case_id = name, h = h,
        max_abs_error = max(abs(estimate$hessian - A)),
        symmetry_error = audit$symmetry_error,
        expected_class = switch(name, positive = "positive",
                                negative = "negative", saddle = "saddle",
                                rank_deficient = "semidefinite_or_flat"),
        observed_class = audit$classification,
        loss_calls = estimate$loss_calls,
        nominal_full_grid_calls = estimate$nominal_full_grid_calls,
        stringsAsFactors = FALSE)
    }
  }
  result <- do.call(rbind, rows)

  # Pre-declared numerical reference gate, avoiding tiny h dominated by
  # cancellation. The h = 1e-8 rows are sensitivity observations only.
  gate <- result[result$h == 1e-3, ]
  stopifnot(nrow(gate) == 4L,
            all(gate$max_abs_error < 1e-7),
            all(gate$observed_class == gate$expected_class),
            all(result$loss_calls == 12L))

  # Invalid-domain and mutation-invariance contracts.
  invalid <- function(expression) inherits(tryCatch(
    { force(expression); NULL }, error = identity), "error")
  stopifnot(invalid(vmar_pilot_hessian(objective, c(0.2, NA_real_))),
            invalid(vmar_pilot_hessian(objective, c(0.2, 0.3), 0)),
            invalid(vmar_pilot_hessian(function(x) NA_real_, c(1, 2))),
            invalid(vmar_pilot_hessian_report(matrix(1:4, 2, 2))))
  state <- c(0.3, -0.7)
  before <- state
  invisible(vmar_pilot_hessian(objective, state))
  stopifnot(identical(state, before))

  file <- file.path(output_dir, "vmar_g08b_curvature_audit.csv")
  utils::write.csv(result, file = file, row.names = FALSE)
  cat("G0.8-B C01 analytic quadratic gate: PASS (4 cases); ",
      nrow(result), " sensitivity rows.\n", sep = "")
  cat("Observed 12 loss calls per 2-parameter Hessian (full source grid: 16).\n")
  cat("No C23 promotion authorized. Results:", file, "\n")
  invisible(result)
}

if (identical(Sys.getenv("VMAR_RUN_CURVATURE_PILOT"), "1")) {
  vmar_run_curvature_pilot()
}
