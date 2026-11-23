# G0.8-B.1 — Curvature observability along confirmed learning states.
# Run from the vMAR C23 project root, with vMAR.Core loaded.
# This reference study never uses curvature to update the learning field.

if (!exists("vmar_pilot_hessian", mode = "function") ||
    !exists("vmar_pilot_hessian_from_gradient", mode = "function")) {
  source("reference/run_R_curvature_pilot.R", local = FALSE)
}

vmar_run_curvature_trajectory <- function(output_dir = "reference") {
  if (!"vMAR.Core" %in% loadedNamespaces()) {
    stop("load vMAR.Core before G0.8-B.1", call. = FALSE)
  }
  if (!is.character(output_dir) || length(output_dir) != 1L ||
      is.na(output_dir) || !dir.exists(output_dir)) {
    stop("output_dir must be an existing directory", call. = FALSE)
  }

  # Fixed small MLP. This descriptor and its initial parameters are shared
  # by both reference integrators, with no stochastic initialization.
  spec <- vMAR.Core::vmar_network_spec(c(2, 2, 1), c("tanh", "tanh"))
  weights <- list(matrix(c(0.3, -0.2, -0.4, 0.5), 2, 2,
                         byrow = TRUE),
                  matrix(c(0.6, -0.7), 1, 2))
  biases <- list(c(0.05, -0.1), 0.03)
  theta0 <- vMAR.Core::vmar_network_pack(spec, weights, biases)
  x <- matrix(c(-1.0, -0.5,
                -0.6,  0.2,
                -0.2,  0.8,
                 0.3, -0.3,
                 0.7,  0.6,
                 1.0, -0.8), ncol = 2, byrow = TRUE)
  target <- c(-0.6, -0.3, 0.1, 0.2, 0.5, 0.65)
  evaluator <- function(theta) vMAR.Core::vmar_network_loss_gradient(
    x, target, spec, theta)
  loss <- function(theta) evaluator(theta)$loss
  gradient <- function(theta) evaluator(theta)$gradient
  field <- vMAR.Core::vmar_network_gradient_field(x, target, spec, eta = 1)

  # A fixed simulation horizon is the comparison basis. Neither integrator
  # is an adaptive learning policy. These are confirmed, not trial states.
  n_steps <- 20L
  dt <- 0.05
  checkpoints <- c(0L, 10L, 20L)
  h_values <- c(1e-3, 1e-4)
  rows <- list()
  idx <- 0L
  states_before <- list()
  states_after <- list()
  for (integrator in c("euler", "rk4")) {
    trajectory <- vMAR.Core::vmar_integrate(
      field, 0, theta0, dt, n_steps, integrator)
    stopifnot(nrow(trajectory$state) == n_steps + 1L,
              isTRUE(all.equal(trajectory$time[n_steps + 1L], 1)),
              identical(as.numeric(trajectory$state[1L, ]), theta0))
    states_before[[integrator]] <- trajectory$state
    for (step_id in checkpoints) {
      theta <- as.numeric(trajectory$state[step_id + 1L, ])
      evaluated <- evaluator(theta)
      for (h in h_values) {
        a <- vmar_pilot_hessian(loss, theta, h)
        b <- vmar_pilot_hessian_from_gradient(gradient, theta, h)
        eig <- eigen(a$hessian, symmetric = TRUE,
                     only.values = TRUE)$values
        report <- vmar_pilot_hessian_report(a$hessian, tol = 1e-6)
        idx <- idx + 1L
        rows[[idx]] <- data.frame(
          scenario = "fixed_tanh_2_2_1", integrator = integrator,
          confirmed_step = step_id, time = trajectory$time[step_id + 1L],
          h = h, loss = evaluated$loss,
          gradient_norm = sqrt(sum(evaluated$gradient^2)),
          lambda_min = min(eig), lambda_max = max(eig),
          curvature_class = report$classification,
          hessian_method_disagreement = max(abs(a$hessian - b$hessian)),
          gradient_hessian_asymmetry = b$unsymmetrized_error,
          loss_calls_hessian = a$loss_calls,
          gradient_calls_hessian = b$gradient_calls,
          parameter_count = length(theta),
          stringsAsFactors = FALSE)
      }
    }
    states_after[[integrator]] <- trajectory$state
  }
  result <- do.call(rbind, rows)

  # Preregistered technical gates. Explanatory value is deliberately NOT
  # asserted: it must be decided after viewing the observed contrasts.
  stopifnot(nrow(result) == 12L, all(is.finite(result$loss)),
            all(is.finite(result$gradient_norm)),
            all(is.finite(result$lambda_min)),
            all(is.finite(result$lambda_max)),
            all(is.finite(result$hessian_method_disagreement)),
            all(result$lambda_min <= result$lambda_max),
            all(result$loss_calls_hessian == 2L * 9L * 10L),
            all(result$gradient_calls_hessian == 2L * 9L),
            all(result$parameter_count == 9L),
            identical(states_before, states_after))
  # The cross-method threshold is a numeric precision gate only.
  reference_rows <- result[result$h == 1e-3, ]
  stopifnot(nrow(reference_rows) == 6L,
            all(reference_rows$hessian_method_disagreement < 1e-3))

  result_file <- file.path(output_dir, "vmar_g08b1_curvature_trajectory.csv")
  utils::write.csv(result, result_file, row.names = FALSE)

  # One row per confirmed checkpoint and integrator, showing whether the
  # second-order observation changes when first-order metrics evolve.
  report_rows <- reference_rows[, c("integrator", "confirmed_step", "time",
    "loss", "gradient_norm", "lambda_min", "lambda_max",
    "curvature_class", "hessian_method_disagreement",
    "loss_calls_hessian", "gradient_calls_hessian")]
  report_file <- file.path(output_dir, "vmar_g08b1_curvature_checkpoints.csv")
  utils::write.csv(report_rows, report_file, row.names = FALSE)
  cat("G0.8-B.1 TECHNICAL GATE PASS: 2 integrators, 3 confirmed checkpoints,",
      "2 Hessian steps; 12 diagnostic rows.\n")
  cat("Scientific utility and FormalSpec evidence: NOT YET ADJUDICATED.\n")
  cat("Results:", result_file, "and", report_file, "\n")
  invisible(list(all = result, checkpoints = report_rows))
}
