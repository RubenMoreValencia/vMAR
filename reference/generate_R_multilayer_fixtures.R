# G0.7-B R oracle for the independent C23 multilayer implementation.
# Run from vMAR/ after devtools::load_all(".../vMAR.Core").
if (!exists("vmar_network_spec", mode = "function")) {
  library(vMAR.Core)
}

cases <- list(
  legacy = list(
    widths = c(2, 1, 1),
    activations = c("identity", "identity"),
    x = matrix(c(1, 2), nrow = 1L),
    target = 16,
    theta = c(2, 3, 1, 2, -1)
  ),
  deep = list(
    widths = c(2, 3, 2, 1),
    activations = c("tanh", "sigmoid", "identity"),
    x = matrix(c(-0.3, 0.7, 0.2, -0.5, 0.9, 0.1),
               ncol = 2L, byrow = TRUE),
    target = c(0.2, -0.1, 0.4),
    theta = 0.12 * sin(seq_len(20L))
  )
)

records <- list()
for (case_id in names(cases)) {
  item <- cases[[case_id]]
  spec <- vmar_network_spec(item$widths, item$activations)
  stopifnot(length(item$theta) == spec$parameter_count)
  initial <- vmar_network_loss_gradient(item$x, item$target, spec, item$theta)
  field <- vmar_network_gradient_field(item$x, item$target, spec, eta = 0.2)
  for (integrator in c("euler", "rk4")) {
    advance <- if (integrator == "euler") vmar_euler_step else vmar_rk4_step
    theta <- item$theta
    for (iteration in seq_len(4L)) {
      theta <- advance(field, (iteration - 1L) * 0.05, theta, 0.05)
    }
    final <- vmar_network_loss_gradient(item$x, item$target, spec, theta)
    records[[length(records) + 1L]] <- data.frame(
      case_id = case_id,
      integrator = integrator,
      steps = 4L,
      step = 0.05,
      parameter_index = seq_along(theta),
      theta_initial = item$theta,
      theta_final = theta,
      gradient_initial = initial$gradient,
      loss_initial = initial$loss,
      loss_final = final$loss
    )
  }
}

fixture <- do.call(rbind, records)
stopifnot(nrow(fixture) == 50L, all(is.finite(fixture$theta_final)),
          all(is.finite(fixture$gradient_initial)))
old_digits <- getOption("digits")
options(digits = 17)
utils::write.csv(fixture, "reference/vmar_multilayer_parity.csv",
                 row.names = FALSE, quote = FALSE)
options(digits = old_digits)
cat("G0.7-B R multilayer fixtures generated:", nrow(fixture),
    "coordinate rows; four trajectories.\n")
