# G0.8-C.1: independent R reference for the C23 2->2->1 tanh Hessian.
# Run from the vMAR root with the vMAR.Core development package loaded.
if (!exists("vmar_pilot_hessian", mode = "function")) {
  source("reference/run_R_curvature_pilot.R", local = FALSE)
}

vmar_generate_R_curvature_parity <- function(
    file = "reference/vmar_curvature_parity.csv") {
  spec <- vMAR.Core::vmar_network_spec(c(2, 2, 1), c("tanh", "tanh"))
  w <- list(matrix(c(0.3, -0.2, -0.4, 0.5), 2, 2, byrow = TRUE),
            matrix(c(0.6, -0.7), 1, 2))
  theta <- vMAR.Core::vmar_network_pack(
    spec, w, list(c(0.05, -0.1), 0.03))
  x <- matrix(c(-1.0, -0.5, -0.6, 0.2, -0.2, 0.8,
                0.3, -0.3, 0.7, 0.6, 1.0, -0.8), ncol = 2, byrow = TRUE)
  target <- c(-0.6, -0.3, 0.1, 0.2, 0.5, 0.65)
  field <- vMAR.Core::vmar_network_gradient_field(x, target, spec, eta = 1)
  loss <- function(th) vMAR.Core::vmar_network_loss_gradient(
    x, target, spec, th)$loss
  records <- list()
  n <- 0L
  for (method in c("euler", "rk4")) {
    trajectory <- vMAR.Core::vmar_integrate(
      field, 0, theta, 0.05, 20L, method)
    for (step_id in c(0L, 10L, 20L)) {
      confirmed <- as.numeric(trajectory$state[step_id + 1L, ])
      for (h in c(1e-3, 1e-4)) {
        estimated <- vmar_pilot_hessian(loss, confirmed, h)
        stopifnot(estimated$loss_calls == 180L)
        n <- n + 1L
        ij <- expand.grid(row = 0:8, col = 0:8)
        records[[n]] <- data.frame(
          method = method, step = step_id, h = h,
          row = ij$row, col = ij$col,
          expected = estimated$hessian[cbind(ij$row + 1L, ij$col + 1L)])
      }
    }
  }
  result <- do.call(rbind, records)
  stopifnot(nrow(result) == 972L, all(is.finite(result$expected)))
  output <- result
  output$expected <- sprintf("%.17g", result$expected)
  utils::write.csv(output, file, row.names = FALSE, quote = FALSE)
  cat("G0.8-C.1 independent R Hessian fixture: ", nrow(result),
      " coordinate rows; 12 confirmed cases.\n", sep = "")
  invisible(result)
}
