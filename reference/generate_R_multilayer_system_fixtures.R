# G0.7-B.1 R oracle for coupled multilayer memory and control in C23.
# Run from vMAR root after loading vMAR.Core >= 0.0.14 development sources.
if (!exists("vmar_network_controlled_step", mode = "function")) {
  stop("Load the G0.7-B.1 vMAR.Core development package first")
}
spec <- vmar_network_spec(c(2, 3, 2, 1),
                          c("tanh", "sigmoid", "identity"))
x <- matrix(c(-0.3, 0.7, 0.2, -0.5, 0.9, 0.1), ncol = 2, byrow = TRUE)
target <- c(0.2, -0.1, 0.4)
p <- spec$parameter_count
initial <- c(0.12 * sin(seq_len(p)), c(0.5, -0.1, rep(0, p - 2L)))
rows <- list()
for (method in c("euler", "rk4")) {
  for (coupling in c(0, 0.2)) {
    for (step in c(0.05, 4)) {
      result <- vmar_network_controlled_step(
        x, target, spec, initial, method = method,
        eta = 0.2, tau = 0.7, coupling = coupling,
        initial_step = step, reduction = 0.5, max_attempts = 4L,
        memory_threshold = 0.2)
      report <- result$report
      rows[[length(rows) + 1L]] <- data.frame(
        case_id = paste(method, coupling, step, sep = "_"),
        method = method, coupling = coupling, proposed_step = step,
        coordinate = seq_along(initial), initial = initial,
        final = result$state, final_time = result$time,
        accepted = as.integer(report$accepted), attempts = report$attempts,
        rejected = report$rejected, effective_step = report$effective_step,
        old_loss = report$old_loss, candidate_loss = report$candidate_loss)
    }
  }
}
fixtures <- do.call(rbind, rows)
stopifnot(nrow(fixtures) == 320L, all(is.finite(fixtures$final)))
old_digits <- getOption("digits")
options(digits = 17)
utils::write.csv(fixtures, "reference/vmar_multilayer_system_parity.csv",
                 row.names = FALSE, quote = FALSE)
options(digits = old_digits)
cat("G0.7-B.1 R fixtures generated:", nrow(fixtures),
    "coordinates across", length(rows), "controlled cases.\n")
