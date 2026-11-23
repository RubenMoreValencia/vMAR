# G0.7-C: selective mechanism/decision parity oracle.
# Run from vMAR C project root after devtools::load_all(vMAR.Core >= 0.0.16).
stopifnot(exists("vmar_network_mechanism_step", mode = "function"))
spec <- vmar_network_spec(c(2, 3, 2, 1),
                          c("tanh", "sigmoid", "identity"))
x <- matrix(c(-0.3, 0.7, 0.2, -0.5, 0.9, 0.1), ncol = 2L, byrow = TRUE)
y <- c(0.2, -0.1, 0.4)
p <- spec$parameter_count
cases <- expand.grid(method = c("euler", "rk4"),
                     mode = c("M0", "M1", "M2", "M3"),
                     memory_sign = c(-1, 1),
                     stringsAsFactors = FALSE)
rows <- lapply(seq_len(nrow(cases)), function(i) {
  z <- cases[i, ]
  initial <- c(0.12 * sin(seq_len(p)), z$memory_sign * 0.5,
               -0.1, rep(0, p - 2L))
  result <- vmar_network_mechanism_step(x, y, spec, initial,
                                        mode = z$mode, method = z$method,
                                        step = 0.05, coupling = 0.2)
  d <- result$decision
  data.frame(method = z$method, mode = z$mode,
             memory_sign = z$memory_sign,
             theta1 = result$state[1], memory1 = result$state[p+1L],
             loss_before = d$loss_before, loss_after = d$loss_after,
             gradient_norm_before = d$gradient_norm_before,
             memory_norm_before = d$memory_norm_before,
             coupling_term_norm_before = d$coupling_term_norm_before,
             requested_step = d$requested_step,
             effective_step = d$effective_step,
             time_after = result$time, accepted = as.integer(d$accepted),
             attempts = d$attempts, rejected = d$rejected,
             field_evaluations = d$field_evaluations)
})
fixture <- do.call(rbind, rows)
stopifnot(nrow(fixture) == 16L, all(is.finite(fixture$theta1)))
old_digits <- getOption("digits")
options(digits = 17)
utils::write.csv(fixture, "reference/vmar_mechanism_parity.csv",
                 row.names = FALSE, quote = FALSE)
options(digits = old_digits)
cat("G0.7-C mechanism parity fixtures:", nrow(fixture), "cases.\n")
