# G0.7-B.2: pre-registered, fixed-horizon causal contrasts.
# Run from vMAR root after loading the current vMAR.Core development package.
stopifnot(exists("vmar_network_controlled_step", mode = "function"))
spec <- vmar_network_spec(c(2, 3, 2, 1),
                          c("tanh", "sigmoid", "identity"))
x <- matrix(c(-0.3, 0.7, 0.2, -0.5, 0.9, 0.1), ncol = 2, byrow = TRUE)
target <- c(0.2, -0.1, 0.4)
p <- spec$parameter_count
base <- 0.12 * sin(seq_len(p))
T <- 0.5
h <- 0.05

evaluate <- function(method, policy, perturb, memory_sign) {
  theta <- base + perturb
  initial_memory <- c(memory_sign * 0.5, rep(0, p - 1L))
  state <- c(theta, initial_memory)
  initial_loss <- vmar_network_loss_gradient(x, target, spec, theta)$loss
  time <- 0
  accepted <- rejected <- attempts <- evaluations <- 0L
  reduction_sum <- 0
  advance <- if (method == "euler") vmar_euler_step else vmar_rk4_step
  coupled <- if (policy %in% c("M2", "M3")) 0.2 else 0
  field <- vmar_network_system_field(x, target, spec, eta = 0.2,
                                     tau = 0.7, coupling = coupled)
  if (policy == "M0") {
    field_theta <- vmar_network_gradient_field(x, target, spec, eta = 0.2)
  }
  while (time < T - 1e-12) {
    step <- min(h, T - time)
    if (policy == "M3") {
      result <- vmar_network_controlled_step(
        x, target, spec, state, time, accepted, method,
        eta = 0.2, tau = 0.7, coupling = 0.2,
        initial_step = step, reduction = 0.5,
        max_attempts = 4L, memory_threshold = 0.2)
      r <- result$report
      attempts <- attempts + r$attempts
      rejected <- rejected + r$rejected
      evaluations <- evaluations + r$field_evaluations
      reduction_sum <- reduction_sum + as.integer(r$effective_step < step)
      if (!r$accepted) break
      state <- result$state
      time <- result$time
    } else {
      if (policy == "M0") {
        state[seq_len(p)] <- advance(field_theta, time,
                                     state[seq_len(p)], step)
      } else {
        state <- advance(field, time, state, step)
      }
      evaluations <- evaluations + if (method == "euler") 1L else 4L
      attempts <- attempts + 1L
      time <- time + step
    }
    accepted <- accepted + 1L
    if (accepted > 500L) stop("controlled horizon could not be reached")
  }
  final_loss <- vmar_network_loss_gradient(x, target, spec,
                                           state[seq_len(p)])$loss
  data.frame(method = method, policy = policy, perturb = perturb,
             memory_sign = memory_sign, initial_loss = initial_loss,
             final_loss = final_loss, final_time = time,
             accepted = accepted, attempts = attempts, rejected = rejected,
             field_evaluations = evaluations,
             interventions = reduction_sum,
             theta1 = state[1], memory1 = state[p+1L],
             reached_target = abs(T - time) <= 1e-10)
}

cases <- expand.grid(method = c("euler", "rk4"),
                     policy = c("M0", "M1", "M2", "M3"),
                     perturb = c(-0.05, 0.05),
                     memory_sign = c(-1, 1),
                     stringsAsFactors = FALSE)
results <- do.call(rbind, lapply(seq_len(nrow(cases)), function(i) {
  z <- cases[i, ]
  evaluate(z$method, z$policy, z$perturb, z$memory_sign)
}))
stopifnot(nrow(results) == 32L, all(is.finite(results$final_loss)))

# M0 theta is independent of observational memory; M1 theta must agree.
reference <- subset(results, policy == "M0")
observed <- subset(results, policy == "M1")
stopifnot(isTRUE(all.equal(reference$theta1, observed$theta1,
                           tolerance = 1e-11)),
          isTRUE(all.equal(reference$final_loss, observed$final_loss,
                           tolerance = 1e-11)))

# Across-policy loss contrasts are only interpretable at the same horizon.
old_digits <- getOption("digits")
options(digits = 17)
utils::write.csv(results, "reference/vmar_multilayer_value_audit.csv",
                 quote = FALSE, row.names = FALSE)
options(digits = old_digits)
cat("G0.7-B.2:", nrow(results), "fixed-horizon policy trajectories;",
    sum(results$reached_target), "reached target.\n")
print(aggregate(cbind(final_loss, field_evaluations, rejected,
                      interventions) ~ method + policy, data = results,
                FUN = mean))
pair_keys <- c("method", "perturb", "memory_sign")
compare_to_M1 <- function(policy) {
  base_case <- subset(results, policy == "M1")
  candidate <- results[results$policy == policy, , drop = FALSE]
  paired <- merge(base_case, candidate, by = pair_keys,
                  suffixes = c(".M1", paste0(".", policy)))
  candidate_loss <- paired[[paste0("final_loss.", policy)]]
  candidate_cost <- paired[[paste0("field_evaluations.", policy)]]
  data.frame(method = paired$method, perturb = paired$perturb,
             memory_sign = paired$memory_sign,
             policy = policy,
             delta_loss_vs_M1 = candidate_loss - paired$final_loss.M1,
             delta_field_calls_vs_M1 = candidate_cost -
               paired$field_evaluations.M1,
             same_time = abs(paired[[paste0("final_time.", policy)]] -
                             paired$final_time.M1) <= 1e-10)
}
paired <- rbind(compare_to_M1("M2"), compare_to_M1("M3"))
utils::write.csv(paired, "reference/vmar_multilayer_value_contrasts.csv",
                 quote = FALSE, row.names = FALSE)
cat("Paired contrasts:", nrow(paired),
    "| positive benefit:", sum(paired$delta_loss_vs_M1 < -1e-10),
    "| ties:", sum(abs(paired$delta_loss_vs_M1) <= 1e-10),
    "| worse:", sum(paired$delta_loss_vs_M1 > 1e-10), "\n")
cat("Scientific gate: compare loss only where reached_target is TRUE;",
    "field-evaluation counts are costs, not an equal-budget design.\n")
