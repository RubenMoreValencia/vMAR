# G0.5-E.1 — run this file with the working directory set to the vMAR C root.
# Requires vMAR.Core (R) functions to be loaded with library(vMAR.Core)
# or devtools::load_all() in the R package project.
if (!exists("vmar_euler_step", mode = "function") ||
    !exists("vmar_rk4_step", mode = "function")) {
  if (requireNamespace("vMAR.Core", quietly = TRUE)) {
    library(vMAR.Core)
  } else {
    stop("Load vMAR.Core in R first (library(vMAR.Core) or devtools::load_all()).")
  }
}

cases <- list()
add_case <- function(id, method, dimension, time, step, n_steps,
                     lambda, equilibrium, forcing, initial) {
  field <- function(t, state) -lambda * (state - equilibrium) + forcing * t
  advance <- if (method == "euler") vmar_euler_step else vmar_rk4_step
  state <- initial
  for (k in seq_len(n_steps)) {
    state <- advance(field, time + (k - 1) * step, state, step)
  }
  data.frame(
    id = id, integrator = method, field = "relaxation",
    dimension = dimension, time = time, step = step, n_steps = n_steps,
    lambda = lambda, equilibrium = equilibrium, forcing = forcing,
    state1 = initial[1], state2 = if (dimension == 2) initial[2] else 0,
    expected1 = state[1], expected2 = if (dimension == 2) state[2] else 0,
    stringsAsFactors = FALSE)
}

for (h in c(0.2, 0.1, 0.05, 0.025)) {
  n <- as.integer(round(1 / h))
  for (method in c("euler", "rk4")) {
    id <- paste0("ode_decay_h", gsub("\\.", "", sprintf("%.3f", h)),
                 "_", method)
    cases[[length(cases) + 1L]] <- add_case(
      id, method, 1, 0, h, n, 1, 0, 0, 1)
  }
}

for (scenario in c("shifted", "forced", "vector")) {
  for (method in c("euler", "rk4")) {
    args <- switch(scenario,
      shifted = list(1L, 0.3, 0.125, 8L, 1.4, 0.2, 0, 0.8),
      forced = list(1L, 0.2, 0.1, 10L, 0.7, -0.1, 0.25, 0.8),
      vector = list(2L, 0, 0.125, 8L, 0.75, 0.2, 0.3, c(0.8, -0.6)))
    cases[[length(cases) + 1L]] <- do.call(
      add_case, c(list(id = paste0(scenario, "_", method), method = method),
                  stats::setNames(args,
                    c("dimension", "time", "step", "n_steps",
                      "lambda", "equilibrium", "forcing", "initial"))))
  }
}

fixtures <- do.call(rbind, cases)
dir.create("reference", showWarnings = FALSE)
utils::write.csv(fixtures, "reference/vmar_parity.csv",
                 row.names = FALSE, quote = FALSE)
pkg_version <- if (requireNamespace("vMAR.Core", quietly = TRUE)) {
  as.character(utils::packageVersion("vMAR.Core"))
} else "loaded from source"
writeLines(c(paste("R version:", R.version.string),
             paste("vMAR.Core:", pkg_version)),
           "reference/vmar_parity_metadata.txt")
cat("G0.5-E.1 R fixtures generated:", nrow(fixtures), "cases\n")
print(fixtures[, c("id", "integrator", "dimension", "expected1", "expected2")],
      row.names = FALSE, digits = 16)
