# G0.7-B.1: recover the exact 14 historical fixture filenames by re-running
# their original R generators. This does not synthesize or copy C output.
# Run only from vMAR root after devtools::load_all('.../vMAR.Core').
if (!file.exists("CMakeLists.txt") || !dir.exists("reference")) {
  stop("Run this script from the vMAR C23 project root", call. = FALSE)
}

generators <- c(
  vmar_neuron_parity.csv = "generate_R_neuron_fixtures.R",
  vmar_memory_parity.csv = "generate_R_memory_fixtures.R",
  vmar_control_parity.csv = "generate_R_control_fixtures.R",
  vmar_stability_control_audit.csv = "run_R_stability_control_audit.R",
  vmar_robustness_parity.csv = "run_R_robustness_audit.R",
  vmar_activation_parity.csv = "generate_R_activation_fixtures.R",
  vmar_custom_activation_parity.csv = "generate_R_custom_activation_fixtures.R",
  vmar_adaptive_parity.csv = "generate_R_adaptive_fixtures.R",
  vmar_coupling_parity.csv = "run_R_coupling_audit.R",
  vmar_regime_parity.csv = "run_R_regime_study.R",
  vmar_intervention_parity.csv = "run_R_intervention_audit.R",
  vmar_transfer_parity.csv = "run_R_transfer_audit.R",
  vmar_adam_parity.csv = "run_R_adam_audit.R",
  vmar_second_moment_parity.csv = "run_R_second_moment_audit.R"
)

recovery <- data.frame(fixture = names(generators),
                       generator = unname(generators),
                       available_before = file.exists(file.path("reference", names(generators))),
                       attempted = FALSE,
                       recovered = FALSE,
                       error = rep("", length(generators)))

for (i in seq_len(nrow(recovery))) {
  destination <- file.path("reference", recovery$fixture[i])
  if (!file.exists(destination)) {
    recovery$attempted[i] <- TRUE
    generator <- file.path("reference", recovery$generator[i])
    error_text <- tryCatch({
      source(generator, local = new.env(parent = globalenv()))
      ""
    }, error = function(e) conditionMessage(e))
    recovery$error[i] <- error_text
  }
  recovery$recovered[i] <- file.exists(destination)
  if (recovery$recovered[i] && !is.finite(file.info(destination)$size)) {
    recovery$recovered[i] <- FALSE
    recovery$error[i] <- "invalid fixture file size"
  }
}

# Record checksums so the regenerated scientific evidence is traceable.
recovery$md5 <- rep("", nrow(recovery))
for (i in which(recovery$recovered)) {
  filename <- file.path("reference", recovery$fixture[i])
  recovery$md5[i] <- unname(tools::md5sum(filename))
}
utils::write.csv(recovery, "reference/historical_fixture_recovery.csv",
                 row.names = FALSE)
print(recovery[, c("fixture", "available_before", "attempted", "recovered")],
      row.names = FALSE)
cat("HISTORICAL FIXTURES:", sum(recovery$recovered), "/14 available\n")
if (!all(recovery$recovered)) {
  warning("Some historical fixtures remain unavailable. Inspect historical_fixture_recovery.csv; do not claim strict parity.")
}
