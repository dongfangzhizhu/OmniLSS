#!/usr/bin/env Rscript
# Generate data and fit models in R, save both data and results

library(gamlss)
library(gamlss.dist)

# Test configurations
test_cases <- list(
  list(family = "NBI", n = 100, formula = "y ~ 1", seed = 42),
  list(family = "NBI", n = 100, formula = "y ~ x1", seed = 42),
  list(family = "NBI", n = 100, formula = "y ~ x1 + x2", seed = 42),
  list(family = "NBI", n = 500, formula = "y ~ x1", seed = 43),
  list(family = "NBI", n = 5000, formula = "y ~ x1", seed = 44),
  list(family = "ZAGA", n = 100, formula = "y ~ 1", seed = 45),
  list(family = "ZAGA", n = 100, formula = "y ~ x1", seed = 45),
  list(family = "ZAGA", n = 100, formula = "y ~ x1 + x2", seed = 45),
  list(family = "ZAGA", n = 500, formula = "y ~ x1", seed = 46),
  list(family = "ZAGA", n = 5000, formula = "y ~ x1", seed = 47)
)

results <- data.frame()

for (test_case in test_cases) {
  family_name <- test_case$family
  n <- test_case$n
  formula_str <- test_case$formula
  seed <- test_case$seed
  
  cat("\n", rep("=", 70), "\n", sep = "")
  cat("Testing", family_name, ": n =", n, ", formula =", formula_str, "\n")
  cat(rep("=", 70), "\n", sep = "")
  
  # Generate data
  set.seed(seed)
  x1 <- rnorm(n)
  x2 <- rnorm(n)
  
  if (family_name == "NBI") {
    mu_true <- exp(1 + 0.5 * x1)
    sigma_true <- 0.5
    y <- rNBI(n, mu = mu_true, sigma = sigma_true)
  } else if (family_name == "ZAGA") {
    nu_true <- 0.3
    mu_true <- exp(1 + 0.5 * x1)
    sigma_true <- 0.5
    y <- rZAGA(n, mu = mu_true, sigma = sigma_true, nu = nu_true)
  }
  
  # Save data
  data_file <- paste0("scripts/diagnostic/data_", family_name, "_n", n, "_seed", seed, ".csv")
  write.csv(data.frame(y = y, x1 = x1, x2 = x2), data_file, row.names = FALSE)
  cat("✓ Data saved to", data_file, "\n")
  
  df <- data.frame(y = y, x1 = x1, x2 = x2)
  
  # Fit model
  tryCatch({
    if (family_name == "NBI") {
      model <- gamlss(
        as.formula(formula_str),
        sigma.formula = ~1,
        family = NBI(),
        data = df,
        trace = FALSE
      )
    } else if (family_name == "ZAGA") {
      model <- gamlss(
        as.formula(formula_str),
        sigma.formula = ~1,
        family = ZAGA(),
        data = df,
        trace = FALSE
      )
    }
    
    result <- data.frame(
      family = family_name,
      n = n,
      formula = formula_str,
      seed = seed,
      deviance = deviance(model),
      iterations = model$iter,
      converged = model$converged,
      mu_intercept = coef(model, "mu")[1],
      sigma_log = coef(model, "sigma")[1],
      sigma = exp(coef(model, "sigma")[1]),
      data_file = data_file,
      status = "SUCCESS"
    )
    
    cat("✓ Converged in", model$iter, "iterations\n")
    cat("  Deviance:", deviance(model), "\n")
    cat("  Sigma:", exp(coef(model, "sigma")[1]), "\n")
    
    # Print all mu coefficients
    mu_coefs <- coef(model, "mu")
    for (i in seq_along(mu_coefs)) {
      cat("  Mu coef", i, ":", mu_coefs[i], "\n")
    }
    
    results <- rbind(results, result)
    
  }, error = function(e) {
    cat("✗ Error:", e$message, "\n")
  })
}

# Save results
write.csv(results, "scripts/diagnostic/r_reference_results.csv", row.names = FALSE)
cat("\n✓ R results saved to scripts/diagnostic/r_reference_results.csv\n")

cat("\n", rep("=", 70), "\n", sep = "")
cat("Summary\n")
cat(rep("=", 70), "\n", sep = "")

for (family in c("NBI", "ZAGA")) {
  family_results <- results[results$family == family, ]
  cat("\n", family, ":\n", sep = "")
  cat("  Tests:", nrow(family_results), "\n")
  cat("  Avg iterations:", round(mean(family_results$iterations), 1), "\n")
  cat("  Avg deviance:", round(mean(family_results$deviance), 2), "\n")
}

cat("\n", rep("=", 70), "\n", sep = "")
cat("Test Complete\n")
cat(rep("=", 70), "\n", sep = "")
