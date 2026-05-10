#!/usr/bin/env Rscript
# Generate R reference results for NBI and ZAGA

library(gamlss)
library(gamlss.dist)

# Test configurations
n_values <- c(100, 500, 5000)
formulas <- list(
  list(mu = "y ~ 1", sigma = "~1"),
  list(mu = "y ~ x1", sigma = "~1"),
  list(mu = "y ~ x1 + x2", sigma = "~1")
)

results <- data.frame()

test_distribution <- function(family_name, n_values, formulas, seed = 42) {
  for (n in n_values) {
    for (formula_spec in formulas) {
      cat("\n", rep("=", 70), "\n", sep = "")
      cat("Testing", family_name, ": n =", n, ", formula =", formula_spec$mu, "\n")
      cat(rep("=", 70), "\n", sep = "")
      
      # Generate data
      set.seed(seed)
      x1 <- rnorm(n)
      x2 <- rnorm(n)
      
      if (family_name == "NBI") {
        # Generate NBI data
        mu_true <- exp(1 + 0.5 * x1)
        sigma_true <- 0.5
        y <- rNBI(n, mu = mu_true, sigma = sigma_true)
      } else if (family_name == "ZAGA") {
        # Generate ZAGA data
        nu_true <- 0.3  # probability of zero
        mu_true <- exp(1 + 0.5 * x1)
        sigma_true <- 0.5
        y <- rZAGA(n, mu = mu_true, sigma = sigma_true, nu = nu_true)
      }
      
      df <- data.frame(y = y, x1 = x1, x2 = x2)
      
      # Fit model
      start_time <- Sys.time()
      tryCatch({
        if (family_name == "NBI") {
          model <- gamlss(
            as.formula(formula_spec$mu),
            sigma.formula = as.formula(formula_spec$sigma),
            family = NBI(),
            data = df,
            trace = FALSE
          )
        } else if (family_name == "ZAGA") {
          model <- gamlss(
            as.formula(formula_spec$mu),
            sigma.formula = as.formula(formula_spec$sigma),
            family = ZAGA(),
            data = df,
            trace = FALSE
          )
        }
        
        fit_time <- as.numeric(difftime(Sys.time(), start_time, units = "secs"))
        
        result <- data.frame(
          family = family_name,
          n = n,
          formula = formula_spec$mu,
          deviance = deviance(model),
          iterations = model$iter,
          converged = model$converged,
          fit_time = fit_time,
          mu_intercept = coef(model, "mu")[1],
          sigma = exp(coef(model, "sigma")[1]),
          status = "SUCCESS"
        )
        
        cat("✓ Converged in", model$iter, "iterations\n")
        cat("  Deviance:", deviance(model), "\n")
        cat("  Fit time:", fit_time, "s\n")
        cat("  Sigma:", exp(coef(model, "sigma")[1]), "\n")
        
        results <<- rbind(results, result)
        
      }, error = function(e) {
        result <- data.frame(
          family = family_name,
          n = n,
          formula = formula_spec$mu,
          deviance = NA,
          iterations = NA,
          converged = FALSE,
          fit_time = NA,
          mu_intercept = NA,
          sigma = NA,
          status = paste("ERROR:", e$message)
        )
        
        cat("✗ Error:", e$message, "\n")
        results <<- rbind(results, result)
      })
    }
  }
}

cat("\n", rep("=", 70), "\n", sep = "")
cat("Comprehensive NBI and ZAGA Test in R\n")
cat(rep("=", 70), "\n", sep = "")

# Test NBI
cat("\n", rep("=", 70), "\n", sep = "")
cat("Testing NBI Distribution\n")
cat(rep("=", 70), "\n", sep = "")
test_distribution("NBI", n_values, formulas, seed = 42)

# Test ZAGA
cat("\n", rep("=", 70), "\n", sep = "")
cat("Testing ZAGA Distribution\n")
cat(rep("=", 70), "\n", sep = "")
test_distribution("ZAGA", n_values, formulas, seed = 42)

# Save results
write.csv(results, "scripts/diagnostic/r_nbi_zaga_results.csv", row.names = FALSE)
cat("\n✓ Results saved to scripts/diagnostic/r_nbi_zaga_results.csv\n")

# Summary
cat("\n", rep("=", 70), "\n", sep = "")
cat("Summary\n")
cat(rep("=", 70), "\n", sep = "")

for (family in c("NBI", "ZAGA")) {
  family_results <- results[results$family == family, ]
  success_count <- sum(family_results$status == "SUCCESS")
  total_count <- nrow(family_results)
  
  cat("\n", family, ":\n", sep = "")
  cat("  Success:", success_count, "/", total_count, "\n")
  
  if (success_count > 0) {
    avg_iterations <- mean(family_results$iterations[!is.na(family_results$iterations)])
    avg_time <- mean(family_results$fit_time[!is.na(family_results$fit_time)])
    cat("  Avg iterations:", round(avg_iterations, 1), "\n")
    cat("  Avg fit time:", round(avg_time, 3), "s\n")
  }
}

cat("\n", rep("=", 70), "\n", sep = "")
cat("Test Complete\n")
cat(rep("=", 70), "\n", sep = "")
