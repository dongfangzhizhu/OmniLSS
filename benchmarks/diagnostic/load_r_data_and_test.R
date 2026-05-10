#!/usr/bin/env Rscript
# Generate data and save for Python to use

library(gamlss)
library(gamlss.dist)

set.seed(42)
n <- 100
x1 <- rnorm(n)

# Generate NBI data
mu_true <- exp(1 + 0.5 * x1)
sigma_true <- 0.5
y <- rNBI(n, mu = mu_true, sigma = sigma_true)

# Save data
write.csv(data.frame(y = y, x1 = x1), "scripts/diagnostic/test_data_nbi.csv", row.names = FALSE)

# Fit model in R
cat("Fitting NBI model in R...\n")
model_r <- gamlss(y ~ x1, sigma.formula = ~1, family = NBI(), data = data.frame(y = y, x1 = x1), trace = TRUE)

cat("\n========================================\n")
cat("R Results:\n")
cat("========================================\n")
cat("Global Deviance:", deviance(model_r), "\n")
cat("Mu coefficients:", coef(model_r, "mu"), "\n")
cat("Sigma coefficient:", coef(model_r, "sigma"), "\n")
cat("Sigma fitted value:", exp(coef(model_r, "sigma")), "\n")
cat("========================================\n")

# Save fitted parameters
write.csv(data.frame(
  parameter = c("deviance", "mu_intercept", "mu_x1", "sigma_log", "sigma"),
  value = c(deviance(model_r), coef(model_r, "mu"), coef(model_r, "sigma"), exp(coef(model_r, "sigma")))
), "scripts/diagnostic/r_fitted_params.csv", row.names = FALSE)
