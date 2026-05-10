#!/usr/bin/env Rscript
# Trace R iterations in detail

library(gamlss)
library(gamlss.dist)

# Load data
df <- read.csv("scripts/diagnostic/test_data_nbi.csv")
y <- df$y
x1 <- df$x1
n <- length(y)

cat("========================================\n")
cat("Tracing R RS Algorithm for NBI\n")
cat("========================================\n")
cat("Sample size:", n, "\n")
cat("Formula: y ~ x1, sigma.formula = ~1\n")
cat("========================================\n\n")

# Fit with trace=TRUE to see iterations
model_r <- gamlss(y ~ x1, sigma.formula = ~1, family = NBI(), 
                  data = data.frame(y = y, x1 = x1), 
                  trace = TRUE,
                  control = gamlss.control(trace = TRUE),
                  i.control = glim.control(glm.trace = TRUE, cc = 1e-4, cyc = 20))

cat("\n========================================\n")
cat("Final Results:\n")
cat("========================================\n")
cat("Global Deviance:", deviance(model_r), "\n")
cat("Mu coefficients:", coef(model_r, "mu"), "\n")
cat("Sigma coefficient:", coef(model_r, "sigma"), "\n")
cat("Sigma fitted value:", exp(coef(model_r, "sigma")), "\n")
cat("Iterations:", model_r$iter, "\n")
cat("========================================\n")
