library(gamlss)

set.seed(42)
n <- 100
x1 <- rnorm(n)
mu_true <- exp(1 + 0.5*x1)
sigma_true <- 0.5
y <- rNBI(n, mu=mu_true, sigma=sigma_true)

# Fit with trace to see convergence
cat("Fitting with trace=TRUE to see convergence details:\n")
cat("=" , rep("=", 59), "\n", sep="")

model <- gamlss(y ~ x1, family=NBI(), trace=TRUE, c.crit=0.001)

cat("\n", rep("=", 60), "\n", sep="")
cat("Final results:\n")
cat("Deviance:", deviance(model), "\n")
cat("Sigma:", unique(fitted(model, "sigma")), "\n")
cat("Convergence criterion (c.crit):", model$c.crit, "\n")
cat("Number of iterations:", model$iter, "\n")
