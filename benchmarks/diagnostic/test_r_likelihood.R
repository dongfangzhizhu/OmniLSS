library(gamlss.dist)

cat("Testing NBI and ZAGA likelihood formulas\n")
cat("=" , rep("=", 59), "\n", sep="")

# NBI test
y <- 5
mu <- 3
sigma <- 0.5
ll_nbi <- dNBI(y, mu=mu, sigma=sigma, log=TRUE)
dev_nbi <- -2 * ll_nbi
cat("\nNBI: y =", y, ", mu =", mu, ", sigma =", sigma, "\n")
cat("  R log-likelihood =", ll_nbi, "\n")
cat("  R deviance =", dev_nbi, "\n")
cat("  Python deviance = 5.1899002267\n")
cat("  Difference =", dev_nbi - 5.1899002267, "\n")

# ZAGA test (positive value)
y <- 2.5
mu <- 4
sigma <- 0.6
nu <- 0.2
ll_zaga <- dZAGA(y, mu=mu, sigma=sigma, nu=nu, log=TRUE)
dev_zaga <- -2 * ll_zaga
cat("\nZAGA (y>0): y =", y, ", mu =", mu, ", sigma =", sigma, ", nu =", nu, "\n")
cat("  R log-likelihood =", ll_zaga, "\n")
cat("  R deviance =", dev_zaga, "\n")
cat("  Python deviance = 3.6826419251\n")
cat("  Difference =", dev_zaga - 3.6826419251, "\n")

# ZAGA test (zero)
y <- 0
ll_zaga_zero <- dZAGA(y, mu=mu, sigma=sigma, nu=nu, log=TRUE)
dev_zaga_zero <- -2 * ll_zaga_zero
cat("\nZAGA (y=0): y =", y, ", mu =", mu, ", sigma =", sigma, ", nu =", nu, "\n")
cat("  R log-likelihood =", ll_zaga_zero, "\n")
cat("  R deviance =", dev_zaga_zero, "\n")
cat("  Python deviance = 3.2188758249\n")
cat("  Difference =", dev_zaga_zero - 3.2188758249, "\n")

cat("\n" , rep("=", 60), "\n", sep="")
