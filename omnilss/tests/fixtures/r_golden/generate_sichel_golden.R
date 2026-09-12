#!/usr/bin/env Rscript
# Generate a SICHEL d/p/q golden reference from R gamlss.dist.
#
# The fixture records R values verbatim (including Inf where R's 1-cdf
# cancellation saturates) so the Python test can distinguish "matches R"
# from "R lost the tail to roundoff while Python kept it".
#
# Usage: Rscript generate_sichel_golden.R [output.json]

suppressPackageStartupMessages(library(gamlss.dist))
suppressPackageStartupMessages(library(jsonlite))

stopifnot(packageVersion("gamlss.dist") >= "6.1.1")

counts <- c(0, 1, 10, 10000, 10001, 50000)
quantile_probabilities <- c(0.1, 0.5, 0.9, 0.999)
upper_tail_probability <- 1e-6

parameter_sets <- list(
  list(mu = 1, sigma = 0.5, nu = 0.2),
  list(mu = 10, sigma = 0.1, nu = 1),
  list(mu = 2, sigma = 1, nu = -0.5)
)

entries <- lapply(parameter_sets, function(p) {
  mu <- p$mu
  sigma <- p$sigma
  nu <- p$nu

  density_log <- dSICHEL(counts, mu = mu, sigma = sigma, nu = nu, log = TRUE)
  cdf <- pSICHEL(counts, mu = mu, sigma = sigma, nu = nu,
                 lower.tail = TRUE, log.p = FALSE)
  cdf_log <- pSICHEL(counts, mu = mu, sigma = sigma, nu = nu,
                     lower.tail = TRUE, log.p = TRUE)
  cdf_upper <- pSICHEL(counts, mu = mu, sigma = sigma, nu = nu,
                       lower.tail = FALSE, log.p = FALSE)
  cdf_upper_log <- pSICHEL(counts, mu = mu, sigma = sigma, nu = nu,
                           lower.tail = FALSE, log.p = TRUE)
  quantiles_lower <- qSICHEL(quantile_probabilities, mu = mu, sigma = sigma,
                             nu = nu, lower.tail = TRUE, log.p = FALSE)
  quantiles_upper <- qSICHEL(upper_tail_probability, mu = mu, sigma = sigma,
                             nu = nu, lower.tail = FALSE, log.p = FALSE)

  list(
    mu = mu, sigma = sigma, nu = nu,
    density_log = as.list(density_log),
    cdf = as.list(cdf),
    cdf_log = as.list(cdf_log),
    cdf_upper = as.list(cdf_upper),
    cdf_upper_log = as.list(cdf_upper_log),
    quantile_probabilities = as.list(quantile_probabilities),
    quantiles_lower = as.list(quantiles_lower),
    quantiles_upper = as.list(quantiles_upper)
  )
})

payload <- list(
  family = "SICHEL",
  r_version = R.version.string,
  gamlss_dist_version = as.character(packageVersion("gamlss.dist")),
  counts = as.list(counts),
  parameters = entries
)

args <- commandArgs(trailingOnly = TRUE)
out <- if (length(args) >= 1) args[[1]] else
  file.path("sichel_dpqr_high_counts_20260912.json")

writeLines(toJSON(payload, auto_unbox = TRUE, digits = NA, pretty = TRUE,
                  null = "null"), out)
cat("wrote", out, "\n")
# Console echo for quick human inspection.
for (e in entries) {
  cat("mu=", e$mu, "sigma=", e$sigma, "nu=", e$nu, "\n", sep = "")
  cat("  density_log:", sprintf("%.6g", e$density_log), "\n")
  cat("  cdf        :", sprintf("%.6g", e$cdf), "\n")
  cat("  cdf_up_log :", sprintf("%.6g", e$cdf_upper_log), "\n")
  cat("  q_lower    :", sprintf("%.6g", e$quantiles_lower), "\n")
  cat("  q_upper    :", sprintf("%.6g", e$quantiles_upper), "\n")
}
