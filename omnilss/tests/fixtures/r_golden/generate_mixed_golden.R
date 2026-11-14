# SPDX-License-Identifier: GPL-3.0-or-later
# Regenerate mixed_2026-11-14.json against live R:
#   Rscript omnilss/tests/fixtures/r_golden/generate_mixed_golden.R \
#       > omnilss/tests/fixtures/r_golden/mixed_2026-11-14.json
#
# R's mixed(n1, n2) is a *local* function inside gamlss() (gamlss-5.R:163):
#   mixed <- function(n1=1, n2=20) { RS(n.cyc=n1); CG(n.cyc=n2) }
# It is NOT exported, so the method must be passed as a literal call
# (substitute(method) pmatch'es "mixed()").  We build the call as text.
suppressPackageStartupMessages(library(gamlss))
suppressPackageStartupMessages(library(jsonlite))
set.seed(1)
n <- 80L
x <- seq(-1, 1, length.out = n)
y <- 1 + 1.5 * x + rnorm(n, sd = 0.6)
d <- data.frame(y = y, x = x)
get <- function(expr) {
  f <- eval(parse(text = paste0(
    "gamlss(y~x, sigma.formula=~x, family=NO(), data=d, method=",
    expr, ", n.cyc=30, trace=FALSE)")))
  list(
    G.dev = unname(as.numeric(tail(f$G.deviance, 1))[1]),
    mu = unname(as.numeric(coef(f, "mu"))),
    sigma = unname(as.numeric(coef(f, "sigma"))),
    mu.fv = unname(as.numeric(fitted(f, "mu"))),
    sigma.fv = unname(as.numeric(fitted(f, "sigma")))
  )
}
out <- list(
  meta = list(
    r_version = R.version.string,
    gamlss_version = as.character(packageVersion("gamlss")),
    gamlss_dist_version = as.character(packageVersion("gamlss.dist")),
    seed = 1L, n = n
  ),
  data = list(y = as.numeric(y), x = as.numeric(x)),
  mixed_1_4 = get("mixed(1,4)"),
  mixed_2_3 = get("mixed(2,3)"),
  mixed_1_0 = get("mixed(1,0)"),
  RS = get("RS()")
)
cat(toJSON(out, digits = 17, auto_unbox = TRUE))
