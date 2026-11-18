"""R reference scripts for the surrounding-package consistency tests (D2.265).

Kept out of the `test_*.py` modules on purpose: the symbol-coverage index scans
test files for identifiers, so an R script embedded in a test would make every
name inside it look like evidence for that symbol (for example MX.control, which
is only used to configure the reference fit, not compared).
"""
from __future__ import annotations

MX_R_SCRIPT = """
suppressPackageStartupMessages(library(gamlss))
suppressPackageStartupMessages(library(jsonlite))
source("R/gamlss.mx/R/gamlssMX-10-11-07.R")
source("R/gamlss.mx/R/gamlssNP.R")
args <- commandArgs(trailingOnly = TRUE)
d <- fromJSON(args[1])
df <- data.frame(y = d$y, x = d$x)
fit <- gamlssMX(y ~ x, family = "NO", K = 2, data = df,
                control = MX.control(seed = 1, n.cyc = 100, trace = FALSE, plot = FALSE))
np <- try(gamlssNP(y ~ x, random = ~1, family = NO(), K = 2, data = df,
                   control = NP.control(EMn.cyc = 60, trace = FALSE, plot.opt = 0)),
          silent = TRUE)
grid <- c(-1, 0, 1)
dvals <- dMX(grid, mu = list(rep(0, 3), rep(3, 3)),
             sigma = list(rep(1, 3), rep(1, 3)), pi = list(0.6, 0.4),
             family = list("NO", "NO"))
out <- list(
  mx = list(dev = fit$G.deviance, prob = as.numeric(fit$prob),
            mu1 = as.numeric(coef(fit$models[[1]], "mu")),
            mu2 = as.numeric(coef(fit$models[[2]], "mu")),
            sigma = c(as.numeric(fit$models[[1]]$sigma.coefficients),
                      as.numeric(fit$models[[2]]$sigma.coefficients))),
  dmx = as.numeric(dvals)
)
if (!inherits(np, "try-error")) {
  out$np <- list(dev = np$G.deviance, prob = as.numeric(np$prob),
                 mass = as.numeric(np$mass.points),
                 mu = as.numeric(coef(np, "mu")),
                 sigma = as.numeric(coef(np, "sigma")),
                 emiter = np$EMiter)
}
cat(toJSON(out, digits = 17, auto_unbox = TRUE))
"""

__all__ = ["MX_R_SCRIPT"]
