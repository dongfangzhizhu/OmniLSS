"""R reference scripts for the surrounding-package consistency tests (D2.265/266).

Kept out of the `test_*.py` modules on purpose: the symbol-coverage index scans
test files for identifiers, so an R script embedded in a test would make every
name inside it look like evidence for that symbol (for example trun.r, which is
sourced only so gen.trun can generate its random function).
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

CENS_R_SCRIPT = """
suppressPackageStartupMessages(library(gamlss.dist))
suppressPackageStartupMessages(library(survival))
suppressPackageStartupMessages(library(jsonlite))
source("R/gamlss.cens/R/cens.R")
source("R/gamlss.cens/R/cens.d.R")
source("R/gamlss.cens/R/cens.p.R")
source("R/gamlss.cens/R/gen.cens.R")
args <- commandArgs(trailingOnly = TRUE)
d <- fromJSON(args[1], simplifyDataFrame = FALSE)
make_surv <- function(cse, type) {
  if (type == "interval") {
    s <- cbind(cse$time, cse$time2, cse$status)
    colnames(s) <- c("t1", "t2", "status")
  } else {
    s <- cbind(cse$time, cse$status)
    colnames(s) <- c("time", "status")
  }
  class(s) <- "Surv"
  s
}
out <- list()
for (type in c("right", "left", "interval")) {
  cse <- d$cases[[type]]
  s <- make_surv(cse, type)
  out[[type]] <- list(
    d = as.numeric(cens.d(family = "NO", type = type)(s, mu = cse$mu, sigma = cse$sigma, log = TRUE)),
    p = as.numeric(cens.p(family = "NO", type = type)(s, mu = cse$mu, sigma = cse$sigma))
  )
}
out$q <- as.numeric(cens.q(family = "NO")(d$probs, mu = 0, sigma = 1))
gen.cens("NO", "right")
sgen <- make_surv(list(time = d$gen_time, status = d$gen_status), "right")
out$gen <- list(
  d = as.numeric(dNOrc(sgen, mu = d$gen_mu, sigma = d$gen_sigma, log = TRUE)),
  p = as.numeric(pNOrc(sgen, mu = d$gen_mu, sigma = d$gen_sigma))
)
cat(toJSON(out, digits = 17, auto_unbox = TRUE))
"""

TR_R_SCRIPT = """
suppressPackageStartupMessages(library(gamlss.dist))
suppressPackageStartupMessages(library(jsonlite))
source("R/gamlss.tr/R/trun.d-30-03-24.R")
source("R/gamlss.tr/R/trun.p-30-03-24.R")
source("R/gamlss.tr/R/trun.q-27-12-12.R")
source("R/gamlss.tr/R/trun.r-28-12-12.R")
source("R/gamlss.tr/R/trun.R")
source("R/gamlss.tr/R/gen.trun-30-04-24.R")
args <- commandArgs(trailingOnly = TRUE)
d <- fromJSON(args[1])
xg <- d$x
mu <- d$mu
sigma <- d$sigma
probs <- d$probs
gen.trun(par = 0, family = "NO", type = "left", print = FALSE)
out <- list()
for (type in c("left", "right", "both")) {
  par <- if (type == "both") c(-0.5, 1.5) else 0
  entry <- list(
    d = as.numeric(trun.d(par, family = "NO", type = type)(xg, mu = mu, sigma = sigma)),
    p = as.numeric(trun.p(par, family = "NO", type = type)(xg, mu = mu, sigma = sigma)),
    q = as.numeric(trun.q(par, family = "NO", type = type)(
          probs, mu = mu[1:length(probs)], sigma = sigma[1:length(probs)]))
  )
  if (type == "left") {
    fam <- trun(par, family = "NO", type = type)()
    entry$fam_dev <- as.numeric(fam$G.dev.incr(xg, mu = mu, sigma = sigma))
  }
  out[[type]] <- entry
}
out$gen <- list(
  d = as.numeric(dNOtr(xg, mu = mu, sigma = sigma)),
  p = as.numeric(pNOtr(xg, mu = mu, sigma = sigma)),
  q = as.numeric(qNOtr(probs, mu = mu[1:length(probs)], sigma = sigma[1:length(probs)]))
)
cat(toJSON(out, digits = 17, auto_unbox = TRUE, na = "null"))
"""

__all__ = ["MX_R_SCRIPT", "CENS_R_SCRIPT", "TR_R_SCRIPT"]
