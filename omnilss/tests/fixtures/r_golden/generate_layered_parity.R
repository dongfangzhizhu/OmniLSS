# SPDX-License-Identifier: GPL-3.0-or-later
# Live-R reference for the RS / CG / MIXED layered parity (D2.238).
#
# Reuses the golden fitting samples (regenerated/<case>.json) so the three
# routes are compared on identical data, then fits each with R's RS(), CG()
# and mixed(1, 100).  Writes the reference consumed by
# tools/layered_parity.py.
#
# Usage:
#   Rscript omnilss/tests/fixtures/r_golden/generate_layered_parity.R \
#       > docs/development/layered-parity-r-reference-2026-11-15.json
suppressPackageStartupMessages(library(gamlss))
suppressPackageStartupMessages(library(jsonlite))

manifest <- fromJSON("omnilss/tests/fixtures/r_golden/regenerated/manifest.json")
cases <- list()
for (name in names(manifest)) {
  cases[[manifest[[name]]$family]] <- name
}

estimated <- function(famobj) {
  names(famobj$parameters)[unlist(famobj$parameters)]
}

fit_route <- function(fam, famobj, df, expr, label) {
  fit <- try(eval(parse(text = paste0(
    "gamlss(y ~ x, family = famobj, data = df, control = gamlss.control(trace = FALSE), method = ",
    expr, ")"
  ))), silent = TRUE)
  if (inherits(fit, "try-error")) {
    return(list(error = substr(as.character(fit), 1, 160)))
  }
  out <- list(
    dev = as.numeric(tail(fit$G.deviance, 1)),
    iter = as.integer(fit$iter),
    converged = isTRUE(fit$converged),
    label = label
  )
  for (p in estimated(famobj)) {
    cf <- try(as.numeric(coef(fit, p)), silent = TRUE)
    if (!inherits(cf, "try-error") && length(cf) > 0 && all(is.finite(cf))) {
      out[[paste0(p, "_coefficients")]] <- as.list(cf)
    }
  }
  out
}

result <- list()
for (fam in names(cases)) {
  case <- cases[[fam]]
  fixture <- fromJSON(file.path(
    "omnilss/tests/fixtures/r_golden/regenerated", paste0(case, ".json")
  ))
  df <- data.frame(y = fixture$data$y, x = fixture$data$x)
  famobj <- if (fam == "NET") NET() else if (fam == "BEINF") BEINF() else gamlss.family(fam)
  result[[fam]] <- list(
    case = case,
    RS = fit_route(fam, famobj, df, "RS(n.cyc=100)", "RS(n.cyc=100)"),
    CG = fit_route(fam, famobj, df, "CG(n.cyc=100)", "CG(n.cyc=100)"),
    MIXED = fit_route(fam, famobj, df, "mixed(1,100)", "mixed(1,100)")
  )
}

cat(toJSON(result, digits = 17, auto_unbox = TRUE))
