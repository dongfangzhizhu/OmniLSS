#!/usr/bin/env Rscript
# Batch parity evaluator: one R process, many families.
#
# Usage:  Rscript tools/r_parity_batch.R jobs.json
#
# The job list is passed as a **file path**, not inline JSON: a full-family
# scan produces a payload far beyond the Windows command-line limit
# (WinError 206, "file name or extension is too long").
# Jobs:   [{kind, family, type, argnames, args, lower_tail, log_p, ...}, ...]
#         kind = "dpqr"  -> call <type><family>() in gamlss.dist
#         kind = "fit"   -> call gamlss() and report deviance + coefficients
# Output: JSON array, one entry per job: {values:[...]} | {deviance, coefficients}
#                                       | {error:"..."}
#
# Keeping every family inside a single R process is deliberate: spawning
# Rscript per family costs ~2s each, which makes a full 100+ family scan
# impractical.

suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(gamlss))
suppressPackageStartupMessages(library(gamlss.dist))

args_all <- commandArgs(trailingOnly = TRUE)
jobs <- fromJSON(args_all[1], simplifyVector = FALSE)

run_dpqr <- function(payload) {
  fname <- paste0(payload$type, payload$family)
  if (!exists(fname, mode = "function")) {
    return(list(error = paste("missing function", fname)))
  }
  argnames <- unlist(payload$argnames)
  vecs <- lapply(payload$args, function(v) as.numeric(unlist(v)))
  n <- max(sapply(vecs, length))
  lower_tail <- if (is.null(payload$lower_tail)) TRUE else as.logical(payload$lower_tail)
  log_p <- if (is.null(payload$log_p)) FALSE else as.logical(payload$log_p)
  tryCatch({
    out <- sapply(seq_len(n), function(i) {
      aa <- lapply(vecs, function(v) if (length(v) == 1) v else v[i])
      names(aa) <- argnames
      if (payload$type %in% c("p", "q")) {
        aa[["lower.tail"]] <- lower_tail
        aa[["log.p"]] <- log_p
      } else if (payload$type == "d") {
        aa[["log"]] <- log_p
      }
      as.numeric(do.call(fname, aa))
    })
    list(values = as.numeric(out))
  }, error = function(e) list(error = conditionMessage(e)))
}

run_fit <- function(payload) {
  df <- as.data.frame(lapply(payload$data, function(v) as.numeric(unlist(v))))
  tryCatch({
    fam <- get(payload$family, mode = "function")
    # Binomial response convention (D2.22).  R's gamlss() rejects a binomial
    # family outright unless the response is the two-column
    # cbind(success, failure) matrix:
    #
    #   Error in gamlss(y ~ x, family = BI(), data = dfb, ...) :
    #     y values must be 0 <= y <= 1
    #
    # even though BI()$y.valid is `all(y >= 0)`.  The scanner therefore emits
    # `fail` alongside `y` for these families, and the formula is rewritten
    # here.  `payload$response_formula` is set by the Python side; older
    # payloads without it keep the single-column path.
    form_text <- if (!is.null(payload$response_formula)) {
      payload$response_formula
    } else {
      payload$formula
    }
    form <- as.formula(form_text)
    sigma_form <- if (is.null(payload$sigma_formula)) ~1 else as.formula(payload$sigma_formula)
    fit <- gamlss(
      form, family = fam, data = df,
      sigma.formula = sigma_form,
      control = gamlss.control(n.cyc = as.numeric(payload$n_cyc), trace = FALSE)
    )
    coefs <- lapply(fit$parameters, function(p) {
      as.numeric(unlist(fit[[paste0(p, ".coefficients")]]))
    })
    names(coefs) <- fit$parameters
    list(
      deviance = as.numeric(fit$G.deviance),
      converged = as.logical(fit$converged),
      coefficients = coefs
    )
  }, error = function(e) list(error = conditionMessage(e)),
     warning = function(w) list(error = paste("warning:", conditionMessage(w))))
}

res <- lapply(jobs, function(p) {
  if (identical(p$kind, "fit")) run_fit(p) else run_dpqr(p)
})
cat(toJSON(res, auto_unbox = TRUE, digits = 15))
