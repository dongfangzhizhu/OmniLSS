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

# --- vendored-source fallback (D2.107) -------------------------------------
# Some families the OmniLSS registry carries are not exported by the
# installed gamlss.dist, but R's own source for them sits in the vendored
# snapshot under R/gamlss.dist/R/.  Sourcing those files makes the same
# gate measurable instead of "no d/p/q counterpart in R".
#
# They are sourced into their own environment, never into globalenv and never
# into the gamlss.dist namespace: the resolver below still prefers the
# installed package, so a vendored file that happens to define a name the
# package also exports (DELAPORT.R defines dDEL) cannot shadow it.
vend_env <- new.env(parent = globalenv())
vend_dir <- normalizePath(file.path(
  dirname(sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])),
  "..", "R", "gamlss.dist", "R"
))
# Only LOGSHASH is wired up so far.  LOGSHASHO is deliberately NOT:
# R dLOGSHASHO is not a density -- it integrates to 0.500 at mu = 0
# (D2.81) -- so sourcing it would turn a clean skip into a fail
# that no honest tolerance can absorb.
vend_files <- c("LOGSHASH.R")
for (vf in vend_files) {
  tryCatch(
    sys.source(file.path(vend_dir, vf), envir = vend_env),
    error = function(e) NULL
  )
}


args_all <- commandArgs(trailingOnly = TRUE)
jobs <- fromJSON(args_all[1], simplifyVector = FALSE)

# Resolve a gamlss.dist function by name, exact match first and
# case-insensitively second.  gamlss.dist spells the "original parameterisation"
# families with a lowercase trailing "o" (PARETO1o, BCCGo, BCPEo, BCTo, JSUo,
# SHASHo, SHASHo2, BEo, EXGAUS, GEOMo, PARETO2o) while the OmniLSS registry
# aliases them with a capital "O".  A plain paste0(type, family) lookup
# therefore failed for every one of those families -- the same defect the
# RTestBus bridge (omnilss/tests/rbus/R_scripts/eval_family.R) had.  Both
# bridges share this resolver's contract.
resolve_gamlss_dist_function <- function(name) {
  ns <- asNamespace("gamlss.dist")
  exports <- getNamespaceExports(ns)
  if (name %in% exports) {
    return(get(name, mode = "function", envir = ns))
  }
  matches <- exports[tolower(exports) == tolower(name)]
  if (length(matches) == 1) {
    return(get(matches[[1]], mode = "function", envir = ns))
  }
  # D2.107: not exported by the installed package -- fall back to the vendored
  # snapshot, which carries R source for a few families the package omits
  # (LOGSHASH, LOGSHASHo, DELAPORT).  The package always wins when both have
  # the name, so nothing that already measured against the package changes.
  if (exists(name, envir = vend_env, mode = "function")) {
    return(get(name, envir = vend_env, mode = "function"))
  }
  vend_names <- ls(vend_env)
  vend_matches <- vend_names[tolower(vend_names) == tolower(name)]
  if (length(vend_matches) == 1) {
    return(get(vend_matches[[1]], envir = vend_env, mode = "function"))
  }
  if (length(matches) == 0 && length(vend_matches) == 0) {
    stop(sprintf("no function '%s' in gamlss.dist", name))
  }
  stop(sprintf(
    "ambiguous function name '%s' (matches %s)",
    name, paste(matches, collapse = ", ")
  ))
}

run_dpqr <- function(payload) {
  fname <- paste0(payload$type, payload$family)
  target <- tryCatch(
    resolve_gamlss_dist_function(fname),
    error = function(e) NULL
  )
  if (is.null(target)) {
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
      as.numeric(do.call(target, aa))
    })
    list(values = as.numeric(out))
  }, error = function(e) list(error = conditionMessage(e)))
}

run_fit <- function(payload) {
  df <- as.data.frame(lapply(payload$data, function(v) as.numeric(unlist(v))))
  tryCatch({
    fam <- resolve_gamlss_dist_function(payload$family)
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
    # Warnings are recorded, not fatal: R emits "Algorithm RS has not yet
    # converged" for any fit that hits its cycle budget, and the fit still
    # returns a deviance worth comparing.  Turning that warning into an error
    # made every such family an unmeasurable rsdev skip (BCTO in D2.58).
    fit_warnings <- character(0)
    fit <- withCallingHandlers(
      gamlss(
        form, family = fam, data = df,
        sigma.formula = sigma_form,
        control = gamlss.control(n.cyc = as.numeric(payload$n_cyc), trace = FALSE)
      ),
      warning = function(w) {
        fit_warnings <<- c(fit_warnings, conditionMessage(w))
        invokeRestart("muffleWarning")
      }
    )
    coefs <- lapply(fit$parameters, function(p) {
      as.numeric(unlist(fit[[paste0(p, ".coefficients")]]))
    })
    names(coefs) <- fit$parameters
    list(
      deviance = as.numeric(fit$G.deviance),
      converged = as.logical(fit$converged),
      coefficients = coefs,
      warnings = if (length(fit_warnings)) fit_warnings else NULL
    )
  }, error = function(e) list(error = conditionMessage(e)))
}

res <- lapply(jobs, function(p) {
  if (identical(p$kind, "fit")) run_fit(p) else run_dpqr(p)
})
cat(toJSON(res, auto_unbox = TRUE, digits = 15))
