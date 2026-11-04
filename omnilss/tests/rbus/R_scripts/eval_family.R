#!/usr/bin/env Rscript

# Rscript that bridges omnilss RTestBus and the gamlss package
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(gamlss))
suppressPackageStartupMessages(library(gamlss.dist))

args_cli <- commandArgs(trailingOnly = TRUE)
if (length(args_cli) == 0) {
  cat(toJSON(list(error = "No arguments file provided"), auto_unbox = TRUE))
  quit(status = 1)
}

arg_file <- args_cli[1]
if (!file.exists(arg_file)) {
    cat(toJSON(list(error = paste("File not found:", arg_file)), auto_unbox = TRUE))
    quit(status = 1)
}

payload <- fromJSON(arg_file)
family_name <- payload$family
func_type <- payload$type  # "d", "p", "q", or "gdev"
func_args <- payload$args

# Families without a d/p/q in gamlss.dist -- the untruncated Box-Cox group
# (BCTuntr / BCCGuntr / BCPEuntr) is the motivating case -- still define a
# G.dev.incr slot, which is -2 * log d and is all there is to compare against.
use_gdev <- identical(func_type, "gdev")

# Resolve a gamlss.dist function by name, exact match first and
# case-insensitively second.  gamlss.dist spells the "original parameterisation"
# families with a lowercase trailing "o" (PARETO1o, BCCGo, BCPEo, BCTo, JSUo,
# SHASHo, SHASHo2, BEo, EXGAUS, GEOMo, PARETO2o) while the OmniLSS registry
# aliases them with a capital "O".  A plain get(paste0(type, family)) therefore
# failed with "could not find function" for every one of those families -- the
# dpqr gate silently skipped them and the R consistency tests hard-errored,
# neither of which is a statement about the families themselves.
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
  if (length(matches) == 0) {
    stop(sprintf("no function '%s' in gamlss.dist", name))
  }
  stop(sprintf(
    "ambiguous function name '%s' (matches %s)",
    name, paste(matches, collapse = ", ")
  ))
}

# Construct the R function name, e.g. "dNO", "pPO"
# D2.163: resolve the function inside the tryCatch below.  Doing it outside made
# a missing symbol kill the whole script (exit 1, no JSON), so the Python side
# reported a bare CalledProcessError instead of "no function 'dDPO1' in
# gamlss.dist" -- which is exactly the message that tells you the family has a
# different R name.
target_func <- tryCatch(if (use_gdev) {
  fam <- resolve_gamlss_dist_function(family_name)
  fam <- fam()
  if (is.null(fam$G.dev.incr)) stop("no G.dev.incr slot")
  fam$G.dev.incr
} else {
  resolve_gamlss_dist_function(paste0(func_type, family_name))
}, error = function(e) e)

# Invoke the function element-wise so that families with imperfect vectorisation
# in gamlss.dist still produce the scalar reference values we want.
result <- tryCatch({
  if (inherits(target_func, "error")) stop(conditionMessage(target_func))
  arg_lengths <- vapply(func_args, length, integer(1))
  target_len <- max(arg_lengths)
  normalized_args <- lapply(func_args, function(arg) {
    if (length(arg) == target_len) {
      arg
    } else if (length(arg) == 1) {
      rep(arg, target_len)
    } else {
      stop("Arguments are not broadcastable to a common length")
    }
  })

  values <- vapply(seq_len(target_len), function(i) {
    scalar_args <- lapply(normalized_args, function(arg) arg[[i]])
    as.numeric(do.call(target_func, scalar_args))
  }, numeric(1))
  list(values = values)
}, error = function(e) {
  list(error = e$message)
})

if (!is.null(result$error)) {
    cat(toJSON(list(error = result$error), auto_unbox = TRUE))
} else {
    # Ensure vectors are preserved
    cat(toJSON(list(values = as.numeric(result$values)), auto_unbox = TRUE, digits = 10))
}
