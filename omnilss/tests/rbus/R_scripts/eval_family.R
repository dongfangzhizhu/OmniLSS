#!/usr/bin/env Rscript

# Rscript that bridges omnilss RTestBus and the gamlss package
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(gamlss))

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
func_type <- payload$type  # "d", "p", "q"
func_args <- payload$args

# Construct the R function name, e.g. "dNO", "pPO"
target_func <- paste0(func_type, family_name)

if (!exists(target_func, mode = "function")) {
  cat(toJSON(list(error = paste("Function not found:", target_func)), auto_unbox = TRUE))
  quit(status = 1)
}

# Invoke the function element-wise so that families with imperfect vectorisation
# in gamlss.dist still produce the scalar reference values we want.
result <- tryCatch({
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
  values
}, error = function(e) {
  e$message
})

if (is.character(result) && length(result) == 1 && inherits(result, "error")) {
    cat(toJSON(list(error = result), auto_unbox = TRUE))
} else {
    # Ensure vectors are preserved
    cat(toJSON(list(values = as.numeric(result)), auto_unbox = TRUE, digits = 10))
}
