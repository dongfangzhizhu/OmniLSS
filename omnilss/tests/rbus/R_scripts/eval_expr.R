#!/usr/bin/env Rscript

# Generic R expression bridge for consistency tests (D2.156).
#
# Payload: {"expr": "as.numeric(quantile(x, probs=probs, type=7))",
#           "args": {"x": [...], "probs": [...]}}
# The named arguments are bound in a fresh environment, the expression is
# evaluated there, and the numeric result is returned as JSON.

suppressPackageStartupMessages(library(jsonlite))

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

result <- tryCatch({
  env <- new.env(parent = globalenv())
  for (name in names(payload$args)) {
    assign(name, payload$args[[name]], envir = env)
  }
  value <- eval(parse(text = payload$expr), envir = env)
  list(values = as.numeric(value))
}, error = function(e) {
  list(error = e$message)
})

if (!is.null(result$error)) {
  cat(toJSON(list(error = result$error), auto_unbox = TRUE))
} else {
  cat(toJSON(list(values = result$values), auto_unbox = TRUE, digits = 17))
}
