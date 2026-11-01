#!/usr/bin/env Rscript
# Dump each gamlss.dist family's response-support predicate (family$y.valid).
#
# Why: R's gamlss() aborts before fitting when
#     if (!family$y.valid(y)) stop("response variable out of range")
# (gamlss/R/gamlss-5.R).  OmniLSS mirrors that check in
# omnilss/src/omnilss/response_domain.py, and the table there may only contain
# predicates R itself declares.  This script produces the authoritative dump
# that tools/generate_response_domain_golden.py pins as a golden artifact.
#
# Usage: Rscript tools/r_y_valid_dump.R <out.json>
#
# Output: {"meta": {...}, "families": {"<R family>": "<deparse of y.valid>"}}
# Families without y.valid are recorded as "<none>".

args <- commandArgs(trailingOnly = TRUE)
if (length(args) < 1L) stop("usage: Rscript tools/r_y_valid_dump.R <out.json>")
out_path <- args[[1L]]

suppressMessages(library(gamlss.dist))
suppressMessages(library(jsonlite))

ns <- asNamespace("gamlss.dist")
families <- list()

for (nm in sort(getNamespaceExports(ns))) {
  f <- tryCatch(get(nm, envir = ns), error = function(e) NULL)
  if (!is.function(f)) next
  # Several constructors print "A log family ... has been generated" to stdout
  # and one of them calls cat(); swallow both so only the file matters.
  fam <- tryCatch(
    suppressWarnings(suppressMessages(utils::capture.output(famobj <- f()))),
    error = function(e) NULL
  )
  if (!exists("famobj", inherits = FALSE)) next
  fam <- famobj
  rm(famobj)
  if (!is.list(fam) || is.null(fam$family)) next
  families[[nm]] <- if (is.null(fam$y.valid)) {
    "<none>"
  } else {
    paste(deparse(body(fam$y.valid)), collapse = " ")
  }
}

meta <- list(
  r_version = R.version.string,
  gamlss = as.character(utils::packageVersion("gamlss")),
  gamlss_dist = as.character(utils::packageVersion("gamlss.dist")),
  generated_on = format(Sys.Date()),
  generator = "tools/r_y_valid_dump.R"
)

writeLines(
  toJSON(list(meta = meta, families = families), auto_unbox = TRUE, pretty = TRUE),
  con = out_path
)
cat("wrote", out_path, "families:", length(families), "\n")
