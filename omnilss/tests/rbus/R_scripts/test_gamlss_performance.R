#!/usr/bin/env Rscript
# R script for GAMLSS performance benchmarking
# This script is called by Python performance tests to measure R gamlss performance

# Load required libraries
suppressPackageStartupMessages({
  library(gamlss)
  library(jsonlite)
})

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Usage: Rscript test_gamlss_performance.R <input.json> <output.json>")
}

input_file <- args[1]
output_file <- args[2]

# Read input data
input <- fromJSON(input_file)

# Convert data to data frame
data_df <- as.data.frame(input$data)

# Parse formulas
mu_formula <- as.formula(input$formula)
sigma_formula <- if (!is.null(input$sigma_formula)) as.formula(input$sigma_formula) else ~1

# Get family
family_name <- input$family
family_obj <- get(family_name)()

# Build gamlss call arguments
gamlss_args <- list(
  formula = mu_formula,
  sigma.formula = sigma_formula,
  family = family_obj,
  data = data_df,
  trace = FALSE
)

# Fit model and measure time
tryCatch({
  start_time <- Sys.time()
  model <- do.call(gamlss, gamlss_args)
  end_time <- Sys.time()
  
  elapsed_seconds <- as.numeric(difftime(end_time, start_time, units = "secs"))
  
  # Extract results
  output <- list(
    success = TRUE,
    time_seconds = elapsed_seconds,
    n_obs = model$noObs,
    converged = model$converged,
    deviance = as.numeric(deviance(model)),
    aic = as.numeric(AIC(model)),
    df_fit = as.numeric(model$df.fit)
  )
  
}, error = function(e) {
  output <<- list(
    success = FALSE,
    error = as.character(e$message),
    time_seconds = 0
  )
})

# Write output
write_json(output, output_file, auto_unbox = TRUE, digits = 16)

# Exit
quit(status = 0)
