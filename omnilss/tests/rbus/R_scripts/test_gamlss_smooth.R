#!/usr/bin/env Rscript
# R script for GAMLSS smooth terms consistency testing
# This script is called by Python tests to fit GAMLSS models with smooth terms in R

# Load required libraries
suppressPackageStartupMessages({
  library(gamlss)
  library(jsonlite)
})

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Usage: Rscript test_gamlss_smooth.R <input.json> <output.json>")
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

# Add nu formula if present
if (!is.null(input$nu_formula)) {
  gamlss_args$nu.formula <- as.formula(input$nu_formula)
}

# Add tau formula if present
if (!is.null(input$tau_formula)) {
  gamlss_args$tau.formula <- as.formula(input$tau_formula)
}

# Fit model
tryCatch({
  model <- do.call(gamlss, gamlss_args)
  
  # Extract results
  output <- list(
    success = TRUE,
    coefficients = list(
      mu = as.numeric(coef(model, "mu"))
    ),
    fitted_values = list(
      mu = as.numeric(fitted(model, "mu"))
    ),
    linear_predictors = list(
      mu = as.numeric(model$mu.lp)
    ),
    deviance = as.numeric(deviance(model)),
    aic = as.numeric(AIC(model)),
    sbc = as.numeric(model$sbc),
    df_fit = as.numeric(model$df.fit),
    df_residual = as.numeric(model$df.residual),
    n = as.numeric(model$noObs),
    converged = model$converged
  )
  
  # Add sigma if present
  if ("sigma" %in% model$parameters) {
    output$coefficients$sigma <- as.numeric(coef(model, "sigma"))
    output$fitted_values$sigma <- as.numeric(fitted(model, "sigma"))
    output$linear_predictors$sigma <- as.numeric(model$sigma.lp)
  }
  
  # Add nu if present
  if ("nu" %in% model$parameters) {
    output$coefficients$nu <- as.numeric(coef(model, "nu"))
    output$fitted_values$nu <- as.numeric(fitted(model, "nu"))
    output$linear_predictors$nu <- as.numeric(model$nu.lp)
  }
  
  # Add tau if present
  if ("tau" %in% model$parameters) {
    output$coefficients$tau <- as.numeric(coef(model, "tau"))
    output$fitted_values$tau <- as.numeric(fitted(model, "tau"))
    output$linear_predictors$tau <- as.numeric(model$tau.lp)
  }
  
  # Extract smooth terms information if present
  if (!is.null(model$mu.s)) {
    mu_lambda <- NULL
    if (is.list(model$mu.s) && length(model$mu.s) > 0) {
      if (!is.null(model$mu.s[[1]]$lambda)) {
        mu_lambda <- as.numeric(model$mu.s[[1]]$lambda)
      }
    }
    output$smooth_info <- list(
      mu = list(
        edf = as.numeric(model$mu.df),
        lambda = mu_lambda
      )
    )
  }
  
  if (!is.null(model$sigma.s)) {
    if (is.null(output$smooth_info)) output$smooth_info <- list()
    sigma_lambda <- NULL
    if (is.list(model$sigma.s) && length(model$sigma.s) > 0) {
      if (!is.null(model$sigma.s[[1]]$lambda)) {
        sigma_lambda <- as.numeric(model$sigma.s[[1]]$lambda)
      }
    }
    output$smooth_info$sigma <- list(
      edf = as.numeric(model$sigma.df),
      lambda = sigma_lambda
    )
  }
  
}, error = function(e) {
  output <<- list(
    success = FALSE,
    error = as.character(e$message)
  )
})

# Write output
write_json(output, output_file, auto_unbox = TRUE, digits = 16)

# Exit
quit(status = 0)
