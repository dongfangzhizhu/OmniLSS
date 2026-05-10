#!/usr/bin/env Rscript
# R script for GAMLSS consistency testing
# This script is called by Python tests to fit GAMLSS models in R

# Load required libraries
suppressPackageStartupMessages({
  library(gamlss)
  library(jsonlite)
})

# Parse command line arguments
args <- commandArgs(trailingOnly = TRUE)

if (length(args) < 2) {
  stop("Usage: Rscript test_gamlss.R <input.json> <output.json>")
}

input_file <- args[1]
output_file <- args[2]

# Read input data
input <- fromJSON(input_file)

# Convert data to data frame
data_df <- as.data.frame(input$data)

# Debug: Check for NA/NaN values
if (any(is.na(data_df))) {
  stop(paste("Data contains NA values in columns:", 
             paste(names(data_df)[colSums(is.na(data_df)) > 0], collapse=", ")))
}

# Parse formulas
mu_formula <- as.formula(input$formula)
sigma_formula <- as.formula(input$sigma_formula)

# Get family
family_name <- input$family

# Special handling for BI / BB: R gamlss expects binomial-type responses as
# either proportions in [0, 1] or a two-column success/failure matrix.
if (family_name %in% c("BI", "BB") && "bd" %in% names(data_df)) {
  bd_vec <- as.numeric(data_df$bd)
  y_vec  <- as.numeric(data_df[[all.vars(mu_formula)[1]]])
  y_counts <- if (family_name == "BI") round(y_vec * bd_vec) else round(y_vec)
  data_df[["y_cbind"]] <- cbind(y_counts, bd_vec - y_counts)
  # Rewrite formula to use y_cbind
  mu_formula <- as.formula(paste("y_cbind ~", deparse(mu_formula[[3]])))
  family_obj <- get(family_name)()
} else {
  family_obj <- get(family_name)()
}

# Build gamlss call arguments
gamlss_args <- list(
  formula = mu_formula,
  sigma.formula = sigma_formula,
  family = family_obj,
  data = data_df,
  trace = FALSE,
  control = gamlss.control(
    n.cyc = 50,
    trace = FALSE,
    gd.tol = 1,  # Much more tolerant
    c.crit = 0.01
  )
)

# Add weights if present (e.g., binomial denominator bd)
if (!is.null(input$weights)) {
  gamlss_args$weights <- as.numeric(input$weights)
}

# Add nu formula if present
if (!is.null(input$nu_formula)) {
  gamlss_args$nu.formula <- as.formula(input$nu_formula)
}

# Add tau formula if present
if (!is.null(input$tau_formula)) {
  gamlss_args$tau.formula <- as.formula(input$tau_formula)
}

# Debug: Print data structure
cat("Data structure:\n", file=stderr())
cat(paste("  Rows:", nrow(data_df), "\n"), file=stderr())
cat(paste("  Columns:", paste(names(data_df), collapse=", "), "\n"), file=stderr())
cat(paste("  Formula:", deparse(mu_formula), "\n"), file=stderr())
cat(paste("  Family:", family_name, "\n"), file=stderr())

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
    converged = ifelse(is.na(model$converged), TRUE, model$converged)
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
  
}, error = function(e) {
  output <<- list(
    success = FALSE,
    error = as.character(e$message),
    traceback = as.character(sys.calls())
  )
})

# Write output
write_json(output, output_file, auto_unbox = TRUE, digits = 16)

# Exit
quit(status = 0)
