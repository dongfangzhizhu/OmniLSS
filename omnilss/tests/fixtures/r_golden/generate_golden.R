# Regenerate reproducible R golden fixtures for OmniLSS R-consistency tests.
#
# Usage:  Rscript generate_golden.R
#
# Writes regenerated fixtures + manifest into
#   omnilss/tests/fixtures/r_golden/regenerated/
# (the checked-in fixtures under r_golden/ are the historical anchors used by
# replay tests; regenerated/ proves the pipeline is reproducible and records
# provenance: R version, gamlss versions, seeds, per-fixture content hashes).
#
# Requires: gamlss, gamlss.dist, jsonlite.

suppressPackageStartupMessages(library(gamlss))
suppressPackageStartupMessages(library(jsonlite))

out_dir <- "omnilss/tests/fixtures/r_golden/regenerated"
dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)

# ---- case definitions (family, formula, seed, n) ---------------------------
# Core eight families (DEVELOPMENT_PLAN Phase 2 task 2.1): NO/GA/PO/BI/NBI/BE/
# ZAGA/GU each get a fixed R golden artifact used as acceptance evidence.
cases <- list(
  list(case = "ig_y_x_seed_20260522", family = "IG",  formula = "y ~ x", seed = 20260522, n = 60),
  list(case = "po_y_x_seed_20260523", family = "PO",  formula = "y ~ x", seed = 20260523, n = 80),
  list(case = "no_y_x_seed_20260524", family = "NO",  formula = "y ~ x", seed = 20260524, n = 90),
  list(case = "ga_y_x_seed_20260525", family = "GA",  formula = "y ~ x", seed = 20260525, n = 85),
  list(case = "bi_y_x_seed_20260809",  family = "BI",  formula = "y ~ x", seed = 20260809, n = 120),
  list(case = "nbi_y_x_seed_20260810", family = "NBI", formula = "y ~ x", seed = 20260810, n = 100),
  list(case = "be_y_x_seed_20260811",  family = "BE",  formula = "y ~ x", seed = 20260811, n = 90),
  list(case = "zaga_y_x_seed_20260812", family = "ZAGA", formula = "y ~ x", seed = 20260812, n = 120),
  list(case = "gu_y_x_seed_20260813",  family = "GU",  formula = "y ~ x", seed = 20260813, n = 100),
  # D2.227 / P1: high-frequency families promoted to golden evidence.  Both
  # sides run n.cyc = 100 here so they reach the same optimum: at the default
  # budget the coefficient gaps for WEI/GEOM/NO2/IGAMMA sat above 2e-4 purely
  # because R stopped earlier on a near-flat ridge, not because the densities
  # differ (deviance matched to ~1e-8 in every case).
  list(case = "wei_y_x_seed_20261115",    family = "WEI",    formula = "y ~ x", seed = 20261115, n = 100, n_cyc = 100),
  list(case = "logno_y_x_seed_20261116",  family = "LOGNO",  formula = "y ~ x", seed = 20261116, n = 90,  n_cyc = 100),
  list(case = "exp_y_x_seed_20261117",    family = "EXP",    formula = "y ~ x", seed = 20261117, n = 90,  n_cyc = 100),
  list(case = "geom_y_x_seed_20261118",   family = "GEOM",   formula = "y ~ x", seed = 20261118, n = 120, n_cyc = 100),
  list(case = "igamma_y_x_seed_20261119", family = "IGAMMA", formula = "y ~ x", seed = 20261119, n = 90,  n_cyc = 100),
  list(case = "no2_y_x_seed_20261120",    family = "NO2",    formula = "y ~ x", seed = 20261120, n = 90,  n_cyc = 100),
  # D2.228 / P1 second batch: LO / RG / PE / NBII / BEINF.  All five matched
  # live R across three seeds at 2e-4 with n.cyc = 100 (see the D2.228 log).
  list(case = "lo_y_x_seed_20261121",     family = "LO",      formula = "y ~ x", seed = 20261121, n = 100, n_cyc = 100),
  list(case = "rg_y_x_seed_20261122",     family = "RG",      formula = "y ~ x", seed = 20261122, n = 100, n_cyc = 100),
  list(case = "pe_y_x_seed_20261123",     family = "PE",      formula = "y ~ x", seed = 20261123, n = 100, n_cyc = 100),
  list(case = "nbii_y_x_seed_20261121",   family = "NBII",    formula = "y ~ x", seed = 20261121, n = 120, n_cyc = 100),
  list(case = "beinf_y_x_seed_20261122",  family = "BEINF",   formula = "y ~ x", seed = 20261122, n = 120, n_cyc = 100)
)

# ---- helpers ---------------------------------------------------------------
make_data <- function(seed, n, family) {
  set.seed(seed)
  x <- rnorm(n, 0, 1)  # standard-normal x, matching the historical fixtures
  if (family == "NO") {
    mu <- 1.0 + 0.4 * x
    y <- rNO(n, mu = mu, sigma = 0.6)
  } else if (family == "GA") {
    mu <- exp(0.4 + 0.3 * x)
    y <- rGA(n, mu = mu, sigma = 0.3)
  } else if (family == "PO") {
    mu <- exp(0.7 + 0.15 * x)
    y <- rPO(n, mu = mu)
  } else if (family == "IG") {
    mu <- exp(0.6 + 0.75 * x)
    y <- rIG(n, mu = mu, sigma = 0.2)
  } else if (family == "BI") {
    # Bernoulli semantics (bd=1), matching test_r_consistency_bi.py
    mu <- plogis(0.2 + 0.6 * x)
    y <- rBI(n, bd = 1, mu = mu)
  } else if (family == "NBI") {
    mu <- exp(0.4 + 0.3 * x)
    y <- rNBI(n, mu = mu, sigma = 0.3)
  } else if (family == "BE") {
    mu <- plogis(0.3 + 0.4 * x)
    y <- rBE(n, mu = mu, sigma = 0.2)
  } else if (family == "ZAGA") {
    mu <- exp(0.5 + 0.2 * x)
    y <- rZAGA(n, mu = mu, sigma = 0.4, nu = 0.2)
  } else if (family == "GU") {
    mu <- 1.0 + 0.5 * x
    y <- rGU(n, mu = mu, sigma = 0.8)
  } else if (family == "WEI") {
    mu <- exp(0.3 + 0.3 * x)
    y <- rWEI(n, mu = mu, sigma = 1.2)
  } else if (family == "LOGNO") {
    mu <- exp(1.0 + 0.3 * x)
    y <- rLOGNO(n, mu = mu, sigma = 0.4)
  } else if (family == "EXP") {
    mu <- exp(0.5 + 0.3 * x)
    y <- rEXP(n, mu = mu)
  } else if (family == "GEOM") {
    mu <- exp(0.8 + 0.2 * x)
    y <- rGEOM(n, mu = mu)
  } else if (family == "IGAMMA") {
    mu <- exp(0.6 + 0.3 * x)
    y <- rIGAMMA(n, mu = mu, sigma = 0.5)
  } else if (family == "NO2") {
    mu <- 1.0 + 0.4 * x
    y <- rNO2(n, mu = mu, sigma = 0.6)
  } else if (family == "LO") {
    mu <- 1.0 + 0.4 * x
    y <- rLO(n, mu = mu, sigma = 0.6)
  } else if (family == "RG") {
    mu <- 1.0 + 0.4 * x
    y <- rRG(n, mu = mu, sigma = 0.6)
  } else if (family == "PE") {
    mu <- 1.0 + 0.4 * x
    y <- rPE(n, mu = mu, sigma = 0.8, nu = 1.4)
  } else if (family == "NBII") {
    mu <- 2.0 + 0.4 * x
    y <- rNBII(n, mu = mu, sigma = 0.5)
  } else if (family == "BEINF") {
    mu <- plogis(0.3 + 0.3 * x)
    y <- rBEINF(n, mu = mu, sigma = 0.3, nu = 0.15, tau = 0.15)
  } else {
    stop("unknown family ", family)
  }
  list(x = x, y = y)
}

content_hash <- function(x) {
  # djb2-style string hash (pure string arithmetic, no 32-bit overflow);
  # hex via a manual conversion (sprintf %x does not accept doubles in R)
  chars <- utf8ToInt(x)
  h <- 5381
  for (b in chars) {
    h <- (h * 33 + b) %% 4294967296
  }
  hex_chars <- c("0","1","2","3","4","5","6","7","8","9","a","b","c","d","e","f")
  out <- ""
  for (i in 1:8) {
    out <- paste0(hex_chars[(h %% 16) + 1], out)
    h <- floor(h / 16)
  }
  out
}

# ---- generate each fixture ------------------------------------------------
manifest <- list()
for (c in cases) {
  dat <- make_data(c$seed, c$n, c$family)
  df <- data.frame(y = dat$y, x = dat$x)

  fam <- gamlss.dist::gamlss.family(c$family)
  ctrl <- if (is.null(c$n_cyc)) gamlss.control(trace = FALSE)
          else gamlss.control(trace = FALSE, n.cyc = c$n_cyc)
  fit <- gamlss(as.formula(c$formula), family = fam, data = df, control = ctrl)

  mu_coef <- as.numeric(coef(fit, "mu"))
  expected <- list(
    mu_coefficients = as.list(mu_coef),
    deviance = as.numeric(deviance(fit))
  )
  # Record coefficients on the link scale for every estimable parameter
  # (sigma for GA/NO/IG/BE/NBI/GU/ZAGA, nu for ZAGA), matching what the
  # OmniLSS replay tests compare against.
  for (p in fit$parameters) {
    if (p != "mu") {
      expected[[paste0(p, "_coefficients")]] <- as.list(as.numeric(coef(fit, p)))
    }
  }

  payload <- list(
    case = c$case,
    family = c$family,
    formula = c$formula,
    data = list(x = dat$x, y = dat$y),
    expected = expected
  )
  if (!is.null(c$n_cyc)) payload$n_cyc <- c$n_cyc

  json_text <- toJSON(payload, digits = 17, auto_unbox = TRUE)
  fpath <- file.path(out_dir, paste0(c$case, ".json"))
  writeLines(json_text, fpath)
  message("wrote ", fpath)

  entry <- list(
    family = c$family,
    formula = c$formula,
    seed = c$seed,
    n = c$n,
    R_version = R.version.string,
    gamlss_version = as.character(packageVersion("gamlss")),
    gamlss_dist_version = as.character(packageVersion("gamlss.dist")),
    content_hash = content_hash(json_text)
  )
  if (!is.null(c$n_cyc)) entry$n_cyc <- c$n_cyc
  manifest[[c$case]] <- entry
}

manifest_path <- file.path(out_dir, "manifest.json")
writeLines(toJSON(manifest, digits = 17, auto_unbox = TRUE, pretty = TRUE), manifest_path)
message("wrote ", manifest_path)
