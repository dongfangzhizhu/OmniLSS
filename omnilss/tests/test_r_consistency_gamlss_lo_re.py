# SPDX-License-Identifier: GPL-3.0-or-later
"""Live-R consistency evidence for ``gamlss.lo``, ``re``, and ``gamlss.re`` (D2.277 / B1b).

Compares:
- ``gamlss_lo`` against live R ``stats::loess`` (degree 1 & 2, weighted &
  unweighted, direct surface, exact trace.hat, and out-of-sample ``xeval``);
- ``re`` + ``gamlss_re`` against live R ``nlme::lme`` (weighted random
  intercept under ML and REML, fixed covariate + random intercept, random
  intercept + slope, ``sigma``, ``logLik``, ``nl.df``, and level 0/1 predict).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss import gamlss_lo, gamlss_re, lo, lo_control, re

_R_SCRIPT = """
suppressPackageStartupMessages(library(nlme))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
d <- fromJSON(args[1])

# 1. LOESS (exact surface & trace.hat, weighted and unweighted)
df_lo <- data.frame(x = d$lo_x, y = d$lo_y, w = d$lo_w)
ctrl <- loess.control(surface = "direct", statistics = "exact", trace.hat = "exact")
fit_d2 <- loess(y ~ x, data = df_lo, weights = w, span = 0.65, degree = 2, control = ctrl)
fit_d1 <- loess(y ~ x, data = df_lo, span = 0.55, degree = 1, control = ctrl)
pred_d2 <- as.numeric(predict(fit_d2, newdata = data.frame(x = d$lo_xeval)))

# 2. Weighted LMM (gamlss.re uses weights = varFixed(~1/w))
df_re <- data.frame(y = d$re_y, x = d$re_x, g = factor(d$re_g), w_inv = 1 / d$re_w)
lme_ml <- lme(y ~ x, data = df_re, random = ~1 | g, weights = varFixed(~w_inv), method = "ML")
lme_reml <- lme(y ~ x, data = df_re, random = ~1 | g, weights = varFixed(~w_inv), method = "REML")
lme_slope <- lme(y ~ x, data = df_re, random = ~1 + x | g, weights = varFixed(~w_inv), method = "REML")

fv_ml <- as.numeric(fitted(lme_ml))
N_ml <- sum(d$re_w != 0)
nl_df_ml <- N_ml - sum(d$re_w * (d$re_y - fv_ml)^2) / (lme_ml$sigma^2) - 1

nd <- data.frame(x = d$re_new_x, g = factor(d$re_new_g, levels = levels(df_re$g)))
pred_l0 <- as.numeric(predict(lme_ml, newdata = nd, level = 0))
pred_l1 <- as.numeric(predict(lme_ml, newdata = nd, level = 1))

out <- list(
  lo_d2_fv = as.numeric(fitted(fit_d2)),
  lo_d2_trace = as.numeric(fit_d2$trace.hat),
  lo_d2_pred = pred_d2,
  lo_d1_fv = as.numeric(fitted(fit_d1)),
  lo_d1_trace = as.numeric(fit_d1$trace.hat),
  re_ml_fixef = as.numeric(fixef(lme_ml)),
  re_ml_fv = fv_ml,
  re_ml_sigma = as.numeric(lme_ml$sigma),
  re_ml_loglik = as.numeric(logLik(lme_ml)),
  re_ml_nldf = as.numeric(nl_df_ml),
  re_ml_pred_l0 = pred_l0,
  re_ml_pred_l1 = pred_l1,
  re_reml_fixef = as.numeric(fixef(lme_reml)),
  re_reml_fv = as.numeric(fitted(lme_reml)),
  re_reml_sigma = as.numeric(lme_reml$sigma),
  re_reml_loglik = as.numeric(logLik(lme_reml)),
  re_slope_fixef = as.numeric(fixef(lme_slope)),
  re_slope_fv = as.numeric(fitted(lme_slope)),
  re_slope_sigma = as.numeric(lme_slope$sigma)
)
cat(toJSON(out, digits = 17, auto_unbox = TRUE))
"""


def _dataset() -> dict:
    rng = np.random.default_rng(20261122)
    n_lo = 45
    lo_x = np.linspace(0.0, 2.0 * np.pi, n_lo)
    lo_y = np.sin(lo_x) + 0.25 * rng.normal(size=n_lo)
    lo_w = rng.uniform(0.5, 2.5, size=n_lo)
    lo_xeval = np.array([0.4, 1.8, 3.3, 5.1])

    n_groups = 12
    per_group = 6
    re_g = np.repeat(np.arange(1, n_groups + 1), per_group)
    re_x = rng.normal(size=n_groups * per_group)
    b0 = rng.normal(scale=1.1, size=n_groups)
    b1 = rng.normal(scale=0.5, size=n_groups)
    re_w = rng.uniform(0.6, 2.0, size=n_groups * per_group)
    re_y = (
        1.5
        + 0.7 * re_x
        + b0[re_g - 1]
        + b1[re_g - 1] * re_x
        + rng.normal(scale=0.4 / np.sqrt(re_w))
    )
    return {
        "lo_x": lo_x,
        "lo_y": lo_y,
        "lo_w": lo_w,
        "lo_xeval": lo_xeval,
        "re_x": re_x,
        "re_y": re_y,
        "re_g": re_g,
        "re_w": re_w,
        "re_new_x": np.array([-0.5, 0.0, 1.2]),
        "re_new_g": np.array([1, 5, 10]),
    }


@pytest.fixture(scope="module")
def r_reference() -> dict:
    rscript = shutil.which("Rscript")
    if rscript is None:
        pytest.skip("Rscript not available")
    data = _dataset()
    payload = {k: v.tolist() for k, v in data.items()}
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        data_file = td_path / "data.json"
        script_file = td_path / "run.R"
        data_file.write_text(json.dumps(payload), encoding="utf-8")
        script_file.write_text(_R_SCRIPT, encoding="utf-8")
        proc = subprocess.run(
            [rscript, str(script_file), str(data_file)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            pytest.fail(f"Rscript failed ({proc.returncode}): {proc.stderr[-500:]}")
        return json.loads(proc.stdout)


def test_gamlss_lo_matches_live_r_loess(r_reference):
    data = _dataset()
    spec = lo("~x", data={"x": data["lo_x"]}, span=0.65, degree=2)
    fit_d2 = gamlss_lo(spec, data["lo_y"], w=data["lo_w"])
    np.testing.assert_allclose(fit_d2.fitted_values, r_reference["lo_d2_fv"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fit_d2.trace_hat, r_reference["lo_d2_trace"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fit_d2.nl_df, r_reference["lo_d2_trace"] - 1.0, rtol=1e-10, atol=1e-10)

    pred_d2 = gamlss_lo(spec, data["lo_y"], w=data["lo_w"], xeval=data["lo_xeval"])
    np.testing.assert_allclose(pred_d2, r_reference["lo_d2_pred"], rtol=1e-10, atol=1e-10)

    fit_d1 = gamlss_lo(data["lo_x"], data["lo_y"], span=0.55, degree=1)
    np.testing.assert_allclose(fit_d1.fitted_values, r_reference["lo_d1_fv"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(fit_d1.trace_hat, r_reference["lo_d1_trace"], rtol=1e-10, atol=1e-10)


def test_gamlss_lo_enp_target_and_symmetric_and_control():
    data = _dataset()
    ctrl = lo_control(span=0.7, degree=2, family="symmetric", iterations=3)
    assert ctrl["family"] == "symmetric"
    fit_sym = gamlss_lo(data["lo_x"], data["lo_y"], control=ctrl)
    assert np.all(np.isfinite(fit_sym.fitted_values))

    fit_enp = gamlss_lo(data["lo_x"], data["lo_y"], enp_target=6.0, degree=2)
    assert abs(fit_enp.trace_hat - 6.0) < 0.25


def test_gamlss_re_ml_and_reml_match_live_r_lme(r_reference):
    data = _dataset()
    fit_data = {"x": data["re_x"], "g": data["re_g"]}

    spec_ml = re(fixed="~x", random="~1 | g", method="ML", data=fit_data)
    res_ml = gamlss_re(spec_ml, data["re_y"], w=data["re_w"])
    np.testing.assert_allclose(res_ml.fixed_effects, r_reference["re_ml_fixef"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res_ml.fitted_values, r_reference["re_ml_fv"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res_ml.sigma, r_reference["re_ml_sigma"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res_ml.log_lik, r_reference["re_ml_loglik"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res_ml.nl_df, r_reference["re_ml_nldf"], rtol=1e-4, atol=1e-4)

    newdata = {"x": data["re_new_x"], "g": data["re_new_g"]}
    pred_l0 = gamlss_re(spec_ml, data["re_y"], w=data["re_w"], xeval=newdata, level=0)
    pred_l1 = gamlss_re(spec_ml, data["re_y"], w=data["re_w"], xeval=newdata, level=1)
    np.testing.assert_allclose(pred_l0, r_reference["re_ml_pred_l0"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(pred_l1, r_reference["re_ml_pred_l1"], rtol=1e-5, atol=1e-5)

    spec_reml = re(fixed="~x", random="~1 | g", method="REML", data=fit_data)
    res_reml = gamlss_re(spec_reml, data["re_y"], w=data["re_w"])
    np.testing.assert_allclose(res_reml.fixed_effects, r_reference["re_reml_fixef"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res_reml.fitted_values, r_reference["re_reml_fv"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res_reml.sigma, r_reference["re_reml_sigma"], rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(res_reml.log_lik, r_reference["re_reml_loglik"], rtol=1e-5, atol=1e-5)


def test_gamlss_re_random_slope_and_explicit_unported_guards(r_reference):
    data = _dataset()
    fit_data = {"x": data["re_x"], "g": data["re_g"]}

    spec_slope = re(fixed="~x", random="~1 + x | g", method="REML", data=fit_data)
    res_slope = gamlss_re(spec_slope, data["re_y"], w=data["re_w"])
    np.testing.assert_allclose(res_slope.fixed_effects, r_reference["re_slope_fixef"], rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(res_slope.fitted_values, r_reference["re_slope_fv"], rtol=2e-3, atol=2e-3)
    np.testing.assert_allclose(res_slope.sigma, r_reference["re_slope_sigma"], rtol=2e-3, atol=2e-3)

    with pytest.raises(NotImplementedError):
        re(fixed="~1", random="~1 | g", correlation="corAR1")
    with pytest.raises(NotImplementedError):
        re(fixed="~1", random="~1 | a/b")
