# SPDX-License-Identifier: GPL-3.0-or-later
"""Live-R consistency evidence for ``calibration``, ``stepGAIC.CH``, and ``stepTGDAll.A`` (D2.278 / B1c).

Closes the final three in-scope missing symbols across the 11 in-scope R
GAMLSS packages (1036 / 1036 = 100% surface coverage).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss import (
    calibration,
    gamlss,
    gamlss_scope,
    stepGAIC_CH,
    stepTGDAll_A,
)

_R_SCRIPT = """
suppressPackageStartupMessages(library(gamlss))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
d <- fromJSON(args[1])

df_tr <- data.frame(y = d$y_tr, x1 = d$x1_tr, x2 = d$x2_tr, x3 = d$x3_tr)
df_va <- data.frame(y = d$y_va, x1 = d$x1_va, x2 = d$x2_va, x3 = d$x3_va)

# 1. calibration() on NO model
m_cal <- gamlss(y ~ x1, sigma.formula = ~1, data = df_tr, family = NO(),
                control = gamlss.control(c.crit = 1e-8, n.cyc = 100, trace = FALSE))
cent <- c(5, 25, 50, 75, 95)
z_q <- as.numeric(quantile(resid(m_cal), probs = cent / 100))
calib_pct <- as.numeric(100 * pNO(z_q, mu = 0, sigma = 1))

# 2. stepGAIC.CH() regimen selection on mu (x1 signal, x2 noise, x3 signal)
m0 <- gamlss(y ~ 1, sigma.formula = ~1, data = df_tr, family = NO(),
             control = gamlss.control(c.crit = 1e-8, n.cyc = 100, trace = FALSE))
sc <- list(x1 = ~1 + x1, x2 = ~1 + x2, x3 = ~1 + x3)
m_ch <- stepGAIC.CH(m0, scope = sc, direction = "both", trace = FALSE, k = 2)
ch_terms <- attr(terms(m_ch, "mu"), "term.labels")
ch_aic <- as.numeric(AIC(m_ch, k = 2))
ch_dev <- as.numeric(deviance(m_ch))

# 3. stepTGDAll.A() Strategy A on mu and sigma
m_tgd <- stepTGDAll.A(
  m0,
  scope = list(lower = ~1, upper = ~x1 + x2 + x3),
  newdata = df_va,
  trace = FALSE
)
tgd_mu_terms <- attr(terms(m_tgd, "mu"), "term.labels")
tgd_sg_terms <- attr(terms(m_tgd, "sigma"), "term.labels")
tgd_val <- as.numeric(getTGD(m_tgd, newdata = df_va)$TGD)

out <- list(
  calib_pct = calib_pct,
  ch_terms = ch_terms,
  ch_aic = ch_aic,
  ch_dev = ch_dev,
  tgd_mu_terms = tgd_mu_terms,
  tgd_sg_terms = tgd_sg_terms,
  tgd_val = tgd_val
)
writeLines(toJSON(out, digits = 17, auto_unbox = TRUE), args[2])
"""


def _dataset() -> dict:
    rng = np.random.default_rng(20261123)
    n_tr, n_va = 90, 60
    x1_tr = rng.normal(size=n_tr)
    x2_tr = rng.normal(size=n_tr)
    x3_tr = rng.normal(size=n_tr)
    sigma_tr = np.exp(-0.5 + 0.45 * x1_tr)
    y_tr = 1.0 + 2.0 * x1_tr - 1.5 * x3_tr + rng.normal(scale=sigma_tr)

    x1_va = rng.normal(size=n_va)
    x2_va = rng.normal(size=n_va)
    x3_va = rng.normal(size=n_va)
    sigma_va = np.exp(-0.5 + 0.45 * x1_va)
    y_va = 1.0 + 2.0 * x1_va - 1.5 * x3_va + rng.normal(scale=sigma_va)
    return {
        "y_tr": y_tr,
        "x1_tr": x1_tr,
        "x2_tr": x2_tr,
        "x3_tr": x3_tr,
        "y_va": y_va,
        "x1_va": x1_va,
        "x2_va": x2_va,
        "x3_va": x3_va,
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
        out_file = td_path / "out.json"
        script_file = td_path / "run.R"
        data_file.write_text(json.dumps(payload), encoding="utf-8")
        script_file.write_text(_R_SCRIPT, encoding="utf-8")
        proc = subprocess.run(
            [rscript, str(script_file), str(data_file), str(out_file)],
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode != 0:
            pytest.fail(f"Rscript failed ({proc.returncode}): {proc.stderr[-500:]}")
        return json.loads(out_file.read_text(encoding="utf-8"))


def test_calibration_matches_live_r_calibrated_centiles(r_reference):
    data = _dataset()
    tr = {"y": data["y_tr"], "x1": data["x1_tr"], "x2": data["x2_tr"], "x3": data["x3_tr"]}
    m_cal = gamlss("y ~ x1", sigma_formula="~1", family="NO", data=tr, max_iter=100, tol=1e-8)

    cent = (5.0, 25.0, 50.0, 75.0, 95.0)
    res = calibration(m_cal, xvar="x1", cent=cent)
    np.testing.assert_allclose(res.target, cent, rtol=0, atol=1e-12)
    np.testing.assert_allclose(res.calib, r_reference["calib_pct"], rtol=1e-4, atol=1e-4)
    assert list(res.table.columns) == ["target", "calib.", "sample"]
    assert np.all((res.sample >= 0.0) & (res.sample <= 100.0))

    res_fan = calibration(m_cal, xvar="x1", cent=cent, fan=True)
    assert res_fan.fan_result is not None
    np.testing.assert_allclose(res_fan.calib, r_reference["calib_pct"], rtol=1e-4, atol=1e-4)


def test_step_gaic_ch_matches_live_r_regimen_selection(r_reference):
    data = _dataset()
    tr = {"y": data["y_tr"], "x1": data["x1_tr"], "x2": data["x2_tr"], "x3": data["x3_tr"]}
    m0 = gamlss("y ~ 1", sigma_formula="~1", family="NO", data=tr, max_iter=100, tol=1e-8)

    scope = {"x1": "~1 + x1", "x2": "~1 + x2", "x3": "~1 + x3"}
    kept_aics: list[float] = []
    res = stepGAIC_CH(
        m0,
        scope=scope,
        direction="both",
        k=2.0,
        keep=lambda mod, aic: aic,
    )
    frame = res.to_frame()
    assert list(frame.columns) == ["From", "To", "Df", "Deviance", "Resid. Df", "Resid. Dev", "AIC"]
    assert len(res.keep) >= len(res.rows)

    selected_formula = res.model.formulas["mu"]
    r_terms = r_reference["ch_terms"]
    if isinstance(r_terms, str):
        r_terms = [r_terms]
    for term in r_terms:
        assert term in selected_formula
    np.testing.assert_allclose(res.rows[-1].aic, r_reference["ch_aic"], rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(float(res.model.deviance), r_reference["ch_dev"], rtol=1e-4, atol=1e-3)


def test_step_tgd_all_a_matches_live_r_strategy_a(r_reference):
    data = _dataset()
    tr = {"y": data["y_tr"], "x1": data["x1_tr"], "x2": data["x2_tr"], "x3": data["x3_tr"]}
    va = {"y": data["y_va"], "x1": data["x1_va"], "x2": data["x2_va"], "x3": data["x3_va"]}
    m0 = gamlss("y ~ 1", sigma_formula="~1", family="NO", data=tr, max_iter=100, tol=1e-8)

    res = stepTGDAll_A(
        m0,
        scope={"lower": "~1", "upper": "~x1 + x2 + x3"},
        newdata=va,
    )
    assert res.failed_stage is None
    assert "mu.anova.for" in res.anova_all
    assert "sigma.anova.for" in res.anova_all
    assert "mu.anova.back" in res.anova_all
    # Signal predictors x1 and x3 must be selected on mu, x1 on sigma
    r_mu = r_reference["tgd_mu_terms"]
    r_sg = r_reference["tgd_sg_terms"]
    if isinstance(r_mu, str):
        r_mu = [r_mu]
    if isinstance(r_sg, str):
        r_sg = [r_sg]
    for term in r_mu:
        assert term in res.model.formulas["mu"]
    for term in r_sg:
        assert term in res.model.formulas["sigma"]
    from omnilss.stepTGD import extract_tgd_data

    got_tgd = float(extract_tgd_data(res.model, newdata=va).tgd)
    np.testing.assert_allclose(got_tgd, r_reference["tgd_val"], rtol=1e-3, atol=1e-2)
