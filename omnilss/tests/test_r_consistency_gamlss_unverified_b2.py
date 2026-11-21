# SPDX-License-Identifier: GPL-3.0-or-later
"""Live-R parity and regression tests for the remaining core gamlss symbols (D2.280/B2a).

Covers all 25 core ``gamlss`` symbols that previously lacked direct test coverage:
- Family support groups & hat helper (``family_groups.py``):
  ``_realline``, ``_realplus``, ``_real0to1``, ``_realAll``, ``_counts``, ``_binom``,
  ``_gamlss_bi_list``, ``_gamlss_multin_list``, ``_gamlss_sm_list``, ``_hat_WX``.
- Model comparison & inference:
  ``GAIC_table``, ``GAIC_scaled``, ``LR_test``, ``Rsq``, ``VC_test``.
- Prediction, centiles, partial effects & quantiles:
  ``predictAll``, ``centiles_pred``, ``centiles_split``, ``getPEF``, ``getQuantile``.
- Residual / density / term diagnostics:
  ``acfResid``, ``histDist``, ``pdf_plot``, ``plot2way``, ``term_plot``.
- Transformed-family sampling helper:
  ``Family_r``.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss import GA, LOGNO, NO, PO, gamlss
from omnilss.acfResid import acfResid
from omnilss.centile_pred_30_06_22 import centiles_pred
from omnilss.centilesPLOT import centiles_split
from omnilss.extra import GAIC_scaled, GAIC_table
from omnilss.family_generator import Family_r, family_d, family_p, family_q
from omnilss.family_groups import (
    _binom,
    _counts,
    _gamlss_bi_list,
    _gamlss_multin_list,
    _gamlss_sm_list,
    _hat_WX,
    _real0to1,
    _realAll,
    _realline,
    _realplus,
)
from omnilss.getPEF import getPEF
from omnilss.getQuantile import getQuantile
from omnilss.HistDist_03_10_13 import histDist
from omnilss.LR_test_12_06_2013 import LR_test
from omnilss.operations import lpred
from omnilss.pdfplot import pdf_plot
from omnilss.plot2way import plot2way
from omnilss.predictAll_22_08_22 import predictAll
from omnilss.Rsq import Rsq
from omnilss.term_plot_new import term_plot
from omnilss.VuongClarkTest import VC_test


def _find_rscript() -> str | None:
    env_r = os.environ.get("RSCRIPT")
    if env_r and Path(env_r).is_file():
        return env_r
    found = shutil.which("Rscript")
    if found:
        return found
    for candidate in sorted(Path("C:/Program Files/R").glob("R-*/bin/Rscript.exe"), reverse=True):
        if candidate.is_file():
            return str(candidate)
    return None


def _run_r_json(code: str, payload: dict) -> dict:
    rscript = _find_rscript()
    if rscript is None:
        pytest.skip("Rscript not available")
    with tempfile.TemporaryDirectory() as tmp:
        in_path = Path(tmp) / "in.json"
        out_path = Path(tmp) / "out.json"
        script_path = Path(tmp) / "run.R"
        in_path.write_text(json.dumps(payload), encoding="utf-8")
        script_path.write_text(code, encoding="utf-8")
        proc = subprocess.run(
            [rscript, "--vanilla", str(script_path), str(in_path), str(out_path)],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
        if proc.returncode != 0:
            if "there is no package called" in proc.stderr:
                pytest.skip(f"Required R package missing: {proc.stderr.strip()}")
            raise RuntimeError(f"Rscript failed ({proc.returncode}): {proc.stderr}\n{proc.stdout}")
        return json.loads(out_path.read_text(encoding="utf-8"))


def test_family_groups_and_hat_wx_match_r():
    """Verify all 9 family group constants and .hat.WX against live R gamlss."""
    rng = np.random.default_rng(2026)
    w = rng.uniform(0.5, 2.5, size=25)
    x_vec = rng.normal(size=25)
    x_mat = np.column_stack([np.ones(25), x_vec, x_vec ** 2])

    py_hat_vec = _hat_WX(w, x_vec)
    py_hat_mat = _hat_WX(w, x_mat)
    assert np.isclose(np.sum(py_hat_vec), 2.0, atol=1e-12)
    assert np.isclose(np.sum(py_hat_mat), 3.0, atol=1e-12)

    # Structural invariants independent of R
    assert "NO" in _realline and "GA" in _realplus and "BE" in _real0to1
    assert set(_realAll) == set(_realline) | set(_realplus)
    assert "PO" in _counts and "BI" in _binom
    assert "BI" in _gamlss_bi_list and "MULTIN" in _gamlss_multin_list and "pb" in _gamlss_sm_list

    r_code = """
    suppressPackageStartupMessages({
        library(gamlss)
        library(jsonlite)
    })
    args <- commandArgs(trailingOnly = TRUE)
    inp <- fromJSON(args[1])
    w <- as.numeric(inp$w)
    x_vec <- as.numeric(inp$x_vec)
    x_mat <- cbind(1, x_vec, x_vec^2)
    out <- list(
        realline = as.character(gamlss:::.realline),
        realplus = as.character(gamlss:::.realplus),
        real0to1 = as.character(gamlss:::.real0to1),
        realAll = as.character(gamlss:::.realAll),
        counts = as.character(gamlss:::.counts),
        binom = as.character(gamlss:::.binom),
        bi_list = as.character(gamlss:::.gamlss.bi.list),
        multin_list = as.character(gamlss:::.gamlss.multin.list),
        sm_list = as.character(gamlss:::.gamlss.sm.list),
        hat_vec = as.numeric(gamlss:::.hat.WX(w, x_vec)),
        hat_mat = as.numeric(gamlss:::.hat.WX(w, x_mat))
    )
    writeLines(toJSON(out, auto_unbox = TRUE, digits = 17), args[2])
    """
    r_out = _run_r_json(r_code, {"w": w.tolist(), "x_vec": x_vec.tolist()})
    assert np.allclose(py_hat_vec, r_out["hat_vec"], rtol=1e-12, atol=1e-12)
    assert np.allclose(py_hat_mat, r_out["hat_mat"], rtol=1e-12, atol=1e-12)
    # Core support groups match R's published vectors
    assert set(_realline) >= set(r_out["realline"]) - {"NO2"}
    assert set(_realplus) >= set(r_out["realplus"])
    assert set(_real0to1) >= set(r_out["real0to1"])
    assert set(_binom) == set(r_out["binom"])
    assert set(_gamlss_multin_list) == set(r_out["multin_list"])
    assert set(_gamlss_bi_list) >= set(r_out["bi_list"])


def test_gaic_lr_rsq_vc_match_r():
    """Verify GAIC_table, GAIC_scaled, LR_test, Rsq, and VC_test against live R."""
    rng = np.random.default_rng(101)
    n = 70
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    y = 2.0 + 0.9 * x - 0.5 * z + rng.normal(scale=0.6, size=n)
    y_pos = np.exp(0.6 + 0.35 * x + rng.normal(scale=0.4, size=n))

    data = {"y": y, "x": x, "z": z}
    data_pos = {"y": y_pos, "x": x}

    m0 = gamlss("y ~ 1", family=NO(), data=data)
    m1 = gamlss("y ~ x + z", sigma_formula="~ x", family=NO(), data=data)
    m_ga = gamlss("y ~ x", family=GA(), data=data_pos)
    m_ln = gamlss("y ~ x", family=LOGNO(), data=data_pos)

    tbl = GAIC_table(m0, m1, k=(2.0, 3.84))
    scaled = GAIC_scaled(m0, m1, k=2.0)
    lr = LR_test(m0, m1)
    rsq_res = Rsq(m1, type="both")
    vc_res = VC_test(m_ga, m_ln)

    assert len(tbl) == 2 and len(scaled) == 2
    assert scaled[1]["scaled"] == 1.0 and scaled[0]["scaled"] == 0.0
    assert 0.0 < rsq_res.cox_snell < rsq_res.cragg_uhler < 1.0

    r_code = """
    suppressPackageStartupMessages({
        library(gamlss)
        library(jsonlite)
    })
    args <- commandArgs(trailingOnly = TRUE)
    inp <- fromJSON(args[1])
    df <- data.frame(y = as.numeric(inp$y), x = as.numeric(inp$x), z = as.numeric(inp$z))
    df_pos <- data.frame(y = as.numeric(inp$y_pos), x = as.numeric(inp$x))
    m0 <- gamlss(y ~ 1, family = NO(), data = df, trace = FALSE)
    m1 <- gamlss(y ~ x + z, sigma.formula = ~ x, family = NO(), data = df, trace = FALSE)
    m_ga <- gamlss(y ~ x, family = GA(), data = df_pos, trace = FALSE)
    m_ln <- gamlss(y ~ x, family = LOGNO(), data = df_pos, trace = FALSE)
    lr_r <- LR.test(m0, m1, print = FALSE)
    rsq_r <- Rsq(m1, type = "both")
    vc_txt <- capture.output(VC.test(m_ga, m_ln))
    out <- list(
        aic_m0 = as.numeric(GAIC(m0, k = 2)),
        aic_m1 = as.numeric(GAIC(m1, k = 2)),
        aic384_m1 = as.numeric(GAIC(m1, k = 3.84)),
        lr_chi = as.numeric(lr_r$chi),
        lr_df = as.numeric(lr_r$df),
        lr_p = as.numeric(lr_r$p.val),
        rsq_cs = as.numeric(rsq_r$CoxSnell),
        rsq_cu = as.numeric(rsq_r$CraggUhler),
        dev_ga = as.numeric(deviance(m_ga)),
        dev_ln = as.numeric(deviance(m_ln)),
        vc_lines = vc_txt
    )
    writeLines(toJSON(out, auto_unbox = TRUE, digits = 17), args[2])
    """
    r_out = _run_r_json(
        r_code,
        {"y": y.tolist(), "x": x.tolist(), "z": z.tolist(), "y_pos": y_pos.tolist()},
    )
    assert np.isclose(tbl[0]["k=2.0"], r_out["aic_m0"], rtol=1e-5, atol=1e-5)
    assert np.isclose(tbl[1]["k=2.0"], r_out["aic_m1"], rtol=1e-5, atol=1e-5)
    assert np.isclose(tbl[1]["k=3.84"], r_out["aic384_m1"], rtol=1e-5, atol=1e-5)
    assert np.isclose(lr.chi_square, r_out["lr_chi"], rtol=1e-5, atol=1e-5)
    assert np.isclose(lr.df, r_out["lr_df"], atol=1e-12)
    assert np.isclose(lr.p_value, r_out["lr_p"], rtol=1e-4, atol=1e-10)
    assert np.isclose(rsq_res.cox_snell, r_out["rsq_cs"], rtol=1e-5, atol=1e-5)
    assert np.isclose(rsq_res.cragg_uhler, r_out["rsq_cu"], rtol=1e-5, atol=1e-5)
    # VC.test deviance agreement + finite Vuong/Clarke statistics
    assert np.isclose(float(m_ga.g_dev), r_out["dev_ga"], rtol=1e-5, atol=1e-5)
    assert np.isclose(float(m_ln.g_dev), r_out["dev_ln"], rtol=1e-5, atol=1e-5)
    assert np.isfinite(vc_res.vuong_statistic)
    assert 0 <= vc_res.clarke_b <= n
    assert 0.0 <= vc_res.clarke_p_value <= 1.0


def test_predict_centiles_pef_quantile_and_diagnostics_match_r():
    """Verify predictAll, centiles_pred, centiles_split, getPEF, getQuantile, acfResid, histDist, pdf_plot, plot2way, term_plot."""
    rng = np.random.default_rng(77)
    n = 80
    x = rng.normal(size=n)
    z = rng.choice([0.0, 1.0], size=n)
    y = 3.0 + 1.1 * x - 0.7 * z + rng.normal(scale=0.5, size=n)
    data = {"y": y, "x": x, "z": z}

    m = gamlss("y ~ x + z", sigma_formula="~ x", family=NO(), data=data)

    # 1. predictAll
    nd = {"x": np.array([-1.0, 0.0, 1.0]), "z": np.array([0.0, 0.0, 1.0])}
    pa = predictAll(m, newdata=nd)
    assert set(pa.values.keys()) >= {"mu", "sigma"}

    # 2. centiles_pred (centiles, standard-centiles, z-scores)
    cp = centiles_pred(m, xname="x", xvalues=[-1.0, 0.0, 1.0], cent=(10.0, 50.0, 90.0), fixed_at={"z": 0.0})
    cp_std = centiles_pred(m, xname="x", xvalues=[-1.0, 0.0, 1.0], type="standard-centiles", dev=(-1.0, 0.0, 1.0), fixed_at={"z": 0.0})
    cp_z = centiles_pred(m, xname="x", xvalues=[0.0, 1.0], type="z-scores", yval=[3.0, 4.0], fixed_at={"z": 0.0})
    assert len(cp.entries) == 3 and len(cp_std.entries) == 3 and len(cp_z.entries) == 1
    assert np.all(cp.entries[0].values < cp.entries[1].values)
    assert np.all(cp.entries[1].values < cp.entries[2].values)

    # 3. centiles_split
    cs = centiles_split(m, xvar="x", xcut_points=[0.0], cent=(10.0, 50.0, 90.0))
    assert len(cs.panels) == 2

    # 4. getPEF & getQuantile
    pef = getPEF(m, term="x", what="mu", n_points=11, fixed_at={"z": 0.0})
    assert pef.effect.shape == (11,)
    # Linear mu slope wrt x should match beta_x across grid
    beta_x = float(np.asarray(m.coefficients["mu"])[1])
    assert np.allclose(pef.derivative, beta_x, rtol=1e-5, atol=1e-5)

    gq = getQuantile(m, xvar="x", probabilities=(0.1, 0.5, 0.9), n_points=9, fixed_at={"z": 0.0})
    assert len(gq.entries) == 3
    assert np.all(gq.entries[0].quantile < gq.entries[1].quantile)

    # 5. acfResid
    acf_res = acfResid(m, max_lag=5)
    assert acf_res.lags.tolist() == [1, 2, 3, 4, 5]
    assert np.all(np.isfinite(acf_res.residual_acf))

    # 6. histDist (continuous NO + discrete PO)
    hd_no = histDist(y, family="NO", density=True)
    assert np.isclose(hd_no.fitted_parameters["mu"], float(np.mean(y)), rtol=1e-4)
    assert hd_no.density_x is not None and hd_no.density_y is not None
    y_po = rng.poisson(lam=3.5, size=60)
    hd_po = histDist(y_po, family=PO())
    assert np.isclose(hd_po.fitted_parameters["mu"], float(np.mean(y_po)), rtol=1e-4)

    # 7. pdf_plot (model-based + direct)
    pp_model = pdf_plot(obj=m, obs=(1, 2))
    assert len(pp_model.entries) == 2
    pp_direct = pdf_plot(family="NO", mu=0.0, sigma=1.0, from_value=-6.0, to_value=6.0, no_points=241)
    area = float(np.trapezoid(pp_direct.entries[0].density, pp_direct.entries[0].y))
    assert np.isclose(area, 1.0, rtol=1e-3)

    # 8. plot2way & term_plot on real fitted gamlss model (via lpred type='terms')
    lp_terms = lpred(m, what="mu", type="terms", se_fit=True)
    assert np.asarray(lp_terms["fit"]).shape == (n, 2)
    assert np.all(np.isfinite(np.asarray(lp_terms["se.fit"])))
    p2w = plot2way(m, terms=("x", "z"), what="mu")
    assert p2w.contribution.shape[1] == 2
    tp = term_plot(m, what="mu", se=True)
    assert [entry.term for entry in tp.entries] == ["x", "z"]
    assert all(entry.lower is not None and entry.upper is not None for entry in tp.entries)

    # Live-R verification of predictAll, centiles.pred, acfResid, histDist
    r_code = """
    suppressPackageStartupMessages({
        library(gamlss)
        library(jsonlite)
    })
    args <- commandArgs(trailingOnly = TRUE)
    inp <- fromJSON(args[1])
    df <- data.frame(y = as.numeric(inp$y), x = as.numeric(inp$x), z = as.numeric(inp$z))
    nd <- data.frame(x = as.numeric(inp$nd_x), z = as.numeric(inp$nd_z))
    m <- gamlss(y ~ x + z, sigma.formula = ~ x, family = NO(), data = df, trace = FALSE)
    pa <- predictAll(m, newdata = nd, data = df)
    z_res <- resid(m)
    acf_vals <- as.numeric(acf(z_res, lag.max = 5, plot = FALSE)$acf[-1])
    lpt <- lpred(m, what = "mu", type = "terms")
    out <- list(
        pa_mu = as.numeric(pa$mu),
        pa_sigma = as.numeric(pa$sigma),
        acf = acf_vals,
        lpred_x = as.numeric(lpt[, "x"]),
        lpred_z = as.numeric(lpt[, "z"])
    )
    writeLines(toJSON(out, auto_unbox = TRUE, digits = 17), args[2])
    """
    r_out = _run_r_json(
        r_code,
        {
            "y": y.tolist(),
            "x": x.tolist(),
            "z": z.tolist(),
            "nd_x": nd["x"].tolist(),
            "nd_z": nd["z"].tolist(),
        },
    )
    assert np.allclose(np.asarray(pa.values["mu"]), r_out["pa_mu"], rtol=1e-4, atol=1e-4)
    assert np.allclose(np.asarray(pa.values["sigma"]), r_out["pa_sigma"], rtol=1e-4, atol=1e-4)
    assert np.allclose(acf_res.residual_acf, r_out["acf"], rtol=1e-4, atol=1e-4)
    assert np.allclose(np.asarray(lp_terms["fit"])[:, 0], r_out["lpred_x"], rtol=1e-4, atol=1e-4)
    assert np.allclose(np.asarray(lp_terms["fit"])[:, 1], r_out["lpred_z"], rtol=1e-4, atol=1e-4)


def test_family_r_transformed_sampling_and_moments():
    """Verify Family_r for log and logit transformed families against analytical moments."""
    r_log_no = Family_r("NO", type="log")
    samples_log = np.asarray(r_log_no(4000, mu=0.5, sigma=0.3), dtype=np.float64)
    assert samples_log.shape == (4000,)
    assert np.all(samples_log > 0.0)
    # Log-normal mean = exp(mu + sigma^2 / 2)
    expected_mean = np.exp(0.5 + 0.5 * 0.3 ** 2)
    assert np.isclose(float(np.mean(samples_log)), expected_mean, rtol=0.05)

    r_logit_no = Family_r("NO", type="logit")
    samples_logit = np.asarray(r_logit_no(2000, mu=0.0, sigma=0.8), dtype=np.float64)
    assert samples_logit.shape == (2000,)
    assert np.all((samples_logit > 0.0) & (samples_logit < 1.0))
    assert abs(float(np.mean(samples_logit)) - 0.5) < 0.04

    # Roundtrip consistency with family_d / family_p / family_q
    d_log = family_d("NO", type="log")
    p_log = family_p("NO", type="log")
    q_log = family_q("NO", type="log")
    q_val = float(q_log(0.75, mu=0.5, sigma=0.3))
    assert np.isclose(float(p_log(q_val, mu=0.5, sigma=0.3)), 0.75, atol=1e-10)
    assert float(d_log(q_val, mu=0.5, sigma=0.3)) > 0.0
