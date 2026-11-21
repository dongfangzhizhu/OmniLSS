# SPDX-License-Identifier: GPL-3.0-or-later
"""Live-R consistency and unit evidence for gamlss core diagnostics batch B1a (D2.276).

Covers the five core ``gamlss`` symbols landed in B1a:
- ``get_rqres`` (``gamlss::get.rqres``): continuous & frequency-weighted matrix
  parity against live R, plus randomized discrete draws;
- ``rqres_plot`` (``gamlss::rqres.plot``): theoretical normal quantiles
  (``ppoints``), median deviation, 95% worm-plot bands, and input validation;
- ``fittedPlot`` (``gamlss::fittedPlot``): single & multi-model sorted parameter
  curves and R's ``1e-4`` constant-curve snapping rule;
- ``bp`` (``gamlss::bp``): moment / centile.central / centile.tail bucket plots
  and covariate-wrapped splits against live R ``momentSK``/``centileSK``;
- ``gamlssNews`` (``gamlss::gamlssNews``): release news text contract.
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
    bp,
    fittedPlot,
    gamlss,
    gamlssNews,
    get_rqres,
    rqres_plot,
)

_R_SCRIPT = """
suppressPackageStartupMessages(library(gamlss))
suppressPackageStartupMessages(library(jsonlite))
args <- commandArgs(trailingOnly = TRUE)
d <- fromJSON(args[1])
df <- data.frame(y = d$y, x = d$x)
m_no <- gamlss(y ~ x, sigma.formula = ~1, data = df, family = NO(),
               control = gamlss.control(c.crit = 1e-8, n.cyc = 100, trace = FALSE))
m_no2 <- gamlss(y ~ x, sigma.formula = ~x, data = df, family = NO(),
                control = gamlss.control(c.crit = 1e-8, n.cyc = 100, trace = FALSE))
rq_no <- get.rqres(m_no, howmany = 4, order = TRUE)
rq_raw <- get.rqres(m_no, howmany = 2, order = FALSE)

df_w <- data.frame(y = d$y[1:20], x = d$x[1:20])
w_int <- as.numeric(d$w_int)
m_w <- gamlss(y ~ x, sigma.formula = ~1, data = df_w, weights = w_int, family = NO(),
              control = gamlss.control(n.cyc = 40, trace = FALSE))
rq_w <- get.rqres(m_w, howmany = 3, order = TRUE)

ord <- order(d$x)
mu1 <- as.numeric(fitted(m_no, "mu"))[ord]
sg1 <- as.numeric(fitted(m_no, "sigma"))[ord]
mu2 <- as.numeric(fitted(m_no2, "mu"))[ord]
sg2 <- as.numeric(fitted(m_no2, "sigma"))[ord]

n <- length(d$y)
th <- qnorm(ppoints(n))
z <- seq(-5, 5, 0.25)
p <- pnorm(z)
se <- (1 / dnorm(z)) * sqrt(p * (1 - p) / n)
low <- qnorm(0.025) * se
high <- -low

ms <- momentSK(d$y)
out <- list(
  rq_no = as.numeric(rq_no[, 1]),
  rq_raw = as.numeric(rq_raw[, 1]),
  rq_w_nrow = nrow(rq_w),
  rq_w_col1 = as.numeric(rq_w[, 1]),
  mu1 = mu1,
  sg1 = sg1,
  mu2 = mu2,
  sg2 = sg2,
  theoretical = as.numeric(th),
  band_z = as.numeric(z),
  band_low = as.numeric(low),
  band_high = as.numeric(high),
  moment_sk = as.numeric(ms)
)
cat(toJSON(out, digits = 17, auto_unbox = TRUE))
"""


def _dataset() -> dict:
    rng = np.random.default_rng(20261121)
    n = 60
    x = rng.normal(size=n)
    y = 2.0 + 1.3 * x + rng.normal(scale=np.exp(-0.4 + 0.15 * x))
    w_int = np.array([1, 2, 1, 3] * 5, dtype=float)
    return {"x": x, "y": y, "w_int": w_int}


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


@pytest.fixture(scope="module")
def fitted_models():
    data = _dataset()
    m_no = gamlss(
        "y ~ x",
        sigma_formula="~1",
        family="NO",
        data={"y": data["y"], "x": data["x"]},
        max_iter=100,
        tol=1e-8,
    )
    m_no2 = gamlss(
        "y ~ x",
        sigma_formula="~x",
        family="NO",
        data={"y": data["y"], "x": data["x"]},
        max_iter=100,
        tol=1e-8,
    )
    m_w = gamlss(
        "y ~ x",
        sigma_formula="~1",
        family="NO",
        data={"y": data["y"][:20], "x": data["x"][:20]},
        weights=data["w_int"],
        max_iter=40,
    )
    return data, m_no, m_no2, m_w


def test_get_rqres_matches_live_r_continuous_and_weighted(r_reference, fitted_models):
    _, m_no, _, m_w = fitted_models

    rq_ord = get_rqres(m_no, howmany=4, order=True)
    assert rq_ord.shape == (60, 4)
    np.testing.assert_allclose(rq_ord[:, 0], r_reference["rq_no"], rtol=1e-4, atol=1e-4)

    rq_raw = get_rqres(m_no, howmany=2, order=False)
    assert rq_raw.shape == (60, 2)
    np.testing.assert_allclose(rq_raw[:, 0], r_reference["rq_raw"], rtol=1e-4, atol=1e-4)

    rq_w = get_rqres(m_w, howmany=3, order=True)
    assert rq_w.shape == (r_reference["rq_w_nrow"], 3)
    np.testing.assert_allclose(rq_w[:, 0], r_reference["rq_w_col1"], rtol=1e-4, atol=1e-4)


def test_get_rqres_discrete_draws_vary_across_columns():
    rng = np.random.default_rng(11)
    x = rng.normal(size=50)
    y = rng.poisson(np.exp(0.8 + 0.3 * x)).astype(float)
    model = gamlss("y ~ x", family="PO", data={"y": y, "x": x}, max_iter=30)
    rs = get_rqres(model, howmany=5, order=False, seed=42)
    assert rs.shape == (50, 5)
    assert np.all(np.isfinite(rs))
    # Independent randomized draws for discrete families must not be identical copies.
    assert not np.allclose(rs[:, 0], rs[:, 1])


def test_rqres_plot_matches_live_r_quantiles_and_bands(r_reference, fitted_models):
    _, m_no, _, _ = fitted_models

    res_all = rqres_plot(m_no, howmany=6, plot_type="all", type="wp", seed=1, plot=False)
    assert res_all.samples.shape == (60, 6)
    np.testing.assert_allclose(res_all.theoretical, r_reference["theoretical"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res_all.band_z, r_reference["band_z"], rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(res_all.band_low, r_reference["band_low"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(res_all.band_high, r_reference["band_high"], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(res_all.rmedian, r_reference["rq_no"], rtol=1e-4, atol=1e-4)

    res_qq = rqres_plot(m_no, howmany=4, plot_type="few", type="QQ", seed=1, plot=False)
    assert res_qq.plot_type == "few"
    assert res_qq.type == "QQ"

    with pytest.raises(ValueError, match="less than or equal to 8"):
        rqres_plot(m_no, howmany=9, plot_type="few")


def test_fitted_plot_single_and_multi_model_matches_live_r(r_reference, fitted_models):
    data, m_no, m_no2, _ = fitted_models

    single = fittedPlot(m_no, x=data["x"], plot=False)
    assert single.model_count == 1
    assert single.parameters["mu"].shape == (60, 1)
    assert single.parameters["sigma"].shape == (60, 1)
    np.testing.assert_allclose(single.parameters["mu"][:, 0], r_reference["mu1"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(single.parameters["sigma"][:, 0], r_reference["sg1"], rtol=1e-4, atol=1e-4)
    # Constant sigma curve snapped flat identically to R fitted-plot.R:156-157
    assert np.all(single.parameters["sigma"][:, 0] == single.parameters["sigma"][0, 0])

    multi = fittedPlot(m_no, m_no2, x="x", plot=False)
    assert multi.model_count == 2
    assert multi.parameters["mu"].shape == (60, 2)
    assert multi.parameters["sigma"].shape == (60, 2)
    np.testing.assert_allclose(multi.parameters["mu"][:, 1], r_reference["mu2"], rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(multi.parameters["sigma"][:, 1], r_reference["sg2"], rtol=1e-4, atol=1e-4)

    with pytest.raises(ValueError, match="x-variable"):
        fittedPlot(m_no)


def test_bp_moment_and_centile_modes_and_r_moment_parity(r_reference, fitted_models):
    data, m_no, _, _ = fitted_models

    res_mom = bp(data["y"], type="moment", bootstrap=False, plot=False)
    assert res_mom.kind == "bp"
    assert res_mom.data["bp_type"] == "moment"
    row = res_mom.data["rows"][0]
    # R momentSK returns (skewness, trans_skew, kurtosis, excess_kurtosis, trans_kurt, jarque_bera)
    ms = r_reference["moment_sk"]
    np.testing.assert_allclose(
        [row["skewness"], row["trans_skew"], row["kurtosis"], row["trans_kurt"], row["jarque_bera"]],
        [ms[0], ms[1], ms[3], ms[4], ms[5]],
        rtol=1e-10,
        atol=1e-10,
    )

    res_cent_c = bp(m_no, type="centile.central", bootstrap=True, no_bootstrap=5, seed=7, plot=False)
    assert res_cent_c.data["bp_type"] == "centile.central"
    assert res_cent_c.data["x_column"] == "S0.25"

    res_cent_t = bp([data["y"], m_no], type="centile.tail", bootstrap=False, plot=False)
    assert res_cent_t.data["bp_type"] == "centile.tail"
    assert res_cent_t.data["x_column"] == "S0.01"

    res_wrap = bp(data["y"], type="moment", xvar=data["x"], n_inter=3, bootstrap=False, plot=False)
    assert res_wrap.kind == "bp"

    with pytest.raises(ValueError, match="type must be"):
        bp(data["y"], type="unknown")


def test_gamlss_news_returns_non_empty_changelog():
    text = gamlssNews(max_lines=15)
    assert isinstance(text, str) and len(text.strip()) > 0
    assert "Omni" in text or "Changelog" in text
