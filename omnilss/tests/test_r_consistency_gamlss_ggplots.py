"""Live-R consistency evidence for gamlss.ggplots (D2.265).

Covers the numerical helpers where R is the reference:

* fit_PB against the vendored P-spline source (lambda / df / sigma / deviance / coef);
* model_GAIC's matrix branch against the vendored function;
* predict_pdf / predict_cdf / quantile_gamlss against dNO / pNO / qNO evaluated at
  the same predicted parameters (isolating the plotting helpers from the fit).
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
    fit_PB,
    gamlss,
    model_GAIC,
    predict_cdf,
    predict_pdf,
    quantile_gamlss,
)
from omnilss.prediction import predict_params

_REPO = Path(__file__).resolve().parents[2]
_R_SCRIPT = """
suppressPackageStartupMessages(library(gamlss.dist))
suppressPackageStartupMessages(library(jsonlite))
source("R/gamlss.ggplots/R/fit_PB.R")
source("R/gamlss.ggplots/R/model_GAIC.R")
args <- commandArgs(trailingOnly = TRUE)
d <- fromJSON(args[1])
pb <- fit_PB(x = d$pb_x, y = d$pb_y, nseg = 20, lambda = 10, order = 2, degree = 3,
             plot = FALSE)
mat <- matrix(d$gaic, ncol = 2, byrow = TRUE,
              dimnames = list(c("a", "b", "c"), c("GAIC", "df")))
gm <- model_GAIC(mat, plot = FALSE, which = 1)
grid <- seq(d$from, d$to, length.out = d$no_points)
dens <- dNO(grid, mu = d$mu[1], sigma = d$sigma[1])
cdf <- pNO(grid, mu = d$mu[1], sigma = d$sigma[1])
q90 <- qNO(0.9, mu = d$mu, sigma = d$sigma)
out <- list(
  pb = list(lambda = pb$lambda, df = pb$df, sigma = pb$sig, dev = pb$deviance,
            coef = as.numeric(pb$coef[1:4])),
  gaic = list(gaic = as.numeric(gm[, "GAIC"]), delta = as.numeric(gm[, "delta"]),
              scaled = as.numeric(gm[, "scaled"])),
  predict = list(dens = as.numeric(dens), cdf = as.numeric(cdf),
                 q90 = as.numeric(q90))
)
cat(toJSON(out, digits = 17, auto_unbox = TRUE))
"""


def _payload() -> dict:
    rng = np.random.default_rng(7)
    n = 300
    pb_x = np.sort(rng.uniform(0.0, 10.0, size=n))
    pb_y = np.sin(pb_x) + rng.normal(scale=0.2, size=n)
    return {
        "pb_x": pb_x.tolist(),
        "pb_y": pb_y.tolist(),
        "gaic": [100.0, 1.0, 101.0, 2.0, 110.0, 3.0],
        "from": -3.0,
        "to": 5.0,
        "no_points": 201,
    }


def _fit_model() -> tuple[object, dict, np.ndarray]:
    rng = np.random.default_rng(2)
    n = 100
    x = rng.normal(size=n)
    y = 1.0 + 0.5 * x + rng.normal(scale=0.4, size=n)
    model = gamlss("y ~ x", family="NO", data={"y": y, "x": x}, max_iter=60)
    new_x = np.array([-1.0, 0.0, 1.0])
    params = predict_params(model, {"x": new_x})
    payload = _payload()
    payload["mu"] = np.asarray(params["mu"], dtype=float).tolist()
    payload["sigma"] = np.asarray(params["sigma"], dtype=float).tolist()
    return model, payload, new_x


def _live_r(payload: dict) -> dict:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.ggplots" / "R" / "fit_PB.R").exists():
        pytest.skip("vendored gamlss.ggplots source not present")
    with tempfile.TemporaryDirectory() as tmp:
        data_file = Path(tmp) / "gg.json"
        data_file.write_text(json.dumps(payload), encoding="utf-8")
        script = Path(tmp) / "ref.R"
        script.write_text(_R_SCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script), str(data_file)],
            capture_output=True,
            text=True,
            timeout=900,
            cwd=str(_REPO),
        )
    if proc.returncode != 0:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def setup():
    model, payload, new_x = _fit_model()
    return model, payload, new_x, _live_r(payload)


@pytest.mark.slow
def test_fit_pb_matches_live_r(setup):
    _, payload, _, reference = setup
    result = fit_PB(
        x=np.asarray(payload["pb_x"], dtype=float),
        y=np.asarray(payload["pb_y"], dtype=float),
        nseg=20, lambda_=10, order=2, degree=3, plot=False,
    )
    expected = reference["pb"]
    assert result.lambda_ == pytest.approx(float(expected["lambda"]), rel=1e-9)
    assert result.df == pytest.approx(float(expected["df"]), rel=1e-9)
    assert result.sigma == pytest.approx(float(expected["sigma"]), rel=1e-9)
    assert result.deviance == pytest.approx(float(expected["dev"]), rel=1e-9)
    np.testing.assert_allclose(
        result.coef[:4], np.asarray(expected["coef"], dtype=float), rtol=1e-8, atol=1e-9
    )


@pytest.mark.slow
def test_model_gaic_matrix_matches_live_r(setup):
    _, _, _, reference = setup
    matrix = np.array([[100.0, 1.0], [101.0, 2.0], [110.0, 3.0]])
    table = model_GAIC(matrix, plot=False)
    expected = reference["gaic"]
    np.testing.assert_allclose(
        table["GAIC"].to_numpy(), np.asarray(expected["gaic"], dtype=float)
    )
    np.testing.assert_allclose(
        table["delta"].to_numpy(), np.asarray(expected["delta"], dtype=float)
    )
    np.testing.assert_allclose(
        table["scaled"].to_numpy(), np.asarray(expected["scaled"], dtype=float)
    )


@pytest.mark.slow
def test_predict_helpers_match_live_r_quantities(setup):
    model, payload, new_x, reference = setup
    pdf = predict_pdf(
        model, {"x": new_x}, from_=payload["from"], to=payload["to"],
        no_points=payload["no_points"], plot=False,
    )
    np.testing.assert_allclose(
        pdf["density"]["1"], np.asarray(reference["predict"]["dens"], dtype=float), rtol=1e-12
    )
    cdf = predict_cdf(
        model, {"x": new_x}, from_=payload["from"], to=payload["to"],
        no_points=payload["no_points"], plot=False,
    )
    np.testing.assert_allclose(
        cdf["cdf"]["1"], np.asarray(reference["predict"]["cdf"], dtype=float), rtol=1e-12
    )
    quantiles = quantile_gamlss(model, 0.9, {"x": new_x})
    np.testing.assert_allclose(
        quantiles, np.asarray(reference["predict"]["q90"], dtype=float), rtol=1e-12
    )
