"""R gamlss cy(): the cyclic P-spline spec and its gamlss.cy fitting step.

The spec test compares the wrapped basis X and the cyclic penalty D with the
attributes that live R's cy() attaches to the covariate.  The fit tests compare
beta / fitted / var / nl.df against R's gamlss.cy() driven with the same X and
D (R's cy() is not called headlessly because it assigns into the gamlss call
frame; the tagged object is built directly instead).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from omnilss.smoother_fits import gamlss_cy
from omnilss.smoother_specs import CYSpline, cbase, cdiff, cy

_REPO = Path(__file__).resolve().parents[2]

_SPEC_SCRIPT = """suppressPackageStartupMessages(library(gamlss))
library(jsonlite)
a <- commandArgs(trailingOnly = TRUE)
d <- fromJSON(a[1])
x <- d$x
s <- cy(x, df = 3)
X <- attr(s, "X"); D <- attr(s, "D")
cat("JSONSTART"); cat(toJSON(list(
  xdim = as.integer(dim(X)), ddim = as.integer(dim(D)),
  X = as.numeric(X), D = as.numeric(D),
  df = as.numeric(attr(s, "df")), inter = as.integer(attr(s, "control")$inter)
), auto_unbox = FALSE, digits = 17))
"""

_FIT_SCRIPT = """suppressPackageStartupMessages(library(gamlss))
library(jsonlite)
a <- commandArgs(trailingOnly = TRUE)
d <- fromJSON(a[1])
x <- d$x; y <- d$y; w <- rep(1, length(x))
X <- matrix(d$X, nrow = d$nr, ncol = d$nc)
D <- matrix(d$D, nrow = d$dr, ncol = d$dc)
run <- function(lam = NULL, df = NULL, method = "ML", k = 2) {
  s <- rep(0, length(x))
  attr(s, "X") <- X
  attr(s, "D") <- D
  attr(s, "lambda") <- lam
  attr(s, "df") <- df
  attr(s, "control") <- list(method = method, order = 2, k = k, start = 10, inter = 10, degree = 3)
  e <- new.env(); assign("start.Lambda.abcd", 10, envir = e)
  attr(s, "gamlss.env") <- e
  attr(s, "NameForLambda") <- "start.Lambda.abcd"
  f <- gamlss.cy(s, y, w)
  list(lambda = f$lambda, nldf = f$nl.df, edf = f$coefSmo$edf,
       beta = as.numeric(f$coefSmo$coef), fv = as.numeric(f$fitted.values),
       var = as.numeric(f$var))
}
out <- list(fixed = run(lam = 3), bydf = run(df = 4), ml = run(method = "ML"),
            gaic = run(method = "GAIC", k = 2), gcv = run(method = "GCV", k = 2))
cat("JSONSTART"); cat(toJSON(out, auto_unbox = FALSE, digits = 17))
"""


def _x(n: int = 60) -> np.ndarray:
    return np.linspace(0.0, 1.0, n)


def _y(seed: int = 7) -> np.ndarray:
    x = _x()
    rs = np.random.RandomState(seed)
    return np.sin(2 * np.pi * x) + rs.normal(0.0, 0.2, len(x))


def _run(script: str, payload: dict) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "d.json"
        data_path.write_text(json.dumps(payload), encoding="utf-8")
        script_path = Path(tmp) / "s.R"
        script_path.write_text(script, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script_path), str(data_path)],
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(_REPO),
        )
    if "JSONSTART" not in proc.stdout:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    return json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())


def _fit_payload() -> dict:
    x = _x()
    y = _y()
    spec = cy(x)
    X, D = spec.X, spec.D
    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "X": X.flatten(order="F").tolist(),
        "D": D.flatten(order="F").tolist(),
        "nr": X.shape[0],
        "nc": X.shape[1],
        "dr": D.shape[0],
        "dc": D.shape[1],
    }


def test_cy_spec_shapes_and_df() -> None:
    spec = cy(_x(), df=3)
    assert isinstance(spec, CYSpline)
    assert spec.X.shape == (60, 10)
    assert spec.D.shape == (10, 10)
    # cy() stores df + 1 (pb stores df + 2).
    assert spec.df == 4.0
    assert spec.control["inter"] == 10


def test_cy_basis_partitions_unity_and_wraps() -> None:
    x = _x()
    spec = cy(x)
    np.testing.assert_allclose(spec.X.sum(axis=1), 1.0, rtol=1e-10)
    # The last `degree` columns of the unwrapped basis are folded into the first.
    from omnilss.smoother_specs import bbase

    b0, _ = bbase(x, 0.0, 1.0, 10, 3)
    np.testing.assert_allclose(spec.X[:, :3], b0[:, :3] + b0[:, 10:13], rtol=1e-12)


def test_cy_penalty_is_cyclic() -> None:
    n = 10
    d = cdiff(n)
    assert d.shape == (n, n)
    assert d[0, 0] == pytest.approx(2.0 * np.cos(2 * np.pi / n), rel=1e-12)
    assert d[0, 1] == pytest.approx(-1.0)
    assert d[0, -1] == pytest.approx(-1.0)
    assert d[-1, 0] == pytest.approx(-1.0)


def test_cbase_shape() -> None:
    x = _x()
    assert cbase(x, 0.0, 1.0, 10, 3).shape == (60, 10)


def test_cy_df_guard_and_negative_df() -> None:
    x = _x()
    with pytest.warns(UserWarning, match="exceed the number of columns"):
        spec = cy(x, df=50)
    assert spec.df == 4.0  # 3 + 1
    with pytest.warns(UserWarning, match="df are set to 1"):
        assert cy(x, df=0.0).df == 1.0


def test_cy_does_not_reduce_intervals_by_distinct_count() -> None:
    x = np.repeat(np.linspace(0.0, 1.0, 20), 3)
    assert cy(x).control["inter"] == 10  # length < 100 -> 10, no distinct reduction


def test_cy_rejects_matrices() -> None:
    with pytest.raises(ValueError, match="must be a vector"):
        cy(np.zeros((5, 2)))


def test_cy_fixed_lambda_fit() -> None:
    x, y = _x(), _y()
    fit = gamlss_cy(cy(x, lambda_=3.0), y, np.ones(60))
    assert fit.name == "cy"
    assert fit.coef.shape == (10,)
    assert fit.lambda_ == 3.0
    # cy() reports nl.df = edf - 1, unlike pb's edf - 2.
    assert fit.nl_df == pytest.approx(fit.edf - 1.0, rel=1e-12)
    np.testing.assert_allclose(fit.residuals, y - fit.fitted_values, rtol=1e-12)


def test_cy_df_target() -> None:
    x, y = _x(), _y()
    fit = gamlss_cy(cy(x, df=3), y, np.ones(60))
    assert fit.edf == pytest.approx(4.0, abs=1e-5)
    assert fit.nl_df == pytest.approx(3.0, abs=1e-5)


def test_cy_selection_methods_return_fits() -> None:
    x, y = _x(), _y()
    base = cy(x)
    for method in ("ML", "ML-1", "EM", "GAIC", "GCV"):
        spec = replace(base, control={**base.control, "method": method})
        fit = gamlss_cy(spec, y, np.ones(60))
        assert 1e-7 <= fit.lambda_ <= 1e7
        assert np.all(np.isfinite(fit.fitted_values))
        assert fit.name == "cy"


def test_cy_prediction() -> None:
    x, y = _x(), _y()
    spec = cy(x, lambda_=3.0)
    grid = np.linspace(0.1, 0.9, 5)
    np.testing.assert_allclose(
        gamlss_cy(spec, y, np.ones(60), xeval=grid),
        gamlss_cy(spec, y, np.ones(60)).predict(grid),
        rtol=1e-12,
    )


def test_cy_rejects_other_specs_and_bad_lengths() -> None:
    x, y = _x(), _y()
    from omnilss.smoother_specs import pb

    with pytest.raises(TypeError, match="returned by cy"):
        gamlss_cy(pb(x), y, np.ones(60))
    with pytest.raises(ValueError, match="different lengths"):
        gamlss_cy(cy(x, lambda_=3.0), np.zeros(5), np.ones(5))


@pytest.mark.slow
def test_cy_spec_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss" / "R" / "cy.R").exists():
        pytest.skip("vendored cy.R not present")
    x = _x()
    ref = _run(_SPEC_SCRIPT, {"x": x.tolist()})
    spec = cy(x, df=3)
    shape = tuple(int(value) for value in ref["xdim"])
    np.testing.assert_allclose(
        spec.X, np.asarray(ref["X"], float).reshape(shape, order="F"), rtol=1e-11, atol=1e-12
    )
    dshape = tuple(int(value) for value in ref["ddim"])
    np.testing.assert_allclose(
        spec.D, np.asarray(ref["D"], float).reshape(dshape, order="F"), rtol=1e-11, atol=1e-12
    )
    assert spec.df == float(np.asarray(ref["df"]).ravel()[0])
    assert spec.control["inter"] == int(np.asarray(ref["inter"]).ravel()[0])


@pytest.mark.slow
def test_cy_fit_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss" / "R" / "cy.R").exists():
        pytest.skip("vendored cy.R not present")
    x, y = _x(), _y()
    w = np.ones(60)
    ref = _run(_FIT_SCRIPT, _fit_payload())

    base = cy(x)
    cases = {
        "fixed": cy(x, lambda_=3.0),
        "bydf": cy(x, df=3),
        "ml": base,
        "gcv": replace(base, control={**base.control, "method": "GCV"}),
    }
    for name, spec in cases.items():
        fit = gamlss_cy(spec, y, w)
        r = ref[name]
        assert fit.lambda_ == pytest.approx(
            float(np.asarray(r["lambda"]).ravel()[0]), rel=1e-5
        ), name
        assert fit.edf == pytest.approx(float(np.asarray(r["edf"]).ravel()[0]), abs=1e-6), name
        assert fit.nl_df == pytest.approx(
            float(np.asarray(r["nldf"]).ravel()[0]), abs=1e-6
        ), name
        np.testing.assert_allclose(
            fit.coef, np.asarray(r["beta"], float), rtol=1e-9, atol=1e-9, err_msg=name
        )
        np.testing.assert_allclose(
            fit.fitted_values, np.asarray(r["fv"], float), rtol=1e-9, atol=1e-9, err_msg=name
        )
        np.testing.assert_allclose(
            fit.var, np.asarray(r["var"], float), rtol=1e-9, atol=1e-9, err_msg=name
        )

    # GAIC is optimizer-dependent: R's nlminb stops at ~2.6e6 while the bounded
    # search reaches the 1e7 cap, but the fitted curves agree to ~1e-7.
    gaic = gamlss_cy(replace(base, control={**base.control, "method": "GAIC"}), y, w)
    np.testing.assert_allclose(
        gaic.fitted_values, np.asarray(ref["gaic"]["fv"], float), atol=1e-6
    )
