"""R gamlss P-spline variants: pbm / pbo / pbp / pbz / pbc / pvc.

One R driver builds all six spec objects and calls the matching gamlss.<name>
fitter with a fixed lambda.  Both the spec attributes (X, D, D1, df, inter) and
the fit outputs (beta, fitted, var, edf, nl.df) are compared with live R; the
fixed-lambda path agrees to machine precision.

Documented deviations (see omnilss.pspline_variants):
* pvc(by=...) varying-coefficient interactions are not ported.
* lambda estimation (ML/GAIC/GCV/df) delegates to gamlss_pb and is approximate;
  only the fixed-lambda path is R-verified here.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss.pspline_variants import (
    PSplineVariant,
    gamlss_pbc,
    gamlss_pbm,
    gamlss_pbo,
    gamlss_pbp,
    gamlss_pbz,
    gamlss_pvc,
    pbc,
    pbm,
    pbo,
    pbp,
    pbz,
    pvc,
)

_REPO = Path(__file__).resolve().parents[2]
_LAMBDA = 3.0

_RSCRIPT = """suppressPackageStartupMessages(library(gamlss))
library(jsonlite)
a <- commandArgs(trailingOnly = TRUE)
d <- fromJSON(a[1])
x <- d$x; y <- d$y; w <- rep(1, length(x)); lam <- d$lambda
specs <- list(pbm = pbm(x, df = 3), pbo = pbo(x, df = 3), pbp = pbp(x, df = 3),
              pbz = pbz(x, df = 3), pbc = pbc(x, df = 3), pvc = pvc(x, df = 3))
pack_spec <- function(s) list(
  Xdim = dim(attr(s, "X")), Ddim = dim(attr(s, "D")),
  X = as.numeric(attr(s, "X")), D = as.numeric(attr(s, "D")),
  df = attr(s, "df"), inter = attr(s, "control")$inter,
  hasD1 = !is.null(attr(s, "D1")),
  D1 = if (!is.null(attr(s, "D1"))) as.numeric(attr(s, "D1")) else NULL)
sp <- lapply(specs, pack_spec)
fit_of <- function(name, s) {
  attr(s, "lambda") <- lam
  attr(s, "df") <- NULL
  fn <- switch(name, pbm = gamlss.pbm, pbo = gamlss.pbo, pbp = gamlss.pbp,
               pbz = gamlss.pbz, pbc = gamlss.pbc, pvc = gamlss.pvc)
  f <- fn(s, y, w)
  list(beta = as.numeric(f$coefSmo$coef), edf = f$coefSmo$edf,
       fv = as.numeric(f$fitted.values), var = as.numeric(f$var),
       nldf = f$nl.df, lambda = f$lambda)
}
ft <- lapply(names(specs), function(nm) fit_of(nm, specs[[nm]]))
names(ft) <- names(specs)
cat("JSONSTART"); cat(toJSON(list(spec = sp, fit = ft), auto_unbox = FALSE, digits = 17))
"""

_SPECS = {"pbm": pbm, "pbo": pbo, "pbp": pbp, "pbz": pbz, "pbc": pbc, "pvc": pvc}
_FITS = {
    "pbm": gamlss_pbm,
    "pbo": gamlss_pbo,
    "pbp": gamlss_pbp,
    "pbz": gamlss_pbz,
    "pbc": gamlss_pbc,
    "pvc": gamlss_pvc,
}


def _data(seed: int = 3, n: int = 80):
    np.random.seed(seed)
    x = np.linspace(0.0, 1.0, n)
    y = np.sin(2 * np.pi * x) + np.random.normal(0.0, 0.2, n)
    return x, y, np.ones(n)


def _run_r(x, y):
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "d.json"
        data_path.write_text(
            json.dumps({"x": x.tolist(), "y": y.tolist(), "lambda": _LAMBDA}), encoding="utf-8"
        )
        script_path = Path(tmp) / "s.R"
        script_path.write_text(_RSCRIPT, encoding="utf-8")
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


def _scalar(value):
    return float(np.asarray(value).ravel()[0])


def test_all_specs_build() -> None:
    x, _, _ = _data()
    for name, constructor in _SPECS.items():
        spec = constructor(x, df=3, lambda_=_LAMBDA)
        assert isinstance(spec, PSplineVariant)
        assert spec.name == name
        assert spec.X.shape[0] == x.shape[0]
        assert spec.X.shape[1] == spec.D.shape[1]
        assert np.all(np.isfinite(spec.X))
        assert spec.lambda_ == _LAMBDA


def test_df_and_shape_conventions() -> None:
    x, _, _ = _data()
    # Five variants use the pb-style 13-column basis; pbc is cyclic with 10.
    for name in ("pbm", "pbo", "pbp", "pbz", "pvc"):
        spec = _SPECS[name](x, df=3)
        assert spec.X.shape[1] == 13
        assert spec.D.shape == (11, 13)
    assert pbc(x, df=3).X.shape[1] == 10
    assert pbc(x, df=3).D.shape == (10, 10)
    # df bookkeeping: +2 for pb-style, +1 for pbc, unchanged for pvc.
    assert _SPECS["pbm"](x, df=3).df == 5.0
    assert _SPECS["pbo"](x, df=3).df == 5.0
    assert _SPECS["pbp"](x, df=3).df == 5.0
    assert _SPECS["pbz"](x, df=3).df == 5.0
    assert pbc(x, df=3).df == 4.0
    assert pvc(x, df=3).df == 3.0


def test_pbz_extra_first_difference_penalty() -> None:
    x, _, _ = _data()
    spec = pbz(x, df=3)
    assert spec.D1 is not None
    assert spec.D1.shape == (12, 13)
    np.testing.assert_allclose(spec.D1[0, :2], [-1.0, 1.0])


def test_pbc_cyclic_penalty_wraps() -> None:
    x, _, _ = _data()
    spec = pbc(x, df=3)
    assert spec.D[0, 0] == pytest.approx(2 * np.cos(2 * np.pi / 10), rel=1e-12)
    assert spec.D[0, -1] == pytest.approx(-1.0)
    assert spec.D[-1, 0] == pytest.approx(-1.0)
    assert spec.max_df == 8.0


def test_inter_bookkeeping() -> None:
    x, _, _ = _data()
    assert all(_SPECS[name](x).control["inter"] == 10 for name in _SPECS)
    # pbo does not reduce inter by the number of distinct values.
    few = np.repeat(np.linspace(0.0, 1.0, 5), 4)
    assert pbo(few).control["inter"] == 10
    assert pbp(few).control["inter"] == 5


def test_df_guard_and_validation() -> None:
    x, _, _ = _data()
    with pytest.warns(UserWarning, match="exceed the number of columns"):
        assert pbm(x, df=50).df == 5.0
    with pytest.raises(ValueError, match="matric declare it as a vector"):
        pbm(np.zeros((5, 2)))
    with pytest.raises(ValueError, match="'arg' should be one of"):
        pbm(x, mono="sideways")


def test_pvc_by_is_not_ported() -> None:
    x, _, _ = _data()
    with pytest.raises(NotImplementedError, match="not ported"):
        pvc(x, by=x)


def test_fixed_lambda_fits_have_the_r_nl_df_offsets() -> None:
    x, y, w = _data()
    offsets = {"pbm": 1.0, "pbo": 2.0, "pbp": 2.0, "pbz": 1.0, "pbc": 1.0, "pvc": 2.0}
    for name, constructor in _SPECS.items():
        fit = _FITS[name](constructor(x, df=3, lambda_=_LAMBDA), y, w)
        assert fit.name == name
        assert fit.nl_df == pytest.approx(fit.edf - offsets[name], rel=1e-12)
        assert fit.lambda_ == _LAMBDA
        np.testing.assert_allclose(fit.residuals, y - fit.fitted_values, rtol=1e-12)


def test_prediction_and_wrong_spec() -> None:
    x, y, w = _data()
    grid = np.linspace(0.1, 0.9, 5)
    for name, constructor in _SPECS.items():
        spec = constructor(x, df=3, lambda_=_LAMBDA)
        fit = _FITS[name](spec, y, w)
        np.testing.assert_allclose(
            _FITS[name](spec, y, w, xeval=grid), fit.predict(grid), rtol=1e-12
        )
    with pytest.raises(TypeError, match="returned by pbm"):
        gamlss_pbm(pbo(x, lambda_=_LAMBDA), y, w)
    with pytest.raises(ValueError, match="different lengths"):
        gamlss_pbo(pbo(x, lambda_=_LAMBDA), np.zeros(5), np.ones(5))


def test_estimation_paths_delegate() -> None:
    x, y, w = _data()
    # No lambda -> delegate to gamlss_pb; documented as approximate but must
    # still return a usable fit with the variant's nl.df offset.
    fit = gamlss_pbo(pbo(x, df=3), y, w)
    assert fit.name == "pbo"
    assert np.all(np.isfinite(fit.fitted_values))
    assert fit.nl_df == pytest.approx(fit.edf - 2.0, rel=1e-12)


@pytest.mark.slow
def test_specs_match_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss" / "R" / "pbm.R").exists():
        pytest.skip("vendored pbm.R not present")
    x, y, _ = _data()
    ref = _run_r(x, y)["spec"]
    for name, constructor in _SPECS.items():
        spec = constructor(x, df=3, lambda_=_LAMBDA)
        r = ref[name]
        shape = tuple(int(v) for v in r["Xdim"])
        np.testing.assert_allclose(
            spec.X, np.asarray(r["X"], float).reshape(shape, order="F"),
            rtol=1e-11, atol=1e-12, err_msg=f"{name} X",
        )
        dshape = tuple(int(v) for v in r["Ddim"])
        np.testing.assert_allclose(
            spec.D, np.asarray(r["D"], float).reshape(dshape, order="F"),
            rtol=1e-11, atol=1e-12, err_msg=f"{name} D",
        )
        assert spec.df == pytest.approx(_scalar(r["df"])), name
        assert spec.control["inter"] == int(_scalar(r["inter"])), name
        assert (spec.D1 is not None) == bool(_scalar(r["hasD1"])), name
        if spec.D1 is not None:
            np.testing.assert_allclose(
                spec.D1.flatten(order="F"), np.asarray(r["D1"], float), atol=1e-12
            )


@pytest.mark.slow
def test_fixed_lambda_fits_match_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss" / "R" / "pbm.R").exists():
        pytest.skip("vendored pbm.R not present")
    x, y, w = _data()
    ref = _run_r(x, y)["fit"]
    for name, constructor in _SPECS.items():
        fit = _FITS[name](constructor(x, df=3, lambda_=_LAMBDA), y, w)
        r = ref[name]
        np.testing.assert_allclose(
            fit.coef, np.asarray(r["beta"], float), rtol=1e-10, atol=1e-11, err_msg=f"{name} beta"
        )
        np.testing.assert_allclose(
            fit.fitted_values, np.asarray(r["fv"], float), rtol=1e-10, atol=1e-11, err_msg=f"{name} fv"
        )
        np.testing.assert_allclose(
            fit.var, np.asarray(r["var"], float), rtol=1e-10, atol=1e-11, err_msg=f"{name} var"
        )
        assert fit.edf == pytest.approx(_scalar(r["edf"]), abs=1e-9), name
        assert fit.nl_df == pytest.approx(_scalar(r["nldf"]), abs=1e-9), name
