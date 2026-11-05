"""R gamlss ps() and cs() smoother constructors.

ps() builds the Eilers-Marx B-spline design (splineDesign) plus the difference
penalty, and is compared element-wise with live R.  cs() only tags the covariate
for gamlss.cs() (the spline is built at fit time), so its contract is the
control list, the class marker and the recorded NA positions.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss.smoother_specs import CSSpline, PSSpline, cs, ps

_REPO = Path(__file__).resolve().parents[2]

_PS_SCRIPT = """suppressPackageStartupMessages(library(gamlss))
x <- seq(0, 1, length.out = 200)
s <- ps(x)
b <- attr(s, "design.matrix"); p <- attr(s, "pen.augment")
s2 <- ps(x, ps.intervals = 10, degree = 2, order = 2)
b2 <- attr(s2, "design.matrix"); p2 <- attr(s2, "pen.augment")
cat(jsonlite::toJSON(list(
  dim = as.integer(dim(b)), pdim = as.integer(dim(p)),
  knots = as.numeric(attr(s, "knots")), df = as.numeric(attr(s, "df")),
  order = as.numeric(attr(s, "order")), B = as.numeric(b), P = as.numeric(p),
  dim2 = as.integer(dim(b2)), pdim2 = as.integer(dim(p2)), B2 = as.numeric(b2), P2 = as.numeric(p2)
), auto_unbox = FALSE, digits = 17))
"""

_CS_SCRIPT = """suppressPackageStartupMessages(library(gamlss))
x <- c(1, 2, NA, 4, 5)
s <- cs(x)
cat(jsonlite::toJSON(list(
  cls = attr(s, "class"),
  low = as.numeric(attr(s, "control")$control.spar$low),
  high = as.numeric(attr(s, "control")$control.spar$high),
  penalty = as.numeric(attr(s, "control")$penalty),
  nas = as.integer(attr(s, "NAs")),
  values = as.numeric(s)
), auto_unbox = FALSE, digits = 17))
"""


def _run(script: str) -> dict:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "spec.R"
        path.write_text(script, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(path)], capture_output=True, text=True, timeout=300
        )
    if proc.returncode != 0:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_ps_shapes_and_settings() -> None:
    spline = ps(np.linspace(0.0, 1.0, 200))
    assert isinstance(spline, PSSpline)
    assert spline.design.shape == (200, 23)
    assert spline.penalty.shape == (20, 23)
    assert spline.df == 5.0
    assert spline.order == 3
    assert spline.X is spline.design


def test_ps_requires_df_or_lambda() -> None:
    with pytest.raises(ValueError, match="df or lambda should be set"):
        ps(np.linspace(0.0, 1.0, 50), df=None, lambda_=None)


def test_ps_order_clipped_to_the_basis() -> None:
    with pytest.warns(UserWarning, match="order was too large"):
        spline = ps(np.linspace(0.0, 1.0, 200), ps_intervals=2, order=10)
    assert spline.order == spline.design.shape[1] - 1


def test_ps_df_and_lambda_guards() -> None:
    with pytest.warns(UserWarning, match="the df are set to 1"):
        assert ps(np.linspace(0.0, 1.0, 100), df=0.5).df == 1.0
    with pytest.warns(UserWarning, match="lambda was negative"):
        assert ps(np.linspace(0.0, 1.0, 100), df=None, lambda_=-1.0).lambda_ == 0.0


def test_ps_rows_sum_to_one() -> None:
    spline = ps(np.linspace(0.0, 1.0, 200))
    np.testing.assert_allclose(spline.design.sum(axis=1), 1.0, rtol=1e-10)


def test_ps_rejects_matrices() -> None:
    with pytest.raises(ValueError, match="declare it as a vector"):
        ps(np.zeros((4, 2)))


def test_cs_records_the_r_attributes() -> None:
    spline = cs([1.0, 2.0, np.nan, 4.0, 5.0])
    assert isinstance(spline, CSSpline)
    assert spline.smooth_class == "smooth"
    assert spline.control["control.spar"] == {"low": -1.5, "high": 2}
    assert spline.control["penalty"] == 1.4
    assert spline.na_positions == (2,)
    assert spline.df == 3.0


def test_cs_c_spar_forms() -> None:
    assert cs([1.0, 2.0, 3.0], c_spar=[-1.0, 1.0]).control["control.spar"] == {"low": -1.0, "high": 1.0}
    assert cs([1.0, 2.0, 3.0], c_spar={"low": -2.0, "high": 3.0}).control["control.spar"] == {
        "low": -2.0,
        "high": 3.0,
    }
    with pytest.raises(ValueError, match="c.spar is not defined properly"):
        cs([1.0, 2.0, 3.0], c_spar=[1.0])


def test_cs_df_guard_and_matrix_rejection() -> None:
    with pytest.warns(UserWarning, match="the df are set to 3"):
        assert cs([1.0, 2.0, 3.0], df=-1).df == 3.0
    with pytest.raises(ValueError, match="bivariate"):
        cs(np.zeros((4, 2)))


@pytest.mark.slow
def test_ps_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss" / "R" / "PS.R").exists():
        pytest.skip("vendored PS.R not present")
    ref = _run(_PS_SCRIPT)

    x = np.linspace(0.0, 1.0, 200)
    spline = ps(x)
    shape = tuple(int(value) for value in ref["dim"])
    np.testing.assert_allclose(
        spline.design,
        np.asarray(ref["B"], dtype=float).reshape(shape, order="F"),
        rtol=1e-12,
        atol=1e-13,
    )
    pshape = tuple(int(value) for value in ref["pdim"])
    np.testing.assert_allclose(
        spline.penalty,
        np.asarray(ref["P"], dtype=float).reshape(pshape, order="F"),
        rtol=0,
        atol=0,
    )
    np.testing.assert_allclose(spline.knots, np.asarray(ref["knots"], dtype=float), rtol=1e-14)
    assert spline.df == float(np.asarray(ref["df"]).ravel()[0])
    assert spline.order == int(np.asarray(ref["order"]).ravel()[0])

    custom = ps(x, ps_intervals=10, degree=2, order=2)
    shape2 = tuple(int(value) for value in ref["dim2"])
    np.testing.assert_allclose(
        custom.design,
        np.asarray(ref["B2"], dtype=float).reshape(shape2, order="F"),
        rtol=1e-12,
        atol=1e-13,
    )
    pshape2 = tuple(int(value) for value in ref["pdim2"])
    np.testing.assert_allclose(
        custom.penalty,
        np.asarray(ref["P2"], dtype=float).reshape(pshape2, order="F"),
        rtol=0,
        atol=0,
    )


@pytest.mark.slow
def test_cs_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss" / "R" / "cubicSplines-10-08-12.R").exists():
        pytest.skip("vendored cubicSplines source not present")
    ref = _run(_CS_SCRIPT)
    spline = cs([1.0, 2.0, np.nan, 4.0, 5.0])
    assert spline.smooth_class in list(ref["cls"])
    assert spline.control["control.spar"]["low"] == float(np.asarray(ref["low"]).ravel()[0])
    assert spline.control["control.spar"]["high"] == float(np.asarray(ref["high"]).ravel()[0])
    assert spline.control["penalty"] == float(np.asarray(ref["penalty"]).ravel()[0])
    assert [index + 1 for index in spline.na_positions] == list(ref["nas"])
    expected_values = np.asarray(
        [np.nan if value == "NA" else value for value in ref["values"]], dtype=float
    )
    assert np.array_equal(np.isnan(spline.values), np.isnan(expected_values))
    np.testing.assert_allclose(
        spline.values[~np.isnan(spline.values)],
        expected_values[~np.isnan(expected_values)],
        rtol=0,
        atol=0,
    )
