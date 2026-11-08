"""R gamlss.ggplots::Copula_ralated.R -- bivariate medians and ECDF_2D parity.

R source: ``R/gamlss.ggplots/R/Copula_ralated.R``.

The live-R oracle sources the vendored R file with one substitution:
``gamlss.foreach::getSVD(df)`` is replaced by a stub because that package is
not installed.  The substitution is safe -- ``median_pc`` overwrites its
``out`` with ``c(median(x), median(y))`` at the end, so the SVD value never
reaches the return value (R dead code, reproduced on the Python side).

Also pinned: R's ``optim`` default ``reltol`` stops ``median_sp`` early;
R's ``reltol=1e-14`` run gives the true geometric median, which the port
matches to ~5e-7 while having a smaller objective value than R's default.
"""
from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss import copula_medians as cm

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = """# median_pc calls gamlss.foreach::getSVD, but its result is overwritten by
# out <- c(median(x), median(y)); a stub is safe for parity.
txt <- readLines('R/gamlss.ggplots/R/Copula_ralated.R')
txt <- gsub('gamlss.foreach::getSVD(df)', 'list(u=matrix(0,nrow(df),2), d=c(1,1))', txt, fixed=TRUE)
eval(parse(text=paste(txt, collapse='\n')), envir=globalenv())
library(jsonlite)
set.seed(7)
x <- c(1.2, 2.5, 3.1, 4.8, 5.0, 7.3, 8.1, 9.6, 10.2, 12.0)
y <- c(3.0, 1.5, 4.2, 2.1, 6.8, 5.5, 7.2, 9.0, 3.3, 8.4)
mat <- cbind(x,y)
fn <- function(par) sum(sqrt((x-par[1])^2+(y-par[2])^2))
r_default <- optim(c(median(x),median(y)), fn)
r_tight <- optim(c(median(x),median(y)), fn, control=list(reltol=1e-14, maxit=10000))

x2 <- c(0.5, 1.5, 2.5, 3.5); y2 <- c(4.0, 2.0, 5.0, 1.0)
out <- list(
  cw = as.numeric(median_cw(x,y)),
  pc = as.numeric(median_pc(x,y)),
  sp = as.numeric(median_sp(x,y)),
  sp_tight = as.numeric(r_tight$par),
  sp_tight_f = as.numeric(r_tight$value),
  f_at_default = as.numeric(fn(r_default$par)),
  xy_cw = as.numeric(xy_median(x,y,"cw")),
  xy_sp = as.numeric(xy_median(x,y,"sp")),
  xy_pc = as.numeric(xy_median(x,y,"pc")),
  xy_default = as.numeric(xy_median(x,y)),
  xy_partial = as.numeric(xy_median(x,y,"s")),
  cw_matrix = as.numeric(median_cw(mat, y=NULL)),
  cw_missing = tryCatch(as.numeric(median_cw(mat)), error=function(e) paste("ERR", conditionMessage(e))),
  cw_x2 = as.numeric(median_cw(x2,y2)),
  pc_x2 = as.numeric(median_pc(x2,y2)),
  sp_x2 = as.numeric(median_sp(x2,y2)),
  ecdf_a = as.numeric(ECDF_2D(5, 5, x, y)),
  ecdf_b = as.numeric(ECDF_2D(0, 0, x, y)),
  ecdf_c = as.numeric(ECDF_2D(20, 20, x, y)),
  ecdf_vec = as.numeric(ECDF_2D(c(2,5,9), c(3,5,7), x, y))
)
cat("JSONSTART"); cat(toJSON(out, auto_unbox=FALSE, digits=17))
"""

_X = np.array([1.2, 2.5, 3.1, 4.8, 5.0, 7.3, 8.1, 9.6, 10.2, 12.0])
_Y = np.array([3.0, 1.5, 4.2, 2.1, 6.8, 5.5, 7.2, 9.0, 3.3, 8.4])
_X2 = np.array([0.5, 1.5, 2.5, 3.5])
_Y2 = np.array([4.0, 2.0, 5.0, 1.0])


def _scalar(value):
    return float(np.asarray(value, dtype=float).ravel()[0])


def _vector(value):
    return np.asarray(value, dtype=float).ravel()


@pytest.fixture(scope="module")
def r_reference():
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.ggplots" / "R" / "Copula_ralated.R").exists():
        pytest.skip("vendored gamlss.ggplots source not present")
    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "medians.R"
        script_path.write_text(_RSCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script_path)],
            capture_output=True,
            text=True,
            timeout=900,
            cwd=str(_REPO),
        )
    if "JSONSTART" not in proc.stdout:
        pytest.skip(f"R failed: {proc.stderr[-400:]}")
    return json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())


# ---------------------------------------------------------------------------
# Live-R parity
# ---------------------------------------------------------------------------


def test_exact_numeric_helpers_match_live_r(r_reference):
    exact = {
        "cw": cm.median_cw(_X, _Y),
        "pc": cm.median_pc(_X, _Y),
        "xy_cw": cm.xy_median(_X, _Y, "cw"),
        "xy_pc": cm.xy_median(_X, _Y, "pc"),
        "xy_default": cm.xy_median(_X, _Y),
        "cw_matrix": cm.median_cw(np.column_stack([_X, _Y])),
        "cw_x2": cm.median_cw(_X2, _Y2),
        "pc_x2": cm.median_pc(_X2, _Y2),
        "ecdf_a": cm.ECDF_2D(5, 5, _X, _Y),
        "ecdf_b": cm.ECDF_2D(0, 0, _X, _Y),
        "ecdf_c": cm.ECDF_2D(20, 20, _X, _Y),
        "ecdf_vec": cm.ECDF_2D(np.array([2, 5, 9]), np.array([3, 5, 7]), _X, _Y),
    }
    for key, value in exact.items():
        np.testing.assert_allclose(
            np.atleast_1d(value),
            _vector(r_reference[key]),
            rtol=0,
            atol=0,
            err_msg=key,
        )


def test_median_sp_r_default_stops_early_and_port_reaches_the_true_minimum(
    r_reference,
):
    """Two-step parity: R's optim default stops early; reltol=1e-14 does not."""
    ours = cm.median_sp(_X, _Y)
    r_default = _vector(r_reference["sp"])
    r_tight = _vector(r_reference["sp_tight"])

    # R's default is ~3e-4 away; the port agrees with R's *tight* run.
    np.testing.assert_allclose(ours, r_default, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(ours, r_tight, rtol=1e-5, atol=1e-5)

    # R's own objective evaluated at our solution is no worse than at R's
    # default solution (and equals R's tight minimum).
    def objective(par):
        return float(np.sum(np.sqrt((_X - par[0]) ** 2 + (_Y - par[1]) ** 2)))

    assert objective(ours) <= _scalar(r_reference["f_at_default"]) + 1e-9
    assert objective(ours) == pytest.approx(
        _scalar(r_reference["sp_tight_f"]), rel=1e-10, abs=1e-9
    )


def test_sp_dispatch_matches_r_default_within_early_stop_tolerance(r_reference):
    for key, value in (
        ("xy_sp", cm.xy_median(_X, _Y, "sp")),
        ("xy_partial", cm.xy_median(_X, _Y, "s")),
        ("sp_x2", cm.median_sp(_X2, _Y2)),
    ):
        np.testing.assert_allclose(
            value, _vector(r_reference[key]), rtol=1e-3, atol=1e-3, err_msg=key
        )


def test_r_requires_explicit_null_for_matrix_form(r_reference):
    """R errors without y (no default); Python accepts the intended form."""
    assert "ERR" in str(r_reference["cw_missing"])
    np.testing.assert_allclose(
        cm.median_cw(np.column_stack([_X, _Y])),
        _vector(r_reference["cw_matrix"]),
        rtol=0,
        atol=0,
    )


# ---------------------------------------------------------------------------
# Reproduced R quirks / pure-Python contracts
# ---------------------------------------------------------------------------


def test_median_pc_dead_code_returns_the_component_wise_median():
    np.testing.assert_allclose(
        cm.median_pc(_X, _Y), cm.median_cw(_X, _Y), rtol=0, atol=0
    )


def test_xy_median_cw_calls_median_pc_not_median_cw(monkeypatch):
    """R bug: method='cw' dispatches to median_pc."""
    calls = {"pc": 0, "cw": 0}
    original_pc = cm.median_pc
    original_cw = cm.median_cw

    def spy_pc(x, y=None):
        calls["pc"] += 1
        return original_pc(x, y)

    def spy_cw(x, y=None):
        calls["cw"] += 1
        return original_cw(x, y)

    monkeypatch.setattr(cm, "median_pc", spy_pc)
    monkeypatch.setattr(cm, "median_cw", spy_cw)
    cm.xy_median(_X, _Y, "cw")
    assert calls == {"pc": 1, "cw": 0}


def test_match_arg_partial_matching_and_invalid_methods():
    assert cm.xy_median(_X, _Y, "c").shape == (2,)
    np.testing.assert_allclose(
        cm.xy_median(_X, _Y, "p"), cm.median_pc(_X, _Y), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        cm.xy_median(_X, _Y, "s"), cm.median_sp(_X, _Y), rtol=0, atol=0
    )
    with pytest.raises(ValueError, match="method must be one of"):
        cm.xy_median(_X, _Y, "xyz")
    with pytest.raises(ValueError, match="method must be one of"):
        cm.xy_median(_X, _Y, "")


def test_two_column_matrix_requires_two_columns():
    with pytest.raises(ValueError, match="2-column matrix"):
        cm.median_cw(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="2-column matrix"):
        cm.median_pc(np.ones((4, 1)))


def test_ecdf_2d_is_between_zero_and_one_and_scalar():
    for vx, vy in ((0, 0), (5, 5), (20, 20), (-1, 100)):
        value = cm.ECDF_2D(vx, vy, _X, _Y)
        assert np.isscalar(value) or np.asarray(value).ndim == 0
        assert 0.0 <= float(value) <= 1.0
    # R recycles vector values and mean() still returns one scalar.
    value = cm.ECDF_2D([2, 5, 9], [3, 5, 7], _X, _Y)
    assert np.asarray(value).ndim == 0
    assert float(value) == pytest.approx(0.4, rel=0, abs=0)


def test_medians_are_finite_two_element_arrays():
    for factory in (cm.median_cw, cm.median_pc, cm.median_sp):
        result = factory(_X, _Y)
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))


def test_module_is_importable_from_package_namespace():
    module = importlib.import_module("omnilss.copula_medians")
    assert module.median_cw is cm.median_cw
