"""R gamlss.mx::get_pdfMX / get_cdfMX / get_pdfNP / get_cdfNP.

R source: ``R/gamlss.mx/R/gamlssMX-10-11-07.R`` (lines 710, 768) and
``R/gamlss.mx/R/gamlssNP.R`` (lines 806, 860).

The four accessors take a fitted mixture object plus an observation number and
return a *function* of y: the fitted pdf/cdf of that observation.  gamlssMX()
and gamlssNP() are not ported yet, so the fitted object is represented by
``MixtureFit`` carrying exactly the fields R's accessors read.  The live-R
oracle below builds the equivalent minimal R objects (with S3 fitted methods
for the component models) and evaluates the *real R accessors* on the same
numbers.

Covered R semantics:

* per-component fitted mu/sigma/nu/tau plus mixing probabilities;
* MX with a ``multinom`` pi model: ``prob[observation, i]`` varies by row;
* NP repeats ``orig.family[1]`` for every component;
* NP shares non-mu parameters (``fitted.gamlssNP`` forces K=1 for sigma/nu/tau),
  so changing the second component's sigma has no effect;
* three-component mixtures with different families (NO + GA + NO).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss.mixtures import (
    MixtureFit,
    get_cdfMX,
    get_cdfNP,
    get_pdfMX,
    get_pdfNP,
)

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = """suppressPackageStartupMessages(library(gamlss))
library(jsonlite)
source('R/gamlss.mx/R/gamlssMX-10-11-07.R')
source('R/gamlss.mx/R/gamlssNP.R')

fitted.mxcomponent <- function(object, parameter=NULL, ...) {
  what <- if (is.null(parameter)) "mu" else parameter
  object[[paste0(what, ".fv")]]
}
mkmodel <- function(mu.fv, sigma.fv) {
  m <- list(parameters=c("mu","sigma"), mu.fv=mu.fv, sigma.fv=sigma.fv)
  class(m) <- "mxcomponent"; m
}
makeMX <- function(models, family, prob, mpi=NULL) {
  obj <- list(K=length(models), N=3, family=family, models=models,
              prob=prob, model.pi=mpi, bd=NULL)
  class(obj) <- "gamlssMX"; obj
}
makeNP <- function(mu.fv, sigma.fv, prob) {
  obj <- list(K=2, N=3, par=c("mu","sigma"), mu.fv=mu.fv, sigma.fv=sigma.fv,
              prob=prob, orig.family="NO", parameters=c("mu","sigma"), bd=NULL)
  class(obj) <- "gamlssNP"; obj
}
y  <- seq(-2, 8, by=0.5)
y2 <- seq(0.25, 8, by=0.25)

m1 <- mkmodel(c(1,2,3), c(0.5,0.6,0.7))
m2 <- mkmodel(c(5,6,7), c(1.0,1.1,1.2))
mx <- makeMX(list(m1,m2), list("NO","NO"), c(0.4,0.6))
np <- makeNP(c(1,2,3, 5,6,7), c(0.5,0.6,0.7, 1.0,1.1,1.2), c(0.4,0.6))

mpi <- list(); class(mpi) <- "multinom"
mxMulti <- makeMX(list(m1,m2), list("NO","NO"),
                  matrix(c(0.1,0.9, 0.5,0.5, 0.8,0.2), nrow=3, ncol=2, byrow=TRUE),
                  mpi=mpi)

ga <- mkmodel(c(2,3,4), c(0.3,0.4,0.5))
m3 <- mkmodel(c(6,7,8), c(1.0,1.1,1.2))
mx3 <- makeMX(list(m1,ga,m3), list("NO","GA","NO"), c(0.2,0.3,0.5))

npShared <- makeNP(c(1,2,3, 5,6,7), c(0.5,0.6,0.7, 9.0,9.1,9.2), c(0.4,0.6))

out <- list(
  y = as.numeric(y), y2 = as.numeric(y2),
  pdfMX1 = as.numeric(get_pdfMX(mx, 1)(y)),
  cdfMX1 = as.numeric(get_cdfMX(mx, 1)(y)),
  pdfMX3 = as.numeric(get_pdfMX(mx, 3)(y)),
  cdfMX3 = as.numeric(get_cdfMX(mx, 3)(y)),
  pdfNP1 = as.numeric(get_pdfNP(np, 1)(y)),
  cdfNP1 = as.numeric(get_cdfNP(np, 1)(y)),
  pdfNP2 = as.numeric(get_pdfNP(np, 2)(y)),
  cdfNP3 = as.numeric(get_cdfNP(np, 3)(y)),
  pdfMulti1 = as.numeric(get_pdfMX(mxMulti, 1)(y)),
  pdfMulti2 = as.numeric(get_pdfMX(mxMulti, 2)(y)),
  cdfMulti3 = as.numeric(get_cdfMX(mxMulti, 3)(y)),
  pdf3fam = as.numeric(get_pdfMX(mx3, 1)(y2)),
  cdf3fam = as.numeric(get_cdfMX(mx3, 2)(y2)),
  pdfNPshared = as.numeric(get_pdfNP(npShared, 1)(y)),
  cdfNPshared = as.numeric(get_cdfNP(npShared, 2)(y))
)
cat("JSONSTART"); cat(toJSON(out, auto_unbox=FALSE, digits=17))
"""

_TWO = (("mu", "sigma"), ("mu", "sigma"))
_FV = {"mu": [1, 2, 3, 5, 6, 7], "sigma": [0.5, 0.6, 0.7, 1.0, 1.1, 1.2]}
_PROB_MULTI = np.array([[0.1, 0.9], [0.5, 0.5], [0.8, 0.2]])
_FV3 = {
    "mu": [1, 2, 3, 2, 3, 4, 6, 7, 8],
    "sigma": [0.5, 0.6, 0.7, 0.3, 0.4, 0.5, 1.0, 1.1, 1.2],
}


def _mx(*, family=("NO", "NO"), prob=(0.4, 0.6), fv=None, params=_TWO,
        model_pi=False, bd=None):
    return MixtureFit(
        K=len(params),
        N=3,
        family=family,
        prob=prob,
        fitted_values=_FV if fv is None else fv,
        parameters=params,
        kind="gamlssMX",
        model_pi=model_pi,
        bd=bd,
    )


def _np(*, prob=(0.4, 0.6), fv=None, orig="NO", params=("mu", "sigma"), bd=None):
    return MixtureFit(
        K=2,
        N=3,
        family=(orig, orig),
        prob=prob,
        fitted_values=_FV if fv is None else fv,
        parameters=params,
        kind="gamlssNP",
        orig_family=orig,
        bd=bd,
    )


@pytest.fixture(scope="module")
def r_reference():
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss.mx" / "R" / "gamlssMX-10-11-07.R").exists():
        pytest.skip("vendored gamlss.mx source not present")
    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "mx.R"
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


def _cases(r_reference):
    y = np.asarray(r_reference["y"], dtype=float)
    y2 = np.asarray(r_reference["y2"], dtype=float)
    mx = _mx()
    npfit = _np()
    multi = _mx(prob=_PROB_MULTI, model_pi=True)
    three = _mx(
        family=("NO", "GA", "NO"),
        prob=(0.2, 0.3, 0.5),
        fv=_FV3,
        params=(("mu", "sigma"),) * 3,
    )
    shared = _np(fv={"mu": [1, 2, 3, 5, 6, 7],
                     "sigma": [0.5, 0.6, 0.7, 9.0, 9.1, 9.2]})
    return {
        "pdfMX1": get_pdfMX(mx, 1)(y),
        "cdfMX1": get_cdfMX(mx, 1)(y),
        "pdfMX3": get_pdfMX(mx, 3)(y),
        "cdfMX3": get_cdfMX(mx, 3)(y),
        "pdfNP1": get_pdfNP(npfit, 1)(y),
        "cdfNP1": get_cdfNP(npfit, 1)(y),
        "pdfNP2": get_pdfNP(npfit, 2)(y),
        "cdfNP3": get_cdfNP(npfit, 3)(y),
        "pdfMulti1": get_pdfMX(multi, 1)(y),
        "pdfMulti2": get_pdfMX(multi, 2)(y),
        "cdfMulti3": get_cdfMX(multi, 3)(y),
        "pdf3fam": get_pdfMX(three, 1)(y2),
        "cdf3fam": get_cdfMX(three, 2)(y2),
        "pdfNPshared": get_pdfNP(shared, 1)(y),
        "cdfNPshared": get_cdfNP(shared, 2)(y),
    }


# ---------------------------------------------------------------------------
# Live-R numerical parity
# ---------------------------------------------------------------------------


def test_all_four_accessors_match_live_r(r_reference):
    ref = r_reference
    ours = _cases(ref)
    for key, values in ours.items():
        expected = np.asarray(ref[key], dtype=float)
        np.testing.assert_allclose(
            values, expected, rtol=1e-12, atol=1e-14, err_msg=key
        )


def test_mx_multinom_pi_varies_by_observation(r_reference):
    """R: PI[[i]] <- object$prob[observation, i] for a multinom pi model."""
    ours = _cases(r_reference)
    assert not np.allclose(ours["pdfMulti1"], ours["pdfMulti2"], rtol=1e-3)
    np.testing.assert_allclose(
        ours["pdfMulti1"], np.asarray(r_reference["pdfMulti1"], float), rtol=1e-12
    )


def test_np_shares_non_mu_parameters_like_r(r_reference):
    """fitted.gamlssNP forces K=1 for sigma, so the 2nd component's sigma is ignored."""
    ours = _cases(r_reference)
    np.testing.assert_allclose(ours["pdfNPshared"], ours["pdfNP1"], rtol=1e-12)
    np.testing.assert_allclose(
        ours["cdfNPshared"], np.asarray(r_reference["cdfNPshared"], float), rtol=1e-12
    )


def test_three_component_mixed_families_match_live_r(r_reference):
    ours = _cases(r_reference)
    for key in ("pdf3fam", "cdf3fam"):
        np.testing.assert_allclose(
            ours[key], np.asarray(r_reference[key], float), rtol=1e-12, atol=1e-14
        )


# ---------------------------------------------------------------------------
# Pure-Python contracts
# ---------------------------------------------------------------------------


def test_accessors_return_callables_of_y():
    y = np.linspace(-2.0, 8.0, 21)
    for factory in (get_pdfMX, get_cdfMX):
        fn = factory(_mx(), 2)
        assert callable(fn)
        assert np.asarray(fn(y)).shape == y.shape
    for factory in (get_pdfNP, get_cdfNP):
        fn = factory(_np(), 2)
        assert callable(fn)
        assert np.asarray(fn(y)).shape == y.shape


def test_default_observation_is_one():
    y = np.linspace(-2.0, 8.0, 21)
    np.testing.assert_allclose(
        get_pdfMX(_mx())(y), get_pdfMX(_mx(), 1)(y), rtol=0, atol=0
    )
    np.testing.assert_allclose(
        get_cdfNP(_np())(y), get_cdfNP(_np(), 1)(y), rtol=0, atol=0
    )


def test_wrong_object_class_is_rejected_with_r_message():
    with pytest.raises(ValueError, match="should be an gamlssMX object"):
        get_pdfMX(_np(), 1)
    with pytest.raises(ValueError, match="should be an gamlssNP object"):
        get_cdfNP(_mx(), 1)


def test_observation_out_of_range_is_rejected():
    with pytest.raises(IndexError):
        get_pdfMX(_mx(), 0)
    with pytest.raises(IndexError):
        get_pdfMX(_mx(), 4)
    with pytest.raises(IndexError):
        get_pdfNP(_np(), -1)


def test_mapping_input_is_accepted():
    payload = {
        "K": 2,
        "N": 3,
        "family": ("NO", "NO"),
        "prob": (0.4, 0.6),
        "fitted_values": _FV,
        "parameters": _TWO,
        "kind": "gamlssMX",
    }
    y = np.linspace(-2.0, 8.0, 21)
    np.testing.assert_allclose(
        get_pdfMX(payload, 1)(y), get_pdfMX(_mx(), 1)(y), rtol=0, atol=0
    )


def test_fitted_values_accept_flat_matrix_and_shared_layouts():
    y = np.linspace(-2.0, 8.0, 21)
    flat = _mx()
    matrix = _mx(fv={k: np.asarray(v).reshape(2, 3) for k, v in _FV.items()})
    shared = _mx(fv={"mu": np.array([1.0, 2.0, 3.0]),
                     "sigma": np.array([0.5, 0.6, 0.7])})
    np.testing.assert_allclose(
        get_pdfMX(flat, 1)(y), get_pdfMX(matrix, 1)(y), rtol=0, atol=0
    )
    # A shared length-N vector means both components use the same values.
    assert not np.allclose(get_pdfMX(flat, 1)(y), get_pdfMX(shared, 1)(y))


def test_binomial_bd_branch_is_reported_not_silently_wrong():
    with pytest.raises(NotImplementedError, match="bd"):
        get_pdfMX(_mx(bd=10), 1)
    with pytest.raises(NotImplementedError, match="bd"):
        get_cdfMX(_mx(family=("BI", "NO")), 1)
    with pytest.raises(NotImplementedError, match="bd"):
        get_pdfNP(_np(orig="BI"), 1)


def test_pdf_is_nonnegative_and_cdf_is_monotone():
    y = np.linspace(-8.0, 12.0, 201)
    for pdf, cdf in (
        (get_pdfMX(_mx(), 1), get_cdfMX(_mx(), 1)),
        (get_pdfNP(_np(), 1), get_cdfNP(_np(), 1)),
    ):
        density = np.asarray(pdf(y), dtype=float)
        cumulative = np.asarray(cdf(y), dtype=float)
        assert np.all(density >= 0.0)
        assert np.all(np.diff(cumulative) >= -1e-12)
        assert cumulative[-1] == pytest.approx(1.0, abs=1e-8)
        assert cumulative[0] == pytest.approx(0.0, abs=1e-8)


def test_mixture_fit_is_immutable():
    from dataclasses import FrozenInstanceError

    fit = _mx()
    with pytest.raises(FrozenInstanceError):
        fit.K = 5  # type: ignore[misc]
