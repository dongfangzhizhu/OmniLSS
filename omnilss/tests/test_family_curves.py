"""gamlss.ggplots ``family_pdf`` / ``family_cdf`` curve data -- live-R parity.

R sources: ``R/gamlss.ggplots/R/family_pdf.R`` / ``family_cdf.R``.  These take
a family + parameter values (no fitted model); a stub-ggplot2 probe captures
the ``data.frame(y.var, matrix)`` handed to ggplot.

Live-R verified (14 cases, all <= 1e-12):

* Continuous grid ``seq(from,to,length=no.points)`` vs Discrete ``seq(by=1)``;
* parameter recycling ``rep(param, length=lobs)`` with ``lobs = max lenghts``;
* ``dFAM`` / ``pFAM`` evaluation with ``lower.tail`` (survival) and 1/2/3-param
  families (NO / PO / GA / NBI / TF);
* R data-frame column naming: ``pdfArr``/``cdfArr`` for one curve, else
  ``X1``, ``X2``, ...

R quirks pinned: single-curve column is NOT ``X1``; ``predict_*``/``fitted_*``
(no fitted model here) and binomial ``bd`` families are out of scope.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

from omnilss.family_curves import FamilyCurve, family_cdf, family_pdf

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = r"""
suppressPackageStartupMessages(library(jsonlite))
suppressPackageStartupMessages(library(gamlss.dist))
store <- new.env(); store$df <- NULL
for (file in c('family_pdf.R','family_cdf.R')) {
  txt <- gsub('ggplot2::','',readLines(paste0('R/gamlss.ggplots/R/',file)),fixed=TRUE)
  stub <- function(...) structure(list(),class='ggstub')
  for (f in c('aes','geom_step','geom_segment','geom_point','geom_line','geom_area',
              'labs','xlim','ggtitle','aes_string','theme_minimal')) assign(f, stub, envir=globalenv())
  rec_ggplot <- function(data=NULL, ...) { store$df <- as.data.frame(data); structure(list(),class='ggstub') }
  Ops.ggstub <- function(e1,e2) e1
  ggplot <- rec_ggplot
  .binom <- c('BI','BB','ZIBI'); .gamlss.bi.list <- .binom
  eval(parse(text=paste(txt, collapse='\n')), envir=globalenv())
}
grab <- function(expr) { store$df<-NULL; force(expr); as.list(store$df) }
out <- list(
  pdf_NO1  = grab(family_pdf('NO', mu=0, sigma=1, from=-3, to=3, no.points=9)),
  pdf_NO2  = grab(family_pdf('NO', mu=c(0,1), sigma=c(1,2), from=-2, to=4, no.points=11)),
  cdf_NO1  = grab(family_cdf('NO', mu=0, sigma=1, from=-3, to=3, no.points=9)),
  cdf_NO2  = grab(family_cdf('NO', mu=c(0,1), sigma=c(1,2), from=-2, to=4, no.points=11)),
  cdf_surv = grab(family_cdf('NO', mu=0, sigma=1, from=-3, to=3, no.points=9, lower.tail=FALSE)),
  pdf_PO   = grab(family_pdf('PO', mu=2, from=0, to=8, no.points=9)),
  cdf_PO   = grab(family_cdf('PO', mu=2, from=0, to=8, no.points=9)),
  pdf_GA   = grab(family_pdf('GA', mu=2, sigma=0.3, from=0.1, to=6, no.points=9)),
  pdf_NBI  = grab(family_pdf('NBI', mu=3, sigma=0.5, from=0, to=10, no.points=11)),
  pdf_TF   = grab(family_pdf('TF', mu=0, sigma=1, nu=10, from=-4, to=4, no.points=9)),
  cdf_TF   = grab(family_cdf('TF', mu=0, sigma=1, nu=10, from=-4, to=4, no.points=9)),
  pdf_recycle = grab(family_pdf('NO', mu=c(-1,0,1), sigma=1, from=-3, to=3, no.points=7)),
  cdf_NBI  = grab(family_cdf('NBI', mu=3, sigma=0.5, from=0, to=10, no.points=11)),
  cdf_GA_surv = grab(family_cdf('GA', mu=2, sigma=0.3, from=0.1, to=6, no.points=9, lower.tail=FALSE))
)
cat("JSONSTART"); cat(toJSON(out, auto_unbox=FALSE, digits=17))
"""


@pytest.fixture(scope="module")
def r_reference():
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    src = _REPO / "R" / "gamlss.ggplots" / "R" / "family_pdf.R"
    if not src.exists():
        pytest.skip("vendored gamlss.ggplots source not present")
    with tempfile.TemporaryDirectory() as tmp:
        script_path = Path(tmp) / "family.R"
        script_path.write_text(_RSCRIPT, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(script_path)], capture_output=True, text=True,
            timeout=900, cwd=str(_REPO),
        )
    # D2.42: a broken R probe must FAIL the gate, never silently skip it.
    assert "JSONSTART" in proc.stdout, f"R probe failed: {proc.stderr[-800:]}"
    return json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())


_CASES = {
    "pdf_NO1": ("d", "NO", dict(mu=0, sigma=1), dict(from_=-3, to=3, no_points=9)),
    "pdf_NO2": ("d", "NO", dict(mu=[0, 1], sigma=[1, 2]), dict(from_=-2, to=4, no_points=11)),
    "cdf_NO1": ("p", "NO", dict(mu=0, sigma=1), dict(from_=-3, to=3, no_points=9)),
    "cdf_NO2": ("p", "NO", dict(mu=[0, 1], sigma=[1, 2]), dict(from_=-2, to=4, no_points=11)),
    "cdf_surv": ("p", "NO", dict(mu=0, sigma=1), dict(from_=-3, to=3, no_points=9, lower_tail=False)),
    "pdf_PO": ("d", "PO", dict(mu=2), dict(from_=0, to=8, no_points=9)),
    "cdf_PO": ("p", "PO", dict(mu=2), dict(from_=0, to=8, no_points=9)),
    "pdf_GA": ("d", "GA", dict(mu=2, sigma=0.3), dict(from_=0.1, to=6, no_points=9)),
    "pdf_NBI": ("d", "NBI", dict(mu=3, sigma=0.5), dict(from_=0, to=10, no_points=11)),
    "pdf_TF": ("d", "TF", dict(mu=0, sigma=1, nu=10), dict(from_=-4, to=4, no_points=9)),
    "cdf_TF": ("p", "TF", dict(mu=0, sigma=1, nu=10), dict(from_=-4, to=4, no_points=9)),
    "pdf_recycle": ("d", "NO", dict(mu=[-1, 0, 1], sigma=1), dict(from_=-3, to=3, no_points=7)),
    "cdf_NBI": ("p", "NBI", dict(mu=3, sigma=0.5), dict(from_=0, to=10, no_points=11)),
    "cdf_GA_surv": ("p", "GA", dict(mu=2, sigma=0.3), dict(from_=0.1, to=6, no_points=9, lower_tail=False)),
}


def _assert_curve(ours: FamilyCurve, ref: dict, key: str, lower_tail: bool) -> None:
    assert isinstance(ours, FamilyCurve)
    assert ours.lower_tail is lower_tail
    keys = [c for c in ref if c != "y.var"]
    assert list(ours.columns) == keys, f"{key}: columns {ours.columns} vs {keys}"
    np.testing.assert_allclose(ours.y, np.asarray(ref["y.var"], float),
                               rtol=0, atol=1e-12, err_msg=f"{key}:y")
    for j, col in enumerate(keys):
        np.testing.assert_allclose(
            ours.values[:, j], np.asarray(ref[col], float),
            rtol=0, atol=1e-12, err_msg=f"{key}:{col}",
        )
    assert ours.values.shape == (len(ours.y), len(keys))


def test_all_family_curves_match_live_r(r_reference):
    for key, (which, fam, params, opts) in _CASES.items():
        fn = family_pdf if which == "d" else family_cdf
        lower_tail = opts.get("lower_tail", True)
        ours = fn(fam, **params, **opts)
        _assert_curve(ours, r_reference[key], key, lower_tail)


def test_continuous_vs_discrete_grid():
    # Continuous: length=no_points; Discrete: seq(from,to,by=1)
    cont = family_pdf("NO", mu=0, sigma=1, from_=-3, to=3, no_points=9)
    np.testing.assert_allclose(cont.y, np.linspace(-3, 3, 9), rtol=0, atol=0)
    disc = family_cdf("PO", mu=2, from_=0, to=8, no_points=9)
    np.testing.assert_allclose(disc.y, np.arange(0, 9, 1.0), rtol=0, atol=0)
    assert len(disc.y) == 9  # by=1 ignores no.points


def test_single_vs_multi_curve_column_names():
    one = family_pdf("NO", mu=0, sigma=1)
    assert one.columns == ("pdfArr",) and one.values.shape[1] == 1
    onec = family_cdf("NO", mu=0, sigma=1)
    assert onec.columns == ("cdfArr",)
    multi = family_pdf("NO", mu=[0, 1, 2], sigma=1)
    assert multi.columns == ("X1", "X2", "X3") and multi.values.shape[1] == 3


def test_survival_is_one_minus_cdf():
    cdf = family_cdf("NO", mu=0, sigma=1, from_=-3, to=3, no_points=15)
    surv = family_cdf("NO", mu=0, sigma=1, from_=-3, to=3, no_points=15,
                      lower_tail=False)
    np.testing.assert_allclose(cdf.values[:, 0] + surv.values[:, 0], 1.0,
                               rtol=0, atol=1e-12)


def test_parameter_recycling_and_missing_params():
    # scalar sigma recycled across vector mu (lobs = 3)
    c = family_pdf("NO", mu=[-1, 0, 1], sigma=1, from_=-2, to=2, no_points=7)
    assert c.values.shape == (7, 3)
    # missing required parameter raises (NO needs mu + sigma)
    with pytest.raises(ValueError):
        family_pdf("NO", mu=0)
    with pytest.raises(ValueError):
        family_cdf("TF", mu=0, sigma=1)  # TF needs nu too
