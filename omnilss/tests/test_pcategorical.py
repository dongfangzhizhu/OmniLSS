"""R gamlss pcat() / gamlss.pcat: penalised categorical smoother.

Compared with live R: the penalty matrix is identical, a fixed lambda and the
ML lambda iteration agree to machine precision, and the Lp = 2 df target agrees
to the root tolerance.

Documented divergence: with Lp != 2 the edf(lambda) curve is *discontinuous*
(the omega iteration switches branch), so a df target has no continuous root.
R's uniroot parks at the jump with edf = 2.899 for a df = 3 target, while a
tight root search lands on the other side (edf = 3.0).  The fitted values then
differ by ~1.4e-2 for that path; the test pins both behaviours down instead of
pretending they match.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from omnilss.pcategorical import (
    PCatFit,
    PCatSpec,
    gamlss_pcat,
    pairwise_differences,
    pcat,
)

_REPO = Path(__file__).resolve().parents[2]

_RSCRIPT = """suppressPackageStartupMessages(library(gamlss))
library(jsonlite)
a <- commandArgs(trailingOnly = TRUE); d <- fromJSON(a[1])
y <- d$y; w <- rep(1, length(y))
fac <- factor(d$g, levels = seq_len(max(d$g)))
run <- function(df = NULL, lam = NULL, Lp = 0, method = "ML") {
  s <- pcat(fac, df = df, lambda = lam, method = method, Lp = Lp)
  f <- gamlss.pcat(s, y, w)
  list(lambda = f$lambda, edf = f$coefSmo$edf, nldf = f$nl.df,
       coef = as.numeric(f$coefSmo$coef), fv = as.numeric(f$fitted.values),
       var = as.numeric(f$var), sigb2 = f$coefSmo$sigb2, sige2 = f$coefSmo$sige2)
}
Dm <- attr(pcat(fac, df = 3), "D")
out <- list(D = as.numeric(Dm), Ddim = c(nrow(Dm), ncol(Dm)),
            fixed = run(lam = 3), bydf0 = run(df = 3), bydf2 = run(df = 3, Lp = 2),
            ml = run())
cat("JSONSTART"); cat(toJSON(out, auto_unbox = FALSE, digits = 17))
"""


def _factor(n: int = 40) -> pd.Series:
    codes = np.repeat([1, 2, 3, 4], [10, 12, 8, 10])[:n]
    return pd.Series(pd.Categorical(codes, categories=[1, 2, 3, 4]))


def _data():
    rs = np.random.RandomState(3)
    fac = _factor()
    codes = fac.cat.codes.to_numpy() + 1
    y = rs.normal(size=codes.shape[0]) + codes * 0.5
    return fac, y, np.ones(codes.shape[0]), codes


def test_pairwise_differences_matches_genD() -> None:
    expected = np.array(
        [
            [-1.0, 1.0, 0.0, 0.0],
            [-1.0, 0.0, 1.0, 0.0],
            [0.0, -1.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0, 1.0],
            [0.0, -1.0, 0.0, 1.0],
            [0.0, 0.0, -1.0, 1.0],
        ]
    )
    np.testing.assert_allclose(pairwise_differences(4), expected, rtol=0, atol=0)
    assert pairwise_differences(3).shape == (3, 3)


def test_pcat_builds_a_one_hot_design() -> None:
    fac, _y, _w, codes = _data()
    spec = pcat(fac, df=3)
    assert isinstance(spec, PCatSpec)
    assert spec.X.shape == (40, 4)
    np.testing.assert_allclose(spec.X.sum(axis=1), 1.0, rtol=0, atol=0)
    assert np.all(np.argmax(spec.X, axis=1) + 1 == codes)
    assert spec.D.shape == (6, 4)
    assert spec.levels == [1, 2, 3, 4]
    assert spec.Lp == 0.0 and spec.kappa == 1e-5 and spec.c_crit == 1e-4


def test_pcat_guards() -> None:
    fac = _factor()
    with pytest.warns(UserWarning, match="lambda was negative"):
        assert pcat(fac, lambda_=-1.0).lambda_ == 1.0
    with pytest.warns(UserWarning, match="the df are set to  4"):
        assert pcat(fac, df=10.0).df == 4.0
    with pytest.warns(UserWarning, match="the df are set to 0"):
        assert pcat(fac, df=-2.0).df == 0.0
    # df == p switches to an unpenalised fit.
    assert pcat(fac, df=4.0).lambda_ == 0.0
    with pytest.raises(ValueError, match="should be one of"):
        pcat(fac, method="GCV")


def test_pcat_fixed_lambda_relations() -> None:
    fac, y, w, _c = _data()
    fit = gamlss_pcat(pcat(fac, lambda_=3.0), y, w)
    assert isinstance(fit, PCatFit)
    assert fit.lambda_ == 3.0
    assert fit.nl_df == pytest.approx(fit.edf - 1.0, rel=1e-12)
    np.testing.assert_allclose(fit.residuals, y - fit.fitted_values, rtol=1e-12)
    np.testing.assert_allclose(fit.se, np.sqrt(fit.var), rtol=1e-12)


def test_pcat_df_target_with_the_eigen_shortcut() -> None:
    fac, y, w, _c = _data()
    fit = gamlss_pcat(pcat(fac, df=3, Lp=2.0), y, w)
    assert fit.edf == pytest.approx(3.0, abs=1e-6)


def test_pcat_df_target_with_the_regpen_path() -> None:
    fac, y, w, _c = _data()
    fit = gamlss_pcat(pcat(fac, df=3), y, w)
    # The edf curve jumps across 3, so the root search returns the upper branch.
    assert fit.edf == pytest.approx(3.0, abs=1e-6)
    assert 0.0 < fit.lambda_ < 1.0


def test_pcat_ml_iteration_selects_lambda() -> None:
    fac, y, w, _c = _data()
    fit = gamlss_pcat(pcat(fac), y, w)
    assert fit.lambda_ > 0.0
    assert fit.sigb2 is not None and fit.sige2 is not None


def test_pcat_predicts_with_the_appended_rows() -> None:
    fac, y, w, codes = _data()
    extra = pd.Series(pd.Categorical([1, 2], categories=[1, 2, 3, 4]))
    base = pcat(fac, lambda_=3.0)
    extra_design = np.zeros((2, 4))
    extra_design[np.arange(2), extra.cat.codes.to_numpy()] = 1.0
    spec = replace(base, X=np.vstack([base.X, extra_design]))
    predicted = gamlss_pcat(spec, y, w, xeval=np.zeros(2))
    fit = gamlss_pcat(base, y, w)
    np.testing.assert_allclose(predicted, extra_design @ fit.coef, rtol=1e-12)


def test_pcat_rejects_other_specs() -> None:
    _fac, y, w, _c = _data()
    with pytest.raises(TypeError, match="returned by pcat"):
        gamlss_pcat(object(), y, w)


@pytest.mark.slow
def test_pcat_matches_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    if not (_REPO / "R" / "gamlss" / "R" / "p-categorical.R").exists():
        pytest.skip("vendored p-categorical.R not present")
    fac, y, w, codes = _data()
    payload = {"y": y.tolist(), "g": codes.tolist()}
    with tempfile.TemporaryDirectory() as tmp:
        data_path = Path(tmp) / "d.json"
        data_path.write_text(json.dumps(payload), encoding="utf-8")
        script_path = Path(tmp) / "pcat.R"
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
    ref = json.loads(proc.stdout.split("JSONSTART", 1)[1].strip())

    reference_penalty = np.asarray(ref["D"], dtype=float).reshape(
        tuple(int(value) for value in ref["Ddim"]), order="F"
    )
    np.testing.assert_allclose(pairwise_differences(4), reference_penalty, rtol=0, atol=0)

    def scalar(block, key):
        return float(np.asarray(block[key], dtype=float).ravel()[0])

    fixed = gamlss_pcat(pcat(fac, lambda_=3.0), y, w)
    np.testing.assert_allclose(
        fixed.coef, np.asarray(ref["fixed"]["coef"], float), rtol=1e-11, atol=1e-12
    )
    np.testing.assert_allclose(
        fixed.fitted_values, np.asarray(ref["fixed"]["fv"], float), rtol=1e-11, atol=1e-12
    )
    assert fixed.edf == pytest.approx(scalar(ref["fixed"], "edf"), rel=1e-10)

    by_df2 = gamlss_pcat(pcat(fac, df=3, Lp=2.0), y, w)
    # R solves this root with uniroot's default 1.22e-4 tolerance.
    assert by_df2.lambda_ == pytest.approx(scalar(ref["bydf2"], "lambda"), rel=1e-4)
    np.testing.assert_allclose(
        by_df2.fitted_values, np.asarray(ref["bydf2"]["fv"], float), rtol=1e-5, atol=1e-7
    )

    ml = gamlss_pcat(pcat(fac), y, w)
    assert ml.lambda_ == pytest.approx(scalar(ref["ml"], "lambda"), rel=1e-9)
    np.testing.assert_allclose(ml.coef, np.asarray(ref["ml"]["coef"], float), rtol=1e-10, atol=1e-12)

    # Lp = 0 with a df target: R stops on the lower side of the jump (edf 2.899),
    # a tight root search lands on the upper side (edf 3.0).
    by_df0 = gamlss_pcat(pcat(fac, df=3), y, w)
    assert scalar(ref["bydf0"], "edf") == pytest.approx(2.8986, abs=1e-3)
    assert by_df0.edf == pytest.approx(3.0, abs=1e-6)
    assert by_df0.lambda_ == pytest.approx(scalar(ref["bydf0"], "lambda"), rel=1e-2)
    np.testing.assert_allclose(
        by_df0.fitted_values, np.asarray(ref["bydf0"]["fv"], float), rtol=0, atol=2e-2
    )