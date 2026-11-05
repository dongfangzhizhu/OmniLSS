"""Tests for R centileSkew and gamlss.scope migrations.

R sources: R/gamlss.dist/R/Functions_for_SK_in_gamlss.R (centileSkew) and
R/gamlss/R/stepGAIC-03-10-13.R (gamlss.scope).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from omnilss.centile_skew import centile_skew
from omnilss.gamlss_scope import gamlss_scope

_X = list(range(1, 11))
_W = [1, 2, 1, 2, 1, 2, 1, 2, 1, 2]


def test_centile_skew_symmetric_is_zero() -> None:
    result = centile_skew(_X, cent=1)
    assert result["p"] == pytest.approx(0.01)
    assert result["Sp"] == pytest.approx(0.0, abs=1e-12)


def test_centile_skew_weighted_matches_r() -> None:
    # Live R 4.6.0 + gamlss.dist: -0.09706546 / -0.03614458 / -0.1111111.
    # R prints 8 significant digits; compare at 1e-6 relative.
    assert centile_skew(_X, 1, weights=_W)["Sp"] == pytest.approx(-0.09706546, rel=1e-6)
    assert centile_skew(_X, 5, weights=_W)["Sp"] == pytest.approx(-0.03614458, rel=1e-6)
    r25 = centile_skew(_X, 25, weights=_W)
    assert r25["p"] == pytest.approx(0.25)
    assert r25["Sp"] == pytest.approx(-0.1111111, rel=1e-6)


def test_centile_skew_rejects_cent_above_50() -> None:
    with pytest.raises(ValueError, match="less than 50"):
        centile_skew(_X, cent=51)


def test_centile_skew_drops_na_with_warning() -> None:
    data = _X + [np.nan]
    with pytest.warns(UserWarning, match="NA"):
        result = centile_skew(data, cent=1)
    assert np.isfinite(result["Sp"])


def test_gamlss_scope_form_and_terms() -> None:
    frame = {"y": [1, 2, 3], "x": [1, 2, 3], "z": [4, 5, 6]}
    forms = gamlss_scope(frame, response=1, smoother="cs", form=True)
    assert forms == {"x": "~1+x+cs(x)", "z": "~1+z+cs(z)"}
    terms = gamlss_scope({"y": [1, 2, 3], "x": [1, 2, 3]}, 1, "pb", "df=5", form=False)
    assert terms == {"x": ["1", "x", "pb(x,df=5)"]}


def test_gamlss_scope_response_by_name() -> None:
    frame = {"y": [1, 2, 3], "x": [1, 2, 3]}
    assert gamlss_scope(frame, response="y") == {"x": "~1+x+cs(x)"}


def test_gamlss_scope_factor_gets_no_smoother() -> None:
    pd = pytest.importorskip("pandas")
    frame = pd.DataFrame({"y": [1, 2, 3], "g": pd.Categorical(["a", "b", "a"]), "x": [1, 2, 3]})
    forms = gamlss_scope(frame, response=1, smoother="cs")
    assert forms["g"] == "~1+g"
    assert forms["x"] == "~1+x+cs(x)"
