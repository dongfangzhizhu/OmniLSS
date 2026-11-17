# SPDX-License-Identifier: GPL-3.0-or-later
"""gamlss.spatial: MRF / MRFA / gmrf / draw.polys / polys2polys (D2.252).

Golden fixture: omnilss/tests/fixtures/spatial/lattice_9.json, written by live R
4.6.0 with the spam wrappers stubbed (as.spam/diag.spam/spam are identity):
a 3x3 rook lattice, n = 200, area effects -1..1.5, noise sd 0.4.

    MRF   sig2e 0.1631551633, sig2b 0.6117357826, df 8.7134320565,
          deviance 196.2512806667
    MRFA  (auto) sig2e 0.1631552009, sig2b 0.6117341758, df 8.7134541487,
          lambda 0.2667093116, deviance 196.2512834676
    MRFA  (df = 6) lambda 4.8106479481, deviance 221.2463447052
    MRFA  (lambda = 2) df 7.3299654745, deviance 202.3393301002
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from omnilss import (
    MRF,
    MRFA,
    draw_polys,
    gamlss,
    gamlss_gmrf,
    gmrf,
    gmrf_control,
    polys2polys,
)

FIXTURE = Path(__file__).parent / "fixtures" / "spatial" / "lattice_9.json"


@pytest.fixture(scope="module")
def fixture() -> dict:
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


def _inputs(fixture):
    return (
        np.asarray(fixture["y"], dtype=float),
        np.asarray(fixture["area"]),
        np.asarray(fixture["G"], dtype=float),
    )


def test_mrf_matches_live_r(fixture):
    y, area, G = _inputs(fixture)
    fit = MRF(y=y, x=area, precision=G)
    reference = fixture["q"]
    np.testing.assert_allclose(fit.beta, reference["beta"], atol=1e-6)
    assert fit.sig2e == pytest.approx(reference["sig2e"], rel=1e-5)
    assert fit.sig2b == pytest.approx(reference["sig2b"], rel=1e-5)
    assert fit.df == pytest.approx(reference["df"], rel=1e-5)
    assert fit.deviance == pytest.approx(reference["dev"], rel=1e-7)
    assert fit.method == "Q-function"


def test_mrfa_auto_matches_live_r(fixture):
    y, area, G = _inputs(fixture)
    fit = MRFA(y=y, x=area, precision=G)
    reference = fixture["a"]
    np.testing.assert_allclose(fit.beta, reference["beta"], atol=1e-9)
    assert fit.sig2e == pytest.approx(reference["sig2e"], rel=1e-9)
    assert fit.sig2b == pytest.approx(reference["sig2b"], rel=1e-9)
    assert fit.df == pytest.approx(reference["df"], rel=1e-9)
    assert fit.lam == pytest.approx(reference["lambda"], rel=1e-9)


def test_mrfa_df_and_lambda_modes_match_live_r(fixture):
    y, area, G = _inputs(fixture)
    by_df = MRFA(y=y, x=area, precision=G, df=6)
    assert by_df.lam == pytest.approx(4.8106479481, rel=1e-8)
    assert by_df.df == pytest.approx(6.0, abs=1e-8)
    assert by_df.deviance == pytest.approx(221.2463447052, rel=1e-9)

    by_lambda = MRFA(y=y, x=area, precision=G, lambda_=2.0)
    assert by_lambda.lam == pytest.approx(2.0, rel=0)
    assert by_lambda.df == pytest.approx(7.3299654745, rel=1e-9)
    assert by_lambda.deviance == pytest.approx(202.3393301002, rel=1e-9)


def test_precision_from_neighbour_matches_the_supplied_matrix(fixture):
    y, area, G = _inputs(fixture)
    levels = sorted(set(area.tolist()))
    neighbour = {"nb": {}}
    for index, name in enumerate(levels):
        neighbour["nb"][name] = tuple(
            str(other + 1) for other in np.nonzero(G[index] < 0)[0]
        )
    from_matrix = MRF(y=y, x=area, precision=G)
    from_neighbour = MRF(y=y, x=area, neighbour=neighbour)
    np.testing.assert_allclose(from_neighbour.beta, from_matrix.beta, atol=1e-9)


def test_gmrf_term_validates_its_inputs(fixture):
    _, area, G = _inputs(fixture)
    with pytest.raises(ValueError, match="needs the area factor"):
        gmrf()
    with pytest.raises(ValueError, match="method"):
        gmrf(area, precision=G, method="X")
    with pytest.raises(ValueError, match="degrees of freedom"):
        gmrf(area, precision=G, df=3, method="Q")
    with pytest.raises(ValueError, match="precision matrix"):
        gmrf(area)
    term = gmrf(area, precision=G, lambda_=2.0, method="A")
    assert term.method == "A" and term.lam == 2.0
    assert gmrf_control().sig2e == 1.0


def test_gmrf_callbacks_return_the_expected_fields(fixture):
    y, area, G = _inputs(fixture)
    term_q = gmrf(area, precision=G)
    fit_q = gamlss_gmrf(term_q, y, None)
    assert fit_q.fitted_values.shape == y.shape
    np.testing.assert_allclose(
        fit_q.residuals, y - fit_q.fitted_values, rtol=0, atol=0
    )
    assert fit_q.nl_df == pytest.approx(fit_q.coef_smo.df - 2.0)
    assert fit_q.lam == pytest.approx(fit_q.coef_smo.sig2e / fit_q.coef_smo.sig2b)
    assert len(fit_q.state) == 3

    term_a = gmrf(area, precision=G, df=6, method="A")
    fit_a = gamlss_gmrf(term_a, y, None)
    assert fit_a.coef_smo.df == pytest.approx(6.0, abs=1e-6)
    assert len(fit_a.state) == 2


def test_gmrf_interface_errors(fixture):
    y, area, G = _inputs(fixture)
    with pytest.raises(TypeError, match="gmrf\\(\\) term"):
        gamlss_gmrf(area, y)
    with pytest.raises(NotImplementedError, match="xeval"):
        gamlss_gmrf(gmrf(area, precision=G), y, None, xeval="new")
    with pytest.raises(NotImplementedError, match="gamlss_gmrf"):
        gamlss("y ~ gmrf(area)", family="NO", data={"y": y, "area": area}, max_iter=5)


def test_polys2polys_and_draw_polys():
    square_a = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
    square_b = square_a + np.array([1.0, 0.0])
    mapping = polys2polys({"A": square_a, "B": square_b})
    assert sorted(mapping) == ["A", "B"]
    named = polys2polys([square_a, square_b], region_id=["A", "B"])
    np.testing.assert_allclose(named["B"], square_b)
    with pytest.raises(ValueError, match="different lengths"):
        polys2polys([square_a], region_id=["A", "B"])

    plot = draw_polys(mapping, object={"A": 0.0, "B": 1.0})
    assert plot.xlim == pytest.approx((0.0, 2.0))
    assert plot.ylim == pytest.approx((0.0, 1.0))
    assert set(plot.colours) == {"A", "B"}
    assert plot.colours["A"][0] == pytest.approx(0.0)
    assert plot.colours["B"][0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="not present in polys"):
        draw_polys(mapping, object={"C": 1.0})
    with pytest.raises(ValueError, match="MRF or a vector"):
        draw_polys(mapping, object=3.0)
