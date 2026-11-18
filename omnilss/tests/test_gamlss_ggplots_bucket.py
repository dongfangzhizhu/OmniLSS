# SPDX-License-Identifier: GPL-3.0-or-later
"""gamlss.ggplots bucket style variants (D2.259).

R's moment_*/centile_* helpers return ggplot layer sets; the port returns a
BucketSpec carrying the layer description plus the theoretical boundary curve
(the reference family curves live in the package extdata, which is not
vendored).  The three plotting functions carry the real statistics and are
pinned against momentSK / centileSK, both verified against live R to ~1e-15.
"""
from __future__ import annotations

import numpy as np
import pytest

from omnilss import (
    BucketSpec,
    centile_bucket,
    centile_bucket_wrap,
    centile_colour_both,
    centile_colour_half,
    centile_gray_both,
    moment_bucket,
    moment_bucket_wrap,
    moment_colour_both,
    moment_colour_half,
    moment_gray_both,
    moment_gray_half,
)
from omnilss.centileSK import centileSK
from omnilss.momentSK import momentSK


def _sample(seed: int = 5, n: int = 200):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = np.exp(0.4 * x) * rng.gamma(2.0, 0.5, size=n) - 0.6
    return x, y


def test_moment_specs_carry_the_theoretical_boundary():
    spec = moment_colour_half()
    assert isinstance(spec, BucketSpec)
    assert spec.kind == "moment" and spec.variant == "colour_half"
    assert spec.legend is True and spec.line_width == 0.5
    boundary = spec.boundary
    assert list(boundary.columns) == ["tskew", "tkurt"]
    assert boundary.shape[0] == 1000
    assert boundary["tskew"].iloc[0] == pytest.approx(0.0)
    assert boundary["tskew"].iloc[-1] == pytest.approx(0.99999)
    # tkurt = ((1 + skew^2) - 3) / (1 + |...|) with skew = t / (1 - t)
    skew = boundary["tskew"].iloc[10] / (1.0 - boundary["tskew"].iloc[10])
    excess = (1.0 + skew**2) - 3.0
    assert boundary["tkurt"].iloc[10] == pytest.approx(excess / (1.0 + abs(excess)))

    assert moment_gray_half().legend is False
    assert moment_gray_half().variant == "gray_half"
    assert moment_colour_both(line_width=2).line_width == 2.0
    assert moment_gray_both().kind == "moment"


def test_centile_specs_validate_the_type():
    tail = centile_colour_half()
    assert tail.kind == "centile" and tail.type == "tail"
    assert tail.boundary is None
    assert centile_colour_both("central").type == "central"
    assert centile_gray_both().line_width == 0.5
    assert centile_colour_both(legend=False).legend is False
    with pytest.raises(ValueError, match="tail"):
        centile_colour_half("middle")
    with pytest.raises(ValueError, match="tail"):
        centile_gray_both("both")


def test_moment_bucket_uses_transformed_coordinates():
    _, y = _sample()
    result = moment_bucket(y, no_bootstrap=5, seed=1)
    row = result.data["rows"][0]
    expected = momentSK(y)
    assert row["trans_skew"] == pytest.approx(expected.trans_skew)
    assert row["trans_kurt"] == pytest.approx(expected.trans_kurt)
    assert row["jarque_bera"] == pytest.approx(expected.jarque_bera)
    assert set(row["polygon"].columns) == {"X", "Y"}
    assert row["bootstrap"].shape == (5, 2)
    # The bootstrap cloud lives on the transformed coordinates too.
    assert np.all(np.abs(row["bootstrap"]) <= 1.0 + 1e-12)


def test_moment_bucket_wrap_reports_points_per_interval():
    x, y = _sample()
    result = moment_bucket_wrap(y, xvar=x, n_inter=4, no_bootstrap=7, seed=2)
    points = result.data["points"]
    bootstrap = result.data["bootstrap"]
    band = result.data["band"]
    assert points["Z"].nunique() == 4
    assert bootstrap.shape[0] == 4 * 7
    assert set(band["Z"]) == set(points["Z"])

    groups = np.asarray(
        __import__("pandas").qcut(x, 4, duplicates="drop").astype(str)
    )
    for level, group in points.groupby("Z", sort=False):
        expected = momentSK(y[groups == level])
        assert group["X"].iloc[0] == pytest.approx(expected.trans_skew)
        assert group["Y"].iloc[0] == pytest.approx(expected.trans_kurt)

    with pytest.raises(ValueError, match="xvar"):
        moment_bucket_wrap(y)


def test_centile_bucket_matches_centileSK_for_each_type():
    _, y = _sample()
    expected = centileSK(y)
    tail = centile_bucket(y, type="tail", no_bootstrap=5, seed=3)
    assert tail.data["x_column"] == "S0.01"
    point = tail.data["points"].iloc[0]
    assert point["S0.01"] == pytest.approx(expected.s001)
    assert point["S0.25"] == pytest.approx(expected.s025)
    assert point["K0.01"] == pytest.approx(expected.trans_k001)
    assert tail.data["bootstrap"].shape == (5, 6)

    central = centile_bucket(y, type="central", no_bootstrap=3, seed=3)
    assert central.data["x_column"] == "S0.25"
    with pytest.raises(ValueError, match="tail"):
        centile_bucket(y, type="both")


def test_centile_bucket_wrap_facets_and_simulates():
    x, y = _sample()
    result = centile_bucket_wrap(
        y, xvar=x, n_inter=3, no_bootstrap=4, sim_test=True,
        no_sim_test=40, seed=4,
    )
    assert result.data["points"]["Z"].nunique() == 3
    assert result.data["bootstrap"].shape[0] == 3 * 4
    simulation = result.data["simulation"]
    assert simulation["Z"].nunique() == 3
    assert simulation.shape[0] == 3 * 100
    # R transforms the ellipse's kurtosis axis into (0, 1).
    assert simulation["y"].abs().max() < 1.0
    assert result.data["spec"].kind == "centile"

    without = centile_bucket_wrap(y, xvar=x, n_inter=2, no_bootstrap=2, seed=4)
    assert without.data["simulation"].empty
    with pytest.raises(ValueError, match="xvar"):
        centile_bucket_wrap(y)


def test_bucket_functions_accept_models_and_weights():
    from omnilss import gamlss

    x, y = _sample()
    model = gamlss("y ~ x", family="NO", data={"y": y, "x": x}, max_iter=30)
    result = moment_bucket_wrap(model, xvar=x, n_inter=2, no_bootstrap=3, seed=5)
    assert result.data["points"].shape[0] == 2

    weights = np.linspace(0.5, 2.0, y.size)
    weighted = momentSK(y, weights=weights)
    manual_mean = float(np.sum(weights * y) / np.sum(weights))
    assert weighted.mean == pytest.approx(manual_mean)
    weighted_cent = centileSK(y, weights=weights)
    plain = centileSK(y)
    assert weighted_cent.k001 != plain.k001

    with pytest.raises(ValueError, match="same length"):
        momentSK(y, weights=np.ones(3))
    with pytest.raises(ValueError, match="same length"):
        centileSK(y, weights=np.ones(3))
    with pytest.raises(ValueError, match="centiles"):
        centileSK(y, centiles=(50, 10))
