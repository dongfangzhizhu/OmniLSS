# SPDX-License-Identifier: GPL-3.0-or-later
"""gamlss.prepdata: diagnostics, associations and maximal correlation (D2.254).

The statistical cores are pinned; plotting is out of scope (structured results
instead of ggplot objects).  y_zscores fits SHASHo by RS: on the normal fixture
(seed 0, n = 120) the residual sd is 1.004 and three points exceed |z| = 2.5,
matching live R's gamlssML (sd 1.0026, three points).
"""
from __future__ import annotations

import numpy as np
import pytest

from omnilss import (
    ACE,
    ACE_iter,
    ZScores,
    association,
    cor_boot,
    cor_perm_test,
    cramer_phi,
    data_association,
    data_bucket,
    data_leverage,
    data_mcor,
    data_outliers,
    data_plot,
    data_response,
    data_xyplot,
    data_zscores,
    mcor,
    y_outliers,
    y_outliers_both,
    y_outliers_by,
    y_outliers_z,
    y_zscores,
)


def _normal(seed: int = 0, n: int = 120):
    rng = np.random.default_rng(seed)
    return rng.normal(size=n)


def _nonlinear(seed: int = 0, n: int = 120):
    rng = np.random.default_rng(seed)
    t = np.linspace(-2.0, 2.0, n)
    return t, np.sin(3.0 * t) + 0.05 * rng.normal(size=n)


def test_y_zscores_fits_shasho_and_returns_quantile_residuals():
    x = _normal()
    scores = y_zscores(x)
    assert isinstance(scores, ZScores)
    assert scores.shape == x.shape
    assert float(np.std(scores, ddof=1)) == pytest.approx(1.0, abs=0.05)
    assert set(scores.parameters) == {"mu", "sigma", "nu", "tau"}
    assert int(np.sum(np.abs(scores) > 2.5)) == 3


def test_y_outliers_flags_the_injected_point():
    x = _normal()
    x[7] = 12.0
    flagged = y_outliers(x, value=2.5, transform=False)
    assert 8 in flagged  # 1-based, as in R
    assert flagged.dtype.kind in "iu"


def test_y_outliers_variants():
    x = _normal()
    x[10] = 10.0
    assert 11 in y_outliers_z(x, value=2.5, transform=False)
    assert 11 in y_outliers_z(x, value=2.5, transform=False, loop=True)
    assert isinstance(y_outliers_both(x, value=2.5), np.ndarray)

    rng = np.random.default_rng(1)
    groups = np.where(rng.random(x.size) < 0.5, "a", "b")
    by = y_outliers_by(x, groups, type="quantile")
    assert set(by) == {"a", "b"}
    for level, indices in by.items():
        indices = np.asarray(indices)
        assert indices.dtype.kind in "iu"
        # Indices are global (1-based) and must belong to that group.
        assert np.all(groups[indices - 1] == level)


def test_data_outliers_and_data_zscores_shapes():
    rng = np.random.default_rng(2)
    frame = {
        "a": rng.normal(size=80),
        "b": rng.normal(size=80) ** 2,
        "g": np.where(rng.random(80) < 0.5, "x", "y"),
    }
    scores = data_zscores(frame)
    assert set(scores) == {"a", "b"}
    assert all(scores[name].shape == (80,) for name in scores)
    flagged = data_outliers(frame, min_distinct=20)
    assert set(flagged) == {"a", "b"}
    assert all(isinstance(value, np.ndarray) for value in flagged.values())


def test_cor_perm_test_behaves_like_a_permutation_test():
    rng = np.random.default_rng(3)
    x = rng.normal(size=80)
    strong = 0.9 * x + 0.1 * rng.normal(size=80)
    result = cor_perm_test(x, strong, B=199, seed=5)
    assert 0.0 <= result.ASL <= 1.0
    assert result.ASL < 0.05
    assert result.sample.shape == (199,)
    repeat = cor_perm_test(x, strong, B=199, seed=5)
    np.testing.assert_array_equal(result.sample, repeat.sample)
    independent = cor_perm_test(x, rng.normal(size=80), B=199, seed=5)
    assert independent.ASL > 0.05
    two_tail = cor_perm_test(x, strong, B=199, seed=5, tail="two")
    assert 0.0 <= two_tail.ASL <= 1.0


def test_cor_boot_reports_the_bootstrap_distribution():
    rng = np.random.default_rng(4)
    x = rng.normal(size=80)
    y = 0.8 * x + rng.normal(scale=0.5, size=80)
    result = cor_boot(x, y, B=199, seed=6)
    assert result.mean == pytest.approx(float(np.corrcoef(x, y)[0, 1]), abs=0.05)
    assert result.sd > 0.0
    assert result.sample.shape == (199,)
    assert result.mean == pytest.approx(float(np.mean(result.sample)))


def test_ace_recovers_non_linear_dependence():
    t, y = _nonlinear()
    fit = ACE(t, y)
    assert fit.mcor > 0.9
    assert fit.mcor > abs(fit.r)
    assert fit.method == "loess"
    assert len(fit.trace) >= 2
    assert mcor(t, y) == pytest.approx(fit.mcor)
    iterate = ACE_iter(t, y)
    assert len(iterate.trace) >= 2


def test_data_mcor_is_a_symmetric_matrix():
    t, y = _nonlinear()
    rng = np.random.default_rng(5)
    matrix = data_mcor({"a": t, "b": y, "c": rng.normal(size=t.size)}, digits=6)
    assert matrix.shape == (3, 3)
    np.testing.assert_allclose(np.diag(matrix), 1.0)
    np.testing.assert_allclose(matrix.to_numpy(), matrix.to_numpy().T)
    assert matrix.loc["a", "b"] > 0.9


def test_associations_use_the_right_statistic():
    rng = np.random.default_rng(6)
    x = rng.normal(size=100)
    y = 0.7 * x + rng.normal(size=100)
    numeric = association(x, y)
    assert numeric["type"] == "correlation"
    assert numeric["assoc"] == pytest.approx(abs(np.corrcoef(x, y)[0, 1]), abs=0.15)

    groups = np.where(x > 0, "pos", "neg")
    mixed = association(x, groups)
    assert mixed["type"] == "anova"
    assert 0.0 <= mixed["assoc"] <= 1.0

    nominal = association(groups, np.where(y > 0, "up", "down"))
    assert nominal["type"] == "cramersV"
    assert 0.0 <= nominal["assoc"] <= 1.0
    assert cramer_phi(groups, groups) == pytest.approx(1.0)

    table = data_association({"a": x, "b": y, "g": groups})
    assert table.shape == (3, 4)
    assert set(table.columns) == {"xname", "yname", "assoc", "type"}


def test_data_bucket_leverage_plot_response_xyplot():
    t, y = _nonlinear()
    rng = np.random.default_rng(7)
    frame = {"y": y, "a": t, "b": rng.normal(size=t.size)}
    bucket = data_bucket(frame)
    assert set(bucket.columns) == {"skewness", "kurtosis", "outliers"}

    leverage = data_leverage(frame, response="y")
    assert leverage["leverage"].shape == (t.size,)
    assert 0.0 < leverage["threshold"] < 1.0
    assert leverage["rank"] >= 2

    plot = data_plot({"a": t, "g": np.where(t > 0, "p", "n")})
    assert plot["a"]["kind"] == "numeric"
    assert plot["g"]["kind"] == "categorical"

    numeric_response = data_response({"y": t}, "y")
    assert numeric_response["class"] == "numeric"
    assert "zscores" in numeric_response
    binary = data_response({"y": np.where(t > 0, "p", "n")}, "y")
    assert binary["class"] == "nominal" and binary["kind"] == "binary"

    pairs = data_xyplot({"y": y, "a": t}, "y")
    assert set(pairs) == {"a"}
    np.testing.assert_allclose(pairs["a"]["y"], y)


def test_validation_errors():
    with pytest.raises(ValueError, match="numeric"):
        y_outliers(np.array(["a", "b", "c"]))
    with pytest.raises(ValueError, match="data.frame"):
        data_zscores([1, 2, 3])
    with pytest.raises(ValueError, match="size of the data set"):
        data_outliers({"a": np.arange(5.0)})
    with pytest.raises(ValueError, match="response"):
        data_leverage({"y": np.arange(30.0)})
    with pytest.raises(ValueError, match="correlation"):
        ACE(np.linspace(0, 1, 30), 2.0 * np.linspace(0, 1, 30))
