"""Dedicated evidence module for gamlss.prepdata (P4-5, D2.254).

Surface evidence comes from the coverage artifact (59/59 public_api + tested);
behaviour evidence covers the three workflows the package exists for:
distribution-based outliers, pairwise association, and the per-variable
diagnostics (bucket / leverage / plot / response / xyplot).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import omnilss
from omnilss import (
    ACE,
    ZScores,
    association,
    cor_boot,
    cor_perm_test,
    cramer_phi,
    data_association,
    data_bucket,
    data_leverage,
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

_REPO = Path(__file__).resolve().parents[2]
_COVERAGE = _REPO / "docs" / "development" / "r-symbol-coverage.json"
_EXPECTED_SYMBOLS = 59


def _package_symbols() -> list[dict]:
    payload = json.loads(_COVERAGE.read_text(encoding="utf-8"))
    return [row for row in payload["symbols"] if row.get("r_package") == "gamlss.prepdata"]


def test_every_prepdata_symbol_is_exported_and_tested():
    symbols = _package_symbols()
    assert len(symbols) == _EXPECTED_SYMBOLS
    for row in symbols:
        layers = row["layers"]
        assert layers["surface"] == "public_api", row["r_symbol"]
        assert layers["kernel"] == "implemented", row["r_symbol"]
        assert layers["parity"] in {"tested", "r_consistency_test"}, row["r_symbol"]
        assert hasattr(omnilss, row["r_symbol"].replace(".", "_")), row["r_symbol"]


def test_zscores_and_outlier_workflow():
    rng = np.random.default_rng(0)
    x = rng.normal(size=120)
    x[7] = 12.0
    scores = y_zscores(x)
    assert isinstance(scores, ZScores)
    assert float(np.std(scores, ddof=1)) == pytest.approx(1.0, abs=0.05)
    assert set(scores.parameters) == {"mu", "sigma", "nu", "tau"}

    flagged = y_outliers(x, value=2.5, transform=False)
    assert 8 in flagged
    assert 8 in y_outliers_z(x, value=2.5, transform=False)
    assert isinstance(y_outliers_both(x, value=2.5), np.ndarray)

    groups = np.where(x > 0, "pos", "neg")
    by_group = y_outliers_by(x, groups, type="quantile")
    assert set(by_group) == {"pos", "neg"}

    frame = {"a": rng.normal(size=80), "b": rng.normal(size=80), "g": groups[:80]}
    assert set(data_zscores(frame)) == {"a", "b"}
    assert set(data_outliers(frame, min_distinct=20)) == {"a", "b"}


def test_association_workflow():
    rng = np.random.default_rng(1)
    x = rng.normal(size=100)
    y = 0.7 * x + rng.normal(size=100)
    numeric = association(x, y)
    assert numeric["type"] == "correlation"
    assert numeric["assoc"] == pytest.approx(abs(float(np.corrcoef(x, y)[0, 1])), abs=0.15)

    mixed = association(x, np.where(x > 0, "pos", "neg"))
    assert mixed["type"] == "anova"
    nominal = association(np.where(x > 0, "pos", "neg"), np.where(y > 0, "up", "down"))
    assert nominal["type"] == "cramersV"
    assert cramer_phi(np.where(x > 0, "a", "b"), np.where(x > 0, "a", "b")) == pytest.approx(1.0)

    table = data_association({"a": x, "b": y, "g": np.where(x > 0, "pos", "neg")})
    assert table.shape == (3, 4)


def test_nonlinear_dependence_and_permutation_workflow():
    rng = np.random.default_rng(2)
    t = np.linspace(-2.0, 2.0, 120)
    y = np.sin(3.0 * t) + 0.05 * rng.normal(size=120)
    fit = ACE(t, y)
    assert fit.mcor > 0.9
    assert fit.mcor > abs(fit.r)
    assert mcor(t, y) == pytest.approx(fit.mcor)

    strong = cor_perm_test(t, y, B=199, seed=5)
    assert strong.ASL < 0.05
    independent = cor_perm_test(t, rng.normal(size=t.size), B=199, seed=5)
    assert independent.ASL > 0.05
    boot = cor_boot(t, y, B=199, seed=6)
    assert boot.mean == pytest.approx(float(np.corrcoef(t, y)[0, 1]), abs=0.05)
    assert boot.sd > 0.0


def test_per_variable_diagnostics_workflow():
    rng = np.random.default_rng(3)
    t = np.linspace(-2.0, 2.0, 120)
    y = np.sin(3.0 * t) + 0.05 * rng.normal(size=120)
    frame = {"y": y, "a": t, "b": rng.normal(size=t.size)}

    bucket = data_bucket(frame)
    assert set(bucket.columns) == {"skewness", "kurtosis", "outliers"}
    leverage = data_leverage(frame, response="y")
    assert leverage["leverage"].shape == (t.size,)
    assert 0.0 < leverage["threshold"] < 1.0

    plot = data_plot({"a": t, "g": np.where(t > 0, "p", "n")})
    assert plot["a"]["kind"] == "numeric" and plot["g"]["kind"] == "categorical"
    assert data_response({"y": t}, "y")["class"] == "numeric"
    assert data_response({"y": np.where(t > 0, "p", "n")}, "y")["kind"] == "binary"
    assert set(data_xyplot({"y": y, "a": t}, "y")) == {"a"}
