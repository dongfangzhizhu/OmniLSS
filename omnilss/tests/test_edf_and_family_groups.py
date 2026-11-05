"""R edf() and the internal support-group constants.

The constants are compared element-for-element with live gamlss 5.5-3
(.realline/.realplus/.real0to1/.realAll/.counts/.binom and the
.gamlss.*.list vectors).  edf() has no direct R counterpart to compare against
without reproducing R's lambda search, so it is checked for internal
consistency: per-term edf must reproduce the model's smooth_edf slot and the
df_fit accounting.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest

import jax

jax.config.update("jax_enable_x64", True)

import omnilss  # noqa: E402
from omnilss import family_groups as fg  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]


def _smooth_model(n: int = 300, seed: int = 0):
    np.random.seed(seed)
    x = np.linspace(0.0, 1.0, n)
    y = np.sin(2 * np.pi * x) + np.random.normal(0.0, 0.2, n)
    return omnilss.gamlss("y ~ pb(x)", family="NO", data={"y": y, "x": x}), y, x


def _two_smooth_model(n: int = 300, seed: int = 0):
    np.random.seed(seed)
    x1 = np.linspace(0.0, 1.0, n)
    x2 = np.linspace(-1.0, 1.0, n)
    y = np.sin(2 * np.pi * x1) + 0.5 * x2**2 + np.random.normal(0.0, 0.2, n)
    return omnilss.gamlss("y ~ pb(x1) + pb(x2)", family="NO", data={"y": y, "x1": x1, "x2": x2})


def test_edf_is_zero_without_smooth_terms() -> None:
    np.random.seed(0)
    x = np.linspace(0.0, 1.0, 200)
    y = 1.0 + 2.0 * x + np.random.normal(0.0, 0.2, 200)
    model = omnilss.gamlss("y ~ x", family="NO", data={"y": y, "x": x})
    assert omnilss.edf(model, print=False) == 0
    assert model.get_smooth_info("mu") == []


def test_edf_matches_the_model_smooth_slot() -> None:
    model, _, _ = _smooth_model()
    per_term = omnilss.edf(model, print=False)
    assert set(per_term) == {"x"}
    assert per_term["x"] == pytest.approx(
        model.additional_slots["smooth_edf"]["mu"], rel=1e-12
    )


def test_edf_matches_get_smooth_info() -> None:
    model, _, _ = _smooth_model()
    info = model.get_smooth_info("mu")
    assert len(info) == 1
    assert info[0]["variable"] == "x"
    assert info[0]["edf"] == pytest.approx(omnilss.edf(model, print=False)["x"], rel=1e-12)


def test_edf_reports_the_fitted_not_the_seed_value() -> None:
    model, _, _ = _smooth_model()
    seed_edf = model.additional_slots["smooth_infos"]["mu"].smooth_fits[0].edf
    fitted_edf = omnilss.edf(model, print=False)["x"]
    # lambda is updated during the RS cycles, so the two must differ here.
    assert fitted_edf != pytest.approx(seed_edf, rel=1e-6)


def test_edf_accounts_for_df_fit() -> None:
    model, _, _ = _smooth_model()
    total_smooth = sum(omnilss.edf(model, print=False).values())
    # df_fit = smooth edf + intercept + sigma intercept.
    assert model.df_fit == pytest.approx(total_smooth + 2.0, rel=1e-10)


def test_edf_with_two_smooth_terms() -> None:
    model = _two_smooth_model()
    per_term = omnilss.edf(model, print=False)
    assert set(per_term) == {"x1", "x2"}
    assert sum(per_term.values()) == pytest.approx(
        model.additional_slots["smooth_edf"]["mu"], rel=1e-10
    )


def test_edf_parameter_alias_overrides_what() -> None:
    model, _, _ = _smooth_model()
    assert omnilss.edf(model, what="sigma", print=False) == 0
    assert omnilss.edf(model, what="mu", parameter="sigma", print=False) == 0


def test_edf_rejects_unknown_parameter() -> None:
    model, _, _ = _smooth_model()
    with pytest.raises(ValueError, match="should be one of"):
        omnilss.edf(model, what="theta", print=False)


def test_support_groups_are_consistent() -> None:
    assert "NO" in fg.REAL_LINE
    assert "GA" in fg.REAL_PLUS
    assert "BE" in fg.REAL_0_TO_1
    assert set(fg.REAL_ALL) == set(fg.REAL_LINE) | set(fg.REAL_PLUS)
    assert len(fg.REAL_ALL) == len(set(fg.REAL_ALL))
    assert "PO" in fg.COUNTS
    assert "BI" in fg.BINOMIAL
    assert "BI" in fg.BI_LIST
    assert "MULTIN" in fg.MULTIN_LIST
    assert "pb" in fg.SM_LIST and "cs" in fg.SM_LIST


def test_r_name_aliases_are_the_same_objects() -> None:
    assert fg._realline is fg.REAL_LINE
    assert fg._realplus is fg.REAL_PLUS
    assert fg._realAll is fg.REAL_ALL
    assert fg._counts is fg.COUNTS
    assert fg._binom is fg.BINOMIAL
    assert fg._gamlss_bi_list is fg.BI_LIST
    assert fg._gamlss_sm_list is fg.SM_LIST


def test_hat_wx_matches_the_hat_matrix_diagonal() -> None:
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    w = np.array([1.0, 2.0, 1.5, 3.0, 0.5, 2.0])
    got = omnilss._hat_WX(w, x)
    design = np.column_stack([np.ones(6), x])
    weighted = design * np.sqrt(w)[:, None]
    expected = np.sum((weighted @ np.linalg.inv(weighted.T @ weighted)) * weighted, axis=1)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)
    assert float(np.sum(got)) == pytest.approx(2.0, abs=1e-10)


def test_hat_wx_accepts_a_matrix() -> None:
    x = np.column_stack([np.ones(6), np.arange(1.0, 7.0), [2.0, 1, 4, 3, 6, 5]])
    w = np.ones(6)
    got = omnilss._hat_WX(w, x)
    assert got.shape == (6,)
    assert float(np.sum(got)) == pytest.approx(3.0, abs=1e-10)


def test_hat_wx_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="same length"):
        omnilss._hat_WX(np.ones(4), np.arange(6.0))


@pytest.mark.slow
def test_support_groups_match_live_r() -> None:
    if shutil.which("Rscript") is None:
        pytest.skip("Rscript not available")
    script = """suppressPackageStartupMessages(library(gamlss))
g <- function(n) get(n, envir = asNamespace("gamlss"))
out <- list(
  realline = as.character(g(".realline")),
  realplus = as.character(g(".realplus")),
  real0to1 = as.character(g(".real0to1")),
  realAll = as.character(g(".realAll")),
  counts = as.character(g(".counts")),
  binom = as.character(g(".binom")),
  bilist = as.character(g(".gamlss.bi.list")),
  multin = as.character(g(".gamlss.multin.list")),
  smlist = as.character(g(".gamlss.sm.list")))
cat(jsonlite::toJSON(out, auto_unbox = FALSE))
"""
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "groups.R"
        path.write_text(script, encoding="utf-8")
        proc = subprocess.run(
            ["Rscript", str(path)], capture_output=True, text=True, timeout=300
        )
    if proc.returncode != 0:
        pytest.skip(f"R failed: {proc.stderr[-300:]}")
    reference = json.loads(proc.stdout.strip().splitlines()[-1])
    assert list(fg.REAL_LINE) == list(reference["realline"])
    assert list(fg.REAL_PLUS) == list(reference["realplus"])
    assert list(fg.REAL_0_TO_1) == list(reference["real0to1"])
    assert list(fg.REAL_ALL) == list(reference["realAll"])
    assert list(fg.COUNTS) == list(reference["counts"])
    assert list(fg.BINOMIAL) == list(reference["binom"])
    assert list(fg.BI_LIST) == list(reference["bilist"])
    assert list(fg.MULTIN_LIST) == list(reference["multin"])
    assert list(fg.SM_LIST) == list(reference["smlist"])
