"""Phase 3.1: pb/ps/cs single managed representation + lambda origin.

design.md Property 8: a smoother has exactly one managed design
representation; the saved Smooth Model State retains whether lambda was
user-fixed, DF-targeted, or method-selected after any update.

SmoothFitInfo.lambda_origin records:
- "user_fixed": explicit lambda argument (pb(x, lambda=...) / ps(x, lambda=...))
- "df_target":  lambda solved from a target df
- "method_selected": lambda chosen by the selection method (ML/GCV/REML/...)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilss import gamlss
from omnilss.smooth_fitting import SmoothFitInfo, update_smooth_lambdas


@pytest.fixture(scope="module")
def smooth_data():
    rng = np.random.default_rng(21)
    n = 60
    x = np.linspace(0, 1, n)
    y = np.sin(4 * x) + rng.normal(scale=0.1, size=n)
    return {"y": y, "x": x}


def _smooth_fits(model) -> list[SmoothFitInfo]:
    slots = model.additional_slots.get("smooth_infos", {})
    info = slots.get("mu")
    if info is None or getattr(info, "smooth_fits", None) is None:
        return []
    return list(info.smooth_fits)


@pytest.mark.parametrize("smooth", ["pb(x, lambda=0.8)", "ps(x, lambda=0.8)"])
def test_user_fixed_lambda_origin(smooth, smooth_data) -> None:
    """Explicit lambda in the formula -> lambda_origin == user_fixed."""
    model = gamlss(f"y ~ {smooth}", family="NO", data=smooth_data)
    fits = _smooth_fits(model)
    assert len(fits) == 1
    assert fits[0].lambda_origin == "user_fixed"
    assert fits[0].lambda_ is not None


@pytest.mark.parametrize("smooth", ["pb(x, df=5)", "ps(x, df=5)"])
def test_df_target_lambda_origin(smooth, smooth_data) -> None:
    """Target df in the formula -> lambda_origin == df_target."""
    model = gamlss(f"y ~ {smooth}", family="NO", data=smooth_data)
    fits = _smooth_fits(model)
    assert len(fits) == 1
    assert fits[0].lambda_origin == "df_target"


@pytest.mark.parametrize("smooth", ["pb(x)", "ps(x)"])
def test_method_selected_lambda_origin(smooth, smooth_data) -> None:
    """No df/lambda given -> lambda_origin == method_selected."""
    model = gamlss(f"y ~ {smooth}", family="NO", data=smooth_data)
    fits = _smooth_fits(model)
    assert len(fits) == 1
    assert fits[0].lambda_origin == "method_selected"


def test_cs_origin_is_df_target(smooth_data) -> None:
    """cs() always targets a df (R default df=3) -> lambda_origin df_target."""
    model = gamlss("y ~ cs(x)", family="NO", data=smooth_data)
    fits = _smooth_fits(model)
    assert len(fits) == 1
    assert fits[0].lambda_origin == "df_target"


def test_smooth_fit_state_single_managed_representation(smooth_data) -> None:
    """Exactly one SmoothFitInfo per term (single managed representation)."""
    model = gamlss("y ~ pb(x) + cs(x)", family="NO", data=smooth_data)
    fits = _smooth_fits(model)
    assert len(fits) == 2
    by_smoother = {f.smoother: f for f in fits}
    assert set(by_smoother) == {"pb", "cs"}
    for info in fits:
        assert info.lambda_ is not None
        assert info.edf is not None and info.edf > 0
        assert info.penalty is not None
        assert info.basis_columns is not None
        assert info.lambda_origin is not None


def test_lambda_update_preserves_origin(smooth_data) -> None:
    """After a lambda update the origin is retained (Property 8)."""
    model = gamlss("y ~ pb(x, lambda=0.8)", family="NO", data=smooth_data)
    fits = _smooth_fits(model)
    assert len(fits) == 1
    original_origin = fits[0].lambda_origin
    assert original_origin == "user_fixed"

    X = np.asarray(model.design_matrices["mu"])
    y = np.asarray(smooth_data["y"])
    beta = np.asarray(model.coefficients["mu"])
    updated = update_smooth_lambdas(
        X=X,
        y=y,
        beta=beta,
        w=np.ones(len(y)),
        smooth_fits=fits,
        method="ML",
    )
    assert len(updated) == 1
    # Origin survives the lambda re-estimation.
    assert updated[0].lambda_origin == original_origin


# --- Phase 3.2: ps lambda/EDF lifecycle (user_fixed / df_target / method) ---

@pytest.mark.parametrize(
    "formula, expected_lambda_origin",
    [
        ("y ~ ps(x, df=5)", "df_target"),
        ("y ~ ps(x, df=10)", "df_target"),
        ("y ~ ps(x, lambda=0.8)", "user_fixed"),
        ("y ~ ps(x)", "method_selected"),
    ],
)
def test_ps_lambda_lifecycle_distinct(
    smooth_data, formula, expected_lambda_origin
) -> None:
    """ps df/lambda inputs must produce distinct, preserved lambda values."""
    models = {
        "df5": gamlss("y ~ ps(x, df=5)", family="NO", data=smooth_data),
        "df10": gamlss("y ~ ps(x, df=10)", family="NO", data=smooth_data),
        "lamb": gamlss("y ~ ps(x, lambda=0.8)", family="NO", data=smooth_data),
        "auto": gamlss("y ~ ps(x)", family="NO", data=smooth_data),
    }
    lambdas = {
        name: _smooth_fits(m)[0].lambda_ for name, m in models.items()
    }
    # Distinct inputs yield distinct managed lambdas (no silent GCV overwrite).
    assert len(set(lambdas.values())) == len(lambdas), lambdas
    # df_target EDF tracks the requested df (R semantics: edf = df + 2).
    assert abs(_smooth_fits(models["df5"])[0].edf - 7.0) < 1e-3
    assert abs(_smooth_fits(models["df10"])[0].edf - 12.0) < 1e-3
    # user_fixed lambda is the exact value requested.
    assert abs(_smooth_fits(models["lamb"])[0].lambda_ - 0.8) < 1e-6


def test_ps_df_target_lambda_not_overwritten_during_fit(smooth_data) -> None:
    """RS iteration must not re-estimate a df-targeted lambda (Property 8)."""
    model = gamlss("y ~ ps(x, df=5)", family="NO", data=smooth_data)
    fit = _smooth_fits(model)[0]
    assert fit.lambda_origin == "df_target"
    # The managed lambda matches the direct df solve (not a GCV/ML estimate).
    from omnilss.smoothers.ps import fit_pspline_smooth

    direct = fit_pspline_smooth(
        np.asarray(smooth_data["x"]),
        np.asarray(smooth_data["y"]),
        weights=None,
        df=5,
        ps_intervals=20,
        degree=3,
    )
    assert abs(fit.lambda_ - direct.lambda_) < 1e-6
