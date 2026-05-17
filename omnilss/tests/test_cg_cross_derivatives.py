"""Regression tests for complete Cole-Green cross-derivative support."""

# ruff: noqa: E402

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from omnilss.cg_derivatives import eta_score_hessian
from omnilss.controls import gamlss_control
from omnilss.distributions import NO
from omnilss.fitting import gamlss
from omnilss.fitting_cg import (
    _compute_full_observed_information_and_score,
    _compute_log_likelihood,
    _initialize_params,
    extract_information_blocks,
    fit_cg,
    parameter_slices,
    zero_cross_information_blocks,
)


def _heteroscedastic_normal_case(n: int = 36):
    x = np.linspace(-1.0, 1.0, n)
    sigma = np.exp(-0.2 + 0.4 * x)
    y = 1.0 + 1.5 * x + sigma * np.sin(np.linspace(0.0, 3.0, n))
    X_mu = jnp.asarray(np.column_stack([np.ones(n), x]), dtype=jnp.float64)
    X_sigma = jnp.asarray(np.column_stack([np.ones(n), x]), dtype=jnp.float64)
    return jnp.asarray(y, dtype=jnp.float64), X_mu, X_sigma, {"y": y, "x": x}


def test_full_hessian_keeps_nonzero_mu_sigma_cross_block():
    y, X_mu, X_sigma, _ = _heteroscedastic_normal_case()
    result = fit_cg(NO(), y, X_mu, X_sigma, max_iter=12, tol=1e-5, return_fisher=True)

    assert result.fisher_matrix is not None
    assert result.param_slices is not None
    blocks = extract_information_blocks(result.fisher_matrix, result.param_slices)
    cross = np.asarray(blocks[("beta_mu", "beta_sigma")])

    assert cross.shape == (2, 2)
    assert np.linalg.norm(cross) > 1e-7
    assert np.allclose(
        cross,
        np.asarray(blocks[("beta_sigma", "beta_mu")]).T,
        atol=1e-5,
    )
    assert result.cg_backend == "CG_FULL_HESSIAN"
    assert result.cross_derivatives == "full_hessian"


def test_zeroing_cross_blocks_changes_cg_update_direction():
    y, X_mu, X_sigma, _ = _heteroscedastic_normal_case()
    family = NO()
    params = _initialize_params(y, X_mu, X_sigma, None, None, family)
    design_matrices = {"X_mu": X_mu, "X_sigma": X_sigma, "X_nu": None, "X_tau": None}
    data = {"y": y, "weights": jnp.ones_like(y), **design_matrices}

    def log_likelihood(params_dict, data_dict):
        return _compute_log_likelihood(params_dict, data_dict, family, design_matrices)

    fisher, score = _compute_full_observed_information_and_score(
        log_likelihood, params, data
    )
    slices = parameter_slices(params)
    block_diag = zero_cross_information_blocks(fisher, slices)
    ridge = 1e-6 * jnp.eye(fisher.shape[0], dtype=fisher.dtype)

    full_delta = np.asarray(jnp.linalg.solve(fisher + ridge, score))
    block_delta = np.asarray(jnp.linalg.solve(block_diag + ridge, score))

    assert np.linalg.norm(full_delta - block_delta) > 1e-6


def test_eta_cross_hessian_shape_symmetry_and_nonzero_cross_terms():
    y, X_mu, X_sigma, _ = _heteroscedastic_normal_case()
    result = fit_cg(NO(), y, X_mu, X_sigma, max_iter=8, tol=1e-5)
    bundle = eta_score_hessian(y, result.fitted_values, NO())

    assert bundle.parameter_order == ("mu", "sigma")
    assert bundle.score.shape == (len(y), 2)
    assert bundle.hessian.shape == (len(y), 2, 2)
    assert np.allclose(
        np.asarray(bundle.hessian),
        np.swapaxes(np.asarray(bundle.hessian), -1, -2),
        atol=1e-5,
    )
    assert np.linalg.norm(np.asarray(bundle.hessian[:, 0, 1])) > 1e-7


def test_gamlss_cg_records_cross_derivative_backend_diagnostics():
    _, _, _, data = _heteroscedastic_normal_case(n=28)
    model = gamlss(
        "y ~ x",
        sigma_formula="~ x",
        family=NO(),
        data=data,
        method="CG",
        control=gamlss_control(n_cyc=8, c_crit=1e-4, trace=False),
        verbose=False,
    )

    slots = model.additional_slots
    assert slots["method"] == "CG"
    assert slots["cg_backend"] == "CG_FULL_HESSIAN"
    assert slots["cg_cross_derivatives"] == "full_hessian"
    assert slots["cg_cross_derivatives"] != "disabled_fallback"
    assert isinstance(slots["cg_line_search_steps"], tuple)
    assert slots["cg_param_slices"] is not None


def test_cg_irls_cross_backend_uses_eta_derivative_bundle():
    _, _, _, data = _heteroscedastic_normal_case(n=30)
    model = gamlss(
        "y ~ x",
        sigma_formula="~ x",
        family=NO(),
        data=data,
        method="CG",
        cg_backend="irls_cross",
        control=gamlss_control(n_cyc=5, c_crit=1e-4, trace=False),
        verbose=False,
    )

    slots = model.additional_slots
    history = slots["deviance_history"]
    assert slots["cg_backend"] == "CG_IRLS_CROSS"
    assert slots["cg_cross_derivatives"] == "eta_correction"
    assert slots["cg_eta_hessian_shape"] == (30, 2, 2)
    assert isinstance(slots["cg_line_search_steps"], tuple)
    assert history[-1] <= history[0]


def test_cg_backend_selection_rejects_unknown_backend():
    _, _, _, data = _heteroscedastic_normal_case(n=12)
    import pytest

    with pytest.raises(ValueError, match="cg_backend"):
        gamlss(
            "y ~ x",
            sigma_formula="~ x",
            family=NO(),
            data=data,
            method="CG",
            cg_backend="not-a-backend",
            control=gamlss_control(n_cyc=2, c_crit=1e-4, trace=False),
            verbose=False,
        )
