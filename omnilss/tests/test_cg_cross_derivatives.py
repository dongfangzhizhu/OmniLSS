"""Regression tests for complete Cole-Green cross-derivative support."""

# ruff: noqa: E402

import numpy as np
import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from omnilss.cg_derivatives import eta_cross_hessian, eta_score_hessian
from omnilss.controls import gamlss_control
from omnilss.distributions import BCCG, BCT, GA, NO, resolve_family
from omnilss.fitting import gamlss
from omnilss.algorithms.cg_algorithm_v2 import cg_outer_step
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


def test_validation_matrix_eta_hessians_have_symmetric_cross_blocks():
    y = jnp.asarray([0.8, 1.0, 1.2, 1.4], dtype=jnp.float64)
    positive_sigma = jnp.asarray([0.2, 0.22, 0.24, 0.26], dtype=jnp.float64)
    base_shape_params = {
        "mu": jnp.asarray([1.0, 1.1, 1.2, 1.3], dtype=jnp.float64),
        "sigma": positive_sigma,
        "nu": jnp.asarray([-0.2, 0.0, 0.2, 0.4], dtype=jnp.float64),
    }

    cases = (
        (GA(), {"mu": base_shape_params["mu"], "sigma": positive_sigma}, (0, 1)),
        (BCCG(), base_shape_params, (0, 2)),
        (
            BCT(),
            {
                **base_shape_params,
                "tau": jnp.asarray([5.0, 5.5, 6.0, 6.5], dtype=jnp.float64),
            },
            (0, 2),
        ),
        (
            resolve_family("SHASH"),
            {
                "mu": base_shape_params["mu"],
                "sigma": jnp.asarray([0.8, 0.9, 1.0, 1.1], dtype=jnp.float64),
                "nu": base_shape_params["nu"],
                "tau": jnp.asarray([0.8, 1.0, 1.2, 1.4], dtype=jnp.float64),
            },
            (0, 3),
        ),
    )

    for family, params, cross_index in cases:
        bundle = eta_score_hessian(y, params, family)
        hessian = np.asarray(bundle.hessian)
        assert hessian.shape == (len(y), len(family.parameters), len(family.parameters))
        assert np.allclose(hessian, np.swapaxes(hessian, -1, -2), atol=1e-8)
        assert np.linalg.norm(hessian[:, cross_index[0], cross_index[1]]) > 1e-8


def test_eta_cross_hessian_alias_respects_parameter_order():
    y = jnp.asarray([0.8, 1.0, 1.2, 1.4], dtype=jnp.float64)
    params = {
        "mu": jnp.asarray([1.0, 1.1, 1.2, 1.3], dtype=jnp.float64),
        "sigma": jnp.asarray([0.4, 0.5, 0.6, 0.7], dtype=jnp.float64),
    }

    bundle = eta_score_hessian(y, params, NO(), parameter_order=("sigma", "mu"))
    alias_hessian = eta_cross_hessian(
        y, params, NO(), parameter_order=("sigma", "mu")
    )

    assert bundle.parameter_order == ("sigma", "mu")
    assert alias_hessian.shape == (len(y), 2, 2)
    assert np.allclose(np.asarray(alias_hessian), np.asarray(bundle.hessian))
    assert np.linalg.norm(np.asarray(alias_hessian[:, 0, 1])) > 1e-8


def test_cg_outer_step_preserves_nonzero_offsets_in_eta_updates():
    y = np.asarray([0.8, 0.9, 1.0, 1.15, 1.25, 1.4], dtype=np.float64)
    x = np.linspace(-1.0, 1.0, len(y))
    design_matrices = {
        "mu": np.column_stack([np.ones_like(x), x]),
        "sigma": np.ones((len(y), 1), dtype=np.float64),
    }
    coefficients = {
        "mu": np.asarray([1.0, 0.1], dtype=np.float64),
        "sigma": np.asarray([-0.5], dtype=np.float64),
    }
    offsets = {
        "mu": np.linspace(-0.05, 0.05, len(y)),
        "sigma": np.linspace(0.02, -0.02, len(y)),
    }
    etas = {
        parameter: design_matrices[parameter] @ coefficients[parameter]
        + offsets[parameter]
        for parameter in coefficients
    }
    family = NO()
    parameter_values = {
        parameter: np.asarray(family.link_inverses[parameter](jnp.asarray(eta)))
        for parameter, eta in etas.items()
    }

    _, proposed_etas, proposed_coefs, bundle = cg_outer_step(
        y=y,
        design_matrices=design_matrices,
        parameter_values=parameter_values,
        etas=etas,
        coefficients=coefficients,
        family=family,
        weights=np.linspace(0.7, 1.3, len(y)),
        step_sizes={"mu": 0.5, "sigma": 0.5},
        offsets=offsets,
    )

    assert bundle.hessian.shape == (len(y), 2, 2)
    for parameter in coefficients:
        reconstructed_eta = (
            design_matrices[parameter] @ proposed_coefs[parameter] + offsets[parameter]
        )
        assert np.allclose(proposed_etas[parameter], reconstructed_eta)

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


def test_cg_irls_cross_intercept_only_with_weights():
    y = np.asarray([0.2, 0.4, 0.6, 0.9, 1.1, 1.3], dtype=np.float64)
    weights = np.linspace(0.5, 1.5, len(y))
    model = gamlss(
        "y ~ 1",
        sigma_formula="~ 1",
        family=NO(),
        data={"y": y},
        weights=weights,
        method="CG",
        cg_backend="irls_cross",
        control=gamlss_control(n_cyc=4, c_crit=1e-4, trace=False),
        verbose=False,
    )

    assert model.additional_slots["cg_backend"] == "CG_IRLS_CROSS"
    assert model.additional_slots["cg_cross_derivatives"] == "eta_correction"
    assert (
        model.additional_slots["deviance_history"][-1]
        <= model.additional_slots["deviance_history"][0]
    )


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
