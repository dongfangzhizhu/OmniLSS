# SPDX-License-Identifier: GPL-3.0-or-later
"""Tests for batched JAX RS fitting helpers."""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from omnilss.algorithms.jax_family_specs import get_jax_spec
from omnilss.algorithms.jax_rs_batch import batch_jax_rs_fit
from omnilss.algorithms.jax_rs_core import jax_rs_fit_core
from omnilss.algorithms.jax_rs_integration import gamlss_rs_jax_batch
from omnilss.distributions import NO


def _make_batch_no_data(k_models: int = 3, n_obs: int = 80):
    rng = np.random.default_rng(123)
    x = np.linspace(-1.0, 1.0, n_obs)
    ys = []
    datasets = []
    for k in range(k_models):
        y = 1.0 + (0.5 + 0.1 * k) * x + rng.normal(0.0, 0.25, n_obs)
        ys.append(y)
        datasets.append({"y": y, "x": x})
    x_mat = jnp.asarray(np.column_stack([np.ones(n_obs), x]), dtype=jnp.float64)
    sigma_mat = jnp.ones((n_obs, 1), dtype=jnp.float64)
    return jnp.asarray(np.stack(ys), dtype=jnp.float64), (x_mat, sigma_mat), datasets


def test_batch_jax_rs_fit_matches_repeated_core_for_broadcast_design():
    ys, designs, _ = _make_batch_no_data()
    spec = get_jax_spec("NO")

    batched = batch_jax_rs_fit(
        ys=ys,
        Xs_per_model=designs,
        family_specs=spec,
        max_outer=12,
        max_inner=1,
        tol=1e-4,
    )
    repeated = [
        jax_rs_fit_core(
            y=ys[idx],
            Xs=designs,
            spec=spec,
            max_outer=12,
            max_inner=1,
            tol=1e-4,
        )
        for idx in range(ys.shape[0])
    ]

    assert len(batched) == ys.shape[0]
    for batch_result, repeated_result in zip(batched, repeated, strict=True):
        np.testing.assert_allclose(batch_result.g_dev, repeated_result.g_dev, rtol=1e-9)
        np.testing.assert_allclose(batch_result.params, repeated_result.params, rtol=1e-8)


def test_batch_jax_rs_fit_accepts_per_model_weights_and_designs():
    ys, designs, _ = _make_batch_no_data(k_models=2)
    spec = get_jax_spec("NO")
    weights = jnp.ones_like(ys)

    results = batch_jax_rs_fit(
        ys=ys,
        Xs_per_model=[designs, designs],
        family_specs=[spec, spec],
        obs_weights=weights,
        max_outer=10,
    )

    assert len(results) == 2
    assert all(np.isfinite(result.g_dev) for result in results)


def test_gamlss_rs_jax_batch_formula_layer_returns_models():
    _, _, datasets = _make_batch_no_data(k_models=2, n_obs=60)

    models = gamlss_rs_jax_batch(
        formula="y ~ x",
        families=NO(),
        datasets=datasets,
        sigma_formula="~ 1",
    )

    assert len(models) == 2
    assert all(model.family.name == "NO" for model in models)
    assert all(np.isfinite(model.g_dev) for model in models)


def test_same_family_same_shape_batch_uses_vmap_path(monkeypatch):
    """The accelerated batch path should not call the per-model fallback."""
    from omnilss.algorithms import jax_rs_batch

    ys, designs, _ = _make_batch_no_data(k_models=2, n_obs=40)
    spec = get_jax_spec("NO")

    def fail_fallback(*args, **kwargs):  # pragma: no cover - should not be called
        raise AssertionError("per-model fallback should not run for same-shape NO batch")

    monkeypatch.setattr(jax_rs_batch, "jax_rs_fit_core", fail_fallback)

    results = batch_jax_rs_fit(
        ys=ys,
        Xs_per_model=designs,
        family_specs=spec,
        max_outer=8,
        max_inner=1,
        tol=1e-4,
    )

    assert len(results) == 2
    assert all(np.isfinite(result.g_dev) for result in results)
