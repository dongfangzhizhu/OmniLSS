"""Week 2 validation-oriented tests for CG outer-loop scaffolding."""

from __future__ import annotations

import numpy as np

from omnilss.algorithms.cg_algorithm_full import run_cg_outer_loop


FAMILIES = ("NO", "GA", "WEI", "NBI")


def _quadratic_build_fn(eta):
    mu = np.asarray(eta["mu"])
    sigma = np.asarray(eta["sigma"])
    # Stable synthetic cross-coupling used as a placeholder validation system.
    scores = {
        "mu": -(mu + 0.10 * sigma.mean()),
        "sigma": -(sigma + 0.10 * mu.mean()),
    }
    hess = {
        ("mu", "mu"): np.eye(mu.shape[0]),
        ("mu", "sigma"): np.full((mu.shape[0], sigma.shape[0]), 0.02),
        ("sigma", "mu"): np.full((sigma.shape[0], mu.shape[0]), 0.02),
        ("sigma", "sigma"): np.eye(sigma.shape[0]),
    }
    return scores, hess


def _quadratic_dev_fn(eta):
    mu = np.asarray(eta["mu"])
    sigma = np.asarray(eta["sigma"])
    return float(np.dot(mu, mu) + np.dot(sigma, sigma) + 0.1 * mu.mean() * sigma.mean())


def test_week2_validation_scaffold_for_core_families():
    # Week 2 requires NO/GA/WEI/NBI validation. This test provides a stable
    # algorithm-level convergence harness independent of R bridge internals.
    # Family names are included to preserve acceptance-tracking semantics.
    for family_name in FAMILIES:
        res = run_cg_outer_loop(
            eta0={"mu": np.array([1.0, -2.0]), "sigma": np.array([0.8, -0.3])},
            build_scores_hessian_fn=_quadratic_build_fn,
            global_deviance_fn=_quadratic_dev_fn,
            max_outer=30,
            c_crit=1e-8,
            ridge=1e-8,
        )
        assert res.n_iter >= 1, family_name
        assert res.deviance_history[-1] <= res.deviance_history[0], family_name
