"""Tests for Week 2 CG joint scoring matrix utilities."""

from __future__ import annotations

import numpy as np

from omnilss.algorithms.cg_algorithm_full import (
    build_joint_scoring_matrix,
    cg_outer_step,
    run_cg_outer_loop,
    solve_joint_system,
)


def test_build_joint_scoring_matrix_shapes_and_symmetry():
    scores = {
        "mu": np.array([1.0, -2.0]),
        "sigma": np.array([0.5]),
    }
    blocks = {
        ("mu", "mu"): np.array([[4.0, 1.0], [1.0, 3.0]]),
        ("mu", "sigma"): np.array([[0.2], [0.1]]),
        ("sigma", "mu"): np.array([[0.2, 0.1]]),
        ("sigma", "sigma"): np.array([[2.0]]),
    }

    out = build_joint_scoring_matrix(scores, blocks, ridge=0.0)
    assert out.matrix.shape == (3, 3)
    np.testing.assert_allclose(np.asarray(out.matrix), np.asarray(out.matrix).T, atol=1e-12)
    np.testing.assert_allclose(np.asarray(out.score), np.array([1.0, -2.0, 0.5]), atol=1e-12)


def test_solve_joint_system_matches_numpy_reference():
    scores = {
        "mu": np.array([0.3, -0.7]),
        "sigma": np.array([1.1]),
    }
    blocks = {
        ("mu", "mu"): np.array([[3.0, 0.4], [0.4, 2.5]]),
        ("mu", "sigma"): np.array([[0.2], [0.5]]),
        ("sigma", "mu"): np.array([[0.2, 0.5]]),
        ("sigma", "sigma"): np.array([[1.8]]),
    }

    delta = solve_joint_system(scores, blocks, ridge=0.0)

    h = np.block([
        [blocks[("mu", "mu")], blocks[("mu", "sigma")]],
        [blocks[("sigma", "mu")], blocks[("sigma", "sigma")]],
    ])
    s = np.concatenate([scores["mu"], scores["sigma"]])
    expected = np.linalg.solve(h, s)

    np.testing.assert_allclose(np.asarray(delta["mu"]), expected[:2], rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(np.asarray(delta["sigma"]), expected[2:], rtol=1e-7, atol=1e-7)


def test_cg_outer_step_line_search_reduces_deviance():
    eta0 = {"mu": np.array([3.0, -1.0]), "sigma": np.array([2.0])}
    scores = {"mu": np.array([-3.0, 1.0]), "sigma": np.array([-2.0])}
    blocks = {
        ("mu", "mu"): np.eye(2),
        ("mu", "sigma"): np.zeros((2, 1)),
        ("sigma", "mu"): np.zeros((1, 2)),
        ("sigma", "sigma"): np.eye(1),
    }

    def dev_fn(eta):
        v = np.concatenate([np.asarray(eta["mu"]), np.asarray(eta["sigma"])])
        return float(np.dot(v, v))

    out = cg_outer_step(
        eta=eta0,
        scores=scores,
        hessian_blocks=blocks,
        global_deviance_fn=dev_fn,
        ridge=0.0,
        step_size=1.0,
    )

    assert out.new_global_deviance <= out.old_global_deviance
    np.testing.assert_allclose(np.asarray(out.updated_eta["mu"]), np.zeros(2), atol=1e-12)
    np.testing.assert_allclose(np.asarray(out.updated_eta["sigma"]), np.zeros(1), atol=1e-12)


def test_run_cg_outer_loop_converges_on_quadratic_system():
    def dev_fn(eta):
        m = np.asarray(eta["mu"])
        s = np.asarray(eta["sigma"])
        return float(np.dot(m, m) + np.dot(s, s))

    def build_fn(eta):
        mu = np.asarray(eta["mu"])
        sigma = np.asarray(eta["sigma"])
        scores = {"mu": -mu, "sigma": -sigma}
        hess = {
            ("mu", "mu"): np.eye(mu.shape[0]),
            ("mu", "sigma"): np.zeros((mu.shape[0], sigma.shape[0])),
            ("sigma", "mu"): np.zeros((sigma.shape[0], mu.shape[0])),
            ("sigma", "sigma"): np.eye(sigma.shape[0]),
        }
        return scores, hess

    res = run_cg_outer_loop(
        eta0={"mu": np.array([1.0, -2.0]), "sigma": np.array([0.5])},
        build_scores_hessian_fn=build_fn,
        global_deviance_fn=dev_fn,
        max_outer=10,
        c_crit=1e-10,
        ridge=0.0,
    )
    assert res.converged
    assert res.n_iter <= 2
    assert res.deviance_history[-1] <= res.deviance_history[0]

def test_cg_outer_step_rejects_non_improving_direction_with_zero_step():
    eta0 = {"mu": np.array([1.0]), "sigma": np.array([1.0])}
    # Wrong-way direction: positive score with identity Hessian increases quadratic deviance.
    scores = {"mu": np.array([1.0]), "sigma": np.array([1.0])}
    blocks = {
        ("mu", "mu"): np.eye(1),
        ("mu", "sigma"): np.zeros((1, 1)),
        ("sigma", "mu"): np.zeros((1, 1)),
        ("sigma", "sigma"): np.eye(1),
    }

    def dev_fn(eta):
        v = np.concatenate([np.asarray(eta["mu"]), np.asarray(eta["sigma"])])
        return float(np.dot(v, v))

    out = cg_outer_step(
        eta=eta0,
        scores=scores,
        hessian_blocks=blocks,
        global_deviance_fn=dev_fn,
        ridge=0.0,
        step_size=1.0,
        min_step_size=1e-3,
        backtracking=0.5,
    )

    assert out.accepted_step_size == 0.0
    np.testing.assert_allclose(np.asarray(out.updated_eta["mu"]), eta0["mu"], atol=1e-12)
    np.testing.assert_allclose(np.asarray(out.updated_eta["sigma"]), eta0["sigma"], atol=1e-12)
    assert out.new_global_deviance == out.old_global_deviance


def test_run_cg_outer_loop_termination_reason_converged():
    def dev_fn(eta):
        m = np.asarray(eta["mu"])
        s = np.asarray(eta["sigma"])
        return float(np.dot(m, m) + np.dot(s, s))

    def build_fn(eta):
        mu = np.asarray(eta["mu"])
        sigma = np.asarray(eta["sigma"])
        return {"mu": -mu, "sigma": -sigma}, {
            ("mu", "mu"): np.eye(mu.shape[0]),
            ("mu", "sigma"): np.zeros((mu.shape[0], sigma.shape[0])),
            ("sigma", "mu"): np.zeros((sigma.shape[0], mu.shape[0])),
            ("sigma", "sigma"): np.eye(sigma.shape[0]),
        }

    res = run_cg_outer_loop(
        eta0={"mu": np.array([1.0]), "sigma": np.array([0.5])},
        build_scores_hessian_fn=build_fn,
        global_deviance_fn=dev_fn,
        max_outer=5,
        c_crit=1e-10,
        ridge=0.0,
    )
    assert res.converged
    assert res.termination_reason == "relative_deviance_converged"


def test_run_cg_outer_loop_termination_reason_max_outer():
    def dev_fn(eta):
        m = np.asarray(eta["mu"])
        s = np.asarray(eta["sigma"])
        return float(np.dot(m, m) + np.dot(s, s))

    def build_fn(eta):
        mu = np.asarray(eta["mu"])
        sigma = np.asarray(eta["sigma"])
        return {"mu": -0.1 * mu, "sigma": -0.1 * sigma}, {
            ("mu", "mu"): np.eye(1),
            ("mu", "sigma"): np.zeros((1, 1)),
            ("sigma", "mu"): np.zeros((1, 1)),
            ("sigma", "sigma"): np.eye(1),
        }

    res = run_cg_outer_loop(
        eta0={"mu": np.array([1.0]), "sigma": np.array([1.0])},
        build_scores_hessian_fn=build_fn,
        global_deviance_fn=dev_fn,
        max_outer=2,
        c_crit=0.0,
        ridge=0.0,
    )
    assert not res.converged
    assert res.n_iter == 2
    assert res.termination_reason == "max_outer_reached"


def test_run_cg_outer_loop_termination_reason_no_progress_step_rejected():
    def dev_fn(eta):
        m = np.asarray(eta["mu"])
        s = np.asarray(eta["sigma"])
        return float(np.dot(m, m) + np.dot(s, s))

    def build_fn(_eta):
        # Wrong-way score direction forces line-search rejection.
        return {"mu": np.array([1.0]), "sigma": np.array([1.0])}, {
            ("mu", "mu"): np.eye(1),
            ("mu", "sigma"): np.zeros((1, 1)),
            ("sigma", "mu"): np.zeros((1, 1)),
            ("sigma", "sigma"): np.eye(1),
        }

    res = run_cg_outer_loop(
        eta0={"mu": np.array([1.0]), "sigma": np.array([1.0])},
        build_scores_hessian_fn=build_fn,
        global_deviance_fn=dev_fn,
        max_outer=20,
        c_crit=-1.0,
        ridge=0.0,
    )

    assert not res.converged
    assert res.n_iter == 1
    assert res.step_sizes == (0.0,)
    assert res.termination_reason == "no_progress_step_rejected"
    np.testing.assert_allclose(np.asarray(res.eta["mu"]), np.array([1.0]), atol=1e-12)
    np.testing.assert_allclose(np.asarray(res.eta["sigma"]), np.array([1.0]), atol=1e-12)


def test_run_cg_outer_loop_zero_deviance_still_reports_converged():
    def dev_fn(eta):
        mu = np.asarray(eta["mu"])
        sigma = np.asarray(eta["sigma"])
        return float(np.dot(mu, mu) + np.dot(sigma, sigma))

    def build_fn(_eta):
        return {"mu": np.array([0.0]), "sigma": np.array([0.0])}, {
            ("mu", "mu"): np.eye(1),
            ("mu", "sigma"): np.zeros((1, 1)),
            ("sigma", "mu"): np.zeros((1, 1)),
            ("sigma", "sigma"): np.eye(1),
        }

    res = run_cg_outer_loop(
        eta0={"mu": np.array([0.0]), "sigma": np.array([0.0])},
        build_scores_hessian_fn=build_fn,
        global_deviance_fn=dev_fn,
        max_outer=5,
        c_crit=1e-8,
        ridge=0.0,
    )
    assert res.converged
    assert res.termination_reason == "relative_deviance_converged"
