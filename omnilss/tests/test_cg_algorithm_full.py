"""Tests for Week 2 CG joint scoring matrix utilities."""

from __future__ import annotations

import numpy as np

from omnilss.algorithms.cg_algorithm_full import (
    build_joint_scoring_matrix,
    cg_outer_step,
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
    # Use a simple convex deviance so Newton direction from identity Hessian is exact.
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
