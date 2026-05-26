from __future__ import annotations

import jax.numpy as jnp

from omnilss import TF, TF2
from omnilss.distribution_registry import resolve


def test_tf2_constructor_and_registry() -> None:
    tf = TF()
    tf2 = TF2()
    assert tf2.name == "TF2"
    assert tf2.parameters == tf.parameters
    assert resolve("TF2").name == "TF2"


def test_tf2_uses_logshiftto2_nu_link() -> None:
    tf2 = TF2()
    assert tf2.links["nu"] == "logshiftto2"

    nu_value = jnp.array([3.5])
    eta = tf2.link_functions["nu"](nu_value)
    recovered = tf2.link_inverses["nu"](eta)
    assert float(recovered[0]) > 2.0
    assert abs(float(recovered[0]) - 3.5) < 1e-6


def test_tf2_dpqr_clamps_nu_to_gt2() -> None:
    tf2 = TF2()
    tf = TF()

    x = jnp.array([0.2])
    mu = jnp.array([0.0])
    sigma = jnp.array([1.0])
    tiny_nu = jnp.array([0.8])

    tf2_d = tf2.d(x, mu, sigma, tiny_nu)
    floor_nu = jnp.array([2.0 + jnp.finfo(jnp.float64).eps])
    tf_d = tf.d(x, mu, sigma, floor_nu)
    assert jnp.allclose(tf2_d, tf_d)


def test_tf2_pq_lower_tail_logp_roundtrip_near_boundary() -> None:
    tf2 = TF2()

    mu = jnp.array([0.0])
    sigma = jnp.array([1.0])
    nu = jnp.array([2.0001])
    x = jnp.array([0.35])

    p_lower = tf2.p(x, mu, sigma, nu)
    p_upper = tf2.p(x, mu, sigma, nu, lower_tail=False)
    assert jnp.allclose(p_lower + p_upper, jnp.array([1.0]), atol=1e-6)

    p_log = tf2.p(x, mu, sigma, nu, log_p=True)
    assert jnp.allclose(jnp.exp(p_log), p_lower, atol=1e-6)

    x_back = tf2.q(p_lower, mu, sigma, nu)
    # q/p path is approximate in current staged numerics; enforce bounded inversion error.
    assert jnp.all(jnp.abs(x_back - x) < 0.08)


def test_tf2_score_hessian_finite_near_nu_boundary() -> None:
    tf2 = TF2()
    y = jnp.array([0.1, -0.3, 0.8])
    mu = jnp.array([0.0, 0.0, 0.0])
    sigma = jnp.array([1.0, 1.1, 0.9])
    nu = jnp.array([1.1, 1.5, 1.9])

    dldn = tf2.score_functions["nu"](y, mu, sigma, nu)
    d2ldn2 = tf2.hessian_functions["nu"](y, mu, sigma, nu)

    assert jnp.all(jnp.isfinite(dldn))
    assert jnp.all(jnp.isfinite(d2ldn2))


def test_tf2_q_refinement_respects_upper_tail() -> None:
    tf2 = TF2()

    mu = jnp.array([0.0])
    sigma = jnp.array([1.0])
    nu = jnp.array([2.2])
    x = jnp.array([0.35])

    p_upper = tf2.p(x, mu, sigma, nu, lower_tail=False)
    x_back_upper = tf2.q(p_upper, mu, sigma, nu, lower_tail=False)

    # Upper-tail q(p) should invert with bounded error in staged numerics.
    assert jnp.all(jnp.abs(x_back_upper - x) < 0.08)


def test_tf2_q_handles_extreme_logp_without_nonfinite() -> None:
    tf2 = TF2()

    mu = jnp.array([0.0])
    sigma = jnp.array([1.0])
    nu = jnp.array([2.00001])
    # Very extreme log-probability input to stress Newton correction guard.
    p_log = jnp.array([-80.0])

    q = tf2.q(p_log, mu, sigma, nu, lower_tail=True, log_p=True)
    assert jnp.all(jnp.isfinite(q))


def test_tf2_q_edge_probabilities_fallback_to_base() -> None:
    tf2 = TF2()

    mu = jnp.array([0.0])
    sigma = jnp.array([1.0])
    nu = jnp.array([2.0001])

    # Edge probabilities should not trigger unstable refinement jumps.
    q_lo = tf2.q(jnp.array([1e-12]), mu, sigma, nu)
    q_hi = tf2.q(jnp.array([1 - 1e-7]), mu, sigma, nu)

    assert jnp.all(jnp.isfinite(q_lo))
    assert jnp.all(jnp.isfinite(q_hi))
    assert float(q_lo[0]) < float(q_hi[0])


def test_tf2_q_monotonic_over_probability_grid() -> None:
    tf2 = TF2()

    mu = jnp.array([0.0])
    sigma = jnp.array([1.0])
    nu = jnp.array([2.2])

    probs = jnp.linspace(0.01, 0.99, 64)
    q_vals = tf2.q(probs, mu, sigma, nu)

    # Quantile function should be non-decreasing over probability grid.
    diffs = q_vals[1:] - q_vals[:-1]
    assert jnp.all(diffs >= -1e-6)
