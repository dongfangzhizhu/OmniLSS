import numpy as np
import jax.numpy as jnp

from omnilss.safe_math import (
    safe_divide,
    safe_exp,
    safe_log,
    safe_log1p,
    safe_sigmoid,
    safe_softplus,
    safe_sqrt,
)


def test_safe_exp_clips_extremes():
    x = jnp.array([-1e6, 0.0, 1e6])
    out = safe_exp(x)
    assert jnp.isfinite(out).all()
    assert np.isclose(float(out[1]), 1.0, rtol=1e-7)


def test_safe_log_avoids_negative_inf():
    x = jnp.array([0.0, 1.0, 10.0])
    out = safe_log(x)
    assert jnp.isfinite(out).all()


def test_safe_log1p_handles_near_minus_one():
    x = jnp.array([-2.0, -0.5, 1.0])
    out = safe_log1p(x)
    assert jnp.isfinite(out).all()


def test_safe_softplus_stable_for_large_values():
    x = jnp.array([-1000.0, 0.0, 1000.0])
    out = safe_softplus(x)
    assert jnp.isfinite(out).all()


def test_safe_sigmoid_bounded():
    x = jnp.array([-1e6, 0.0, 1e6])
    out = safe_sigmoid(x)
    assert jnp.all(out >= 0.0)
    assert jnp.all(out <= 1.0)


def test_safe_divide_handles_zero_denominator():
    num = jnp.array([1.0, 0.0, -1.0])
    den = jnp.array([0.0, 0.0, 0.0])
    out = safe_divide(num, den)
    assert jnp.isfinite(out).all()


def test_safe_sqrt_handles_negative_values():
    x = jnp.array([-1.0, 0.0, 4.0])
    out = safe_sqrt(x)
    assert jnp.isfinite(out).all()
    assert float(out[0]) == 0.0
