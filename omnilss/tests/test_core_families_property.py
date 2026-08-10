"""Phase 2.1 core-family property tests (E3 gate acceptance).

Eight core families: NO, GA, PO, BI, NBI, BE, ZAGA, GU.

Each family gets hypothesis property tests (default 100 examples each):
- d: density finite and non-negative on its support
- p: CDF monotone in q, values within [0, 1]
- q/p round-trip: q(p(x)) ~= x for continuous families
- r: seeded sampling returns finite values of the right length
- boundary: documented boundary behaviour (support edges, zero mass)

Alignment: DEVELOPMENT_PLAN Phase 2 task 2.1 acceptance — "每族 100 次属性测试
+ 固定 R artifact；仅全过才晋级 E3".
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.random as jrandom

from omnilss.distributions import resolve_family

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st

# --- per-family parameterisation metadata --------------------------------

# (param_names, params strategy, support x strategy, is_continuous, has_mass_point)
FAMILIES: dict[str, tuple[tuple[str, ...], st.SearchStrategy, st.SearchStrategy, bool, bool]] = {
    "NO": (
        ("mu", "sigma"),
        st.tuples(st.floats(-10.0, 10.0), st.floats(1e-3, 10.0)),
        st.floats(-20.0, 20.0),
        True,
        False,
    ),
    "GA": (
        ("mu", "sigma"),
        st.tuples(st.floats(0.05, 20.0), st.floats(1e-3, 2.0)),
        st.floats(1e-6, 50.0),
        True,
        False,
    ),
    "PO": (
        ("mu",),
        st.tuples(st.floats(0.05, 50.0)),
        st.integers(0, 200),
        False,
        False,
    ),
    "BI": (
        ("mu",),
        st.tuples(st.floats(1e-6, 1.0 - 1e-6)),
        st.integers(0, 1),
        False,
        False,
    ),
    "NBI": (
        ("mu", "sigma"),
        st.tuples(st.floats(0.05, 50.0), st.floats(1e-3, 5.0)),
        st.integers(0, 300),
        False,
        False,
    ),
    "BE": (
        ("mu", "sigma"),
        st.tuples(st.floats(1e-6, 1.0 - 1e-6), st.floats(1e-3, 0.99)),
        st.floats(1e-6, 1.0 - 1e-6),
        True,
        False,
    ),
    "ZAGA": (
        ("mu", "sigma", "nu"),
        st.tuples(st.floats(0.05, 20.0), st.floats(1e-3, 2.0), st.floats(1e-6, 1.0 - 1e-6)),
        st.floats(0.0, 50.0),
        True,  # continuous hurdle (mass at 0)
        True,
    ),
    "GU": (
        ("mu", "sigma"),
        st.tuples(st.floats(-10.0, 10.0), st.floats(1e-3, 10.0)),
        st.floats(-20.0, 30.0),
        True,
        False,
    ),
}


def _call(fam, method: str, x, params, **extra):
    kwargs = dict(zip(FAMILIES[fam.name][0], params))
    fn = getattr(fam, method)
    if method == "r":
        return fn(jrandom.PRNGKey(1234), 64, **kwargs)
    return fn(x, **kwargs, **extra)


@pytest.mark.parametrize("fam_name", sorted(FAMILIES))
class TestCoreFamilyProperties:
    def test_d_finite_nonnegative(self, fam_name):
        fam = resolve_family(fam_name)
        param_names, params_st, x_st, _, _ = FAMILIES[fam_name]
        strategy = st.tuples(params_st, x_st)

        @given(data=st.data())
        @settings(deadline=None)
        def _check(data):
            params, x = data.draw(strategy)
            d = np.asarray(_call(fam, "d", jnp.array([x]), params), dtype=np.float64)
            assert np.isfinite(d).all(), (fam_name, params, x, d)
            assert (d >= 0).all(), (fam_name, params, x, d)

        _check()

    def test_p_monotone_bounded(self, fam_name):
        fam = resolve_family(fam_name)
        param_names, params_st, x_st, _, _ = FAMILIES[fam_name]
        strategy = st.tuples(params_st, x_st, x_st)

        @given(data=st.data())
        @settings(deadline=None)
        def _check(data):
            params, x1, x2 = data.draw(strategy)
            lo, hi = sorted([float(x1), float(x2)])
            p_lo = float(np.asarray(_call(fam, "p", jnp.array([lo]), params))[0])
            p_hi = float(np.asarray(_call(fam, "p", jnp.array([hi]), params))[0])
            assert p_lo <= p_hi + 1e-12, (fam_name, params, lo, hi, p_lo, p_hi)
            assert 0.0 - 1e-12 <= p_lo <= 1.0 + 1e-12
            assert 0.0 - 1e-12 <= p_hi <= 1.0 + 1e-12

        _check()

    def test_q_p_roundtrip(self, fam_name):
        fam = resolve_family(fam_name)
        param_names, params_st, x_st, is_continuous, has_mass_point = FAMILIES[fam_name]
        if not is_continuous:
            pytest.skip("discrete family: q/p round-trip not applicable")
        strategy = st.tuples(params_st, x_st)

        @given(data=st.data())
        @settings(deadline=None)
        def _check(data):
            params, x = data.draw(strategy)
            x = float(x)
            p_val = float(np.asarray(_call(fam, "p", jnp.array([x]), params))[0])
            if p_val <= 1e-8 or p_val >= 1.0 - 1e-8:
                return  # extreme tail: inverse unreliable, skip
            if has_mass_point:
                nu = params[2]
                # ZAGA: near the zero-mass point nu the float cancellation of
                # (p - nu) makes gamma-tail inversion unreliable (e.g. x in
                # the far-left gamma tail -> p_adj ~ 1e-14).  Same limitation
                # exists in R's qZAGA.
                if abs(p_val - nu) / max(nu, 1.0 - nu) < 1e-8:
                    return
            q_val = float(np.asarray(_call(fam, "q", jnp.array([p_val]), params))[0])
            tol = 1e-2 * max(1.0, abs(x))
            assert abs(q_val - x) <= tol, (fam_name, params, x, p_val, q_val)

        _check()

    def test_r_sampling(self, fam_name):
        fam = resolve_family(fam_name)
        param_names, params_st, x_st, is_continuous, _ = FAMILIES[fam_name]

        @given(data=st.data())
        @settings(deadline=None)
        def _check(data):
            params = data.draw(params_st)
            samples = np.asarray(_call(fam, "r", None, params), dtype=np.float64)
            assert samples.shape == (64,)
            assert np.isfinite(samples).all(), (fam_name, params, samples)
            if not is_continuous:
                assert (samples >= 0).all()
                assert np.allclose(samples, np.round(samples)), "discrete samples must be integer-valued"

        _check()

    def test_boundary(self, fam_name):
        fam = resolve_family(fam_name)
        if fam_name == "ZAGA":
            # mass at zero equals nu
            d0 = float(np.asarray(_call(fam, "d", jnp.array([0.0]), (2.0, 0.5, 0.3)))[0])
            assert np.isclose(d0, 0.3, atol=1e-6), d0
            p0 = float(np.asarray(_call(fam, "p", jnp.array([0.0]), (2.0, 0.5, 0.3)))[0])
            assert np.isclose(p0, 0.3, atol=1e-6), p0
        elif fam_name == "GA":
            # density at 0 is finite for shape alpha > 1 (sigma < 1)
            d0 = float(np.asarray(_call(fam, "d", jnp.array([0.0]), (2.0, 0.5)))[0])
            assert np.isfinite(d0) and d0 >= 0.0
        elif fam_name == "BE":
            d0 = float(np.asarray(_call(fam, "d", jnp.array([1e-12]), (0.5, 0.3)))[0])
            assert np.isfinite(d0) and d0 >= 0.0
        elif fam_name in ("PO", "NBI"):
            d0 = float(np.asarray(_call(fam, "d", jnp.array([0]), (3.0,) if fam_name == "PO" else (3.0, 0.3)))[0])
            assert 0.0 < d0 <= 1.0  # positive zero mass
        elif fam_name == "BI":
            d0 = float(np.asarray(_call(fam, "d", jnp.array([0]), (0.3,)))[0])
            d1 = float(np.asarray(_call(fam, "d", jnp.array([1]), (0.3,)))[0])
            assert np.isclose(d0 + d1, 1.0, atol=1e-6)
        elif fam_name in ("NO", "GU"):
            d0 = float(np.asarray(_call(fam, "d", jnp.array([0.0]), (0.0, 1.0)))[0])
            assert 0.0 < d0 < 1.0


def test_core_families_fit_smoke():
    """At least one fit case per core family (E3 acceptance)."""
    from omnilss.fitting import gamlss
    from omnilss.controls import gamlss_control

    rng = np.random.default_rng(20260729)
    n = 80
    x = rng.uniform(-1, 1, n)
    control = gamlss_control(n_cyc=30, trace=False)

    cases = {
        "NO": ("NO", {"y": 1.0 + 0.5 * x + rng.normal(scale=0.3, size=n), "x": x}),
        "GA": ("GA", {"y": rng.gamma(2.0, 2.0, n), "x": x}),
        "PO": ("PO", {"y": rng.poisson(np.exp(0.5 + 0.4 * x)), "x": x}),
        "BI": ("BI", {"y": (rng.uniform(size=n) < 0.5).astype(float), "x": x}),
        "NBI": ("NBI", {"y": rng.negative_binomial(5, p=0.4, size=n), "x": x}),
        "BE": ("BE", {"y": rng.beta(2.0, 2.0, n), "x": x}),
        "ZAGA": ("ZAGA", {"y": np.where(rng.uniform(size=n) < 0.2, 0.0, rng.gamma(2.0, 2.0, n)), "x": x}),
        "GU": ("GU", {"y": rng.gumbel(2.0, 1.0, n), "x": x}),
    }
    for fam_name, (fam, data) in cases.items():
        model = gamlss("y ~ x", family=fam, data=data, control=control)
        assert np.isfinite(model.deviance), f"{fam_name} deviance not finite"
