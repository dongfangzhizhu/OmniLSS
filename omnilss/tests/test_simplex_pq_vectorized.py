"""SIMPLEX p/q: vectorized parameters and live-R parity (D2.31).

History: pSIMPLEX/qSIMPLEX existed and were wired into the family, but the
CDF integration built a single ``(1, n_points)`` grid around scalar
parameters and qSIMPLEX solved per element with scipy ``brentq`` over the
whole parameter vector -- so any *vector* parameter call raised a shape
error and the parity scanner reported ``p: no Python function``.  The
existing tests only ever called with scalar parameters, which is exactly
the "dpqr pass under constant parameters != vectorized path correct" trap
recorded in D2.30 (R's dZAP recycle artifact).
"""

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilss.dpqr_functions import pSIMPLEX, qSIMPLEX, rSIMPLEX
from tests.rbus.core import RTestBus


R_HAS_SIMPLEX = True


class TestVectorizedParameters:
    """The defect that motivated D2.31: vector params must not crash."""

    def test_p_survives_vector_parameters(self) -> None:
        q = np.linspace(0.05, 0.95, 8)
        out = np.asarray(pSIMPLEX(q, np.full(8, 0.4), np.full(8, 0.8)))
        assert out.shape == (8,)
        assert np.all(np.isfinite(out))
        assert np.all(np.diff(out) >= 0), "CDF must stay monotone across the grid"

    def test_q_survives_vector_parameters(self) -> None:
        p = np.linspace(0.1, 0.9, 8)
        out = np.asarray(qSIMPLEX(p, np.full(8, 0.4), np.full(8, 0.8)))
        assert out.shape == (8,)
        assert np.all(np.isfinite(out))
        assert np.all((out > 0) & (out < 1))
        assert np.all(np.diff(out) >= 0), "quantile must stay monotone"

    def test_scalar_input_keeps_scalar_shape(self) -> None:
        assert np.asarray(pSIMPLEX(0.4, 0.4, 0.8)).shape == ()
        assert np.asarray(qSIMPLEX(0.5, 0.4, 0.8)).shape == ()

    def test_recycling_matches_r(self) -> None:
        """R recycles q/mu/sigma to a common length; check one mixed call."""
        # live R: pSIMPLEX(0.4, mu=c(0.3, 0.5, 0.7), sigma=0.8)
        expected = np.array([0.9010687821454302, 0.1537170829635939, 1.026625538167297e-04])
        out = np.asarray(pSIMPLEX(0.4, np.array([0.3, 0.5, 0.7]), 0.8))
        np.testing.assert_allclose(out, expected, rtol=1e-7, atol=1e-9)


class TestLiveRParity:
    """Interior CDF / quantile against live R (RTestBus, element-wise)."""

    def test_p_matches_live_r_on_vector_params(self) -> None:
        q = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 0.9, 0.99]
        mu = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
        sigma = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]
        expected = RTestBus().eval_family(
            "SIMPLEX", "p", dict(q=q, mu=mu, sigma=sigma, **{"lower.tail": [True] * 8, "log.p": [False] * 8})
        )
        actual = np.asarray(pSIMPLEX(np.asarray(q), np.asarray(mu), np.asarray(sigma)))
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)

    def test_q_agrees_with_r_within_r_s_own_uniroot_error(self) -> None:
        # R's qSIMPLEX uses uniroot with the default tol (~1.2e-4); its own
        # roots carry up to ~2e-5 absolute error (evidence table in
        # tools/rs_family_parity_scan.py _DPQR_TOLERANCE_NOTE).  Compare with
        # that slack -- not with a tighter one, which would demand we beat R
        # at being R.
        p = [0.1, 0.3, 0.5, 0.7, 0.9]
        expected = RTestBus().eval_family(
            "SIMPLEX", "q", dict(p=p, mu=[0.4] * 5, sigma=[0.8] * 5)
        )
        actual = np.asarray(qSIMPLEX(np.asarray(p), 0.4, 0.8))
        np.testing.assert_allclose(actual, expected, atol=2.5e-5)

    def test_q_is_the_inverse_of_p(self) -> None:
        p = np.array([0.05, 0.25, 0.5, 0.75, 0.95])
        q = np.asarray(qSIMPLEX(p, 0.45, 1.1))
        back = np.asarray(pSIMPLEX(q, 0.45, 1.1))
        np.testing.assert_allclose(back, p, atol=1e-9)


class TestBoundarySemantics:
    """R's explicit ifelse: q <= 0 or q >= 1 gives 0, for both tails."""

    def test_boundary_gives_zero_for_both_tails(self) -> None:
        out = np.asarray(pSIMPLEX(np.array([-1.0, 0.0, 1.0, 2.0]), 0.4, 0.8))
        np.testing.assert_array_equal(out, np.zeros(4))

    def test_boundary_gives_zero_even_with_log_p(self) -> None:
        # live R: pSIMPLEX(c(-1, 2), mu=0.4, sigma=0.8, log.p=TRUE) -> c(0, 0)
        # (0, not -inf -- the overwrite happens after the log transform).
        out = np.asarray(pSIMPLEX(np.array([-1.0, 2.0]), 0.4, 0.8, log_p=True))
        np.testing.assert_array_equal(out, np.zeros(2))

    def test_lower_tail_false_matches_live_r(self) -> None:
        q = [0.2, 0.6, 0.8]
        expected = RTestBus().eval_family(
            "SIMPLEX", "p",
            dict(q=q, mu=[0.4] * 3, sigma=[0.8] * 3, **{"lower.tail": [False] * 3, "log.p": [False] * 3}),
        )
        actual = np.asarray(pSIMPLEX(np.asarray(q), 0.4, 0.8, lower_tail=False))
        np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)


class TestRandomGeneration:
    def test_r_stays_in_support(self) -> None:
        samples = np.asarray(rSIMPLEX(jax.random.PRNGKey(0), 50, 0.5, 1.0))
        assert np.all(np.isfinite(samples))
        assert np.all((samples > 0) & (samples < 1))
