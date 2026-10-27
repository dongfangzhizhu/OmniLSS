"""Bessel callbacks under vmap, and the Sichel recursion bound (D2.76).

`log_bessel_kv` / `log_bessel_iv` reach scipy through `jax.pure_callback`.
With `vmap_method="sequential"` a vmap over n observations ran n host
callbacks per evaluation; the score for a Sichel family evaluates four of
them per observation, so one score call over 48 rows cost a second and a
ZASICHEL fit on 12 rows took 216 s for a single outer cycle.  `expand_dims`
batches the whole vmap into one call because the scipy kernels broadcast.

`tofySICHEL` had the matching problem on the other axis: its `lax.scan` ran
the full 100000-count budget for every observation, while the scan only ever
selects `j == floor(y)`.  It is now bounded by the largest count in the call.

Both changes are supposed to be value-neutral.  These tests pin that: the
batched Bessel matches scipy element for element, and the bounded recursion
answers the same thing in a batch as it does alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import ive, kve

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnilss.bessel import log_bessel_iv, log_bessel_kv  # noqa: E402
from omnilss.dpqr_functions import tofySICHEL  # noqa: E402


def _expected_log_kv(nu, x):
    scaled = kve(np.asarray(nu, float), np.asarray(x, float))
    return np.log(np.maximum(scaled, np.finfo(np.float64).tiny)) - np.asarray(x, float)


def _expected_log_iv(nu, x):
    scaled = ive(np.asarray(nu, float), np.asarray(x, float))
    return np.log(np.maximum(scaled, np.finfo(np.float64).tiny)) + np.asarray(x, float)


class TestBatchedCallbacks:
    """`expand_dims` must not change a single value."""

    @pytest.mark.parametrize("nu_value", [-1.5, -0.5, 0.0, 0.75, 2.0])
    def test_log_kv_under_vmap_matches_scipy(self, nu_value):
        x = np.linspace(0.5, 30.0, 17)
        nu = np.full_like(x, nu_value)
        ours = np.asarray(log_bessel_kv(jnp.asarray(nu), jnp.asarray(x)))
        np.testing.assert_allclose(ours, _expected_log_kv(nu, x), rtol=1e-13)

    @pytest.mark.parametrize("nu_value", [-1.5, -0.5, 0.0, 0.75, 2.0])
    def test_log_iv_under_vmap_matches_scipy(self, nu_value):
        x = np.linspace(0.5, 30.0, 17)
        nu = np.full_like(x, nu_value)
        ours = np.asarray(log_bessel_iv(jnp.asarray(nu), jnp.asarray(x)))
        np.testing.assert_allclose(ours, _expected_log_iv(nu, x), rtol=1e-13)

    def test_a_nu_array_matches_the_per_element_answer(self):
        """The callback gets arrays of different orders; scipy must broadcast."""
        x = np.linspace(1.0, 20.0, 11)
        nu = np.linspace(-1.5, 2.0, 11)
        ours = np.asarray(log_bessel_kv(jnp.asarray(nu), jnp.asarray(x)))
        expected = np.array(
            [_expected_log_kv(nv, xv) for nv, xv in zip(nu, x)], dtype=float
        )
        np.testing.assert_allclose(ours, expected, rtol=1e-13)


class TestTheRecursionIsBounded:
    """Bounding the scan must not change tofySICHEL(y)."""

    def _tofy(self, y):
        y = np.asarray(y, dtype=float)
        n = y.size
        return np.asarray(
            tofySICHEL(
                jnp.asarray(y),
                jnp.full(n, 3.0),
                jnp.full(n, 0.5),
                jnp.full(n, -0.5),
            )
        )

    def test_a_batch_agrees_with_a_single_point(self):
        batch = self._tofy([0.0, 1.0, 2.0, 5.0])
        for index, y in enumerate([0.0, 1.0, 2.0, 5.0]):
            alone = self._tofy([y])[0]
            np.testing.assert_allclose(batch[index], alone, rtol=1e-13)

    def test_a_large_count_still_resolves(self):
        """The bound is max(y), so a wide batch must not clip the last row."""
        values = self._tofy([0.0, 3.0, 40.0])
        assert np.isfinite(values).all()
        alone = self._tofy([40.0])[0]
        np.testing.assert_allclose(values[-1], alone, rtol=1e-13)