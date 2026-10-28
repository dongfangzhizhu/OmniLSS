"""ZAP normalises elementwise; R does not (D2.86).

Same route as ZANBI (D2.85) through a different base: R dZAP takes
fy0 <- dPO(0, mu), and R dPO does not recycle the scalar x against a vector
mu -- measured on live R, length(dPO(0, mu = c(2, 8))) is 1.  So R shares one
f0 across the whole vector while ours evaluates each row own.

These tests pin OUR behaviour so that matching R cannot be mistaken for a fix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from omnilss.distribution_registry import resolve  # noqa: E402


def _density(y, mu, sigma):
    family = resolve("ZAP")
    n = len(np.atleast_1d(y))
    return np.asarray(
        family.d(
            jnp.asarray(np.atleast_1d(y), dtype=jnp.float64),
            jnp.asarray(np.atleast_1d(mu), dtype=jnp.float64),
            jnp.full(n, sigma, dtype=jnp.float64),
            log=True,
        ),
        dtype=float,
    )


def test_two_rows_with_different_mu_do_not_share_one_f0():
    """If these agree, we have adopted R behaviour -- the gate would look green
    for the wrong reason."""
    y = np.array([2.0, 5.0])
    mu = [2.0, 8.0]
    sigma = 0.3
    ours = _density(y, mu, sigma)
    shared_f0 = _density(y, [2.0, 2.0], sigma)[1]
    # Row 2 differs from the "carry row 1 forward" answer by the f0(2) vs
    # f0(8) normalisation, which is exactly the term R loses.
    both_same = abs(ours[1] - shared_f0) < 1e-12
    assert not both_same


def test_the_zero_atom_is_the_sigma_parameter_on_its_logit_scale():
    """ZAP names the zero mass "sigma"; the parameter is a probability."""
    family = resolve("ZAP")
    assert family.parameters == ("mu", "sigma")
    assert family.links["sigma"] == "logit"
    value = _density(0.0, 3.0, 0.25)
    np.testing.assert_allclose(value, [np.log(0.25)], rtol=0, atol=1e-14)