from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from omnilss import GA, NO
from omnilss.distributions3 import cdf, is_continuous, is_discrete, kurtosis, log_pdf, mean, pdf, quantile, skewness, support, variance


def test_pdf_logpdf_and_cdf_no() -> None:
    fam = NO()
    x = np.array([0.0, 1.0])
    mu = np.array([0.0, 0.0])
    sigma = np.array([1.0, 1.0])

    d = pdf(fam, x, mu, sigma)
    ld = log_pdf(fam, x, mu, sigma)
    p = cdf(fam, x, mu, sigma)

    assert d.shape == (2,)
    assert np.allclose(ld, np.log(d), atol=1e-7)
    assert np.all((p >= 0) & (p <= 1))


def test_quantile_no() -> None:
    fam = NO()
    probs = np.array([0.2, 0.8])
    q = quantile(fam, probs, np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    assert q.shape == (2,)


def test_type_and_support_helpers() -> None:
    assert is_continuous(NO())
    assert not is_discrete(NO())
    assert support(NO()) == "(-∞,∞)"
    assert support(GA()) == "(0,∞)"


def test_moment_helpers_for_no() -> None:
    fam = NO()
    mu = np.array([1.0, 3.0])
    sigma = np.array([2.0, 2.0])
    assert mean(fam, mu, sigma) == 2.0
    assert variance(fam, mu, sigma) == 4.0
    assert skewness(fam, mu, sigma) == 0.0
    assert kurtosis(fam, mu, sigma) == 3.0
