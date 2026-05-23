from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from omnilss import BI, EXP, GA, NO, PO
from omnilss.distributions3 import cdf, is_continuous, is_discrete, kurtosis, log_pdf, mean, moment_availability, pdf, quantile, skewness, support, variance


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


def test_moment_helpers_sample_fallback_for_po() -> None:
    fam = PO()
    mu = np.array([3.0] * 200)
    m = mean(fam, mu, fallback="sample", n=4000, seed=7)
    v = variance(fam, mu, fallback="sample", n=4000, seed=7)
    s = skewness(fam, mu, fallback="sample", n=4000, seed=7)
    k = kurtosis(fam, mu, fallback="sample", n=4000, seed=7)
    assert 2.6 < m < 3.4
    assert 2.2 < v < 3.8
    assert s > 0.3
    assert k > 2.0


def test_moment_helpers_analytic_for_ga_exp_po_bi() -> None:
    ga = GA()
    mu = np.array([2.0, 4.0])
    sigma = np.array([0.5, 0.5])
    assert mean(ga, mu, sigma) == 3.0
    assert variance(ga, mu, sigma) == 2.5
    assert skewness(ga, mu, sigma) == 1.0
    assert kurtosis(ga, mu, sigma) == 4.5

    exp = EXP()
    mu_exp = np.array([2.0, 2.0])
    assert mean(exp, mu_exp) == 2.0
    assert variance(exp, mu_exp) == 4.0
    assert skewness(exp, mu_exp) == 2.0
    assert kurtosis(exp, mu_exp) == 9.0

    po = PO()
    mu_po = np.array([4.0, 4.0])
    assert mean(po, mu_po) == 4.0
    assert variance(po, mu_po) == 4.0
    assert np.isclose(skewness(po, mu_po), 0.5)
    assert np.isclose(kurtosis(po, mu_po), 3.25)

    bi = BI()
    mu_bi = np.array([0.25, 0.25])
    assert mean(bi, mu_bi) == 0.25
    assert np.isclose(variance(bi, mu_bi), 0.1875)
    assert np.isclose(skewness(bi, mu_bi), 1.1547005383792517)
    assert np.isclose(kurtosis(bi, mu_bi), 2.333333333333333)


def test_moment_availability_declaration() -> None:
    no_tiers = moment_availability(NO())
    assert no_tiers == {
        "mean": "analytic",
        "variance": "analytic",
        "skewness": "analytic",
        "kurtosis": "analytic",
    }

    ga_tiers = moment_availability(GA())
    assert ga_tiers["mean"] == "analytic"
    assert ga_tiers["variance"] == "analytic"
    assert ga_tiers["skewness"] == "analytic"
    assert ga_tiers["kurtosis"] == "analytic"
