"""build_ad_family() can differentiate a family closure, not just a kernel.

A family that is constructed at run time -- trun(), cens(), or a
caller-defined FamilyDefinition -- has no hand-written scalar log-density and
no entry in any name-keyed table.  build_ad_family(source_family=...) derives
the AD kernel from the family's own d() closure, inherits the structural
metadata (links, type, fixed parameters, dataclass extras) and keeps p/q/r, so
the AD-ified object is the same distribution with exact score functions.

Why hessian_from_score defaults to True here: the raw AD second derivative of a
log-density is the *observed* information and can be positive (truncated WEI3
sits at d2l/dmu2 = +0.099 at the optimum).  IRLS needs the expected negative
information; gamlss uses -(dldx)**2 for every numerically differentiated
family, and the working weights stay positive only that way.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from omnilss.ad import build_ad_family
from omnilss.distribution_registry import resolve
from omnilss.families import FamilyDefinition
from omnilss.gamlss_tr import trun
from omnilss.links import (
    identity_derivative,
    identity_inverse,
    identity_link,
    log_derivative,
    log_inverse,
    log_link,
)

_Y = np.array([2.5, 3.2, 4.1, 5.0, 6.3, 7.7, 9.1])
_MU = np.full(_Y.shape, 5.0)
_SIGMA = np.full(_Y.shape, 2.5)


def _truncated():
    return trun(par=2.0, family="WEI3", type="left")


def _runtime_laplace() -> FamilyDefinition:
    """A family that exists only as a closure: no registry entry, no name table."""

    def d(y, mu, sigma, log=False):
        z = (y - mu) / sigma
        value = -jnp.log(sigma) - jnp.abs(z)
        return value if log else jnp.exp(value)

    def g_dev_inc(y, **params):
        return -2.0 * d(y, log=True, **params)

    return FamilyDefinition(
        name="RUNTIME_LAPLACE",
        parameters=("mu", "sigma"),
        g_dev_inc=g_dev_inc,
        type="Continuous",
        link_functions={"mu": identity_link, "sigma": log_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative},
        d=d,
    )


def _central_diff(function, value, step=1e-6):
    return (function(value + step) - function(value - step)) / (2.0 * step)


def test_a_family_can_be_ad_ified_from_its_own_density() -> None:
    base = resolve("WEI3")
    built = build_ad_family(source_family=base)

    assert built.name == base.name
    assert built.parameters == base.parameters
    assert built.type == base.type
    assert built.links == base.links
    np.testing.assert_allclose(
        np.asarray(built.g_dev_inc(y=_Y, mu=_MU, sigma=_SIGMA)),
        np.asarray(base.g_dev_inc(y=_Y, mu=_MU, sigma=_SIGMA)),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        np.asarray(built.d(_Y, log=True, mu=_MU, sigma=_SIGMA)),
        np.asarray(base.d(_Y, log=True, mu=_MU, sigma=_SIGMA)),
        rtol=1e-12,
        atol=1e-12,
    )


def test_ad_scores_of_an_ad_ified_family_match_its_analytic_scores() -> None:
    base = resolve("WEI3")
    built = build_ad_family(source_family=base)
    for parameter in ("mu", "sigma"):
        np.testing.assert_allclose(
            np.asarray(built.score_functions[parameter](y=_Y, mu=_MU, sigma=_SIGMA)),
            np.asarray(base.score_functions[parameter](y=_Y, mu=_MU, sigma=_SIGMA)),
            rtol=1e-8,
            atol=1e-10,
        )


def test_closure_families_use_the_scoring_hessian_by_default() -> None:
    built = build_ad_family(source_family=resolve("WEI3"))
    for parameter in ("mu", "sigma"):
        kwargs = {"y": _Y, "mu": _MU, "sigma": _SIGMA}
        score = np.asarray(built.score_functions[parameter](**kwargs))
        hessian = np.asarray(built.hessian_functions[parameter](**kwargs))
        np.testing.assert_allclose(hessian, -(score**2), rtol=1e-14, atol=0)


def test_raw_ad_hessian_is_available_on_request() -> None:
    base = resolve("WEI3")
    built = build_ad_family(source_family=base, hessian_from_score=False)
    kwargs = {"y": _Y, "mu": _MU, "sigma": _SIGMA}
    hessian = np.asarray(built.hessian_functions["mu"](**kwargs))
    score = np.asarray(built.score_functions["mu"](**kwargs))
    # The raw second derivative is not the -score^2 convention ...
    assert not np.allclose(hessian, -(score**2), rtol=1e-6)
    # ... and it agrees with a finite difference of the log-density.
    log_pdf = lambda mu: np.asarray(base.d(_Y, log=True, mu=mu, sigma=_SIGMA))
    second = (
        log_pdf(_MU + 1e-4) - 2.0 * log_pdf(_MU) + log_pdf(_MU - 1e-4)
    ) / (1e-4**2)
    np.testing.assert_allclose(hessian, second, rtol=1e-4, atol=1e-4)


def test_a_truncated_family_is_differentiated_through_its_truncation() -> None:
    truncated = _truncated()
    built = build_ad_family(source_family=truncated)

    # The truncation is part of the differentiated expression.
    score = np.asarray(built.score_functions["mu"](y=_Y, mu=_MU, sigma=_SIGMA))
    log_pdf = lambda mu: np.asarray(truncated.d(_Y, log=True, mu=mu, sigma=_SIGMA))
    np.testing.assert_allclose(score, _central_diff(log_pdf, _MU), rtol=1e-6)

    base_score = np.asarray(
        resolve("WEI3").score_functions["mu"](y=_Y, mu=_MU, sigma=_SIGMA)
    )
    assert not np.allclose(score, base_score, rtol=1e-6, atol=1e-8)


def test_the_truncated_cdf_and_density_are_kept() -> None:
    """A truncation-blind copy would silently revert residuals to the base CDF."""

    truncated = _truncated()
    built = build_ad_family(source_family=truncated)

    assert built.p is truncated.p
    assert built.d is truncated.d
    np.testing.assert_allclose(
        np.asarray(built.p(_Y, mu=_MU, sigma=_SIGMA)),
        np.asarray(truncated.p(_Y, mu=_MU, sigma=_SIGMA)),
        rtol=1e-14,
        atol=0,
    )


def test_dataclass_extras_survive_the_ad_ification() -> None:
    truncated = _truncated()
    built = build_ad_family(source_family=truncated)

    # base_name is what the start-value mapping keys on; dropping it sends a
    # truncated family back to the generic target and exp(y) on a log link.
    assert built.base_name == truncated.base_name == "WEI3"
    assert built.truncation_type == truncated.truncation_type == "left"
    assert built.truncation_parameters == truncated.truncation_parameters == (2.0,)
    assert built.support() == truncated.support()


def test_a_runtime_family_without_a_registry_entry_is_supported() -> None:
    from omnilss.family_capabilities import family_capability_names

    runtime = _runtime_laplace()
    assert runtime.name.upper() not in family_capability_names()

    built = build_ad_family(source_family=runtime)
    mu = np.array([3.0, 4.0, 5.5])
    sigma = np.array([2.0, 2.0, 2.0])
    y = np.array([2.0, 5.0, 4.5])

    score_mu = np.asarray(built.score_functions["mu"](y=y, mu=mu, sigma=sigma))
    np.testing.assert_allclose(score_mu, np.sign(y - mu) / sigma, rtol=1e-10)

    score_sigma = np.asarray(
        built.score_functions["sigma"](y=y, mu=mu, sigma=sigma)
    )
    expected = -1.0 / sigma + np.abs(y - mu) / sigma**2
    np.testing.assert_allclose(score_sigma, expected, rtol=1e-10)


def test_explicit_overrides_still_win() -> None:
    base = resolve("WEI3")
    sentinel_score = lambda **kwargs: jnp.zeros_like(jnp.asarray(kwargs["y"]))
    sentinel_hessian = lambda **kwargs: -jnp.ones_like(jnp.asarray(kwargs["y"]))
    built = build_ad_family(
        source_family=base,
        score_overrides={"mu": sentinel_score},
        hessian_overrides={"mu": sentinel_hessian},
    )
    kwargs = {"y": _Y, "mu": _MU, "sigma": _SIGMA}
    np.testing.assert_allclose(
        np.asarray(built.score_functions["mu"](**kwargs)), np.zeros_like(_Y)
    )
    np.testing.assert_allclose(
        np.asarray(built.hessian_functions["mu"](**kwargs)), -np.ones_like(_Y)
    )


def test_a_family_without_a_density_is_rejected() -> None:
    empty = FamilyDefinition(
        name="NO_DENSITY",
        parameters=("mu",),
        g_dev_inc=lambda y, **params: y,
    )
    with pytest.raises(TypeError, match=r"no d\(\) closure"):
        build_ad_family(source_family=empty)


def test_the_kernel_api_is_unchanged() -> None:
    """The original four-argument form still builds a plain AD family."""

    def log_pdf(y, mu, sigma):
        return -jnp.log(sigma) - 0.5 * ((y - mu) / sigma) ** 2

    classic = build_ad_family(
        FamilyDefinition, "PLAIN_NO", ("mu", "sigma"), log_pdf
    )
    score = np.asarray(
        classic.score_functions["mu"](y=_Y, mu=_MU, sigma=_SIGMA)
    )
    np.testing.assert_allclose(score, (_Y - _MU) / _SIGMA**2, rtol=1e-10)
    # Without source_family the hessian stays the raw AD second derivative.
    hessian = np.asarray(
        classic.hessian_functions["mu"](y=_Y, mu=_MU, sigma=_SIGMA)
    )
    np.testing.assert_allclose(hessian, -1.0 / _SIGMA**2, rtol=1e-10)


def test_the_ad_ified_truncated_family_reaches_the_same_optimum() -> None:
    from omnilss.fitting import gamlss

    fits = {}
    for tag, family in (("numeric", _truncated()), ("ad", build_ad_family(source_family=_truncated()))):
        fits[tag] = gamlss("y ~ 1", family=family, data={"y": _Y}, method="RS")

    for parameter in ("mu", "sigma"):
        numeric_value = float(np.ravel(fits["numeric"].fitted_values[parameter])[0])
        ad_value = float(np.ravel(fits["ad"].fitted_values[parameter])[0])
        assert ad_value == pytest.approx(numeric_value, rel=2e-2)
    assert float(fits["ad"].g_dev) == pytest.approx(float(fits["numeric"].g_dev), abs=1e-2)