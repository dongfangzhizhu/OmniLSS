"""Batch 11: GIG and GB1 distributions."""

from __future__ import annotations

# Enable float64 precision for numerical accuracy
from dataclasses import dataclass
import jax.numpy as jnp
from jax.scipy.special import gammaln, betaln

from .ad import build_ad_family
from .families import FamilyDefinition
from .links import (
    log_link,
    log_inverse,
    log_derivative,
    identity_link,
    identity_inverse,
    identity_derivative,
)

# ============================================================================
# GIG - Generalized Inverse Gaussian Distribution
# ============================================================================


@dataclass(frozen=True)
class GIGFamily(FamilyDefinition):
    """Generalized Inverse Gaussian distribution family."""

    pass


def _gig_log_pdf(y, mu, sigma, nu):
    """Generalized Inverse Gaussian log-PDF in R's parameterization.

    Delegates to :func:`omnilss.gig.log_density`, the parity-verified
    transcription of ``dGIG`` from the installed gamlss.dist 6.1.1
    (``c = K_{nu+1}(1/sigma^2)/K_nu(1/sigma^2)``; see the gig module
    docstring for the full contract and the parameterization correction
    this replaces -- the previous ``chi=1/sigma^2, psi=1/(mu^2 sigma^2)``
    form was a different distribution, off by 8.08e-02 rel against live R).

    Scalar kernel: ``build_ad_family`` vmaps it, so no broadcasting or
    ``_prepare_jax_args`` here (D2.45 shape contract).
    """
    from .gig import log_density

    return log_density(y, mu, sigma, nu)


def GIG():
    """Generalized Inverse Gaussian distribution.

    A three-parameter distribution that includes inverse Gaussian
    and gamma as special cases.

    Parameters:
    - mu: location (> 0)
    - sigma: scale (> 0)
    - nu: shape (can be negative)

    Link functions:
    - mu: log
    - sigma: log
    - nu: identity

    Notes
    -----
    Delivered against live R (installed gamlss.dist 6.1.1) in the D2.48
    batch: density parameterization corrected to R's ``c`` form, p/q/r
    wired (numerical quadrature CDF + log-space bisection quantile),
    and R's own derivative slots transcribed as overrides -- including
    the expected-Hessian ``d2ldm2`` and the ``-score^2`` conventions for
    ``d2ldd2``/``d2ldv2``, which is what RS parity requires.  See
    ``omnilss/gig.py`` for the full contract.
    """
    from .gig import GIG_HESSIANS, GIG_SCORES, dGIG, pGIG, qGIG, rGIG

    return build_ad_family(
        family_class=GIGFamily,
        name="GIG",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=_gig_log_pdf,
        type_="Continuous",
        links={"mu": "log", "sigma": "log", "nu": "identity"},
        link_functions={
            "mu": log_link,
            "sigma": log_link,
            "nu": identity_link,
        },
        link_inverses={
            "mu": log_inverse,
            "sigma": log_inverse,
            "nu": identity_inverse,
        },
        link_derivatives={
            "mu": log_derivative,
            "sigma": log_derivative,
            "nu": identity_derivative,
        },
        score_overrides=GIG_SCORES,
        hessian_overrides=GIG_HESSIANS,
        d=dGIG,
        p=pGIG,
        q=qGIG,
        r=rGIG,
    )


# ============================================================================
# GB1 - Generalized Beta Type 1 Distribution
# ============================================================================


@dataclass(frozen=True)
class GB1Family(FamilyDefinition):
    """Generalized Beta Type 1 distribution family."""

    pass


def _gb1_log_pdf(y, mu, sigma, nu, tau):
    """Shared differentiable density in R's bounded GB1 parameterization."""
    from .gb1 import log_density

    return log_density(y, mu, sigma, nu, tau)


def GB1():
    """Generalized Beta Type 1 distribution.

    A four-parameter extension of the beta distribution for
    modeling bounded data with flexible shapes.

    Parameters:
    - mu: location (0 < mu < 1)
    - sigma: scale (0 < sigma < 1)
    - nu: shape 1 (> 0)
    - tau: shape 2 (> 0)

    Link functions:
    - mu: logit
    - sigma: logit
    - nu: log
    - tau: log
    """
    from .links import logit_link, logit_inverse, logit_derivative
    from .gb1 import dGB1, pGB1, qGB1, rGB1

    return build_ad_family(
        family_class=GB1Family,
        name="GB1",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_gb1_log_pdf,
        type_="Continuous",
        d=dGB1,
        p=pGB1,
        q=qGB1,
        r=rGB1,
        links={"mu": "logit", "sigma": "logit", "nu": "log", "tau": "log"},
        link_functions={
            "mu": logit_link,
            "sigma": logit_link,
            "nu": log_link,
            "tau": log_link,
        },
        link_inverses={
            "mu": logit_inverse,
            "sigma": logit_inverse,
            "nu": log_inverse,
            "tau": log_inverse,
        },
        link_derivatives={
            "mu": logit_derivative,
            "sigma": logit_derivative,
            "nu": log_derivative,
            "tau": log_derivative,
        },
    )
