"""Initial distribution support for staged `gamlss` migration.

R source references:
- file: `gamlss/R/gamlss-5.R`
- default family usage: `family = NO()`
- related family concepts in the `gamlss.dist` companion package
"""

from __future__ import annotations

# Enable float64 precision for numerical accuracy
from dataclasses import dataclass
from functools import lru_cache
import math

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln, betaln
from jax import jit, vmap

from .families import FamilyDefinition
from .links import (
    identity_derivative,
    identity_inverse,
    identity_link,
    log_derivative,
    log_inverse,
    log_link,
    logit_derivative,
    logit_inverse,
    logit_link,
    logshiftto2_derivative,
    logshiftto2_inverse,
    logshiftto2_link,
)
from .dpqr_functions import (
    dNO,
    pNO,
    qNO,
    rNO,
    dPO,
    pPO,
    qPO,
    rPO,
    dBI,
    pBI,
    qBI,
    rBI,
    dGA,
    pGA,
    qGA,
    rGA,
    dLOGNO,
    pLOGNO,
    qLOGNO,
    rLOGNO,
    dEXP,
    pEXP,
    qEXP,
    rEXP,
    dTF,
    pTF,
    qTF,
    rTF,
    dWEI,
    pWEI,
    qWEI,
    rWEI,
    dLO,
    pLO,
    qLO,
    rLO,
    dBE,
    pBE,
    qBE,
    rBE,
    dGEOM,
    pGEOM,
    qGEOM,
    rGEOM,
    dNBI,
    pNBI,
    qNBI,
    rNBI,
    dBCT,
    pBCT,
    qBCT,
    rBCT,
    dBCPE,
    pBCPE,
    qBCPE,
    rBCPE,
    dJSU,
    pJSU,
    qJSU,
    rJSU,
    dBCCG,
    pBCCG,
    qBCCG,
    rBCCG,
    dZIP,
    pZIP,
    qZIP,
    rZIP,
)

# The Box-Cox power-exponential family's standardized kernel.  Its scoring
# path used to omit the standardization constant entirely, so g_dev_inc and
# dpqr_functions.dBCPE described different densities -- importing the same
# helpers the d/p/q/r functions use is what keeps them one density.
from .dpqr_functions import _bcpe_c, _bcpe_ftrunc_log


@dataclass(frozen=True)
class NormalFamily(FamilyDefinition):
    """Minimal Normal (`NO`) family for the staged Python port."""


@dataclass(frozen=True)
class PoissonFamily(FamilyDefinition):
    """Minimal Poisson (`PO`) family for the staged Python port."""


@dataclass(frozen=True)
class BinomialFamily(FamilyDefinition):
    """Minimal Binomial (`BI`) family for the staged Python port."""


@dataclass(frozen=True)
class GammaFamily(FamilyDefinition):
    """Minimal Gamma (`GA`) family for the staged Python port."""


@dataclass(frozen=True)
class ExponentialFamily(FamilyDefinition):
    """Minimal Exponential (`EXP`) family for the staged Python port."""


@dataclass(frozen=True)
class LogNormalFamily(FamilyDefinition):
    """Minimal Log-Normal (`LOGNO`) family for the staged Python port."""


@dataclass(frozen=True)
class NegativeBinomialFamily(FamilyDefinition):
    """Minimal Negative Binomial (`NBI`) family for the staged Python port."""


@dataclass(frozen=True)
class InverseGaussianFamily(FamilyDefinition):
    """Minimal Inverse Gaussian (`IG`) family for the staged Python port."""


@dataclass(frozen=True)
class LogisticFamily(FamilyDefinition):
    """Minimal Logistic (`LO`) family for the staged Python port."""


@dataclass(frozen=True)
class BetaFamily(FamilyDefinition):
    """Minimal Beta (`BE`) family for the staged Python port."""


@dataclass(frozen=True)
class WeibullFamily(FamilyDefinition):
    """Minimal Weibull (`WEI`) family for the staged Python port."""


@dataclass(frozen=True)
class GeometricFamily(FamilyDefinition):
    """Minimal Geometric (`GEOM`) family for the staged Python port."""


@dataclass(frozen=True)
class ZeroInflatedPoissonFamily(FamilyDefinition):
    """Minimal Zero-Inflated Poisson (`ZIP`) family for the staged Python port."""


@dataclass(frozen=True)
class StudentTFamily(FamilyDefinition):
    """Minimal Student-t (`TF`) family for the staged Python port."""


@dataclass(frozen=True)
class TF2Family(FamilyDefinition):
    """Compatibility Student-t family shell for `TF2`."""


@dataclass(frozen=True)
class JohnsonSUFamily(FamilyDefinition):
    """Minimal Johnson SU (`JSU`) family for the staged Python port."""


@dataclass(frozen=True)
class BoxCoxColeGreenFamily(FamilyDefinition):
    """Minimal Box-Cox Cole-Green (`BCCG`) family for the staged Python port."""


@dataclass(frozen=True)
class BoxCoxTFamily(FamilyDefinition):
    """Minimal Box-Cox t (`BCT`) family for the staged Python port."""


@dataclass(frozen=True)
class BoxCoxPowerExponentialFamily(FamilyDefinition):
    """Minimal Box-Cox power exponential (`BCPE`) family for the staged Python port."""


@lru_cache(maxsize=1)
def NO() -> NormalFamily:
    """R reference: `gamlss/R/gamlss-5.R` default family `NO()`."""

    def _sigma_floor(sigma: jnp.ndarray) -> jnp.ndarray:
        # sigma == 0 is outside the family's support; clamping keeps the
        # first/second derivatives finite (fail-closed at the deviance level)
        # instead of producing inf - inf -> NaN inside the IRLS step.
        return jnp.maximum(sigma, jnp.finfo(jnp.float64).eps)

    def g_dev_inc(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = _sigma_floor(jnp.asarray(sigma, dtype=jnp.float64))
        log_density = (
            -0.5 * jnp.log(2.0 * math.pi)
            - jnp.log(sigma)
            - 0.5 * jnp.square((y - mu) / sigma)
        )
        return -2.0 * log_density

    def dldm(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = _sigma_floor(jnp.asarray(sigma, dtype=jnp.float64))
        return (y - mu) / jnp.square(sigma)

    def d2ldm2(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        sigma = _sigma_floor(jnp.asarray(sigma, dtype=jnp.float64))
        return -1.0 / jnp.square(sigma)

    def dldd(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = _sigma_floor(jnp.asarray(sigma, dtype=jnp.float64))
        return -1.0 / sigma + jnp.square(y - mu) / jnp.power(sigma, 3)

    def d2ldd2(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        """Expected Hessian wrt sigma (Fisher information), aligned with R gamlss."""
        sigma = _sigma_floor(jnp.asarray(sigma, dtype=jnp.float64))
        return -2.0 / jnp.square(sigma)

    return NormalFamily(
        name="NO",
        parameters=("mu", "sigma"),
        g_dev_inc=g_dev_inc,
        type="Continuous",
        links={"mu": "identity", "sigma": "log"},
        link_functions={"mu": identity_link, "sigma": log_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative},
        score_functions={"mu": dldm, "sigma": dldd},
        hessian_functions={"mu": d2ldm2, "sigma": d2ldd2},
        d=dNO,
        p=pNO,
        q=qNO,
        r=rNO,
    )


@lru_cache(maxsize=1)
def PO() -> PoissonFamily:
    """Poisson family compatible with the staged GAMLSS protocol."""

    def g_dev_inc(y: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        log_density = (
            y * jnp.log(jnp.maximum(mu, jnp.finfo(jnp.float64).eps))
            - mu
            - gammaln(y + 1.0)
        )
        return -2.0 * log_density

    def dldm(y: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        return y / mu - 1.0

    def d2ldm2(y: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        return -y / jnp.square(mu)

    return PoissonFamily(
        name="PO",
        parameters=("mu",),
        g_dev_inc=g_dev_inc,
        type="Discrete",
        links={"mu": "log"},
        link_functions={"mu": log_link},
        link_inverses={"mu": log_inverse},
        link_derivatives={"mu": log_derivative},
        score_functions={"mu": dldm},
        hessian_functions={"mu": d2ldm2},
        d=dPO,
        p=pPO,
        q=qPO,
        r=rPO,
    )


@lru_cache(maxsize=1)
def BI() -> BinomialFamily:
    """Binomial family compatible with the staged GAMLSS protocol.

    Supports both Bernoulli (bd=1) and general Binomial (bd>1) cases.

    Deviance calculation:
    - Follows R GAMLSS: -2 * log L(model)
    - Uses dbinom(y, size=bd, prob=mu) when bd is provided
    - Defaults to Bernoulli (bd=1) when bd is not provided
    """

    def g_dev_inc(
        y: jnp.ndarray, mu: jnp.ndarray, bd: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Deviance for Binomial distribution.

        Following R GAMLSS implementation:
        G.dev.incr = -2 * dBI(y, bd, mu, log=TRUE)

        R's binomial response is a **count** of successes (``gamlss()`` builds
        it from a two-column ``cbind(success, failure)`` response).  It is not
        a proportion, so the count is used as given -- scaling it by ``bd``
        double-counts the denominator.

        For Bernoulli (bd=1 or bd=None):
        - y should be 0 or 1
        - log L = y*log(mu) + (1-y)*log(1-mu)
        - deviance = -2 * log L
        """
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.clip(mu, eps, 1.0 - eps)

        if bd is None or jnp.all(bd == 1.0):
            # Bernoulli case (bd=1)
            log_density = y * jnp.log(mu) + (1.0 - y) * jnp.log(1.0 - mu)
        else:
            # General Binomial case: y is already a count of successes,
            # matching R's `dbinom(y, size = bd, prob = mu)`.
            bd = jnp.asarray(bd, dtype=jnp.float64)
            y_counts = jnp.round(y)

            from jax.scipy.special import gammaln

            # log(C(bd, y_counts)) = log(bd!) - log(y_counts!) - log((bd-y_counts)!)
            log_comb = (
                gammaln(bd + 1.0)
                - gammaln(y_counts + 1.0)
                - gammaln(bd - y_counts + 1.0)
            )

            # log L = log_comb + y_counts*log(mu) + (bd-y_counts)*log(1-mu)
            log_density = (
                log_comb + y_counts * jnp.log(mu) + (bd - y_counts) * jnp.log(1.0 - mu)
            )

        return -2.0 * log_density

    def dldm(
        y: jnp.ndarray, mu: jnp.ndarray, bd: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Score function for mu parameter.

        R formula: dldm = function(y, mu, bd) (y-bd*mu)/(mu*(1-mu))

        ``y`` is the count of successes, so it enters the formula directly.
        """
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.clip(mu, eps, 1.0 - eps)

        if bd is None or jnp.all(bd == 1.0):
            # Bernoulli case (bd=1)
            return y / mu - (1.0 - y) / (1.0 - mu)
        else:
            # General Binomial case
            bd = jnp.asarray(bd, dtype=jnp.float64)
            return (y - bd * mu) / (mu * (1.0 - mu))

    def d2ldm2(
        y: jnp.ndarray, mu: jnp.ndarray, bd: jnp.ndarray | None = None
    ) -> jnp.ndarray:
        """Hessian function for mu parameter.

        R formula: d2ldm2 = function(mu,bd) -(bd/(mu*(1-mu)))
        """
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.clip(mu, eps, 1.0 - eps)

        if bd is None:
            # Bernoulli case (bd=1)
            return -(y / jnp.square(mu) + (1.0 - y) / jnp.square(1.0 - mu))
        else:
            # General Binomial case
            bd = jnp.asarray(bd, dtype=jnp.float64)
            return -bd / (mu * (1.0 - mu))

    return BinomialFamily(
        name="BI",
        parameters=("mu",),
        g_dev_inc=g_dev_inc,
        type="Discrete",
        links={"mu": "logit"},
        link_functions={"mu": logit_link},
        link_inverses={"mu": logit_inverse},
        link_derivatives={"mu": logit_derivative},
        score_functions={"mu": dldm},
        hessian_functions={"mu": d2ldm2},
        fixed_parameters=("bd",),  # bd is a fixed parameter (binomial denominator)
        d=dBI,
        p=pBI,
        q=qBI,
        r=rBI,
    )


@lru_cache(maxsize=1)
def GA() -> GammaFamily:
    """Gamma family compatible with the staged GAMLSS protocol.

    R source references:
    - companion package family: `GA`

    Staged parameterization:
    - `E(Y) = mu`
    - `Var(Y) = sigma^2 * mu^2`
    """

    def g_dev_inc(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        y = jnp.maximum(y, eps)
        mu = jnp.maximum(mu, eps)
        sigma = jnp.maximum(sigma, eps)
        shape = 1.0 / jnp.square(sigma)
        scale = mu * jnp.square(sigma)
        log_density = (
            (shape - 1.0) * jnp.log(y)
            - y / scale
            - gammaln(shape)
            - shape * jnp.log(scale)
        )
        return -2.0 * log_density

    def dldm(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.maximum(mu, eps)
        sigma = jnp.maximum(sigma, eps)
        return (y - mu) / (jnp.square(mu) * jnp.square(sigma))

    def d2ldm2(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.maximum(mu, eps)
        sigma = jnp.maximum(sigma, eps)
        return -1.0 / (jnp.square(mu) * jnp.square(sigma))

    def dlds(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        """Score function for sigma parameter.

        R formula: dldd = (2/sigma^3)*((y/mu)-log(y)+log(mu)+log(sigma^2)-1+digamma(1/(sigma^2)))
        """
        from jax.scipy.special import digamma

        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        y = jnp.maximum(y, eps)
        mu = jnp.maximum(mu, eps)
        sigma = jnp.maximum(sigma, eps)

        shape = 1.0 / jnp.square(sigma)
        return (2.0 / jnp.power(sigma, 3)) * (
            (y / mu)
            - jnp.log(y)
            + jnp.log(mu)
            + jnp.log(jnp.square(sigma))
            - 1.0
            + digamma(shape)
        )

    def d2lds2(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        """Hessian function for sigma parameter.

        R formula: d2ldd2 = (4/sigma^4)-(4/sigma^6)*trigamma((1/sigma^2))
        """
        from jax.scipy.special import polygamma

        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        sigma = jnp.maximum(sigma, eps)

        shape = 1.0 / jnp.square(sigma)
        # trigamma(x) = polygamma(1, x)
        return (4.0 / jnp.power(sigma, 4)) - (4.0 / jnp.power(sigma, 6)) * polygamma(
            1, shape
        )

    return GammaFamily(
        name="GA",
        parameters=("mu", "sigma"),
        g_dev_inc=g_dev_inc,
        type="Continuous",
        links={"mu": "log", "sigma": "log"},
        link_functions={"mu": log_link, "sigma": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative},
        score_functions={"mu": dldm, "sigma": dlds},
        hessian_functions={"mu": d2ldm2, "sigma": d2lds2},
        d=dGA,
        p=pGA,
        q=qGA,
        r=rGA,
    )


def EXP() -> ExponentialFamily:
    """Exponential family compatible with the staged GAMLSS protocol.

    Staged parameterization:
    - `E(Y) = mu`
    - `Var(Y) = mu^2`
    """

    def g_dev_inc(y: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        y = jnp.maximum(y, eps)
        mu = jnp.maximum(mu, eps)
        log_density = -jnp.log(mu) - y / mu
        return -2.0 * log_density

    def dldm(y: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.maximum(mu, eps)
        return (y - mu) / jnp.square(mu)

    def d2ldm2(y: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
        mu = jnp.asarray(mu, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.maximum(mu, eps)
        return -1.0 / jnp.square(mu)

    return ExponentialFamily(
        name="EXP",
        parameters=("mu",),
        g_dev_inc=g_dev_inc,
        type="Continuous",
        links={"mu": "log"},
        link_functions={"mu": log_link},
        link_inverses={"mu": log_inverse},
        link_derivatives={"mu": log_derivative},
        score_functions={"mu": dldm},
        hessian_functions={"mu": d2ldm2},
        d=dEXP,
        p=pEXP,
        q=qEXP,
        r=rEXP,
    )


@lru_cache(maxsize=1)
def LOGNO() -> LogNormalFamily:
    """Log-Normal family compatible with the staged GAMLSS protocol.

    Staged parameterization:
    - `log(Y) ~ Normal(mu, sigma)`
    - `mu` uses identity link
    - `sigma` uses log link
    """

    def g_dev_inc(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        y = jnp.maximum(y, eps)
        log_y = jnp.log(y)
        log_density = (
            -jnp.log(y)
            - 0.5 * jnp.log(2.0 * math.pi)
            - jnp.log(sigma)
            - 0.5 * jnp.square((log_y - mu) / sigma)
        )
        return -2.0 * log_density

    def dldm(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        log_y = jnp.log(jnp.maximum(y, eps))
        return (log_y - mu) / jnp.square(sigma)

    def d2ldm2(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        return -1.0 / jnp.square(sigma)

    def dlds(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        log_y = jnp.log(jnp.maximum(y, eps))
        return -1.0 / sigma + jnp.square(log_y - mu) / jnp.power(sigma, 3)

    def d2lds2(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        """Expected (Fisher) second derivative w.r.t. sigma -- R parity.

        R's LOGNO declares ``d2ldd2 = function(sigma) -2/sigma^2``.  Note the
        signature takes **only sigma**: gamlss' RS algorithm uses the *expected*
        Hessian, i.e. ``E[d2/dsigma^2]`` with ``E[(log y - mu)^2] = sigma^2``:

            1/sigma^2 - 3 sigma^2 / sigma^4 = -2/sigma^2

        The observed Hessian (``1/sigma^2 - 3(log y - mu)^2/sigma^4``) was used
        here before.  Away from the optimum it can turn positive, which makes
        the RS working weights negative and drives sigma to ~1e-18 while the
        relative-deviance test still reports ``converged=True``
        (observed: g_dev 8.9e12 vs R's 173.05).
        """
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        return -2.0 / jnp.square(sigma)

    return LogNormalFamily(
        name="LOGNO",
        parameters=("mu", "sigma"),
        g_dev_inc=g_dev_inc,
        type="Continuous",
        links={"mu": "identity", "sigma": "log"},
        link_functions={"mu": identity_link, "sigma": log_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative},
        score_functions={"mu": dldm, "sigma": dlds},
        hessian_functions={"mu": d2ldm2, "sigma": d2lds2},
        d=dLOGNO,
        p=pLOGNO,
        q=qLOGNO,
        r=rLOGNO,
    )


@lru_cache(maxsize=1)
def NBI() -> NegativeBinomialFamily:
    """Negative Binomial family compatible with the staged GAMLSS protocol.

    Staged parameterization:
    - `E(Y) = mu`
    - `Var(Y) = mu + sigma * mu^2`
    """

    def _log_density(
        y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray
    ) -> jnp.ndarray:
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.maximum(mu, eps)
        sigma = jnp.maximum(sigma, eps)
        size = 1.0 / sigma
        return (
            gammaln(y + size)
            - gammaln(size)
            - gammaln(y + 1.0)
            + size * jnp.log(size / (size + mu))
            + y * jnp.log(mu / (size + mu))
        )

    def g_dev_inc(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        return -2.0 * _log_density(y, mu, sigma)

    def dldm(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.maximum(mu, eps)
        sigma = jnp.maximum(sigma, eps)
        size = 1.0 / sigma
        return y / mu - (y + size) / (size + mu)

    def d2ldm2(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.maximum(mu, eps)
        sigma = jnp.maximum(sigma, eps)
        size = 1.0 / sigma
        return -y / jnp.square(mu) + (y + size) / jnp.square(size + mu)

    def dlds(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        """Score function for sigma parameter.

        R formula: dldd = -((1/sigma)^2)* (digamma(y+(1/sigma))-digamma(1/sigma)
                          -log(1+mu*sigma)-(y-mu)*sigma/(1+mu*sigma))
        """
        from jax.scipy.special import digamma

        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.maximum(mu, eps)
        sigma = jnp.maximum(sigma, eps)

        size = 1.0 / sigma
        return -(size**2) * (
            digamma(y + size)
            - digamma(size)
            - jnp.log(1.0 + mu * sigma)
            - (y - mu) * sigma / (1.0 + mu * sigma)
        )

    def d2lds2(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        """Hessian function for sigma parameter.

        R formula: d2ldd2 = -dldd^2 (using observed information)
        with a floor of -1e-15 for numerical stability
        """
        dldd = dlds(y, mu, sigma)
        d2ldd2 = -jnp.square(dldd)
        # Apply floor for numerical stability (from R code)
        d2ldd2 = jnp.where(d2ldd2 < -1e-15, d2ldd2, -1e-15)
        return d2ldd2

    return NegativeBinomialFamily(
        name="NBI",
        parameters=("mu", "sigma"),
        g_dev_inc=g_dev_inc,
        type="Discrete",
        links={"mu": "log", "sigma": "log"},
        link_functions={"mu": log_link, "sigma": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative},
        score_functions={"mu": dldm, "sigma": dlds},
        hessian_functions={"mu": d2ldm2, "sigma": d2lds2},
        d=dNBI,
        p=pNBI,
        q=qNBI,
        r=rNBI,
    )


def _ig_log_pdf(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Log PDF for Inverse Gaussian distribution.

    Parameterization:
    - E(Y) = mu
    - Var(Y) = sigma^2 * mu^3
    - lambda = 1 / sigma^2
    """
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)
    lam = 1.0 / jnp.square(sigma)
    return 0.5 * (
        jnp.log(lam) - jnp.log(2.0 * math.pi) - 3.0 * jnp.log(y)
    ) - lam * jnp.square(y - mu) / (2.0 * jnp.square(mu) * y)


def _build_inverse_gaussian(name: str, family_class, log_pdf_func=_ig_log_pdf):
    """Shared builder for IG and IGA (they are the same distribution).

    D2.77: IGA names the same law as IG -- with lambda = 1/sigma^2 the two
    log-densities agree to 4.4e-16 -- so it must also get the expected-
    information hessians.  Built with plain AD hessians, an IGA fit diverged to
    deviance 8.07e+21 where R answers 153.5; with these it matches.  Keeping
    one builder is what makes "IGA is IG" true in behaviour and not only on
    paper.
    """
    from .ad import build_ad_family

    family = build_ad_family(
        family_class=family_class,
        name=name,
        parameters=("mu", "sigma"),
        log_pdf_func=log_pdf_func,
        type_="Continuous",
        links={"mu": "log", "sigma": "log"},
        link_functions={"mu": log_link, "sigma": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative},
    )

    # Expected information (Fisher information): these do not depend on y,
    # which is what makes the IRLS step stable for this family.
    def d2ldm2_expected(
        y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, **kwargs
    ) -> jnp.ndarray:
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.maximum(mu, eps)
        sigma = jnp.maximum(sigma, eps)
        return -1.0 / (jnp.square(sigma) * jnp.power(mu, 3))

    def d2lds2_expected(
        y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, **kwargs
    ) -> jnp.ndarray:
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        sigma = jnp.maximum(sigma, eps)
        return -2.0 / jnp.square(sigma)

    from .dpqr_functions import dIG, pIG, qIG, rIG

    return family_class(
        name=family.name,
        parameters=family.parameters,
        g_dev_inc=family.g_dev_inc,
        type=family.type,
        links=family.links,
        link_functions=family.link_functions,
        link_inverses=family.link_inverses,
        link_derivatives=family.link_derivatives,
        score_functions=family.score_functions,
        hessian_functions={"mu": d2ldm2_expected, "sigma": d2lds2_expected},
        d=dIG,
        p=pIG,
        q=qIG,
        r=rIG,
    )


def IG() -> InverseGaussianFamily:
    """Inverse Gaussian family compatible with the staged GAMLSS protocol.

    Staged parameterization:
    - E(Y) = mu
    - Var(Y) = sigma^2 * mu^3

    Uses JAX auto-differentiation for score functions (first derivatives)
    but uses expected information (Fisher information) for hessians to ensure
    numerical stability, following the R GAMLSS implementation.

    R source reference:
    - file: gamlss.dist/R/IG.R
    - function: IG()
    """
    return _build_inverse_gaussian("IG", InverseGaussianFamily)


def _lo_log_pdf(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Log PDF for Logistic distribution.

    Parameterization:
    - E(Y) = mu
    - Var(Y) = sigma^2 * pi^2 / 3
    """
    eps = jnp.finfo(jnp.float64).eps
    sigma = jnp.maximum(sigma, eps)
    z = (y - mu) / sigma
    return -jnp.log(sigma) - z - 2.0 * jnp.log1p(jnp.exp(-z))


def LO() -> LogisticFamily:
    """Logistic family compatible with the staged GAMLSS protocol.

    Uses JAX auto-differentiation for score functions (first derivatives)
    but uses expected information (Fisher information) for hessians to ensure
    numerical stability, following the R GAMLSS implementation.

    R source reference:
    - file: `gamlss.dist/R/LO.R`
    - function: `LO()`
    """
    from .ad import build_ad_family

    # Build family with AD for scores
    family = build_ad_family(
        family_class=LogisticFamily,
        name="LO",
        parameters=("mu", "sigma"),
        log_pdf_func=_lo_log_pdf,
        type_="Continuous",
        links={"mu": "identity", "sigma": "log"},
        link_functions={"mu": identity_link, "sigma": log_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative},
    )

    # Override hessians with expected information (Fisher information)
    # For logistic distribution:
    # - Var(logistic(0,1)) = pi^2/3
    # - E[logistic(z) * (1-logistic(z))] = 1/3
    def d2ldm2_expected(
        y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, **kwargs
    ) -> jnp.ndarray:
        """Expected second derivative wrt mu (Fisher information)."""
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        sigma = jnp.maximum(sigma, eps)
        # E[d2ldm2] = -2 * E[logistic * (1-logistic)] / sigma^2
        # E[logistic * (1-logistic)] = 1/3 for standard logistic
        return -2.0 / (3.0 * jnp.square(sigma))

    def d2lds2_expected(
        y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, **kwargs
    ) -> jnp.ndarray:
        """Expected second derivative wrt sigma (Fisher information)."""
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        sigma = jnp.maximum(sigma, eps)
        # E[d2lds2] = -2 / sigma^2
        # This comes from E[z^2 * logistic * (1-logistic)] = 1/3 (variance of standard logistic)
        return -2.0 / jnp.square(sigma)

    # Replace hessian functions with expected versions
    hessian_functions = {
        "mu": d2ldm2_expected,
        "sigma": d2lds2_expected,
    }

    # Create new family with updated hessians
    return LogisticFamily(
        name=family.name,
        parameters=family.parameters,
        g_dev_inc=family.g_dev_inc,
        type=family.type,
        links=family.links,
        link_functions=family.link_functions,
        link_inverses=family.link_inverses,
        link_derivatives=family.link_derivatives,
        score_functions=family.score_functions,  # Keep AD scores
        hessian_functions=hessian_functions,  # Use expected hessians
        d=dLO,
        p=pLO,
        q=qLO,
        r=rLO,
    )


@jit
def _be_log_pdf_single(y_i: float, mu_i: float, sigma_i: float) -> float:
    """JIT-compiled log PDF for single Beta observation.

    This function is vectorized for better performance.
    """
    eps = jnp.finfo(jnp.float64).eps
    y_i = jnp.clip(y_i, eps, 1.0 - eps)
    mu_i = jnp.clip(mu_i, eps, 1.0 - eps)
    sigma_i = jnp.clip(sigma_i, eps, 1.0 - eps)

    # R parameterization
    a = mu_i * (1.0 - jnp.square(sigma_i)) / jnp.square(sigma_i)
    b = (1.0 - mu_i) * (1.0 - jnp.square(sigma_i)) / jnp.square(sigma_i)

    a = jnp.maximum(a, eps)
    b = jnp.maximum(b, eps)

    return (a - 1.0) * jnp.log(y_i) + (b - 1.0) * jnp.log(1.0 - y_i) - betaln(a, b)


# Vectorize for all observations. Defining the jitted/vmapped callable is
# cheap; compilation is left to the first real call so importing omnilss does
# not pay a BE-specific XLA compile (and does not require x64 at import time).
_be_log_pdf_vectorized = jit(vmap(_be_log_pdf_single, in_axes=(0, 0, 0)))


def _be_log_pdf(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Log PDF for Beta distribution.

    Parameterization (matching R gamlss.dist):
    - E(Y) = mu, where 0 < mu < 1
    - a = mu * (1 - sigma^2) / sigma^2
    - b = (1 - mu) * (1 - sigma^2) / sigma^2
    - 0 < sigma < 1 is a dispersion parameter

    Note: This is different from the standard Beta(alpha, beta) parameterization!

    R source reference:
    - file: `gamlss.dist/R/BE.R`
    - function: `dBE()`

    Performance: Uses JIT-compiled vectorized computation for 5-10x speedup.
    """
    # Ensure inputs are arrays (handle scalar case)
    y = jnp.atleast_1d(y)
    mu = jnp.atleast_1d(mu)
    sigma = jnp.atleast_1d(sigma)

    # Use vectorized version for better performance
    result = _be_log_pdf_vectorized(y, mu, sigma)

    # Return scalar if input was scalar
    return result.squeeze() if result.shape == (1,) else result


@lru_cache(maxsize=1)
def BE() -> BetaFamily:
    """Beta family with expected Hessian (Fisher information).

    Uses the correct R gamlss.dist parameterization and Fisher information
    formulas for both mu and sigma parameters.

    Staged parameterization (matching R):
    - `0 < Y < 1`
    - `E(Y) = mu`
    - `a = mu * (1 - sigma^2) / sigma^2`
    - `b = (1 - mu) * (1 - sigma^2) / sigma^2`
    - `0 < sigma < 1` is a dispersion parameter

    Fisher information:
    - E[∂²log L/∂mu²] = -((1 - sigma^2)^2 / sigma^4) * (trigamma(a) + trigamma(b))
    - E[∂²log L/∂sigma²] = -(4 / sigma^6) * (mu^2 * trigamma(a) + (1-mu)^2 * trigamma(b) - trigamma(a+b))

    R source reference:
    - file: `gamlss.dist/R/BE.R`
    - function: `BE()`
    - d2ldm2: Expected Hessian for mu
    - d2ldd2: Expected Hessian for sigma
    """
    from .ad import build_ad_family
    from jax.scipy.special import polygamma

    # Build family with AD for scores
    family = build_ad_family(
        family_class=BetaFamily,
        name="BE",
        parameters=("mu", "sigma"),
        log_pdf_func=_be_log_pdf,
        type_="Continuous",
        links={"mu": "logit", "sigma": "logit"},  # sigma uses logit to keep < 1
        link_functions={"mu": logit_link, "sigma": logit_link},
        link_inverses={"mu": logit_inverse, "sigma": logit_inverse},
        link_derivatives={"mu": logit_derivative, "sigma": logit_derivative},
    )

    # Define expected Hessian functions (Fisher information) matching R
    def d2ldm2_expected(
        y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, **kwargs
    ) -> jnp.ndarray:
        """Expected second derivative wrt mu (Fisher information).

        R formula: -(((1 - sigma^2)^2) / sigma^4) * (trigamma(a) + trigamma(b))
        """
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.clip(mu, eps, 1.0 - eps)
        sigma = jnp.clip(sigma, eps, 1.0 - eps)

        # R parameterization
        a = mu * (1.0 - jnp.square(sigma)) / jnp.square(sigma)
        b = (1.0 - mu) * (1.0 - jnp.square(sigma)) / jnp.square(sigma)
        a = jnp.maximum(a, eps)
        b = jnp.maximum(b, eps)

        # trigamma(x) = polygamma(1, x)
        trigamma_a = polygamma(1, a)
        trigamma_b = polygamma(1, b)

        # R formula
        d2ldm2 = -(jnp.square(1.0 - jnp.square(sigma)) / jnp.power(sigma, 4)) * (
            trigamma_a + trigamma_b
        )

        # Ensure negative
        d2ldm2 = jnp.where(d2ldm2 < -1e-15, d2ldm2, -1e-15)

        return d2ldm2

    def d2lds2_expected(
        y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, **kwargs
    ) -> jnp.ndarray:
        """Expected second derivative wrt sigma (Fisher information).

        R formula: -(4 / sigma^6) * (mu^2 * trigamma(a) + (1-mu)^2 * trigamma(b) - trigamma(a+b))
        """
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.clip(mu, eps, 1.0 - eps)
        sigma = jnp.clip(sigma, eps, 1.0 - eps)

        # R parameterization
        a = mu * (1.0 - jnp.square(sigma)) / jnp.square(sigma)
        b = (1.0 - mu) * (1.0 - jnp.square(sigma)) / jnp.square(sigma)
        a = jnp.maximum(a, eps)
        b = jnp.maximum(b, eps)

        # trigamma(x) = polygamma(1, x)
        trigamma_a = polygamma(1, a)
        trigamma_b = polygamma(1, b)
        trigamma_ab = polygamma(1, a + b)

        # R formula
        d2lds2 = -(4.0 / jnp.power(sigma, 6)) * (
            jnp.square(mu) * trigamma_a
            + jnp.square(1.0 - mu) * trigamma_b
            - trigamma_ab
        )

        # Ensure negative
        d2lds2 = jnp.where(d2lds2 < -1e-15, d2lds2, -1e-15)

        return d2lds2

    # Replace hessian functions with expected versions (both mu and sigma)
    hessian_functions = {
        "mu": d2ldm2_expected,
        "sigma": d2lds2_expected,
    }

    # Create new family with updated hessians
    return BetaFamily(
        name=family.name,
        parameters=family.parameters,
        g_dev_inc=family.g_dev_inc,
        type=family.type,
        links=family.links,
        link_functions=family.link_functions,
        link_inverses=family.link_inverses,
        link_derivatives=family.link_derivatives,
        score_functions=family.score_functions,  # Keep AD scores
        hessian_functions=hessian_functions,  # Use expected hessians for both
        d=dBE,
        p=pBE,
        q=qBE,
        r=rBE,
    )


def _wei_log_pdf(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
    """Log PDF for Weibull distribution.

    **IMPORTANT**: R's parameterization is NOT E[Y] = mu!

    R's actual parameterization (from gamlss.dist/R/WEI.R):
    - Uses dweibull(x, scale=mu, shape=sigma) directly
    - mu = λ (Weibull scale parameter)
    - sigma = k (Weibull shape parameter)
    - E[Y] = mu * Γ(1 + 1/sigma) ≠ mu

    Standard Weibull PDF:
    f(y|k,λ) = (k/λ) * (y/λ)^(k-1) * exp(-(y/λ)^k)

    R source reference:
    - file: `gamlss.dist/R/WEI.R`
    - function: `dWEI()` calls `dweibull(x, scale=mu, shape=sigma)`
    """
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, eps)
    mu = jnp.maximum(mu, eps)  # scale parameter
    sigma = jnp.maximum(sigma, eps)  # shape parameter

    # R uses: dweibull(x, scale=mu, shape=sigma)
    # Standard Weibull: f(y|k,λ) = (k/λ) * (y/λ)^(k-1) * exp(-(y/λ)^k)
    shape = sigma  # k = sigma (NOT 1/sigma!)
    scale = mu  # λ = mu (NOT mu/Γ(1+sigma)!)

    return (
        jnp.log(shape)
        - jnp.log(scale)
        + (shape - 1.0) * (jnp.log(y) - jnp.log(scale))
        - jnp.power(y / scale, shape)
    )


def WEI() -> WeibullFamily:
    """Weibull family compatible with the staged GAMLSS protocol.

    **IMPORTANT**: R's parameterization is NOT E[Y] = mu!

    R's actual parameterization (from gamlss.dist/R/WEI.R):
    - mu = λ (Weibull scale parameter)
    - sigma = k (Weibull shape parameter)
    - E[Y] = mu * Γ(1 + 1/sigma) ≠ mu
    - Uses dweibull(x, scale=mu, shape=sigma) directly

    Hessian method:
    - R uses expected Hessian (Fisher information)
    - d2ldm2 = -sigma^2 / mu^2
    - d2ldd2 = -1.82368 / sigma^2 (constant from trigamma function)
    - Cross Hessian d2ldmdd is ignored (set to 0 in GAMLSS)

    R source reference:
    - file: `gamlss.dist/R/WEI.R`
    - function: `WEI()`
    """
    from .ad import build_ad_family

    # Build base family with AD for score functions
    base_family = build_ad_family(
        family_class=WeibullFamily,
        name="WEI",
        parameters=("mu", "sigma"),
        log_pdf_func=_wei_log_pdf,
        type_="Continuous",
        links={"mu": "log", "sigma": "log"},
        link_functions={"mu": log_link, "sigma": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative},
    )

    # Override with expected Hessian (Fisher information) as R does
    eps = jnp.finfo(jnp.float64).eps

    def d2ldm2_expected(y, mu, sigma, **kwargs):
        """Expected Hessian for mu.

        R formula: -sigma^2 / mu^2
        Does not depend on y (Fisher information).
        """
        mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), eps)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        d2ldm2 = -jnp.square(sigma) / jnp.square(mu)
        # Ensure negative for numerical stability
        d2ldm2 = jnp.where(d2ldm2 < -eps, d2ldm2, -eps)
        return d2ldm2

    def d2lds2_expected(y, mu, sigma, **kwargs):
        """Expected Hessian for sigma.

        R formula: -1.82368 / sigma^2
        Constant 1.82368 ≈ trigamma(1) = π²/6
        Does not depend on y or mu (Fisher information).
        """
        sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps)
        # trigamma(1) = π²/6 ≈ 1.644934, but R uses 1.82368
        # This might be an empirical constant or approximation
        d2lds2 = -1.82368 / jnp.square(sigma)
        # Ensure negative for numerical stability
        d2lds2 = jnp.where(d2lds2 < -eps, d2lds2, -eps)
        return d2lds2

    # Create new family with expected Hessian
    return WeibullFamily(
        name="WEI",
        parameters=("mu", "sigma"),
        g_dev_inc=base_family.g_dev_inc,
        type="Continuous",
        links={"mu": "log", "sigma": "log"},
        link_functions={"mu": log_link, "sigma": log_link},
        link_inverses={"mu": log_inverse, "sigma": log_inverse},
        link_derivatives={"mu": log_derivative, "sigma": log_derivative},
        score_functions={
            "mu": base_family.score_functions["mu"],
            "sigma": base_family.score_functions["sigma"],
        },
        hessian_functions={
            "mu": d2ldm2_expected,
            "sigma": d2lds2_expected,
        },
        d=dWEI,
        p=pWEI,
        q=qWEI,
        r=rWEI,
    )


def GEOM() -> GeometricFamily:
    """Geometric family compatible with the staged GAMLSS protocol.

    Staged parameterization:
    - `Y in {0, 1, 2, ...}`
    - `E(Y) = mu`
    """

    def g_dev_inc(y: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.maximum(mu, eps)
        log_density = y * jnp.log(mu) - (y + 1.0) * jnp.log1p(mu)
        return -2.0 * log_density

    def dldm(y: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.maximum(mu, eps)
        return y / mu - (y + 1.0) / (1.0 + mu)

    def d2ldm2(y: jnp.ndarray, mu: jnp.ndarray) -> jnp.ndarray:
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.maximum(mu, eps)
        return -y / jnp.square(mu) + (y + 1.0) / jnp.square(1.0 + mu)

    return GeometricFamily(
        name="GEOM",
        parameters=("mu",),
        g_dev_inc=g_dev_inc,
        type="Discrete",
        links={"mu": "log"},
        link_functions={"mu": log_link},
        link_inverses={"mu": log_inverse},
        link_derivatives={"mu": log_derivative},
        score_functions={"mu": dldm},
        hessian_functions={"mu": d2ldm2},
        d=dGEOM,
        p=pGEOM,
        q=qGEOM,
        r=rGEOM,
    )


@lru_cache(maxsize=1)
def ZIP() -> ZeroInflatedPoissonFamily:
    """Zero-Inflated Poisson family compatible with the staged GAMLSS protocol.

    Staged parameterization:
    - `Y in {0, 1, 2, ...}`
    - `mu > 0`: Poisson mean parameter
    - `0 < sigma < 1`: probability of extra zeros
    - `E[Y] = (1 - sigma) * mu`

    Hessian method:
    - R uses -score^2 approximation (same as NBII)
    - d2ldm2 = -dldm * dldm
    - d2ldd2 = -dldd * dldd

    R source reference:
    - file: `gamlss.dist/R/ZIP.R`
    - function: `ZIP()`
    """

    def _zero_prob(mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), eps)
        sigma = jnp.clip(jnp.asarray(sigma, dtype=jnp.float64), eps, 1.0 - eps)
        return sigma + (1.0 - sigma) * jnp.exp(-mu)

    def g_dev_inc(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.maximum(mu, eps)
        sigma = jnp.clip(sigma, eps, 1.0 - eps)
        zero_prob = _zero_prob(mu, sigma)
        positive_log_density = (
            jnp.log1p(-sigma) + y * jnp.log(mu) - mu - gammaln(y + 1.0)
        )
        log_density = jnp.where(y == 0.0, jnp.log(zero_prob), positive_log_density)
        return -2.0 * log_density

    def dldm(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        """Score function for mu.

        R formula:
        dldm0 <- -(1 - sigma) * (((1 - sigma) + sigma * exp(mu))^(-1))
        dldm <- ifelse(y == 0, dldm0, (y/mu) - 1)
        """
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.maximum(mu, eps)
        sigma = jnp.clip(sigma, eps, 1.0 - eps)

        # For y = 0
        zero_prob = _zero_prob(mu, sigma)
        zero_score = -((1.0 - sigma) * jnp.exp(-mu)) / zero_prob

        # For y > 0
        positive_score = y / mu - 1.0

        return jnp.where(y == 0.0, zero_score, positive_score)

    def dlds(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        """Score function for sigma.

        R formula:
        dldd0 <- (1 - exp(-mu)) * ((sigma + (1 - sigma) * exp(-mu))^(-1))
        dldd <- ifelse(y == 0, dldd0, -1/(1 - sigma))
        """
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        eps = jnp.finfo(jnp.float64).eps
        mu = jnp.maximum(mu, eps)
        sigma = jnp.clip(sigma, eps, 1.0 - eps)

        # For y = 0
        zero_prob = _zero_prob(mu, sigma)
        zero_score = (1.0 - jnp.exp(-mu)) / zero_prob

        # For y > 0
        positive_score = -1.0 / (1.0 - sigma)

        return jnp.where(y == 0.0, zero_score, positive_score)

    def d2ldm2(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        """Hessian for mu using -score^2 approximation.

        R formula:
        d2ldm2 <- -dldm * dldm
        d2ldm2 <- ifelse(d2ldm2 < -1e-15, d2ldm2, -1e-15)
        """
        eps = jnp.finfo(jnp.float64).eps
        score = dldm(y, mu, sigma)
        d2ldm2 = -jnp.square(score)
        # Ensure negative for numerical stability
        d2ldm2 = jnp.where(d2ldm2 < -eps, d2ldm2, -eps)
        return d2ldm2

    def d2lds2(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray) -> jnp.ndarray:
        """Hessian for sigma using -score^2 approximation.

        R formula:
        d2ldd2 <- -dldd * dldd
        d2ldd2 <- ifelse(d2ldd2 < -1e-15, d2ldd2, -1e-15)
        """
        eps = jnp.finfo(jnp.float64).eps
        score = dlds(y, mu, sigma)
        d2lds2 = -jnp.square(score)
        # Ensure negative for numerical stability
        d2lds2 = jnp.where(d2lds2 < -eps, d2lds2, -eps)
        return d2lds2

    return ZeroInflatedPoissonFamily(
        name="ZIP",
        parameters=("mu", "sigma"),
        g_dev_inc=g_dev_inc,
        type="Discrete",
        links={"mu": "log", "sigma": "logit"},
        link_functions={"mu": log_link, "sigma": logit_link},
        link_inverses={"mu": log_inverse, "sigma": logit_inverse},
        link_derivatives={"mu": log_derivative, "sigma": logit_derivative},
        score_functions={"mu": dldm, "sigma": dlds},
        hessian_functions={"mu": d2ldm2, "sigma": d2lds2},
        d=dZIP,
        p=pZIP,
        q=qZIP,
        r=rZIP,
    )


def TF() -> StudentTFamily:
    """Student-t family: mu (location), sigma (scale), nu (df > 0).

    R source: gamlss.dist, TF family
    Link: mu=identity, sigma=log, nu=log (nu > 0, df = exp(eta_nu))

    Note: R's TF uses nu link = log(nu), not log(nu-2).
    This means nu can be any positive value.
    """
    import jax

    eps = jnp.finfo(jnp.float64).eps

    # nu link: log(nu), inverse: exp(eta)
    # Use standard log link on nu
    def nu_link(x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x, dtype=jnp.float64)
        return jnp.log(jnp.maximum(x, eps))

    def nu_inverse(eta: jnp.ndarray) -> jnp.ndarray:
        eta = jnp.asarray(eta, dtype=jnp.float64)
        return jnp.exp(jnp.clip(eta, -10.0, 10.0))

    def nu_derivative(eta: jnp.ndarray) -> jnp.ndarray:
        eta = jnp.asarray(eta, dtype=jnp.float64)
        return jnp.exp(jnp.clip(eta, -10.0, 10.0))

    def _log_density(
        y: jnp.ndarray,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
        nu: jnp.ndarray,
    ) -> jnp.ndarray:
        """Scalar log density for autodiff.

        Mirrors ``dpqr_functions.dTF`` including its ``nu > 1e6`` normal-limit
        switch.  The two must agree: the RS path differentiates *this* while the
        deviance increment evaluates ``dTF``, so a branch present in only one
        of them reintroduces exactly the density/derivative split D2.20 exists
        to close -- scores that belong to a different function than the one
        being minimized.
        """
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps)
        nu = jnp.maximum(jnp.asarray(nu, dtype=jnp.float64), eps)  # nu > 0
        z = (y - mu) / sigma
        log_t = (
            gammaln((nu + 1.0) / 2.0)
            - gammaln(nu / 2.0)
            - 0.5 * jnp.log(nu * math.pi)
            - jnp.log(sigma)
            - ((nu + 1.0) / 2.0) * jnp.log1p(jnp.square(z) / nu)
        )
        log_no = -0.5 * jnp.square(z) - 0.5 * jnp.log(2.0 * math.pi) - jnp.log(sigma)
        return jnp.where(nu > 1e6, log_no, log_t)

    def g_dev_inc(
        y: jnp.ndarray,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
        nu: jnp.ndarray,
    ) -> jnp.ndarray:
        return -2.0 * _log_density(y, mu, sigma, nu)

    # Use JAX autodiff for exact score and hessian
    def _log_density_scalar(mu_s, sigma_s, nu_s, y_s):
        """Scalar log density for autodiff."""
        sigma_s = jnp.maximum(sigma_s, eps)
        nu_s = jnp.maximum(nu_s, eps)  # nu > 0
        z = (y_s - mu_s) / sigma_s
        return (
            gammaln((nu_s + 1.0) / 2.0)
            - gammaln(nu_s / 2.0)
            - 0.5 * jnp.log(nu_s * math.pi)
            - jnp.log(sigma_s)
            - ((nu_s + 1.0) / 2.0) * jnp.log1p(jnp.square(z) / nu_s)
        )

    # Vectorised score via vmap of grad
    _grad_mu = jax.vmap(jax.grad(_log_density_scalar, argnums=0))
    _grad_sigma = jax.vmap(jax.grad(_log_density_scalar, argnums=1))
    _grad_nu = jax.vmap(jax.grad(_log_density_scalar, argnums=2))
    _hess_mu = jax.vmap(jax.grad(jax.grad(_log_density_scalar, argnums=0), argnums=0))
    _hess_sigma = jax.vmap(
        jax.grad(jax.grad(_log_density_scalar, argnums=1), argnums=1)
    )
    _hess_nu = jax.vmap(jax.grad(jax.grad(_log_density_scalar, argnums=2), argnums=2))

    def dldm(y, mu, sigma, nu):
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps)
        nu = jnp.maximum(jnp.asarray(nu, dtype=jnp.float64), eps)
        return _grad_mu(mu, sigma, nu, y)

    def d2ldm2(y, mu, sigma, nu):
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps)
        nu = jnp.maximum(jnp.asarray(nu, dtype=jnp.float64), eps)
        return _hess_mu(mu, sigma, nu, y)

    def dldd(y, mu, sigma, nu):
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps)
        nu = jnp.maximum(jnp.asarray(nu, dtype=jnp.float64), eps)
        return _grad_sigma(mu, sigma, nu, y)

    def d2ldd2(y, mu, sigma, nu):
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps)
        nu = jnp.maximum(jnp.asarray(nu, dtype=jnp.float64), eps)
        return _hess_sigma(mu, sigma, nu, y)

    def dldn(y, mu, sigma, nu):
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps)
        nu = jnp.maximum(jnp.asarray(nu, dtype=jnp.float64), eps)
        return _grad_nu(mu, sigma, nu, y)

    def d2ldn2(y, mu, sigma, nu):
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps)
        nu = jnp.maximum(jnp.asarray(nu, dtype=jnp.float64), eps)
        return _hess_nu(mu, sigma, nu, y)

    return StudentTFamily(
        name="TF",
        parameters=("mu", "sigma", "nu"),
        g_dev_inc=g_dev_inc,
        type="Continuous",
        links={"mu": "identity", "sigma": "log", "nu": "log"},
        link_functions={"mu": identity_link, "sigma": log_link, "nu": nu_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse, "nu": nu_inverse},
        link_derivatives={
            "mu": identity_derivative,
            "sigma": log_derivative,
            "nu": nu_derivative,
        },
        score_functions={"mu": dldm, "sigma": dldd, "nu": dldn},
        hessian_functions={"mu": d2ldm2, "sigma": d2ldd2, "nu": d2ldn2},
        d=dTF,
        p=pTF,
        q=qTF,
        r=rTF,
    )


def TF2() -> TF2Family:
    """Alternative Student-t constructor compatible with R `gamlss.dist::TF2`.

    R reference: `gamlss.dist/R/TF2.R::TF2`.

    TF2 keeps TF numerics for d/p/q/r in the current staged migration, but
    uses the dedicated ``logshiftto2`` link for ``nu`` so fitting paths enforce
    ``nu > 2`` semantics expected by TF2.
    """
    fam = TF()
    eps = jnp.finfo(jnp.float64).eps
    # R's d2ldv2 uses trigamma; jax.scipy.special.polygamma(1, .) is the same
    # function (agrees with scipy to 1e-16).
    from jax.scipy.special import polygamma

    # D2.143: the floor used to be 2.0 + eps with eps = finfo(float64).eps,
    # which is exactly half an ulp at 2.0 -- 2.0 + eps rounds back to 2.0, so
    # the clamp was a no-op: sigma1 = sqrt((nu-2)/nu) became 0, and every
    # density/score at nu <= 2 came back nan or inf (the very boundary the
    # clamp exists for).  1e-8 is representable and keeps sigma1 ~ 7e-5.
    _NU_FLOOR = 2.0 + 1e-8

    def _nu_gt2(nu):
        return jnp.maximum(jnp.asarray(nu, dtype=jnp.float64), _NU_FLOOR)

    def _sigma_scaled(sigma, nu):
        # R TF2.R: sigma1 = sqrt((nu-2)/nu) * sigma  (variance-1 t rescaled)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        nu_safe = _nu_gt2(nu)
        return sigma * jnp.sqrt((nu_safe - 2.0) / nu_safe)

    def _ds1dd(nu):
        """d(sigma1)/d(sigma) = sqrt((nu-2)/nu)  -- R TF2.R's `ds1dd`."""
        return jnp.sqrt((_nu_gt2(nu) - 2.0) / _nu_gt2(nu))

    def _ds1dv(sigma, nu):
        """d(sigma1)/d(nu) = sqrt(nu/(nu-2)) * sigma / nu^2  -- R's `ds1dv`.

        Note this is R's own expression, differentiated by hand and not by
        calculus: the true derivative of sqrt((nu-2)/nu) is
        `-1 / (nu^2 * sqrt((nu-2)/nu))`.  Multiplying that by sigma gives
        `-sigma / (nu^2 * sqrt((nu-2)/nu))`, which equals R's
        `sqrt(nu/(nu-2)) * sigma / nu^2` -- the same value, but R reaches it
        through the reciprocal form.  Kept verbatim because parity is judged
        against R, not against the prettiest derivation.
        """
        nu_safe = _nu_gt2(nu)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        return jnp.sqrt(nu_safe / (nu_safe - 2.0)) * sigma / jnp.square(nu_safe)

    def g_dev_inc_tf2(y, mu, sigma, nu):
        return fam.g_dev_inc(y, mu, _sigma_scaled(sigma, nu), _nu_gt2(nu))

    def d_tf2(x, mu=0.0, sigma=1.0, nu=3.0, log=False):
        return fam.d(x, mu, _sigma_scaled(sigma, nu), _nu_gt2(nu), log=log)

    def p_tf2(q, mu=0.0, sigma=1.0, nu=3.0, lower_tail=True, log_p=False):
        return fam.p(
            q,
            mu,
            _sigma_scaled(sigma, nu),
            _nu_gt2(nu),
            lower_tail=lower_tail,
            log_p=log_p,
        )

    def q_tf2(p, mu=0.0, sigma=1.0, nu=3.0, lower_tail=True, log_p=False):
        """Quantile wrapper with one-step refinement for staged TF2 numerics."""
        nu_safe = _nu_gt2(nu)

        # Normalize all paths to lower-tail probability space.
        p_arr = jnp.asarray(p, dtype=jnp.float64)
        log_half = -jnp.log(jnp.array(2.0, dtype=p_arr.dtype))
        if log_p:
            # log-probability path accepts finite values <= 0 and -Inf (p=0).
            valid_input = ~jnp.isnan(p_arr)
            valid_input = valid_input & (p_arr <= 0.0)
            # Clamp only for internal staged numerics: values above 0 are routed
            # to a conservative interior point and still masked out in output.
            p_internal = jnp.minimum(p_arr, 0.0)
        else:
            valid_input = jnp.isfinite(p_arr) & (p_arr >= 0.0) & (p_arr <= 1.0)
            p_internal = p_arr
        # Keep downstream staged numerics stable even for invalid inputs by
        # routing invalid entries through a conservative interior probability.
        # Branch at Python level to avoid constructing unused fill tensors.
        if log_p:
            logp_fill = jnp.full_like(p_arr, log_half)
            p_arr_safe = jnp.where(valid_input, p_internal, logp_fill)
            p_raw = jnp.exp(p_arr_safe)
        else:
            prob_fill = jnp.full_like(p_arr, 0.5)
            p_arr_safe = jnp.where(valid_input, p_internal, prob_fill)
            p_raw = p_arr_safe
        p_target = jnp.where(lower_tail, p_raw, 1.0 - p_raw)
        # Keep exact boundary semantics for legal log.p points:
        # log.p==0 maps to p=1 (or upper-tail p=0) before clip.
        if log_p:
            is_logp_zero = valid_input & (p_arr == 0.0)
            p_target = jnp.where(
                is_logp_zero,
                jnp.where(
                    lower_tail, jnp.ones_like(p_target), jnp.zeros_like(p_target)
                ),
                p_target,
            )
        p_target = jnp.clip(p_target, eps, 1.0 - eps)

        # Base quantile evaluation in stable lower-tail mode.
        sigma_s = _sigma_scaled(sigma, nu_safe)
        q0 = fam.q(p_target, mu, sigma_s, nu_safe, lower_tail=True, log_p=False)

        # One Newton-style correction improves p/q self-consistency near nu boundary.
        p0 = fam.p(q0, mu, sigma_s, nu_safe, lower_tail=True, log_p=False)
        d0 = fam.d(q0, mu, sigma_s, nu_safe, log=False)
        d0 = jnp.where(jnp.isfinite(d0), d0, eps)
        d0 = jnp.maximum(d0, eps)

        step = (p0 - p_target) / d0
        step = jnp.clip(step, -5.0, 5.0)
        edge_mask = (p_target <= 1e-8) | (p_target >= 1.0 - 1e-8)
        step = jnp.where(edge_mask, 0.0, step)
        step = jnp.where(jnp.isfinite(step), step, 0.0)

        q1 = q0 - step
        q1 = jnp.where(jnp.isfinite(q1), q1, q0)

        # Weak monotonicity across a sorted probability grid is only a real
        # invariant when every element shares the same parameters.  D2.143b:
        # the guard used to fire on any sorted p vector, so a call like
        # q(p_grid, mu_grid, sigma_grid, nu_grid) -- each element with its own
        # parameters, which is exactly how the R-consistency harness calls it --
        # had its quantiles forced through maximum.accumulate and came back up
        # to 3.6e-2 away in probability space (measured at p=0.087, nu=2.25,
        # where the unguarded value is scipy-exact to 1e-15).
        p_flat = jnp.ravel(p_target)
        q_flat = jnp.ravel(q1)
        if q_flat.ndim == 1:
            p_diff = jnp.diff(p_flat)
            is_monotone_grid = p_flat.size <= 1 or bool(jnp.all(p_diff >= 0.0))

            def _constant(parameter) -> bool:
                values = jnp.ravel(
                    jnp.broadcast_to(
                        jnp.asarray(parameter, dtype=jnp.float64), p_target.shape
                    )
                )
                return values.size <= 1 or bool(jnp.all(values == values[0]))

            if (
                is_monotone_grid
                and _constant(mu)
                and _constant(sigma)
                and _constant(nu_safe)
            ):
                q_flat = jnp.maximum.accumulate(q_flat)
                q1 = jnp.reshape(q_flat, q1.shape)
        if log_p:
            is_logp_zero = valid_input & (p_arr == 0.0)
            is_logp_neginf = valid_input & jnp.isneginf(p_arr)
            q1 = jnp.where(
                is_logp_zero,
                jnp.where(lower_tail, jnp.inf, -jnp.inf),
                q1,
            )
            q1 = jnp.where(
                is_logp_neginf,
                jnp.where(lower_tail, -jnp.inf, jnp.inf),
                q1,
            )
        return jnp.where(valid_input, q1, jnp.nan)

    def r_tf2(n, mu=0.0, sigma=1.0, nu=3.0):
        return fam.r(n, mu, sigma, _nu_gt2(nu))

    # The score/hessian forwarding below used to pass the *raw* sigma straight
    # through to TF's derivatives.  That is wrong: TF2's density is TF's
    # density in sigma1 = sqrt((nu-2)/nu) * sigma, so the chain rule contributes
    # ds1/dd and ds1/dv that were simply missing.  Measured against our own
    # density at (y=0.3, mu=0, sigma=1, nu=3), every parameter was off:
    #
    #     mu  analytic 0.38834951vs FD 1.10091743  (diff 7.1e-01)
    #     sigma analytic -0.88349515  vs FD -0.66972477 (diff 2.1e-01)
    #     nu  analytic 0.03111859   vs FD -0.18480405 (diff 2.2e-01)
    #
    # The hessians still agreed with the *forwarded* scores to 1e-11, so RS was
    # iterating a gradient that was internally self-consistent but did not
    # belong to the density being minimized -- deviance 3399 vs R's 144.172.
    #
    # R's own formulas (gamlss.dist TF2.R), transcribed including its
    # deliberately coarse hessian approximations:
    #
    #   dldm   = TF$dldm (y, mu, sigma1, nu)
    #   dldd   = ds1dd* TF$dldd (y, mu, sigma1, nu)
    #   dldv   = TF$dldv (y, mu, sigma1, nu) + ds1dv * TF$dldd(y, mu, sigma1, nu)
    #   d2ldm2 = -(nu+1) / ((nu+3) * sigma1^2)
    #   d2ldd2 = -ds1dd^2 * (2*nu) / ((nu+3) * sigma1^2)
    #   d2ldv2 = [trigamma((nu+1)/2) - trigamma(nu/2) + 2(nu+5)/(nu(nu+1)(nu+3))]/4
    #            + ds1dv^2 * TF$d2ldd2(sigma1, nu),floored at-1e-15
    #
    # RS only consumes the diagonal Hessian, so d2ldmdd and friends are not
    # transcribed.
    def dldm_tf2(y, mu, sigma, nu):
        return fam.score_functions["mu"](y, mu, _sigma_scaled(sigma, nu), _nu_gt2(nu))

    def dldd_tf2(y, mu, sigma, nu):
        return _ds1dd(nu) * fam.score_functions["sigma"](
            y, mu, _sigma_scaled(sigma, nu), _nu_gt2(nu)
        )

    def dldn_tf2(y, mu, sigma, nu):
        nu_safe = _nu_gt2(nu)
        sigma1 = _sigma_scaled(sigma, nu)
        return fam.score_functions["nu"](y, mu, sigma1, nu_safe) + _ds1dv(
            sigma, nu
        ) * fam.score_functions["sigma"](y, mu, sigma1, nu_safe)

    def d2ldm2_tf2(y, mu, sigma, nu):
        nu_safe = _nu_gt2(nu)
        sigma1 = _sigma_scaled(sigma, nu_safe)
        return -(nu_safe + 1.0) / ((nu_safe + 3.0) * jnp.square(sigma1))

    def d2ldd2_tf2(y, mu, sigma, nu):
        nu_safe = _nu_gt2(nu)
        sigma1 = _sigma_scaled(sigma, nu_safe)
        s2 = jnp.square(sigma1)
        return -jnp.square(_ds1dd(nu_safe)) * ((2.0 * nu_safe) / ((nu_safe + 3.0) * s2))

    def d2ldn2_tf2(y, mu, sigma, nu):
        nu_safe = _nu_gt2(nu)
        ds1dv = _ds1dv(sigma, nu_safe)
        v2 = nu_safe / 2.0
        v3 = (nu_safe + 1.0) / 2.0
        val = (
            polygamma(1, v3)
            - polygamma(1, v2)
            + (2.0 * (nu_safe + 5.0) / (nu_safe * (nu_safe + 1.0) * (nu_safe + 3.0)))
        )
        val = val / 4.0
        # R writes TF()$d2ldd2(sigma1, nu) here -- a two-argument closed form
        # `-2*nu / ((nu+3) * sigma1^2)`.  Forwarding our own TF hessian instead
        # (which is autodiff-derived and takes (y, mu, sigma, nu)) silently
        # substituted a different function: measured at nu = 3 it returns
        # -1.0 instead of R's -3.0, and the resulting d2ldv2 came out -0.00983
        # against R's -0.12802 -- a13x error in the only term that steers the
        # nu direction, which is why the fit ran into the nu > 2 floor
        # (nu_eta = -36.0) instead of R's optimum at nu_eta = 2.22.
        # Transcribed rather than derived.
        val = val + jnp.square(ds1dv) * (
            -(2.0 * nu_safe)
            / ((nu_safe + 3.0) * jnp.square(_sigma_scaled(sigma, nu_safe)))
        )
        return jnp.where(val < -1e-15, val, -1e-15)

    return TF2Family(
        name="TF2",
        parameters=fam.parameters,
        g_dev_inc=g_dev_inc_tf2,
        type=fam.type,
        links={"mu": "identity", "sigma": "log", "nu": "logshiftto2"},
        link_functions={"mu": identity_link, "sigma": log_link, "nu": logshiftto2_link},
        link_inverses={
            "mu": identity_inverse,
            "sigma": log_inverse,
            "nu": logshiftto2_inverse,
        },
        link_derivatives={
            "mu": identity_derivative,
            "sigma": log_derivative,
            "nu": logshiftto2_derivative,
        },
        score_functions={"mu": dldm_tf2, "sigma": dldd_tf2, "nu": dldn_tf2},
        hessian_functions={"mu": d2ldm2_tf2, "sigma": d2ldd2_tf2, "nu": d2ldn2_tf2},
        d=d_tf2,
        p=p_tf2,
        q=q_tf2,
        r=r_tf2,
    )


def JSU() -> JohnsonSUFamily:
    """Staged Johnson SU family with parameters `mu`, `sigma`, `nu`, `tau`."""

    eps = jnp.finfo(jnp.float64).eps

    def _log_density(
        y: jnp.ndarray,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
        nu: jnp.ndarray,
        tau: jnp.ndarray,
    ) -> jnp.ndarray:
        """R JSU.R dJSU 的 log 密度（D2.34 逐字转写，与 dpqr_functions.dJSU 同步）。

        旧实现是 ``z=(y-mu)/sigma, w=nu+tau*asinh(z)`` 的无中心简化版——
        那是另一个分布：dpqr 三关 rel 0.3~1、rsdev 拟合偏大，全部同根。
        R 的参数化带 SU 校正项：w=exp(1/tau^2), omega=-nu/tau,
        c=(0.5*(w-1)*(w*cosh(2*omega)+1))^(-0.5)。
        """
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps)
        nu = jnp.asarray(nu, dtype=jnp.float64)
        tau = jnp.maximum(jnp.asarray(tau, dtype=jnp.float64), eps)

        rtau = 1.0 / tau
        w = jnp.where(rtau < 1e-7, 1.0, jnp.exp(jnp.square(rtau)))
        omega = -nu * rtau
        c = jnp.power(0.5 * (w - 1.0) * (w * jnp.cosh(2.0 * omega) + 1.0), -0.5)

        loc = mu + c * sigma * jnp.sqrt(w) * jnp.sinh(omega)
        z = (y - loc) / (c * sigma)
        r = -nu + jnp.arcsinh(z) / rtau

        return (
            -jnp.log(sigma)
            - jnp.log(c)
            - jnp.log(rtau)
            - 0.5 * jnp.log1p(jnp.square(z))
            - 0.5 * jnp.log(2.0 * math.pi)
            - 0.5 * jnp.square(r)
        )

    def g_dev_inc(
        y: jnp.ndarray,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
        nu: jnp.ndarray,
        tau: jnp.ndarray,
    ) -> jnp.ndarray:
        return -2.0 * _log_density(y, mu, sigma, nu, tau)

    def _numeric_score(
        parameter: str,
        y: jnp.ndarray,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
        nu: jnp.ndarray,
        tau: jnp.ndarray,
    ) -> jnp.ndarray:
        step = jnp.array(1e-5, dtype=jnp.float64)
        if parameter == "mu":
            plus = _log_density(y, mu + step, sigma, nu, tau)
            minus = _log_density(y, mu - step, sigma, nu, tau)
            return (plus - minus) / (2.0 * step)
        if parameter == "sigma":
            plus = _log_density(y, mu, jnp.maximum(sigma + step, eps), nu, tau)
            minus = _log_density(y, mu, jnp.maximum(sigma - step, eps), nu, tau)
            return (plus - minus) / (2.0 * step)
        if parameter == "nu":
            plus = _log_density(y, mu, sigma, nu + step, tau)
            minus = _log_density(y, mu, sigma, nu - step, tau)
            return (plus - minus) / (2.0 * step)
        plus = _log_density(y, mu, sigma, nu, jnp.maximum(tau + step, eps))
        minus = _log_density(y, mu, sigma, nu, jnp.maximum(tau - step, eps))
        return (plus - minus) / (2.0 * step)

    def _numeric_hessian(
        parameter: str,
        y: jnp.ndarray,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
        nu: jnp.ndarray,
        tau: jnp.ndarray,
    ) -> jnp.ndarray:
        step = jnp.array(1e-4, dtype=jnp.float64)
        center = _log_density(y, mu, sigma, nu, tau)
        if parameter == "mu":
            plus = _log_density(y, mu + step, sigma, nu, tau)
            minus = _log_density(y, mu - step, sigma, nu, tau)
        elif parameter == "sigma":
            plus = _log_density(y, mu, jnp.maximum(sigma + step, eps), nu, tau)
            minus = _log_density(y, mu, jnp.maximum(sigma - step, eps), nu, tau)
        elif parameter == "nu":
            plus = _log_density(y, mu, sigma, nu + step, tau)
            minus = _log_density(y, mu, sigma, nu - step, tau)
        else:
            plus = _log_density(y, mu, sigma, nu, jnp.maximum(tau + step, eps))
            minus = _log_density(y, mu, sigma, nu, jnp.maximum(tau - step, eps))
        return (plus - 2.0 * center + minus) / jnp.square(step)

    return JohnsonSUFamily(
        name="JSU",
        parameters=("mu", "sigma", "nu", "tau"),
        g_dev_inc=g_dev_inc,
        type="Continuous",
        links={"mu": "identity", "sigma": "log", "nu": "identity", "tau": "log"},
        link_functions={
            "mu": identity_link,
            "sigma": log_link,
            "nu": identity_link,
            "tau": log_link,
        },
        link_inverses={
            "mu": identity_inverse,
            "sigma": log_inverse,
            "nu": identity_inverse,
            "tau": log_inverse,
        },
        link_derivatives={
            "mu": identity_derivative,
            "sigma": log_derivative,
            "nu": identity_derivative,
            "tau": log_derivative,
        },
        score_functions={
            "mu": lambda y, mu, sigma, nu, tau: _numeric_score(
                "mu", y, mu, sigma, nu, tau
            ),
            "sigma": lambda y, mu, sigma, nu, tau: _numeric_score(
                "sigma", y, mu, sigma, nu, tau
            ),
            "nu": lambda y, mu, sigma, nu, tau: _numeric_score(
                "nu", y, mu, sigma, nu, tau
            ),
            "tau": lambda y, mu, sigma, nu, tau: _numeric_score(
                "tau", y, mu, sigma, nu, tau
            ),
        },
        hessian_functions={
            "mu": lambda y, mu, sigma, nu, tau: _numeric_hessian(
                "mu", y, mu, sigma, nu, tau
            ),
            "sigma": lambda y, mu, sigma, nu, tau: _numeric_hessian(
                "sigma", y, mu, sigma, nu, tau
            ),
            "nu": lambda y, mu, sigma, nu, tau: _numeric_hessian(
                "nu", y, mu, sigma, nu, tau
            ),
            "tau": lambda y, mu, sigma, nu, tau: _numeric_hessian(
                "tau", y, mu, sigma, nu, tau
            ),
        },
        d=dJSU,
        p=pJSU,
        q=qJSU,
        r=rJSU,
    )


def BCCG() -> BoxCoxColeGreenFamily:
    """Staged Box-Cox Cole-Green family with parameters `mu`, `sigma`, `nu`."""

    eps = jnp.finfo(jnp.float64).eps

    def _z_transform(
        y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, nu: jnp.ndarray
    ) -> jnp.ndarray:
        y = jnp.maximum(jnp.asarray(y, dtype=jnp.float64), eps)
        mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), eps)
        sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps)
        nu = jnp.asarray(nu, dtype=jnp.float64)
        ratio = y / mu
        use_log = jnp.abs(nu) < 1e-6
        z_boxcox = (jnp.power(ratio, nu) - 1.0) / (nu * sigma)
        z_log = jnp.log(ratio) / sigma
        return jnp.where(use_log, z_log, z_boxcox)

    def _truncation_log_term(sigma: jnp.ndarray, nu: jnp.ndarray) -> jnp.ndarray:
        """``-log(pnorm(1/(sigma*|nu|)))`` -- R BCCG.R's truncation term.

        Same defect shape as D2.12 found on BCT: ``dpqr_functions.dBCCG``
        carries this term and the fitting log-density did not.  Measured at
        sigma=0.65, nu=1.7 the two log-densities differed by exactly
        ``0.20179639 = -log(pnorm(1/(sigma*|nu|)))``, constant in ``y`` --
        harmless for the deviance, but it shifts every score and Hessian
        because the term depends on ``sigma`` and ``nu``.
        """
        from jax.scipy.stats import norm

        return -norm.logcdf(1.0 / (sigma * jnp.maximum(jnp.abs(nu), eps)))

    def _log_density(
        y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, nu: jnp.ndarray
    ) -> jnp.ndarray:
        y = jnp.maximum(jnp.asarray(y, dtype=jnp.float64), eps)
        mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), eps)
        sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps)
        nu = jnp.asarray(nu, dtype=jnp.float64)
        z = _z_transform(y, mu, sigma, nu)
        return (
            -0.5 * jnp.log(2.0 * math.pi)
            - jnp.log(sigma)
            - nu * jnp.log(mu)
            + (nu - 1.0) * jnp.log(y)
            - 0.5 * jnp.square(z)
            + _truncation_log_term(sigma, nu)
        )

    def g_dev_inc(
        y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, nu: jnp.ndarray
    ) -> jnp.ndarray:
        return -2.0 * _log_density(y, mu, sigma, nu)

    def _score(
        parameter: str,
        y: jnp.ndarray,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
        nu: jnp.ndarray,
    ) -> jnp.ndarray:
        """R BCCG.R's closed-form scores ``dldm`` / ``dldd`` / ``dldv``.

        Transcribed verbatim.  ``dldd`` and ``dldv`` both need the normal
        inverse Mills ratio ``h = dnorm(1/(sigma*|nu|)) / pnorm(1/(sigma*|nu|))``
        because of the truncation term.  Note the ``1/sqrt(2*pi)`` in ``dnorm``:
        dropping it inflates both scores by 2.5066.
        """
        from jax.scipy.stats import norm as _norm

        y = jnp.maximum(jnp.asarray(y, dtype=jnp.float64), eps)
        mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), eps)
        sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps)
        nu = jnp.asarray(nu, dtype=jnp.float64)

        if parameter == "mu":
            z = _z_transform(y, mu, sigma, nu)
            return ((z / sigma) + nu * (z * z - 1.0)) / mu

        arg = 1.0 / (sigma * jnp.maximum(jnp.abs(nu), eps))
        h = _norm.pdf(arg) / jnp.maximum(_norm.cdf(arg), eps)
        z = _z_transform(y, mu, sigma, nu)
        if parameter == "sigma":
            return (z * z - 1.0) / sigma + h / (sigma * sigma * jnp.abs(nu))
        log_ratio = jnp.log(y / mu)
        value = (z - (log_ratio / sigma)) * (z / nu) - log_ratio * (z * z - 1.0)
        return value + jnp.sign(nu) * h / (sigma * nu * nu)

    def _hessian(
        parameter: str,
        y: jnp.ndarray,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
        nu: jnp.ndarray,
    ) -> jnp.ndarray:
        """R BCCG.R's diagonal second derivatives, transcribed verbatim.

        ``d2ldv2 = -7*sigma^2/4`` is as coarse here as it is on BCT -- it is
        what R feeds its IRLS, so copying it is what parity means.
        """
        y = jnp.maximum(jnp.asarray(y, dtype=jnp.float64), eps)
        mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), eps)
        sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps)
        nu = jnp.asarray(nu, dtype=jnp.float64)

        if parameter == "mu":
            return -(1.0 + 2.0 * nu * nu * sigma * sigma) / (mu * mu * sigma * sigma)
        if parameter == "sigma":
            return -2.0 / (sigma * sigma)
        return -7.0 * sigma * sigma / 4.0

    return BoxCoxColeGreenFamily(
        name="BCCG",
        parameters=("mu", "sigma", "nu"),
        g_dev_inc=g_dev_inc,
        type="Continuous",
        # Link functions transcribed from gamlss.dist::BCCG():
        #   mu.link = "identity", sigma.link = "log", nu.link = "identity"
        # mu was "log" here.  With the analytic scores in place the fit
        # converges either way (deviance 199.567 vs R's 199.565), but the
        # reported coefficient is on a different scale: log-link mu = 0.9996
        # against R's identity-link 2.7684 (exp(0.9996) = 2.717, the same
        # optimum, different parameterisation).
        links={"mu": "identity", "sigma": "log", "nu": "identity"},
        link_functions={"mu": identity_link, "sigma": log_link, "nu": identity_link},
        link_inverses={
            "mu": identity_inverse,
            "sigma": log_inverse,
            "nu": identity_inverse,
        },
        link_derivatives={
            "mu": identity_derivative,
            "sigma": log_derivative,
            "nu": identity_derivative,
        },
        score_functions={
            "mu": lambda y, mu, sigma, nu: _score("mu", y, mu, sigma, nu),
            "sigma": lambda y, mu, sigma, nu: _score("sigma", y, mu, sigma, nu),
            "nu": lambda y, mu, sigma, nu: _score("nu", y, mu, sigma, nu),
        },
        hessian_functions={
            "mu": lambda y, mu, sigma, nu: _hessian("mu", y, mu, sigma, nu),
            "sigma": lambda y, mu, sigma, nu: _hessian("sigma", y, mu, sigma, nu),
            "nu": lambda y, mu, sigma, nu: _hessian("nu", y, mu, sigma, nu),
        },
        d=dBCCG,
        p=pBCCG,
        q=qBCCG,
        r=rBCCG,
    )


# BCT (Box-Cox t) Hessian floors per parameter — heuristic lower bounds on
# the second derivative to prevent IWLS collapse.  Kept as module constants
# so they can be tuned without touching the fitting code.
_BCT_HESS_FLOOR = {"tau": -0.1, "sigma": -0.5, "nu": -0.2, "mu": -1e-10}


def BCT() -> BoxCoxTFamily:
    """Staged Box-Cox t family with parameters `mu`, `sigma`, `nu`, `tau`."""

    eps = jnp.finfo(jnp.float64).eps

    def tau_link(x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x, dtype=jnp.float64)
        return jnp.log(jnp.maximum(x, eps))

    def tau_inverse(eta: jnp.ndarray) -> jnp.ndarray:
        eta = jnp.asarray(eta, dtype=jnp.float64)
        return jnp.maximum(jnp.exp(eta), eps)

    def tau_derivative(eta: jnp.ndarray) -> jnp.ndarray:
        eta = jnp.asarray(eta, dtype=jnp.float64)
        return jnp.maximum(jnp.exp(eta), eps)

    def _z_transform(
        y: jnp.ndarray,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
        nu: jnp.ndarray,
    ) -> jnp.ndarray:
        y = jnp.maximum(jnp.asarray(y, dtype=jnp.float64), eps)
        mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), eps)
        sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps)
        nu = jnp.asarray(nu, dtype=jnp.float64)
        ratio = y / mu
        use_log = jnp.abs(nu) < 1e-6
        z_boxcox = (jnp.power(ratio, nu) - 1.0) / (nu * sigma)
        z_log = jnp.log(ratio) / sigma
        return jnp.where(use_log, z_log, z_boxcox)

    def _lgamma_array(x: jnp.ndarray) -> jnp.ndarray:
        flat = np.asarray(x, dtype=np.float64).reshape(-1)
        values = np.array(
            [math.lgamma(float(value)) for value in flat], dtype=np.float64
        )
        return jnp.asarray(values.reshape(np.asarray(x).shape), dtype=jnp.float64)

    def _log_pt_trunc(
        sigma: jnp.ndarray, nu: jnp.ndarray, tau: jnp.ndarray
    ) -> jnp.ndarray:
        """``log(pt(1/(sigma*|nu|), df = tau))`` -- R BCT.R's truncation CDF.

        JAX ships no Student-t CDF (only ``t.pdf`` / ``t.logpdf``), so use the
        regularized incomplete beta identity already used by ``BCTo``::

            pt(t, df) = 1 - 0.5 * I_{df/(df+t^2)}(df/2, 1/2)     for t > 0

        ``|nu|`` is floored rather than allowed to reach zero so the ``1/|nu|``
        blow-up stays finite (and differentiable) instead of producing inf/NaN.

        Accuracy caveat: when ``df`` is large *and* the argument is small the
        regularized beta saturates towards 1, and the ``1 - ...`` cancellation
        amplifies the absolute error of ``betainc``.  R's own ``dldt`` only ever
        differentiates this quantity with a 0.01 forward step, so the induced
        noise floor (~1e-3 relative) is R's, not ours.
        """
        from jax.scipy.special import betainc as _betainc

        arg = 1.0 / (sigma * jnp.maximum(jnp.abs(nu), eps))
        x_beta = tau / (tau + arg * arg)
        tail = 0.5 * _betainc(tau / 2.0, 0.5, x_beta)
        return jnp.log(jnp.maximum(1.0 - tail, eps))

    def _truncation_log_term(
        sigma: jnp.ndarray, nu: jnp.ndarray, tau: jnp.ndarray
    ) -> jnp.ndarray:
        """``-log(pt(1/(sigma*|nu|), df = tau))`` -- R BCT.R's truncation term.

        The Box-Cox t family is *truncated*: R's ``dBCT`` subtracts the upper
        tail beyond ``1/(sigma*|nu|)``.  That term depends on ``sigma`` and
        ``nu``, so leaving it out of the fitting log-density does not merely
        shift the deviance by a constant -- it also makes every score and
        Hessian wrong, which is what the parity scanner flags for this family.
        """
        return -_log_pt_trunc(sigma, nu, tau)

    def _trunc_hazard(
        sigma: jnp.ndarray, nu: jnp.ndarray, tau: jnp.ndarray
    ) -> jnp.ndarray:
        """``h = dt(1/(sigma*|nu|), tau) / pt(1/(sigma*|nu|), tau)``.

        Inverse Mills ratio of the Student-t at the truncation point -- the term
        R's ``dldd`` and ``dldv`` both need for the truncation constant.
        """
        from jax.scipy.special import gammaln as _gammaln

        arg = 1.0 / (sigma * jnp.maximum(jnp.abs(nu), eps))
        log_norm = (
            _gammaln((tau + 1.0) / 2.0)
            - _gammaln(tau / 2.0)
            - 0.5 * jnp.log(tau * math.pi)
        )
        pdf = jnp.exp(log_norm) * jnp.power(1.0 + arg * arg / tau, -(tau + 1.0) / 2.0)
        return pdf * jnp.exp(-_log_pt_trunc(sigma, nu, tau))

    def _log_density(
        y: jnp.ndarray,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
        nu: jnp.ndarray,
        tau: jnp.ndarray,
    ) -> jnp.ndarray:
        y = jnp.maximum(jnp.asarray(y, dtype=jnp.float64), eps)
        mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), eps)
        sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps)
        nu = jnp.asarray(nu, dtype=jnp.float64)
        tau = jnp.maximum(jnp.asarray(tau, dtype=jnp.float64), eps)
        z = _z_transform(y, mu, sigma, nu)
        log_norm = (
            _lgamma_array((tau + 1.0) / 2.0)
            - _lgamma_array(tau / 2.0)
            - 0.5 * jnp.log(tau * math.pi)
        )
        jacobian = -jnp.log(sigma) - nu * jnp.log(mu) + (nu - 1.0) * jnp.log(y)
        kernel = -((tau + 1.0) / 2.0) * jnp.log1p(jnp.square(z) / tau)
        truncation = _truncation_log_term(sigma, nu, tau)
        return log_norm + jacobian + kernel + truncation

    def g_dev_inc(
        y: jnp.ndarray,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
        nu: jnp.ndarray,
        tau: jnp.ndarray,
    ) -> jnp.ndarray:
        return -2.0 * _log_density(y, mu, sigma, nu, tau)

    def _score(
        parameter: str,
        y: jnp.ndarray,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
        nu: jnp.ndarray,
        tau: jnp.ndarray,
    ) -> jnp.ndarray:
        """R BCT.R's closed-form scores ``dldm``/``dldd``/``dldv``/``dldt``.

        Transcribed verbatim from ``gamlss.dist::BCT()`` rather than derived.
        Two of them are only *approximately* right even in R (``dldt`` even
        differentiates ``log pt`` with a hard-coded 0.01 forward step), so
        "matches the density to 1e-12" is not an achievable target here and is
        not what parity means for this family -- matching R is.
        """
        from jax.scipy.special import digamma as _digamma

        y = jnp.maximum(jnp.asarray(y, dtype=jnp.float64), eps)
        mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), eps)
        sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps)
        nu = jnp.asarray(nu, dtype=jnp.float64)
        tau = jnp.maximum(jnp.asarray(tau, dtype=jnp.float64), eps)

        if parameter == "mu":
            z = _z_transform(y, mu, sigma, nu)
            w = (tau + 1.0) / (tau + z * z)
            return (w * z) / (mu * sigma) + (nu / mu) * (w * z * z - 1.0)
        if parameter == "sigma":
            z = _z_transform(y, mu, sigma, nu)
            w = (tau + 1.0) / (tau + z * z)
            h = _trunc_hazard(sigma, nu, tau)
            return (w * z * z - 1.0) / sigma + h / (sigma * sigma * jnp.abs(nu))
        if parameter == "nu":
            z = _z_transform(y, mu, sigma, nu)
            log_ratio = jnp.log(jnp.maximum(y, eps) / jnp.maximum(mu, eps))
            w = (tau + 1.0) / (tau + z * z)
            h = _trunc_hazard(sigma, nu, tau)
            value = (w * z * z) / nu - log_ratio * (
                w * z * z + (w * z) / (sigma * nu) - 1.0
            )
            return value + jnp.sign(nu) * h / (sigma * nu * nu)

        z = _z_transform(y, mu, sigma, nu)
        w = (tau + 1.0) / (tau + z * z)
        # R differentiates log pt with a fixed 0.01 forward step; copied as-is
        # because a more accurate j would drift *away* from R's own dldt.
        j = (
            _log_pt_trunc(sigma, nu, tau + 0.01) - _log_pt_trunc(sigma, nu, tau)
        ) / 0.01
        value = -0.5 * jnp.log1p(z * z / tau) + (w * z * z) / (2.0 * tau)
        return (
            value
            + 0.5 * _digamma((tau + 1.0) / 2.0)
            - 0.5 * _digamma(tau / 2.0)
            - 1.0 / (2.0 * tau)
            - j
        )

    def _hessian(
        parameter: str,
        y: jnp.ndarray,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
        nu: jnp.ndarray,
        tau: jnp.ndarray,
    ) -> jnp.ndarray:
        """R BCT.R's diagonal second derivatives.

        ``d2ldm2``/``d2ldd2``/``d2ldv2``/``d2ldt2`` are transcribed verbatim.
        They are coarse (``d2ldv2`` is literally ``-7*sigma^2/4``) but they are
        what R feeds to its IRLS, so parity means copying them.

        No ``_BCT_HESS_FLOOR`` clamping here any more: that floor existed to
        stop the finite-difference Hessian from going unstable, and
        ``compute_working_weights_and_response`` already applies R's own
        ``-1e-15`` floor plus a weight clip at the point of use.
        """
        from jax.scipy.special import polygamma as _polygamma

        y = jnp.maximum(jnp.asarray(y, dtype=jnp.float64), eps)
        mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), eps)
        sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps)
        nu = jnp.asarray(nu, dtype=jnp.float64)
        tau = jnp.maximum(jnp.asarray(tau, dtype=jnp.float64), eps)

        if parameter == "mu":
            return (
                -(tau + 2.0 * nu * nu * sigma * sigma * tau + 1.0)
                / (tau + 3.0)
                / (mu * mu * sigma * sigma)
            )
        if parameter == "sigma":
            return -2.0 * tau / (sigma * sigma * (tau + 3.0))
        if parameter == "nu":
            return -7.0 * sigma * sigma / 4.0

        value = (
            _polygamma(1, (tau + 1.0) / 2.0)
            - _polygamma(1, tau / 2.0)
            + 2.0 * (tau + 5.0) / (tau * (tau + 1.0) * (tau + 3.0))
        ) / 4.0
        return jnp.where(value < -1e-15, value, -1e-15)

    return BoxCoxTFamily(
        name="BCT",
        parameters=("mu", "sigma", "nu", "tau"),
        g_dev_inc=g_dev_inc,
        type="Continuous",
        # Link functions transcribed from gamlss.dist::BCT():
        #   mu.link = "identity", sigma.link = "log",
        #   nu.link = "identity", tau.link = "log"
        # The old mu link was "log" and tau was "logshiftto2" (i.e. log(tau-2)).
        # That is not a cosmetic difference: R's dBCT only requires tau > 0, so
        # under "logshiftto2" RS ran the eta to -35 and stalled against the
        # implied tau = 2 floor (deviance 209.53 against R's 199.56).  With R's
        # links the tau direction can actually reach the tau -> infinity
        # (Box-Cox Gaussian) limit that is BCT's deviance optimum on this data.
        links={"mu": "identity", "sigma": "log", "nu": "identity", "tau": "log"},
        link_functions={
            "mu": identity_link,
            "sigma": log_link,
            "nu": identity_link,
            "tau": tau_link,
        },
        link_inverses={
            "mu": identity_inverse,
            "sigma": log_inverse,
            "nu": identity_inverse,
            "tau": tau_inverse,
        },
        link_derivatives={
            "mu": identity_derivative,
            "sigma": log_derivative,
            "nu": identity_derivative,
            "tau": tau_derivative,
        },
        score_functions={
            "mu": lambda y, mu, sigma, nu, tau: _score("mu", y, mu, sigma, nu, tau),
            "sigma": lambda y, mu, sigma, nu, tau: _score(
                "sigma", y, mu, sigma, nu, tau
            ),
            "nu": lambda y, mu, sigma, nu, tau: _score("nu", y, mu, sigma, nu, tau),
            "tau": lambda y, mu, sigma, nu, tau: _score("tau", y, mu, sigma, nu, tau),
        },
        hessian_functions={
            "mu": lambda y, mu, sigma, nu, tau: _hessian("mu", y, mu, sigma, nu, tau),
            "sigma": lambda y, mu, sigma, nu, tau: _hessian(
                "sigma", y, mu, sigma, nu, tau
            ),
            "nu": lambda y, mu, sigma, nu, tau: _hessian("nu", y, mu, sigma, nu, tau),
            "tau": lambda y, mu, sigma, nu, tau: _hessian("tau", y, mu, sigma, nu, tau),
        },
        d=dBCT,
        p=pBCT,
        q=qBCT,
        r=rBCT,
    )


def BCPE() -> BoxCoxPowerExponentialFamily:
    """Staged Box-Cox power exponential family with `mu`, `sigma`, `nu`, `tau`."""

    eps = jnp.finfo(jnp.float64).eps

    def tau_link(x: jnp.ndarray) -> jnp.ndarray:
        x = jnp.asarray(x, dtype=jnp.float64)
        return jnp.log(jnp.maximum(x, eps))

    def tau_inverse(eta: jnp.ndarray) -> jnp.ndarray:
        eta = jnp.asarray(eta, dtype=jnp.float64)
        return jnp.maximum(jnp.exp(eta), eps)

    def tau_derivative(eta: jnp.ndarray) -> jnp.ndarray:
        eta = jnp.asarray(eta, dtype=jnp.float64)
        return jnp.maximum(jnp.exp(eta), eps)

    def _z_transform(
        y: jnp.ndarray,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
        nu: jnp.ndarray,
    ) -> jnp.ndarray:
        y = jnp.maximum(jnp.asarray(y, dtype=jnp.float64), eps)
        mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), eps)
        sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps)
        nu = jnp.asarray(nu, dtype=jnp.float64)
        ratio = y / mu
        use_log = jnp.abs(nu) < 1e-6
        z_boxcox = (jnp.power(ratio, nu) - 1.0) / (nu * sigma)
        z_log = jnp.log(ratio) / sigma
        return jnp.where(use_log, z_log, z_boxcox)

    def _log_density(
        y: jnp.ndarray,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
        nu: jnp.ndarray,
        tau: jnp.ndarray,
    ) -> jnp.ndarray:
        y = jnp.maximum(jnp.asarray(y, dtype=jnp.float64), eps)
        mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), eps)
        sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps)
        nu = jnp.asarray(nu, dtype=jnp.float64)
        tau = jnp.maximum(jnp.asarray(tau, dtype=jnp.float64), eps)
        z = _z_transform(y, mu, sigma, nu)
        c = jnp.maximum(_bcpe_c(tau), eps)
        # R BCPE.R's kernel is  -0.5 * (|z/c|^tau), NOT -0.5*|z|^tau.  The
        # standardization constant c is what makes the distribution
        # variance-standardised, and it appears in three places: log_norm,
        # the kernel, and (via its derivative) the tau score.
        log_c = jnp.log(c)
        log_norm = (
            jnp.log(tau) - log_c - (1.0 + 1.0 / tau) * jnp.log(2.0) - gammaln(1.0 / tau)
        )
        jacobian = -jnp.log(sigma) - nu * jnp.log(mu) + (nu - 1.0) * jnp.log(y)
        kernel = -0.5 * jnp.power(jnp.abs(z) / c, tau)
        # Truncation: R ends with - log(F.T(1/(sigma*|nu|))).  It depends on
        # sigma and nu, so leaving it out shifts the scores as well.
        truncation = -_bcpe_ftrunc_log(
            1.0 / (sigma * jnp.maximum(jnp.abs(nu), eps)), tau
        )
        return log_norm + jacobian + kernel + truncation

    def g_dev_inc(
        y: jnp.ndarray,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
        nu: jnp.ndarray,
        tau: jnp.ndarray,
    ) -> jnp.ndarray:
        return -2.0 * _log_density(y, mu, sigma, nu, tau)

    def _bcpe_trunc_hazard(
        sigma: jnp.ndarray, nu: jnp.ndarray, tau: jnp.ndarray
    ) -> jnp.ndarray:
        """R BCPE.R's ``h = f.T(1/(sigma*|nu|)) / F.T(1/(sigma*|nu|))``.

        Both halves are needed in closed form: ``f.T`` is the standardized
        power-exponential density, ``F.T`` its CDF.  ``|nu|`` is floored so the
        ``1/|nu|`` blow-up stays finite.
        """
        arg = 1.0 / (sigma * jnp.maximum(jnp.abs(nu), eps))
        c = jnp.maximum(_bcpe_c(tau), eps)
        log_pdf = (
            jnp.log(tau)
            - jnp.log(c)
            - 0.5 * jnp.power(jnp.abs(arg) / c, tau)
            - (1.0 + 1.0 / tau) * jnp.log(2.0)
            - gammaln(1.0 / tau)
        )
        return jnp.exp(log_pdf - _bcpe_ftrunc_log(arg, tau))

    def _score(
        parameter: str,
        y: jnp.ndarray,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
        nu: jnp.ndarray,
        tau: jnp.ndarray,
    ) -> jnp.ndarray:
        """R BCPE.R's closed-form scores, transcribed verbatim.

        Every term is expressed through the standardization constant
        ``c = exp(0.5*(-(2/tau)*log(2) + lgamma(1/tau) - lgamma(3/tau)))``;
        ``dldt`` additionally needs ``d log(c)/d tau`` and, like R's own
        ``dldt``, differences ``log F.T`` with a hard-coded 0.001 step.
        """
        from jax.scipy.special import digamma as _digamma, gammainc as _gammainc

        y = jnp.maximum(jnp.asarray(y, dtype=jnp.float64), eps)
        mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), eps)
        sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps)
        nu = jnp.asarray(nu, dtype=jnp.float64)
        tau = jnp.maximum(jnp.asarray(tau, dtype=jnp.float64), eps)

        z = _z_transform(y, mu, sigma, nu)
        c = jnp.maximum(_bcpe_c(tau), eps)

        if parameter == "mu":
            return (tau / (2.0 * mu * sigma * c * c)) * (
                z + sigma * nu * z * z
            ) * jnp.power(jnp.abs(z / c), tau - 2.0) - (nu / mu)

        h = _bcpe_trunc_hazard(sigma, nu, tau)
        if parameter == "sigma":
            return (1.0 / sigma) * (
                (tau / 2.0) * jnp.power(jnp.abs(z / c), tau) - 1.0
            ) + h / (sigma * sigma * jnp.abs(nu))

        log_ratio = jnp.log(y / mu)
        if parameter == "nu":
            value = (
                -(tau / (2.0 * nu * c * c))
                * jnp.power(jnp.abs(z / c), tau - 2.0)
                * z
                * ((nu * z + 1.0 / sigma) * log_ratio - z)
            )
            return value + log_ratio + jnp.sign(nu) * h / (sigma * nu * nu)

        d_log_c_dt = (1.0 / (2.0 * tau * tau)) * (
            2.0 * math.log(2.0) - _digamma(1.0 / tau) + 3.0 * _digamma(3.0 / tau)
        )

        arg = 1.0 / (sigma * jnp.maximum(jnp.abs(nu), eps))

        # R differentiates log F.T in tau with a 0.001 forward step; copied
        # verbatim because a more accurate j would drift away from R's dldt.
        def log_ft_tau(tau_value: jnp.ndarray) -> jnp.ndarray:
            c_val = jnp.maximum(_bcpe_c(tau_value), eps)
            s = 0.5 * jnp.power(jnp.abs(arg) / c_val, tau_value)
            cdf = 0.5 * (1.0 + _gammainc(1.0 / tau_value, s) * jnp.sign(arg))
            return jnp.log(jnp.maximum(cdf, eps))

        j = (log_ft_tau(tau + 0.001) - log_ft_tau(tau)) / 0.001
        return (
            (1.0 / tau)
            - 0.5 * jnp.log(jnp.abs(z / c)) * jnp.power(jnp.abs(z / c), tau)
            + (1.0 / (tau * tau)) * (math.log(2.0) + _digamma(1.0 / tau))
            + ((tau / 2.0) * jnp.power(jnp.abs(z / c), tau) - 1.0) * d_log_c_dt
            - j
        )

    def _hessian(
        parameter: str,
        y: jnp.ndarray,
        mu: jnp.ndarray,
        sigma: jnp.ndarray,
        nu: jnp.ndarray,
        tau: jnp.ndarray,
    ) -> jnp.ndarray:
        """R BCPE.R's diagonal second derivatives, transcribed verbatim.

        ``d2ldm2`` carries R's branch ``if (any(tau < 1.05)) -dldm^2``, which
        switches to the squared score when the curvature formula degenerates.
        The branch is on a *scalar* tau, so it is expressed with ``jnp.where``
        on the whole array -- R's vectorised ``ifelse`` would collapse.
        """
        from jax.scipy.special import digamma as _digamma, polygamma as _polygamma

        y = jnp.maximum(jnp.asarray(y, dtype=jnp.float64), eps)
        mu = jnp.maximum(jnp.asarray(mu, dtype=jnp.float64), eps)
        sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), eps)
        nu = jnp.asarray(nu, dtype=jnp.float64)
        tau = jnp.maximum(jnp.asarray(tau, dtype=jnp.float64), eps)

        if parameter == "mu":
            value = -(tau * tau) * jnp.exp(
                gammaln(2.0 - 1.0 / tau) + gammaln(3.0 / tau) - 2.0 * gammaln(1.0 / tau)
            ) / (mu * mu * sigma * sigma) - (tau * nu * nu) / (mu * mu)
            # R: if (any(tau < 1.05)) -dldm^2  else the closed form above.
            # The fallback is the *squared score*, so it has to come from
            # _score("mu", ...), not from a placeholder -- that recursion is
            # what makes the degenerate branch meaningful.
            dldm = _score("mu", y, mu, sigma, nu, tau)
            return jnp.where(tau < 1.05, -dldm * dldm, value)
        if parameter == "sigma":
            return -tau / (sigma * sigma)
        if parameter == "nu":
            return -sigma * sigma * (3.0 * tau + 1.0) / 4.0

        p = (tau + 1.0) / tau
        d_log_c_dt = (1.0 / (2.0 * tau * tau)) * (
            2.0 * math.log(2.0) - _digamma(1.0 / tau) + 3.0 * _digamma(3.0 / tau)
        )
        part1 = p * _polygamma(1, p) + 2.0 * _digamma(p) ** 2
        part2 = _digamma(p) * (math.log(2.0) + 3.0 - 3.0 * _digamma(3.0 / tau) - tau)
        part3 = -3.0 * _digamma(3.0 / tau) * (1.0 + math.log(2.0))
        part4 = -(tau + math.log(2.0)) * math.log(2.0)
        part5 = -tau + (tau**4) * d_log_c_dt**2
        value = -(part1 + part2 + part3 + part4 + part5) / (tau * tau * tau)
        return jnp.where(value < -1e-15, value, -1e-15)

    return BoxCoxPowerExponentialFamily(
        name="BCPE",
        parameters=("mu", "sigma", "nu", "tau"),
        g_dev_inc=g_dev_inc,
        type="Continuous",
        # gamlss.dist::BCPE(): mu.link = "identity", sigma.link = "log",
        # nu.link = "identity", tau.link = "log".  mu was "log" here.
        links={"mu": "identity", "sigma": "log", "nu": "identity", "tau": "log"},
        link_functions={
            "mu": identity_link,
            "sigma": log_link,
            "nu": identity_link,
            "tau": tau_link,
        },
        link_inverses={
            "mu": identity_inverse,
            "sigma": log_inverse,
            "nu": identity_inverse,
            "tau": tau_inverse,
        },
        link_derivatives={
            "mu": identity_derivative,
            "sigma": log_derivative,
            "nu": identity_derivative,
            "tau": tau_derivative,
        },
        score_functions={
            "mu": lambda y, mu, sigma, nu, tau: _score("mu", y, mu, sigma, nu, tau),
            "sigma": lambda y, mu, sigma, nu, tau: _score(
                "sigma", y, mu, sigma, nu, tau
            ),
            "nu": lambda y, mu, sigma, nu, tau: _score("nu", y, mu, sigma, nu, tau),
            "tau": lambda y, mu, sigma, nu, tau: _score("tau", y, mu, sigma, nu, tau),
        },
        hessian_functions={
            "mu": lambda y, mu, sigma, nu, tau: _hessian("mu", y, mu, sigma, nu, tau),
            "sigma": lambda y, mu, sigma, nu, tau: _hessian(
                "sigma", y, mu, sigma, nu, tau
            ),
            "nu": lambda y, mu, sigma, nu, tau: _hessian("nu", y, mu, sigma, nu, tau),
            "tau": lambda y, mu, sigma, nu, tau: _hessian("tau", y, mu, sigma, nu, tau),
        },
        d=dBCPE,
        p=pBCPE,
        q=qBCPE,
        r=rBCPE,
    )


def _resolve_family_legacy(family_name: str) -> FamilyDefinition:
    """Deprecated compatibility shim for historical private resolver imports.

    Family lookup remains registry-backed; this private helper no longer owns
    a separate if/elif resolver table.
    """

    from .distribution_registry import resolve

    return resolve(family_name)


def resolve_family(family: str | FamilyDefinition | None) -> FamilyDefinition:
    """Resolve a family name/object into a FamilyDefinition instance."""

    if family is None:
        return NO()

    from .distribution_registry import resolve

    return resolve(family)
