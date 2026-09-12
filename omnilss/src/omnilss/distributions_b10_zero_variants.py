"""Zero-Inflated and Zero-Altered distribution variants.

This module provides systematic implementations of ZA* and ZI* distributions
by wrapping existing base distributions.
"""

from __future__ import annotations
from dataclasses import dataclass, replace
import jax.numpy as jnp
import jax

from .ad import build_ad_family
from .families import FamilyDefinition
from .links import logit_link, logit_inverse, logit_derivative


def make_zero_altered_family(base_name: str, base_log_pdf_func, base_params: tuple, base_links: dict):
    """Factory to create zero-altered distributions.
    
    Zero-altered (hurdle): Pr(Y=0) = nu, Pr(Y=y|y>0) = (1-nu)*f(y)/[1-f(0)]
    """
    
    @dataclass(frozen=True)
    class ZeroAlteredFamily(FamilyDefinition):
        pass
    
    ZeroAlteredFamily.__name__ = f"ZeroAltered{base_name}Family"
    
    def za_log_pdf(y, *params):
        """Zero-altered log-PDF."""
        if base_name == "SICHEL":
            return _zero_sichel_log_pdf(y, *params, altered=True)
        # Last parameter is nu (zero probability)
        base_params_vals = params[:-1]
        nu = params[-1]
        
        eps = jnp.finfo(jnp.float64).eps
        nu = jnp.clip(nu, eps, 1.0 - eps)
        
        # Base distribution log-PDF for y > 0
        log_base = base_log_pdf_func(y, *base_params_vals)
        
        # f(0) for base distribution
        log_f0 = base_log_pdf_func(jnp.zeros_like(y), *base_params_vals)
        
        # Zero-altered probabilities
        log_at0 = jnp.log(nu)
        log_cont = jnp.log1p(-nu) + log_base - jnp.log1p(-jnp.exp(log_f0))
        
        return jnp.where(y <= 0, log_at0, log_cont)
    
    # Add the zero probability parameter.  R uses "nu" for the zero mass, but
    # some base families already have a "nu" shape parameter (e.g. SICHEL),
    # so use "tau" to keep the parameter tuple unique (Phase 4.4 contract fix).
    zero_param = "tau" if "nu" in base_params else "nu"
    za_params = base_params + (zero_param,)
    za_links = {**base_links, zero_param: "logit"}
    
    from .links import log_link, log_inverse, log_derivative, identity_link, identity_inverse, identity_derivative
    
    link_funcs = {}
    link_invs = {}
    link_derivs = {}
    
    for param in za_params:
        if param == zero_param:
            link_funcs[param] = logit_link
            link_invs[param] = logit_inverse
            link_derivs[param] = logit_derivative
        else:
            link_type = za_links.get(param, "identity")
            if link_type == "log":
                link_funcs[param] = log_link
                link_invs[param] = log_inverse
                link_derivs[param] = log_derivative
            elif link_type == "logit":
                link_funcs[param] = logit_link
                link_invs[param] = logit_inverse
                link_derivs[param] = logit_derivative
            else:  # identity
                link_funcs[param] = identity_link
                link_invs[param] = identity_inverse
                link_derivs[param] = identity_derivative
    
    return build_ad_family(
        family_class=ZeroAlteredFamily,
        name=f"ZA{base_name}",
        parameters=za_params,
        log_pdf_func=za_log_pdf,
        type_="Mixed",
        links=za_links,
        link_functions=link_funcs,
        link_inverses=link_invs,
        link_derivatives=link_derivs,
    )


def make_zero_inflated_family(base_name: str, base_log_pdf_func, base_params: tuple, base_links: dict):
    """Factory to create zero-inflated distributions.
    
    Zero-inflated: Pr(Y=0) = nu + (1-nu)*f(0), Pr(Y=y|y>0) = (1-nu)*f(y)
    """
    
    @dataclass(frozen=True)
    class ZeroInflatedFamily(FamilyDefinition):
        pass
    
    ZeroInflatedFamily.__name__ = f"ZeroInflated{base_name}Family"
    
    def zi_log_pdf(y, *params):
        """Zero-inflated log-PDF."""
        if base_name == "SICHEL":
            return _zero_sichel_log_pdf(y, *params, altered=False)
        base_params_vals = params[:-1]
        nu = params[-1]
        
        eps = jnp.finfo(jnp.float64).eps
        nu = jnp.clip(nu, eps, 1.0 - eps)
        
        # Base distribution log-PDF
        log_base = base_log_pdf_func(y, *base_params_vals)
        
        # f(0) for base distribution
        log_f0 = base_log_pdf_func(jnp.zeros_like(y), *base_params_vals)
        
        # Zero-inflated probabilities
        log_at0 = jnp.log(nu + (1.0 - nu) * jnp.exp(log_f0))
        log_cont = jnp.log1p(-nu) + log_base
        
        return jnp.where(y <= 0, log_at0, log_cont)
    
    zero_param = "tau" if "nu" in base_params else "nu"
    zi_params = base_params + (zero_param,)
    zi_links = {**base_links, zero_param: "logit"}
    
    from .links import log_link, log_inverse, log_derivative, identity_link, identity_inverse, identity_derivative
    
    link_funcs = {}
    link_invs = {}
    link_derivs = {}
    
    for param in zi_params:
        if param == zero_param:
            link_funcs[param] = logit_link
            link_invs[param] = logit_inverse
            link_derivs[param] = logit_derivative
        else:
            link_type = zi_links.get(param, "identity")
            if link_type == "log":
                link_funcs[param] = log_link
                link_invs[param] = log_inverse
                link_derivs[param] = log_derivative
            elif link_type == "logit":
                link_funcs[param] = logit_link
                link_invs[param] = logit_inverse
                link_derivs[param] = logit_derivative
            else:
                link_funcs[param] = identity_link
                link_invs[param] = identity_inverse
                link_derivs[param] = identity_derivative
    
    return build_ad_family(
        family_class=ZeroInflatedFamily,
        name=f"ZI{base_name}",
        parameters=zi_params,
        log_pdf_func=zi_log_pdf,
        type_="Discrete",
        links=zi_links,
        link_functions=link_funcs,
        link_inverses=link_invs,
        link_derivatives=link_derivs,
    )


# Now create the variants using existing distributions

# Import base distributions
from .distributions_b6 import _sichel_log_pdf, _pig_log_pdf
from .distributions_b7 import _bb_log_pdf
from .distributions_b9 import _lg_log_pdf, _zipf_log_pdf

# Define binomial log-PDF (not exported from distributions.py)
def _binomial_log_pdf(y, mu):
    """Binomial log-PDF for Bernoulli case."""
    import jax.numpy as jnp
    eps = jnp.finfo(jnp.float64).eps
    mu = jnp.clip(mu, eps, 1.0 - eps)
    return y * jnp.log(mu) + (1.0 - y) * jnp.log1p(-mu)

def _zero_sichel_log_pdf(x, mu, sigma, nu, tau, *, altered):
    """Shared scalar PMF kernel for density and family likelihoods."""
    x, mu, sigma, nu, tau = map(jnp.asarray, (x, mu, sigma, nu, tau))
    valid = ((mu > 0) & (sigma > 0) & jnp.isfinite(mu)
             & jnp.isfinite(sigma) & jnp.isfinite(nu)
             & (tau >= 0) & (tau <= 1))
    log_base = _sichel_log_pdf(x, mu, sigma, nu)
    log_f0 = _sichel_log_pdf(0., mu, sigma, nu)
    positive = jnp.log1p(-tau) + log_base
    if altered:
        positive = positive - jnp.log(-jnp.expm1(log_f0))
        at_zero = jnp.log(tau)
    else:
        at_zero = jnp.logaddexp(jnp.log(tau), jnp.log1p(-tau) + log_f0)
    value = jnp.where(x == 0, at_zero, positive)
    support = (x >= 0) & jnp.isfinite(x) & (x == jnp.floor(x))
    value = jnp.where(support, value, -jnp.inf)
    undefined = jnp.isnan(x) | (support & (x > 10000))
    return jnp.where(valid & ~undefined, value, jnp.nan)


def _zero_sichel_density(x, mu, sigma, nu, tau, *, altered, log):
    args = jnp.broadcast_arrays(*[
        jnp.asarray(v, dtype=jnp.float64) for v in (x, mu, sigma, nu, tau)
    ])
    values = jax.vmap(lambda *v: _zero_sichel_log_pdf(*v, altered=altered))(
        *(v.ravel() for v in args)).reshape(args[0].shape)
    return values if log else jnp.exp(values)


# ZASICHEL - Zero-Altered Sichel
def ZASICHEL():
    """Zero-Altered Sichel distribution."""
    family = make_zero_altered_family(
        "SICHEL",
        _sichel_log_pdf,
        ("mu", "sigma", "nu"),
        {"mu": "log", "sigma": "log", "nu": "identity"}
    )
    from .dpqr_functions import pSICHEL, qSICHEL
    def p(q, mu=1., sigma=1., nu=-.5, tau=.5, lower_tail=True, log_p=False):
        base_log = pSICHEL(q, mu, sigma, nu, log_p=True)
        f0_log = pSICHEL(0., mu, sigma, nu, log_p=True)
        valid = (tau >= 0) & (tau <= 1) & jnp.isfinite(tau)
        f0 = jnp.exp(f0_log)
        base = jnp.exp(base_log)
        value = jnp.where(q < 0, 0., tau + (1. - tau) * (base - f0) / (1. - f0))
        value = jnp.where(q >= jnp.inf, 1., value)
        if not lower_tail:
            value = jnp.where(q <= 0, 1. - tau, (1. - tau) * (1. - base) / (1. - f0))
        return jnp.where(valid, jnp.log(value) if log_p else value, jnp.nan)
    def q(prob, mu=1., sigma=.5, nu=-.5, tau=.5, lower_tail=True, log_p=False):
        prob = jnp.exp(prob) if log_p else jnp.asarray(prob)
        prob = prob if lower_tail else 1. - prob
        f0 = pSICHEL(0., mu, sigma, nu)
        denom = jnp.where(tau == 1., 1., 1. - tau)
        target = f0 + (1. - f0) * (prob - tau) / denom
        target = jnp.where(prob == 1., 1., target)
        valid = (tau >= 0) & (tau <= 1) & jnp.isfinite(tau) & (prob >= 0) & (prob <= 1)
        return jnp.where(valid, jnp.where(prob <= tau, 0., qSICHEL(target, mu, sigma, nu)), jnp.nan)
    def r(key, n, mu=1., sigma=.5, nu=-.5, tau=.5):
        if not isinstance(n, int) or n < 0:
            raise ValueError('n must be a nonnegative integer')
        mu, sigma, nu, tau = jnp.broadcast_arrays(*map(jnp.asarray, (mu, sigma, nu, tau)))
        return q(jax.random.uniform(key, (n,) + mu.shape), mu, sigma, nu, tau)
    def d(x, mu=1., sigma=1., nu=-.5, tau=.5, log=False):
        return _zero_sichel_density(x, mu, sigma, nu, tau, altered=True, log=log)
    return replace(family, d=d, p=p, q=q, r=r)

# ZISICHEL - Zero-Inflated Sichel  
def ZISICHEL():
    """Zero-Inflated Sichel distribution."""
    family = make_zero_inflated_family(
        "SICHEL",
        _sichel_log_pdf,
        ("mu", "sigma", "nu"),
        {"mu": "log", "sigma": "log", "nu": "identity"}
    )
    from .dpqr_functions import pSICHEL, qSICHEL
    def p(q, mu=1., sigma=1., nu=-.5, tau=.5, lower_tail=True, log_p=False):
        base_log = pSICHEL(q, mu, sigma, nu, log_p=True)
        base = jnp.exp(base_log)
        value = jnp.where(q < 0, 0., tau + (1. - tau) * base)
        value = jnp.where(q >= jnp.inf, 1., value)
        if not lower_tail:
            value = jnp.where(q < 0, 1., (1. - tau) * (1. - base))
        return jnp.where((tau >= 0) & (tau <= 1) & jnp.isfinite(tau),
                         jnp.log(value) if log_p else value, jnp.nan)
    def q(prob, mu=1., sigma=.5, nu=-.5, tau=.5, lower_tail=True, log_p=False):
        prob = jnp.exp(prob) if log_p else jnp.asarray(prob)
        prob = prob if lower_tail else 1. - prob
        f0 = pSICHEL(0., mu, sigma, nu)
        threshold = tau + (1. - tau) * f0
        denom = jnp.where(tau == 1., 1., 1. - tau)
        target = (prob - tau) / denom
        target = jnp.where(prob == 1., 1., target)
        valid = (tau >= 0) & (tau <= 1) & jnp.isfinite(tau) & (prob >= 0) & (prob <= 1)
        return jnp.where(valid, jnp.where(prob <= threshold, 0., qSICHEL(target, mu, sigma, nu)), jnp.nan)
    def r(key, n, mu=1., sigma=.5, nu=-.5, tau=.5):
        if not isinstance(n, int) or n < 0:
            raise ValueError('n must be a nonnegative integer')
        mu, sigma, nu, tau = jnp.broadcast_arrays(*map(jnp.asarray, (mu, sigma, nu, tau)))
        return q(jax.random.uniform(key, (n,) + mu.shape), mu, sigma, nu, tau)
    def d(x, mu=1., sigma=1., nu=-.5, tau=.5, log=False):
        return _zero_sichel_density(x, mu, sigma, nu, tau, altered=False, log=log)
    return replace(family, d=d, p=p, q=q, r=r)

# ZAPIG - Zero-Altered Poisson Inverse Gaussian
def ZAPIG():
    """Zero-Altered Poisson Inverse Gaussian distribution."""
    return make_zero_altered_family(
        "PIG",
        _pig_log_pdf,
        ("mu", "sigma"),
        {"mu": "log", "sigma": "log"}
    )

# ZIPIG - Zero-Inflated Poisson Inverse Gaussian
def ZIPIG():
    """Zero-Inflated Poisson Inverse Gaussian distribution."""
    return make_zero_inflated_family(
        "PIG",
        _pig_log_pdf,
        ("mu", "sigma"),
        {"mu": "log", "sigma": "log"}
    )

# ZABB - Zero-Altered Beta Binomial
# ZIBB - Zero-Inflated Beta Binomial
# NOTE: BB requires 'bd' (binomial denominator) parameter
# These need special handling with bd as a fixed parameter

def ZABB(bd=10):
    """Zero-Altered Beta Binomial distribution.
    
    Args:
        bd: Binomial denominator (fixed parameter, default=10)
    """
    from .distributions_b7 import _bb_log_pdf
    
    def zabb_log_pdf(y, mu, sigma, nu):
        """ZABB log-PDF with fixed bd."""
        eps = jnp.finfo(jnp.float64).eps
        nu = jnp.clip(nu, eps, 1.0 - eps)
        
        # Base BB log-PDF
        log_base = _bb_log_pdf(y, mu, sigma, bd)
        
        # f(0) for BB
        log_f0 = _bb_log_pdf(jnp.zeros_like(y), mu, sigma, bd)
        
        # Zero-altered probabilities
        log_at0 = jnp.log(nu)
        log_cont = jnp.log1p(-nu) + log_base - jnp.log1p(-jnp.exp(log_f0))
        
        return jnp.where(y <= 0, log_at0, log_cont)
    
    from .links import logit_link, logit_inverse, logit_derivative, log_link, log_inverse, log_derivative
    
    @dataclass(frozen=True)
    class ZABBFamily(FamilyDefinition):
        pass
    
    return build_ad_family(
        family_class=ZABBFamily,
        name="ZABB",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=zabb_log_pdf,
        type_="Discrete",
        links={"mu": "logit", "sigma": "log", "nu": "logit"},
        link_functions={
            "mu": logit_link,
            "sigma": log_link,
            "nu": logit_link,
        },
        link_inverses={
            "mu": logit_inverse,
            "sigma": log_inverse,
            "nu": logit_inverse,
        },
        link_derivatives={
            "mu": logit_derivative,
            "sigma": log_derivative,
            "nu": logit_derivative,
        },
    )


def ZIBB(bd=10):
    """Zero-Inflated Beta Binomial distribution.
    
    Args:
        bd: Binomial denominator (fixed parameter, default=10)
    """
    from .distributions_b7 import _bb_log_pdf
    
    def zibb_log_pdf(y, mu, sigma, nu):
        """ZIBB log-PDF with fixed bd."""
        eps = jnp.finfo(jnp.float64).eps
        nu = jnp.clip(nu, eps, 1.0 - eps)
        
        # Base BB log-PDF
        log_base = _bb_log_pdf(y, mu, sigma, bd)
        
        # f(0) for BB
        log_f0 = _bb_log_pdf(jnp.zeros_like(y), mu, sigma, bd)
        
        # Zero-inflated probabilities
        log_at0 = jnp.log(nu + (1.0 - nu) * jnp.exp(log_f0))
        log_cont = jnp.log1p(-nu) + log_base
        
        return jnp.where(y <= 0, log_at0, log_cont)
    
    from .links import logit_link, logit_inverse, logit_derivative, log_link, log_inverse, log_derivative
    
    @dataclass(frozen=True)
    class ZIBBFamily(FamilyDefinition):
        pass
    
    return build_ad_family(
        family_class=ZIBBFamily,
        name="ZIBB",
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=zibb_log_pdf,
        type_="Discrete",
        links={"mu": "logit", "sigma": "log", "nu": "logit"},
        link_functions={
            "mu": logit_link,
            "sigma": log_link,
            "nu": logit_link,
        },
        link_inverses={
            "mu": logit_inverse,
            "sigma": log_inverse,
            "nu": logit_inverse,
        },
        link_derivatives={
            "mu": logit_derivative,
            "sigma": log_derivative,
            "nu": logit_derivative,
        },
    )

# ZABI - Zero-Altered Binomial
def ZABI():
    """Zero-Altered Binomial distribution."""
    return make_zero_altered_family(
        "BI",
        _binomial_log_pdf,
        ("mu",),
        {"mu": "logit"}
    )

# ZIBI - Zero-Inflated Binomial
def ZIBI():
    """Zero-Inflated Binomial distribution."""
    return make_zero_inflated_family(
        "BI",
        _binomial_log_pdf,
        ("mu",),
        {"mu": "logit"}
    )

# ZAZIPF - Zero-Altered Zipf
def ZAZIPF():
    """Zero-Altered Zipf distribution."""
    return make_zero_altered_family(
        "ZIPF",
        _zipf_log_pdf,
        ("mu",),
        {"mu": "log"}
    )

# Note: ZIZIPF doesn't make sense as Zipf already starts at 1


# ============================================================================
# Additional Zero-Variants for NB and BNB
# ============================================================================

# Import additional base distributions
from .distributions_b7 import _bnb_log_pdf

# Helper to create NBI log-PDF wrapper
def _nbi_log_pdf(y, mu, sigma):
    """NBI log-PDF wrapper for zero-variant factories."""
    import jax.numpy as jnp
    from jax.scipy.special import gammaln
    
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


# ZANBI - Zero-Altered Negative Binomial
def ZANBI():
    """Zero-Altered Negative Binomial distribution."""
    return make_zero_altered_family(
        "NBI",
        _nbi_log_pdf,
        ("mu", "sigma"),
        {"mu": "log", "sigma": "log"}
    )

# ZINBI already exists in distributions.py, so we skip it here

# ZABNB - Zero-Altered Beta Negative Binomial
def ZABNB():
    """Zero-Altered Beta Negative Binomial distribution."""
    return make_zero_altered_family(
        "BNB",
        _bnb_log_pdf,
        ("mu", "sigma", "nu"),
        {"mu": "log", "sigma": "log", "nu": "log"}
    )

# ZIBNB - Zero-Inflated Beta Negative Binomial
def ZIBNB():
    """Zero-Inflated Beta Negative Binomial distribution."""
    return make_zero_inflated_family(
        "BNB",
        _bnb_log_pdf,
        ("mu", "sigma", "nu"),
        {"mu": "log", "sigma": "log", "nu": "log"}
    )

# ZINBF - Zero-Inflated Negative Binomial (Famoye parameterization)
# Note: This would require NBF distribution to be implemented first
# Skipping for now
