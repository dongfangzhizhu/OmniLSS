"""Zero-Inflated and Zero-Altered distribution variants.

This module provides systematic implementations of ZA* and ZI* distributions
by wrapping existing base distributions.
"""

from __future__ import annotations
from dataclasses import dataclass, replace
import jax.numpy as jnp
import jax
import jax.random as jrandom

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
    
    extra = {}
    if base_name == "BNB":
        from .dpqr_functions import pBNB, qBNB, rBNB

        def zi_p(q, *params, lower_tail=True, log_p=False):
            base = params[:-1]
            tau = jnp.clip(params[-1], 0.0, 1.0)
            prob = pBNB(q, *base, lower_tail=True, log_p=False)
            f0 = pBNB(0.0, *base, lower_tail=True, log_p=False)
            out = jnp.where(q < 0, 0.0, jnp.where(q < 0.5, tau + (1-tau)*f0, tau + (1-tau)*prob))
            if not lower_tail:
                out = 1.0 - out
            return jnp.log(out) if log_p else out

        def zi_q(p, *params, lower_tail=True, log_p=False):
            base = params[:-1]
            tau = jnp.clip(params[-1], 0.0, 1.0)
            prob = jnp.exp(p) if log_p else p
            prob = 1.0 - prob if not lower_tail else prob
            invalid = (prob < 0) | (prob > 1) | jnp.isnan(prob)
            at_zero = prob == 0
            at_one = prob == 1
            base0 = pBNB(0.0, *base)
            zi0 = tau + (1.0 - tau) * base0
            base_prob = (prob - tau) / (1.0 - tau)
            result = jnp.where(prob <= zi0, 0.0, qBNB(base_prob, *base))
            result = jnp.where(at_zero, 0.0, result)
            result = jnp.where(at_one, jnp.inf, result)
            return jnp.where(invalid, jnp.nan, result)

        def zi_r(key, n, *params):
            base = params[:-1]
            tau = jnp.clip(params[-1], 0.0, 1.0)
            tau = jnp.asarray(tau)
            batch_shape = jnp.broadcast_shapes(*[jnp.asarray(v).shape for v in (*base, tau)])
            base = jnp.broadcast_arrays(*[jnp.asarray(v) for v in base])
            tau = jnp.broadcast_to(tau, batch_shape)
            k1, k2 = jrandom.split(key)
            inflated = jrandom.uniform(k1, (n,) + batch_shape) < tau
            base_samples = rBNB(k2, n, *base)
            return jnp.where(inflated | (tau == 1), 0, base_samples)
        extra = {"p": zi_p, "q": zi_q, "r": zi_r}

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
        **extra,
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
    # The point mass needs no base-density approximation or support cutoff.
    value = jnp.where(tau == 1, jnp.where(x == 0, 0., -jnp.inf), value)
    from .distributions_b6 import SICHEL_DENSITY_MAX_COUNT
    undefined = (jnp.isnan(x) | (support & (x > SICHEL_DENSITY_MAX_COUNT)
                                 & (tau != 1)))
    return jnp.where(valid & ~undefined, value, jnp.nan)


def _zero_sichel_density(x, mu, sigma, nu, tau, *, altered, log):
    args = jnp.broadcast_arrays(*[
        jnp.asarray(v, dtype=jnp.float64) for v in (x, mu, sigma, nu, tau)
    ])
    values = jax.vmap(lambda *v: _zero_sichel_log_pdf(*v, altered=altered))(
        *(v.ravel() for v in args)).reshape(args[0].shape)
    return values if log else jnp.exp(values)


def _zero_sichel_log_mass_sequence(mu, sigma, nu, tau, *, altered, max_count):
    """Log PMF of a zero variant for counts 0..max_count in one base scan.

    Equivalent to calling `_zero_sichel_log_pdf` on every count, but the
    base SICHEL recursion runs once instead of once per candidate, which
    keeps probability/quantile scans linear in the support budget.
    """
    from .distributions_b6 import _sichel_log_pmf_sequence
    base = _sichel_log_pmf_sequence(mu, sigma, nu, max_count)
    log_f0 = base[0]
    log_positive = jnp.log1p(-tau) + base
    if altered:
        log_positive = log_positive - jnp.log(-jnp.expm1(log_f0))
        at_zero = jnp.log(tau)
    else:
        at_zero = jnp.logaddexp(jnp.log(tau), jnp.log1p(-tau) + log_f0)
    counts = jnp.arange(max_count + 1, dtype=jnp.float64)
    mass = jnp.concatenate((jnp.reshape(at_zero, (1,)), log_positive[1:]))
    # A tau of 1 collapses everything onto the point mass at zero.
    mass = jnp.where(tau == 1, jnp.where(counts == 0, 0., -jnp.inf), mass)
    return mass


def _zero_sichel_probability(value, mu, sigma, nu, tau, *, altered,
                             lower_tail, log_p, quantile=False,
                             max_value=100000):
    """Evaluate tails/invert them directly, without probability complements.

    Like SICHEL, the numerical support is limited to the configured
    probability budget. Quantile scans use linear space rather than a square
    cumulative-sum matrix. max_value is a static Python integer in the same
    configured range limiting candidates,
    not upper-tail summation. Unresolved interior probabilities return NaN.
    """
    from .distributions_b6 import SICHEL_PROBABILITY_MAX_COUNT
    if isinstance(max_value, bool) or not isinstance(max_value, int) or not 0 <= max_value <= SICHEL_PROBABILITY_MAX_COUNT:
        raise ValueError(f"max_value must be an integer in [0, {SICHEL_PROBABILITY_MAX_COUNT}]")
    args = jnp.broadcast_arrays(*[
        jnp.asarray(v, dtype=jnp.float64) for v in (value, mu, sigma, nu, tau)
    ])
    # Upper tails still need masses beyond the candidate search cap.
    support = jnp.arange((max_value if lower_tail else SICHEL_PROBABILITY_MAX_COUNT) + 1, dtype=jnp.float64)

    def scalar(v, mm, ss, nn, tt):
        mass = _zero_sichel_log_mass_sequence(
            mm, ss, nn, tt, altered=altered, max_count=support.shape[0] - 1)
        valid = ((mm > 0) & (ss > 0) & jnp.isfinite(mm)
                 & jnp.isfinite(ss) & jnp.isfinite(nn)
                 & (tt >= 0) & (tt <= 1) & ~jnp.isnan(v))
        if quantile:
            valid = valid & ((v <= 0) if log_p else ((v >= 0) & (v <= 1)))
            target = v if log_p else jnp.log(v)
            if lower_tail:
                cumulative = jax.lax.associative_scan(jnp.logaddexp, mass)
            else:
                inclusive = jax.lax.associative_scan(jnp.logaddexp, mass, reverse=True)
                cumulative = jnp.concatenate((inclusive[1:], jnp.array([-jnp.inf])))
            # Only compensate for floating-point roundoff at discrete jumps.
            tolerance = 8 * jnp.finfo(jnp.float64).eps * jnp.maximum(1., jnp.abs(target))
            tolerance = jnp.where(jnp.isfinite(target), tolerance, 0.)
            reached = (cumulative >= target - tolerance) if lower_tail else (cumulative <= target + tolerance)
            reached = reached & (support <= max_value)
            result = jnp.where(jnp.any(reached), jnp.argmax(reached).astype(jnp.float64), jnp.nan)
            at_zero = jnp.isneginf(target) if lower_tail else (target == 0)
            at_infinity = (target == 0) if lower_tail else jnp.isneginf(target)
            result = jnp.where(at_zero | (tt == 1), 0., jnp.where(at_infinity, jnp.inf, result))
        else:
            selected = (support <= v) if lower_tail else (support > v)
            result = jax.scipy.special.logsumexp(jnp.where(selected, mass, -jnp.inf))
            result = jnp.minimum(result, 0.)
            result = jnp.where(v < 0, -jnp.inf if lower_tail else 0., result)
            result = jnp.where(jnp.isposinf(v), 0. if lower_tail else -jnp.inf, result)
            point_mass = jnp.where(v < 0, -jnp.inf, 0.) if lower_tail else jnp.where(v < 0, 0., -jnp.inf)
            result = jnp.where(tt == 1, point_mass, result)
            valid = valid & ~((v > max_value) & jnp.isfinite(v) & (tt != 1))
            result = result if log_p else jnp.exp(result)
        return jnp.where(valid, result, jnp.nan)

    return jax.vmap(scalar)(*(v.ravel() for v in args)).reshape(args[0].shape)


# ZASICHEL - Zero-Altered Sichel
def ZASICHEL():
    """Zero-Altered Sichel distribution."""
    family = make_zero_altered_family(
        "SICHEL",
        _sichel_log_pdf,
        ("mu", "sigma", "nu"),
        {"mu": "log", "sigma": "log", "nu": "identity"}
    )
    def p(q, mu=1., sigma=1., nu=-.5, tau=.5, lower_tail=True, log_p=False):
        return _zero_sichel_probability(q, mu, sigma, nu, tau, altered=True,
                                        lower_tail=lower_tail, log_p=log_p)
    def q(prob, mu=1., sigma=.5, nu=-.5, tau=.5, lower_tail=True, log_p=False, max_value=100000):
        return _zero_sichel_probability(prob, mu, sigma, nu, tau, altered=True,
                                        lower_tail=lower_tail, log_p=log_p, quantile=True, max_value=max_value)
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
    def p(q, mu=1., sigma=1., nu=-.5, tau=.5, lower_tail=True, log_p=False):
        return _zero_sichel_probability(q, mu, sigma, nu, tau, altered=False,
                                        lower_tail=lower_tail, log_p=log_p)
    def q(prob, mu=1., sigma=.5, nu=-.5, tau=.5, lower_tail=True, log_p=False, max_value=100000):
        return _zero_sichel_probability(prob, mu, sigma, nu, tau, altered=False,
                                        lower_tail=lower_tail, log_p=log_p, quantile=True, max_value=max_value)
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
    from .zabb import dZABB, pZABB, qZABB, rZABB, _check_bd
    _check_bd(bd)
    
    def zabb_log_pdf(y, mu, sigma, nu):
        """ZABB log-PDF with fixed bd."""
        return dZABB(y,mu,sigma,nu,bd,log=True)
    
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
        d=lambda x,mu=.5,sigma=.1,nu=.1,log=False: dZABB(x,mu,sigma,nu,bd,log),
        p=lambda q,mu=.5,sigma=.1,nu=.1,lower_tail=True,log_p=False: pZABB(q,mu,sigma,nu,bd,lower_tail,log_p),
        q=lambda p,mu=.5,sigma=.1,nu=.1,lower_tail=True,log_p=False: qZABB(p,mu,sigma,nu,bd,lower_tail,log_p),
        r=lambda key,n,mu=.5,sigma=.1,nu=.1: rZABB(key,n,mu,sigma,nu,bd),
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
    from .zibb import dZIBB, pZIBB, qZIBB, rZIBB
    
    def zibb_log_pdf(y, mu, sigma, nu):
        """ZIBB log-PDF with fixed bd."""
        return dZIBB(y,mu,sigma,nu,bd,log=True)
    
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
        d=lambda x,mu=.5,sigma=.5,nu=.1,log=False: dZIBB(x,mu,sigma,nu,bd,log),
        p=lambda q,mu=.5,sigma=.5,nu=.1,lower_tail=True,log_p=False: pZIBB(q,mu,sigma,nu,bd,lower_tail,log_p),
        q=lambda p,mu=.5,sigma=.5,nu=.1,lower_tail=True,log_p=False: qZIBB(p,mu,sigma,nu,bd,lower_tail,log_p),
        r=lambda key,n,mu=.5,sigma=.5,nu=.1: rZIBB(key,n,mu,sigma,nu,bd),
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
