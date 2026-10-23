"""Zero-Inflated and Zero-Altered distribution variants.

This module provides systematic implementations of ZA* and ZI* distributions
by wrapping existing base distributions.
"""

from __future__ import annotations
from dataclasses import dataclass, replace
import jax.numpy as jnp
import jax
import jax.random as jrandom
import numpy as np
import numpy as np

from .ad import build_ad_family, _prepare_jax_args
from .families import FamilyDefinition
from .links import logit_link, logit_inverse, logit_derivative


def _base_response_type(base_name: str) -> str:
    """Return the response ``type`` a ZA/ZI variant inherits from its base.

    R's rule, read off live R for all 22 zero-variant families (D2.25): a
    variant takes the base family's ``type`` when the base is discrete, and is
    marked ``"Mixed"`` when the base is continuous -- gluing a point mass at
    zero onto a positive continuous density is exactly what "Mixed" means.
    ``ZAPIG()$type`` is ``"Discrete"`` with ``y.valid = all(y >= 0)``, while
    ``ZAGA()$type`` is ``"Mixed"``.

    This was previously hard-coded to ``"Mixed"`` for zero-*altered* families
    and ``"Discrete"`` for zero-*inflated* ones.  Both constants were wrong for
    some base: the six ``ZA`` families over a discrete base (``ZAPIG``,
    ``ZAZIPF``, ``ZABI``, ``ZANBI``, ``ZABNB``, ``ZASICHEL``) came out
    ``Mixed`` where R says ``Discrete``, while their ``ZI`` siblings came out
    right -- the ZA/ZI split was never the distinction R draws.  The
    consequence was not cosmetic: a ``Mixed`` response grid is continuous, the
    discrete ``PIG`` log-pdf returns ``-inf`` off the integers, and the RS
    deviance was ``inf`` from step 1.  Deriving the type from the base makes
    that whole class of mismatch unrepresentable.

    Resolution failures are deliberately *not* swallowed: an unknown base name
    must fail loudly here rather than silently inherit a default.  An
    unrecognised base ``type`` likewise raises -- a silent fallback is the
    defect being fixed.
    """
    from .distribution_registry import resolve as _resolve

    base_type = _resolve(base_name).type
    if base_type == "Discrete":
        return "Discrete"
    if base_type == "Continuous":
        return "Mixed"
    raise ValueError(
        f"cannot derive a response type for a zero-variant over {base_name!r}: "
        f"base type is {base_type!r}, expected 'Discrete' or 'Continuous'"
    )


# R subtracts a small epsilon from the rescaled probability before inverting it
# through the base quantile function.  The value is a per-family constant in R,
# not a derived quantity: sixteen of the eighteen `q` functions read off live R
# subtract 1e-10, while `qZIBNB` (and `qZINBI`, which does not come from the
# factory) subtract 1e-7.  Parity means reproducing the constant, so the two
# values are named rather than smoothed into one.
_QUANTILE_EPSILON = 1e-10
_QUANTILE_EPSILON_1E7 = 1e-7
#: Families whose live-R `q` subtracts 1e-7 instead of 1e-10.
_ZI_FAMILIES_WITH_THE_COARSE_EPSILON = frozenset({"ZIBNB"})


def _zero_variant_tails(base_name: str, *, altered: bool, epsilon: float):
    """Compose a zero variant's ``p``/``q``/``r`` from its base family's.

    R derives both variants' distributions from the base CDF by a closed-form
    affine map, and every one of the eighteen families that has a ``p`` in
    ``gamlss.dist`` uses one of exactly two formulas (read off live R 4.6.0):

    **zero-altered (hurdle)** -- the positive support is the base conditioned
    on ``Y > 0``, with a point mass ``tau`` glued back at zero::

        p(q) = tau                                     if q == 0
             = tau + (1-tau)*(F(q) - F(0))/(1 - F(0))   otherwise

    **zero-inflated** -- the base CDF scaled into ``[tau, 1]``::

        p(q) = tau + (1-tau)*F(q)

    Inverting the second is direct; inverting the first additionally has to
    undo the conditioning, which R writes as ``pnew2 <- F(0)*(1-pnew) + pnew``.
    Note that ``ZAZIPF`` -- whose base ``ZIPF`` starts at 1, so ``F(0) == 0`` --
    collapses onto the second formula, and the composition below reproduces it
    without a special case.  That is the reason to compose from the base CDF
    rather than to transcribe each family: the degenerate base falls out.

    The tail flags follow R's order, which is not the order a reader expects
    (``ZAPIG.R`` and the vendored snapshot alike)::

        cdf <- ifelse(q == 0, tau, cdf3)
        if (!lower.tail) cdf <- 1 - cdf
        if (log.p)      cdf <- log(cdf)
        cdf <- ifelse(q < 0, 0, cdf)     # <- after the log

    so ``p(-1, log.p = TRUE)`` is ``0`` rather than ``-Inf``.  Transcribing the
    flags in the "natural" order would put a ``-Inf`` where R has a ``0``.

    Parameters mirror the base family's parameter order followed by the zero
    mass, which is the order the family objects themselves use.
    """
    from .distribution_registry import resolve as _resolve

    base = _resolve(base_name)
    base_p, base_q = base.p, base.q
    # Keyword dispatch, not positional: the base `p`/`q` signatures interleave
    # fixed parameters (`BI.p(q, mu, bd)`), so a positional call can silently
    # hand a parameter to the wrong slot.
    base_names = tuple(base.parameters)

    def variant_p(x, *params, lower_tail=True, log_p=False):
        base_kwargs = dict(zip(base_names, params[:-1]))
        tau = params[-1]
        x = jnp.asarray(x, dtype=jnp.float64)
        cdf = base_p(x, **base_kwargs)
        if altered:
            # R evaluates the base CDF at a zero of the same shape as q, so a
            # vectorised parameter set gives one hurdle point per element.
            cdf0 = base_p(jnp.zeros_like(x), **base_kwargs)
            cdf = tau + (1.0 - tau) * (cdf - cdf0) / (1.0 - cdf0)
            cdf = jnp.where(x == 0, tau, cdf)
        else:
            cdf = tau + (1.0 - tau) * cdf
        if not lower_tail:
            cdf = 1.0 - cdf
        if log_p:
            cdf = jnp.log(cdf)
        # R clamps *after* taking the log; see the docstring.
        return jnp.where(x < 0, 0.0, cdf)

    def variant_q(p, *params, lower_tail=True, log_p=False):
        base_kwargs = dict(zip(base_names, params[:-1]))
        tau = params[-1]
        prob = jnp.exp(jnp.asarray(p, dtype=jnp.float64)) if log_p else jnp.asarray(p, dtype=jnp.float64)
        prob = prob if lower_tail else 1.0 - prob
        rescaled = (prob - tau) / (1.0 - tau) - epsilon
        if altered:
            cdf0 = base_p(jnp.zeros_like(rescaled), **base_kwargs)
            rescaled = cdf0 * (1.0 - rescaled) + rescaled
        rescaled = jnp.where(rescaled > 0, rescaled, 0.0)
        result = base_q(rescaled, **base_kwargs)
        # R's `q` functions abort on p outside [0, 1]; vectorised code cannot,
        # and a quantile there is undefined, so report NaN as pBNI/qZIPF do.
        return jnp.where((prob >= 0) & (prob <= 1), result, jnp.nan)

    def variant_r(key, n, *params):
        # Every R `r` for these families is `as.integer(q(runif(n)))`, including
        # the hurdle ones -- inversion, not a mixture draw.  The one exception is
        # the BNB block this replaced, which drew a mixture directly; both agree
        # in distribution, and inversion is what the other eight R functions do.
        return variant_q(jrandom.uniform(key, (n,), dtype=jnp.float64), *params)

    return {"p": variant_p, "q": variant_q, "r": variant_r}


def make_zero_altered_family(base_name: str, base_log_pdf_func, base_params: tuple, base_links: dict,
                             score_overrides=None, hessian_overrides=None):
    """Factory to create zero-altered distributions.

    Zero-altered (hurdle): Pr(Y=0) = nu, Pr(Y=y|y>0) = (1-nu)*f(y)/[1-f(0)]

    ``score_overrides`` / ``hessian_overrides`` are forwarded to
    ``build_ad_family``; the SICHEL-based variants need them because their AD
    scores differentiate through the defective Bessel JVP (see
    ``distributions_b6._sichel_dldm``).
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
        type_=_base_response_type(base_name),
        links=za_links,
        link_functions=link_funcs,
        link_inverses=link_invs,
        link_derivatives=link_derivs,
        score_overrides=score_overrides,
        hessian_overrides=hessian_overrides,
        **_zero_variant_tails(base_name, altered=True,
                              epsilon=_QUANTILE_EPSILON),
    )


def make_zero_inflated_family(base_name: str, base_log_pdf_func, base_params: tuple, base_links: dict,
                              score_overrides=None, hessian_overrides=None):
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
    
    # D2.27: this used to carry a hand-written BNB-only `p`/`q`/`r` block.  It
    # computed the same thing as the general composition, only with the zero
    # branch spelled `q < 0.5` instead of `q == 0` -- equivalent for an integer
    # support, because `pBNB(0)` *is* `f0`.  The composition in
    # `_zero_variant_tails` replaces all three, and it is what the other eight
    # factory-built variants had been missing entirely (their `p`/`q` were the
    # `build_ad_family` placeholders, i.e. all-NaN).
    epsilon = (_QUANTILE_EPSILON_1E7
               if f"ZI{base_name}" in _ZI_FAMILIES_WITH_THE_COARSE_EPSILON
               else _QUANTILE_EPSILON)

    return build_ad_family(
        family_class=ZeroInflatedFamily,
        name=f"ZI{base_name}",
        parameters=zi_params,
        log_pdf_func=zi_log_pdf,
        type_=_base_response_type(base_name),
        links=zi_links,
        link_functions=link_funcs,
        link_inverses=link_invs,
        link_derivatives=link_derivs,
        score_overrides=score_overrides,
        hessian_overrides=hessian_overrides,
        **_zero_variant_tails(base_name, altered=False, epsilon=epsilon),
    )


# Now create the variants using existing distributions

# Import base distributions
from .distributions_b6 import _sichel_log_pdf, _pig_log_pdf
from .distributions_b6 import _sichel_dldm as _sichel_base_dldm
from .distributions_b6 import _sichel_dldd as _sichel_base_dldd
from .distributions_b6 import _r_negative_information
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


# ------------------------------------------------------------------
# R's own derivative slots for ZISICHEL / ZASICHEL (ZISICHEL.R / ZASICHEL.R).
#
# Same root cause as SICHEL itself: the AD scores differentiate
# ``_zero_sichel_log_pdf`` -> ``_sichel_log_pdf`` -> the defective Bessel JVP
# (bessel.py ``_log_bessel_kv_jvp``).  Gate 2 measured mu off by 6.1e-01
# (ZISICHEL) / 5.5e-01 (ZASICHEL), sigma by 7.9e-01 / 1.1e+00, nu by
# 6.3e-01 / 1.3e+00 rel.  R's slots are transcribed below, zero-variant
# wrappers included, in exactly R's structure (the y == 0 branches call the
# SICHEL slots at zero).
#
# ZISICHEL (R, verbatim):
#   dldm0 = (1-tau)*f0/(tau + (1-tau)*f0) * SICHEL$dldm(0, ...)
#   dldm  = y == 0 ? dldm0 : SICHEL$dldm(y, ...)
#   dldd0 = (1-tau)*f0/(tau + (1-tau)*f0) * SICHEL$dldd(0, ...)
#   dldd  = y == 0 ? dldd0 : SICHEL$dldd(y, ...)
#   dldv  = numeric.deriv(dZISICHEL(y, ..., log=TRUE), "nu", delta=0.001)
#   dldt0 = (1 - f0)/(tau + (1-tau)*f0);  dldt = y == 0 ? dldt0 : -1/(1-tau)
#
# ZASICHEL (R, verbatim):
#   dldm0 = SICHEL$dldm(y, ...) + f0*SICHEL$dldm(0, ...)/(1 - f0)
#   dldm  = y == 0 ? 0 : dldm0
#   dldd0 = SICHEL$dldd(y, ...) + f0*SICHEL$dldd(0, ...)/(1 - f0)   [sigma floored at 1e-6]
#   dldd  = y == 0 ? 0 : dldd0
#   dldv  = numeric.deriv(dZASICHEL(y, ..., log=TRUE), "nu", delta=0.001)
#   dldt  = y == 0 ? 1/tau : -1/(1-tau)
#
# with f0 = dSICHEL(0, mu, sigma, nu).  Every d2ldX2 is -score^2 clamped at
# -1e-15, except ZASICHEL's d2ldt2, which R writes as -1/(tau*(1-tau)).
# gamlss's numeric.deriv is a *forward* difference (see b6._sichel_dldv).
# ------------------------------------------------------------------

_sichel_log_pdf_v = jax.vmap(_sichel_log_pdf)
_R_NUMERIC_DERIV_DELTA = 0.001


def _zero_sichel_f0(mu, sigma, nu):
    """``dSICHEL(0, mu, sigma, nu)`` -- the base density at zero."""
    return jnp.exp(_sichel_log_pdf_v(jnp.zeros_like(mu), mu, sigma, nu))


def _neg_info(score_fn):
    """Wrap a score slot into R's ``-score^2`` clamped at ``-1e-15``."""
    def hess(*args):
        return _r_negative_information(score_fn(*args))
    hess.__name__ = f"d2_{getattr(score_fn, '__name__', 'slot')}"
    return hess


def _zisichel_dldm(y, mu, sigma, nu, tau):
    (y, mu, sigma, nu, tau), scalar = _prepare_jax_args((y, mu, sigma, nu, tau))
    zeros = jnp.zeros_like(y)
    f0 = _zero_sichel_f0(mu, sigma, nu)
    d0 = (1.0 - tau) * f0 / (tau + (1.0 - tau) * f0) * _sichel_base_dldm(zeros, mu, sigma, nu)
    out = jnp.where(y == 0, d0, _sichel_base_dldm(y, mu, sigma, nu))
    return out[0] if scalar else out


def _zisichel_dldd(y, mu, sigma, nu, tau):
    (y, mu, sigma, nu, tau), scalar = _prepare_jax_args((y, mu, sigma, nu, tau))
    zeros = jnp.zeros_like(y)
    f0 = _zero_sichel_f0(mu, sigma, nu)
    d0 = (1.0 - tau) * f0 / (tau + (1.0 - tau) * f0) * _sichel_base_dldd(zeros, mu, sigma, nu)
    out = jnp.where(y == 0, d0, _sichel_base_dldd(y, mu, sigma, nu))
    return out[0] if scalar else out


def _zisichel_dldv(y, mu, sigma, nu, tau):
    (y, mu, sigma, nu, tau), scalar = _prepare_jax_args((y, mu, sigma, nu, tau))
    delta = _R_NUMERIC_DERIV_DELTA
    f = jax.vmap(lambda a, b, c, d, e: _zero_sichel_log_pdf(a, b, c, d, e, altered=False))
    out = (f(y, mu, sigma, nu + delta, tau) - f(y, mu, sigma, nu, tau)) / delta
    return out[0] if scalar else out


def _zisichel_dldt(y, mu, sigma, nu, tau):
    (y, mu, sigma, nu, tau), scalar = _prepare_jax_args((y, mu, sigma, nu, tau))
    f0 = _zero_sichel_f0(mu, sigma, nu)
    d0 = (1.0 - f0) / (tau + (1.0 - tau) * f0)
    out = jnp.where(y == 0, d0, -1.0 / (1.0 - tau))
    return out[0] if scalar else out


def _zasichel_dldm(y, mu, sigma, nu, tau):
    (y, mu, sigma, nu, tau), scalar = _prepare_jax_args((y, mu, sigma, nu, tau))
    zeros = jnp.zeros_like(y)
    f0 = _zero_sichel_f0(mu, sigma, nu)
    d0 = _sichel_base_dldm(y, mu, sigma, nu) \
        + f0 * _sichel_base_dldm(zeros, mu, sigma, nu) / (1.0 - f0)
    out = jnp.where(y == 0, 0.0, d0)
    return out[0] if scalar else out


def _zasichel_dldd(y, mu, sigma, nu, tau):
    (y, mu, sigma, nu, tau), scalar = _prepare_jax_args((y, mu, sigma, nu, tau))
    zeros = jnp.zeros_like(y)
    # R floors sigma at 1e-6 inside its dldd/d2ldd2 before calling SICHEL's.
    sigma = jnp.maximum(sigma, 1e-6)
    f0 = _zero_sichel_f0(mu, sigma, nu)
    d0 = _sichel_base_dldd(y, mu, sigma, nu) \
        + f0 * _sichel_base_dldd(zeros, mu, sigma, nu) / (1.0 - f0)
    out = jnp.where(y == 0, 0.0, d0)
    return out[0] if scalar else out


def _zasichel_dldv(y, mu, sigma, nu, tau):
    (y, mu, sigma, nu, tau), scalar = _prepare_jax_args((y, mu, sigma, nu, tau))
    delta = _R_NUMERIC_DERIV_DELTA
    f = jax.vmap(lambda a, b, c, d, e: _zero_sichel_log_pdf(a, b, c, d, e, altered=True))
    out = (f(y, mu, sigma, nu + delta, tau) - f(y, mu, sigma, nu, tau)) / delta
    return out[0] if scalar else out


def _zasichel_dldt(y, mu, sigma, nu, tau):
    (y, mu, sigma, nu, tau), scalar = _prepare_jax_args((y, mu, sigma, nu, tau))
    out = jnp.where(y == 0, 1.0 / tau, -1.0 / (1.0 - tau))
    return out[0] if scalar else out


def _zasichel_d2ldt2(y, mu, sigma, nu, tau):
    """R's ZASICHEL ``d2ldt2`` is *not* ``-dldt^2``: it is -1/(tau*(1-tau))."""
    (y, mu, sigma, nu, tau), scalar = _prepare_jax_args((y, mu, sigma, nu, tau))
    out = -1.0 / (tau * (1.0 - tau))
    out = jnp.where(out < -1e-15, out, -1e-15)
    return out[0] if scalar else out


_ZISICHEL_SCORES = {
    "mu": _zisichel_dldm,
    "sigma": _zisichel_dldd,
    "nu": _zisichel_dldv,
    "tau": _zisichel_dldt,
}
_ZISICHEL_HESSIANS = {
    "mu": _neg_info(_zisichel_dldm),
    "sigma": _neg_info(_zisichel_dldd),
    "nu": _neg_info(_zisichel_dldv),
    "tau": _neg_info(_zisichel_dldt),
}
_ZASICHEL_SCORES = {
    "mu": _zasichel_dldm,
    "sigma": _zasichel_dldd,
    "nu": _zasichel_dldv,
    "tau": _zasichel_dldt,
}
_ZASICHEL_HESSIANS = {
    "mu": _neg_info(_zasichel_dldm),
    "sigma": _neg_info(_zasichel_dldd),
    "nu": _neg_info(_zasichel_dldv),
    "tau": _zasichel_d2ldt2,
}


# ZASICHEL - Zero-Altered Sichel
def ZASICHEL():
    """Zero-Altered Sichel distribution."""
    family = make_zero_altered_family(
        "SICHEL",
        _sichel_log_pdf,
        ("mu", "sigma", "nu"),
        {"mu": "log", "sigma": "log", "nu": "identity"},
        score_overrides=_ZASICHEL_SCORES,
        hessian_overrides=_ZASICHEL_HESSIANS,
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
        {"mu": "log", "sigma": "log", "nu": "identity"},
        score_overrides=_ZISICHEL_SCORES,
        hessian_overrides=_ZISICHEL_HESSIANS,
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

def _static_bd(value):
    """Collapse an R-style bd vector to the static Python int ZABB needs.

    The d/p/q contract carries bd per observation (R's functions rep() it to
    the response length), while the ZABB kernels size an internal grid with
    bd and therefore require one Python int -- the scanner's probe passes
    np.full(n, 10.0) and the calibration passes the float 10.0, both of which
    the family lambda must accept.  Raises when the vector is not constant or
    not a positive integer.
    """
    arr = np.asarray(value, dtype=np.float64)
    first = float(arr.ravel()[0]) if arr.size else float("nan")
    if arr.size and not np.all(arr == first):
        raise ValueError("bd must be constant across observations")
    if not np.isfinite(first) or first != int(first) or first < 1:
        raise ValueError("bd must be a positive integer")
    return int(first)


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
        constructor_parameters={"bd": int(bd)},
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=zabb_log_pdf,
        type_="Discrete",
        fixed_parameters=("bd",),
        d=lambda x,mu=.5,sigma=.1,nu=.1,bd=bd,log=False: dZABB(x,mu,sigma,nu,_static_bd(bd),log),
        p=lambda q,mu=.5,sigma=.1,nu=.1,bd=bd,lower_tail=True,log_p=False: pZABB(q,mu,sigma,nu,_static_bd(bd),lower_tail,log_p),
        q=lambda p,mu=.5,sigma=.1,nu=.1,bd=bd,lower_tail=True,log_p=False: qZABB(p,mu,sigma,nu,_static_bd(bd),lower_tail,log_p),
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
        constructor_parameters={"bd": bd},
        parameters=("mu", "sigma", "nu"),
        log_pdf_func=zibb_log_pdf,
        type_="Discrete",
        fixed_parameters=("bd",),
        d=lambda x,mu=.5,sigma=.5,nu=.1,bd=bd,log=False: dZIBB(x,mu,sigma,nu,_static_bd(bd),log),
        p=lambda q,mu=.5,sigma=.5,nu=.1,bd=bd,lower_tail=True,log_p=False: pZIBB(q,mu,sigma,nu,_static_bd(bd),lower_tail,log_p),
        q=lambda p,mu=.5,sigma=.5,nu=.1,bd=bd,lower_tail=True,log_p=False: qZIBB(p,mu,sigma,nu,_static_bd(bd),lower_tail,log_p),
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
