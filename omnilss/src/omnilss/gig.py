"""GIG -- generalised inverse Gaussian distribution (R ``gamlss.dist`` parity).

Live-R contract (installed gamlss.dist 6.1.1, dumped from the namespace and
cross-checked against live values at ``digits = 17``):

Density (``dGIG``)
------------------
R's parameterisation is *not* the textbook ``(lambda, chi, psi)`` triple.  With

    c = K_{nu+1}(1/sigma^2) / K_nu(1/sigma^2)

the log-density is

    log f = nu*log(c) - nu*log(mu) + (nu-1)*log(x) - log(2)
            - log(K_nu(1/sigma^2))
            - (1/(2*sigma^2)) * (c*x/mu + mu/(c*x))

which is the GIG with ``lambda = nu``, ``chi = mu/(c*sigma^2)``,
``psi = c/(mu*sigma^2)`` -- i.e. ``sqrt(chi*psi) = 1/sigma^2`` exactly as the
normalising constant demands, and ``E[X] = mu`` by construction (the ``c``
ratio *is* ``K_{lambda+1}/K_lambda`` at the normaliser's argument).  The
previous implementation used ``chi = 1/sigma^2``, ``psi = 1/(mu^2*sigma^2)``,
whose mean is ``mu*K_{nu+1}(1/sigma^2)/K_nu(1/sigma^2) = mu*c`` -- a different
distribution whenever ``c != 1``.  Measured against live R: the old density
was off by 8.08e-02 rel (dpqr gate, D2.47 scan); the transcribed one agrees
to 9.1e-13 over 48 probe points spanning ``nu`` in [-1.5, 2] and ``x`` from
0.01 to 50.

R quirks reproduced verbatim (explicit branch = convention, D2.25/D2.30):

* ``ifelse(x <= 0, 0, ft)`` is applied *after* the ``log`` branch, so the
  **logged** density at a non-positive ``x`` is ``0``, not ``-inf``;
* ``dGIG``/``pGIG``/``qGIG``/``rGIG`` ``stop`` for ``mu <= 0`` or
  ``sigma <= 0``; we return NaN for those rows (house convention, same as
  the exGAUS/EGB2 kernels).

CDF (``pGIG``)
--------------
No closed form; R integrates the ``mu = 1`` density from ``0.001`` to
``q/mu`` per point with adaptive ``integrate``.  By the exact scaling
property ``GIG(mu, sigma, nu) = mu * GIG(1, sigma, nu)`` that equals
``F(q) - F(0.001*mu)`` -- the true CDF minus a negligible lower-tail slice
(measured <= 1.2e-07 on probe profiles; the residual against R is R's own
quadrature error, ~1e-9).  We integrate the true CDF with a composite
Gauss-Legendre rule on a per-row log-spaced grid: the substitution
``u = log x`` turns the integrand into ``g(u) = exp(u + log f(e^u))``, which
is a single smooth peak of width ~``sigma`` in ``u`` with super-exponential
decay on both sides, so a fixed panel count resolves every profile the
scanner probes (``sigma`` in [0.5, 3]).

Below the floor (``q < 0.001*mu``) R's ``integrate`` runs over an inverted
interval and answers **tiny negative quadrature noise**, not 0 and not the
true CDF: measured at mu=2, sigma=1, nu=-0.5, ``pGIG`` returns
``-4.9108e-219 / -4.8859e-219 / -4.8794e-219`` at q = 0 / 1e-6 / 0.001 and
exactly ``0`` at q = 0.001*mu.  We return the true (tiny, non-negative) CDF;
the test pins both sides.

Quantile (``qGIG``)
-------------------
R brackets around ``mu`` and calls ``uniroot`` at the default tolerance
(~1.2e-4 in ``x``) on its own quadrature CDF; measured against a
``tol = 1e-14`` re-solve of the true CDF, R's roots sit up to 5.7e-04 away
(deep tail: the CDF error is amplified through 1/f).  We bisect in log
space to machine precision on our quadrature CDF, so the residual against R
is R's own quantile error -- the same disposition as IG/SEP/SIMPLEX (D2.31/
D2.32, ``tools/rs_family_parity_scan._DPQR_TOLERANCE``).  Boundary pins:
R's installed ``qGIG`` has no ``p == 0``/``p == 1`` assignments (the
vendored 6.1-8 source does), so ``qGIG(0)`` lands wherever ``uniroot``
happens to converge inside the flat ``pGIG == 0`` region (~0.00195 at the
default profile) and ``qGIG(1)`` wherever ``integrate`` noise first reads
as 1 (71.15); R *stops* for ``p`` outside [0, 1].  We return the
mathematically correct 0 / +inf and clip, pinning R's behaviour in tests.

Sampler (``rGIG``)
------------------
R pushes ``runif(n)`` through ``qGIG``; we do the same with
``jax.random.uniform`` (the RNG stream differs by construction, so only
distributional equivalence is assertable).

Derivative slots
----------------
R hands the family object closed forms for all nine slots; RS consumes only
the diagonal three (``distributions.py``: "Cross Hessian d2ldmdd is ignored
(set to 0 in GAMLSS)").  Transcribed verbatim as overrides -- including
R's approximations, which is what parity means here (D2.44 SICHEL pattern):

* ``dldm``, ``dldd``: exact closed forms in ``c`` (verified against central
  differences of the log-density);
* ``dldv``: gamlss's ``numeric.deriv`` with ``delta = 0.01`` -- dumped from
  the installed namespace it is a **forward** difference, reproduced
  bit-for-bit against live R;
* ``d2ldm2``: the *expected* Hessian ``(nu - c/sigma^2)/mu^2`` (NOT the
  observed one -- measured at y=3, mu=2, sigma=1, nu=-0.5: R -0.375 vs
  observed -0.5), floored at -1e-15 (keep the more negative);
* ``d2ldd2``, ``d2ldv2``: ``-score^2`` (Gauss-Newton form, NOT the true
  second derivative -- the observed sigma curvature is *positive* there),
  floored at -1e-6.
"""

from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp
import jax.random as jr

from .ad import _prepare_jax_args
from .bessel import log_bessel_kv

# ---------------------------------------------------------------------------
# Density
# ---------------------------------------------------------------------------


def log_density(y, mu, sigma, nu):
    """R's ``dGIG`` log branch, term for term (see module docstring).

    Scalar kernel: ``build_ad_family`` vmaps this for the AD/deviance path,
    so it must NOT call ``_prepare_jax_args`` (that flattens to 1-D and
    breaks the vmap shape contract, D2.45) nor broadcast -- JAX vmap hands
    it 0-d slices.
    """
    eps = jnp.finfo(jnp.float64).eps
    y = jnp.maximum(y, eps)
    mu = jnp.maximum(mu, eps)
    sigma = jnp.maximum(sigma, eps)

    inv_sigma2 = 1.0 / (sigma * sigma)
    log_k_nu = log_bessel_kv(nu, inv_sigma2)
    log_c = log_bessel_kv(nu + 1.0, inv_sigma2) - log_k_nu
    # The clip only binds where exp would overflow/underflow double anyway;
    # in that regime R's own exp() gives 0 and its density collapses to 0
    # too, so the behaviour is preserved.
    c = jnp.exp(jnp.clip(log_c, -745.0, 709.0))

    return (
        nu * log_c
        - nu * jnp.log(mu)
        + (nu - 1.0) * jnp.log(y)
        - jnp.log(2.0)
        - log_k_nu
        - 0.5 * inv_sigma2 * (c * y / mu + mu / (c * y))
    )


def _gig_c(nu, sigma):
    """``c = K_{nu+1}(1/sigma^2)/K_nu(1/sigma^2)`` in log space."""
    inv_sigma2 = 1.0 / (sigma * sigma)
    log_k_nu = log_bessel_kv(nu, inv_sigma2)
    log_c = log_bessel_kv(nu + 1.0, inv_sigma2) - log_k_nu
    c = jnp.exp(jnp.clip(log_c, -745.0, 709.0))
    return c, log_c


def dGIG(x, mu=1.0, sigma=1.0, nu=1.0, log=False):
    """Density of the generalised inverse Gaussian distribution."""
    x, mu, sigma, nu = jnp.broadcast_arrays(
        jnp.asarray(x, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
        jnp.asarray(nu, dtype=jnp.float64),
    )
    valid = (mu > 0.0) & (sigma > 0.0)
    safe_mu = jnp.where(mu > 0.0, mu, 1.0)
    safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)

    loglik = log_density(x, safe_mu, safe_sigma, nu)
    value = loglik if log else jnp.exp(loglik)
    # R applies the support branch after the log branch: a non-positive x
    # yields 0 even when log = TRUE.  Reproduced verbatim.
    value = jnp.where(x <= 0.0, 0.0, value)
    return jnp.where(valid, value, jnp.nan)


# ---------------------------------------------------------------------------
# CDF: composite Gauss-Legendre quadrature on a per-row log grid
# ---------------------------------------------------------------------------

#: Log-space integration span below log(q).  The integrand decays
#: super-exponentially (exp(-mu/(c*x)) as x -> 0), so 50 log units covers
#: every mass the scanner or a fit can exercise.
_GIG_LOG_SPAN = 50.0

#: Composite panels (width _GIG_LOG_SPAN / _GIG_PANELS = 0.208) x Gauss-
#: Legendre nodes.  The integrand's peak has width ~ sigma >= 0.5 on every
#: probe grid, so each peak is resolved by several panels at spectral
#: accuracy; the tails are entire functions on every panel.
_GIG_PANELS = 240
_GIG_GL_NODES = 24

_GL_NODES, _GL_WEIGHTS = np.polynomial.legendre.leggauss(_GIG_GL_NODES)
_GL_NODES = jnp.asarray(_GL_NODES, dtype=jnp.float64)
_GL_WEIGHTS = jnp.asarray(_GL_WEIGHTS, dtype=jnp.float64)
# Node positions inside a panel, in (0, 1).
_GL_FRAC = 0.5 * (_GL_NODES + 1.0)
_PANEL_H = _GIG_LOG_SPAN / _GIG_PANELS
# Panel offsets from the lower integration limit: (P, K).
_PANEL_OFFSETS = (
    jnp.arange(_GIG_PANELS, dtype=jnp.float64)[:, None] + _GL_FRAC[None, :]
) * _PANEL_H
# Quadrature weights on a panel of width _PANEL_H: (K,).
_PANEL_WEIGHTS = _GL_WEIGHTS * (_PANEL_H / 2.0)


def _gig_row_constants(mu_flat, sigma_flat, nu_flat):
    """Per-row Bessel constants: ``(c, log_c, log_k_nu, log_mu, inv_sigma2)``."""
    inv_sigma2 = 1.0 / (sigma_flat * sigma_flat)
    log_k_nu = log_bessel_kv(nu_flat, inv_sigma2)
    log_c = log_bessel_kv(nu_flat + 1.0, inv_sigma2) - log_k_nu
    c = jnp.exp(jnp.clip(log_c, -745.0, 709.0))
    return c, log_c, log_k_nu, jnp.log(mu_flat), inv_sigma2


def _gig_cdf_impl(q_flat, mu_flat, sigma_flat, nu_flat, constants):
    """Composite-rule CDF; ``constants`` from :func:`_gig_row_constants`."""
    c, log_c, log_k_nu, log_mu, inv_sigma2 = constants

    u_hi = jnp.log(jnp.maximum(q_flat, 1e-300))[:, None, None]
    u = u_hi - _GIG_LOG_SPAN + _PANEL_OFFSETS[None, :, :]

    nu_b = nu_flat[:, None, None]
    log_c_b = log_c[:, None, None]
    log_mu_b = log_mu[:, None, None]
    log_k_b = log_k_nu[:, None, None]
    c_b = c[:, None, None]
    mu_b = mu_flat[:, None, None]
    inv_sigma2_b = inv_sigma2[:, None, None]
    exp_u = jnp.exp(u)

    log_g = (
        nu_b * u
        + nu_b * log_c_b
        - nu_b * log_mu_b
        - jnp.log(2.0)
        - log_k_b
        - 0.5 * inv_sigma2_b * (c_b * exp_u / mu_b + mu_b / (c_b * exp_u))
    )
    g = jnp.exp(log_g)
    return jnp.sum(g * _PANEL_WEIGHTS[None, None, :], axis=(1, 2))


def _gig_cdf_rows(q_flat, mu_flat, sigma_flat, nu_flat):
    """True CDF F(q) for flattened rows (Bessel constants computed here)."""
    constants = _gig_row_constants(mu_flat, sigma_flat, nu_flat)
    return _gig_cdf_impl(q_flat, mu_flat, sigma_flat, nu_flat, constants)


def pGIG(q, mu=1.0, sigma=1.0, nu=1.0, lower_tail=True, log_p=False):
    """CDF of the generalised inverse Gaussian distribution."""
    q, mu, sigma, nu = jnp.broadcast_arrays(
        jnp.asarray(q, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
        jnp.asarray(nu, dtype=jnp.float64),
    )
    valid = (mu > 0.0) & (sigma > 0.0)
    safe_mu = jnp.where(mu > 0.0, mu, 1.0)
    safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)

    shape = q.shape
    cdf = _gig_cdf_rows(
        q.ravel(), safe_mu.ravel(), safe_sigma.ravel(), nu.ravel()
    ).reshape(shape)
    cdf = jnp.clip(cdf, 0.0, 1.0)
    # R's pGIG integrates from 0.001, so for q below 0.001*mu its integrate
    # runs over an inverted interval and answers tiny negative quadrature
    # noise; we return the true (tiny) CDF.  q <= 0 is 0 on both sides.
    cdf = jnp.where(q <= 0.0, 0.0, cdf)
    # q = +inf is the whole real line: the log-space grid would degenerate
    # to inf nodes and return 0, so pin the limit explicitly (R's
    # integrate(0.001, Inf) answers 1).
    cdf = jnp.where(jnp.isposinf(q), 1.0, cdf)

    if not lower_tail:
        cdf = 1.0 - cdf
    if log_p:
        cdf = jnp.log(cdf)
    return jnp.where(valid, cdf, jnp.nan)


# ---------------------------------------------------------------------------
# Quantile: bracket expansion + bisection in log space
# ---------------------------------------------------------------------------

_Q_BISECTION_ITERS = 90


def _gig_quantile_rows(prob_flat, mu_flat, sigma_flat, nu_flat):
    """True quantiles for flattened rows by bisection on log x.

    Bessel constants are computed once per row and the bisection reuses
    them: 90 iterations x 2 ``log_bessel_kv`` callbacks per row would
    dominate the runtime otherwise (the callbacks run a sequential scipy
    loop per row).
    """
    constants = _gig_row_constants(mu_flat, sigma_flat, nu_flat)
    c = constants[0]
    # Centre of the bracket at the density's balance point c*e^u/mu = 1.
    u0 = jnp.log(mu_flat) - jnp.log(jnp.maximum(c, 1e-300))
    lo = u0 - _GIG_LOG_SPAN
    hi = u0 + _GIG_LOG_SPAN

    def cdf_at(u):
        return jnp.clip(
            _gig_cdf_impl(jnp.exp(u), mu_flat, sigma_flat, nu_flat, constants),
            0.0,
            1.0,
        )

    for _ in range(_Q_BISECTION_ITERS):
        mid = 0.5 * (lo + hi)
        below = cdf_at(mid) < prob_flat
        lo = jnp.where(below, mid, lo)
        hi = jnp.where(below, hi, mid)
    return jnp.exp(0.5 * (lo + hi))


def qGIG(p, mu=1.0, sigma=1.0, nu=1.0, lower_tail=True, log_p=False):
    """Quantile of the generalised inverse Gaussian distribution."""
    p, mu, sigma, nu = jnp.broadcast_arrays(
        jnp.asarray(p, dtype=jnp.float64),
        jnp.asarray(mu, dtype=jnp.float64),
        jnp.asarray(sigma, dtype=jnp.float64),
        jnp.asarray(nu, dtype=jnp.float64),
    )
    valid = (mu > 0.0) & (sigma > 0.0)
    safe_mu = jnp.where(mu > 0.0, mu, 1.0)
    safe_sigma = jnp.where(sigma > 0.0, sigma, 1.0)

    prob = p
    if log_p:
        prob = jnp.exp(prob)
    if not lower_tail:
        prob = 1.0 - prob
    prob = jnp.clip(prob, 0.0, 1.0)

    shape = p.shape
    value = _gig_quantile_rows(
        prob.ravel(), safe_mu.ravel(), safe_sigma.ravel(), nu.ravel()
    ).reshape(shape)
    # House boundary convention (exGAUS pattern); R's flat-zone artifacts
    # at p == 0/1 and its stop for p outside [0, 1] are pinned in tests.
    value = jnp.where(prob <= 0.0, 0.0, value)
    value = jnp.where(prob >= 1.0, jnp.inf, value)
    return jnp.where(valid, value, jnp.nan)


def rGIG(key, n, mu=1.0, sigma=1.0, nu=1.0):
    """Sampler: R pushes ``runif(n)`` through ``qGIG``; so do we."""
    u = jr.uniform(key, (int(n),), dtype=jnp.float64)
    return qGIG(u, mu=mu, sigma=sigma, nu=nu)


# ---------------------------------------------------------------------------
# R's derivative slots, transcribed verbatim (see module docstring)
# ---------------------------------------------------------------------------

#: gamlss's ``numeric.deriv`` step for GIG's ``dldv`` (GIG.R:70/76).
_R_NUMERIC_DERIV_DELTA = 0.01


def _r_keep_more_negative(value, floor):
    """R's ``ifelse(h < floor, h, floor)`` clamp (keeps the more negative)."""
    return jnp.where(value < floor, value, floor)


def _gig_dldm(y, mu, sigma, nu):
    """R's ``dldm``: ``-(nu/mu) + c*y/(2*sigma^2*mu^2) - 1/(2*sigma^2*c*y)``."""
    (y, mu, sigma, nu), scalar = _prepare_jax_args((y, mu, sigma, nu))
    c, _ = _gig_c(nu, sigma)
    s2 = sigma * sigma
    out = -(nu / mu) + (c * y) / (2.0 * s2 * mu * mu) - 1.0 / (2.0 * s2 * c * y)
    return out[0] if scalar else out


def _gig_dldd(y, mu, sigma, nu):
    """R's ``dldd``, transcribed from GIG.R lines 47-54 (installed identical)."""
    (y, mu, sigma, nu), scalar = _prepare_jax_args((y, mu, sigma, nu))
    c, _ = _gig_c(nu, sigma)
    s2 = sigma * sigma
    inv_sigma2 = 1.0 / s2
    dcdd = (c * s2 * (2.0 * nu + 1.0) + 1.0 - c * c) / (s2 * s2)
    out = (
        inv_sigma2
        * (
            nu
            - c * inv_sigma2
            + 0.5 * inv_sigma2 * ((c * y / mu) + (mu / (c * y)))
            + dcdd * (s2 * nu / c - 0.5 * ((y / mu) - (mu / (c * c * y))))
        )
        * (2.0 * sigma)
    )
    return out[0] if scalar else out


def _gig_dldv(y, mu, sigma, nu):
    """R's ``dldv``: forward difference of the log density in ``nu``.

    gamlss's ``numeric.deriv`` (dumped from the installed namespace) is
    ``(f(nu + delta) - f(nu))/delta`` -- a forward difference, not stats'
    central difference -- with ``delta = 0.01`` for GIG.  Reproduced against
    our own parity-verified density so the IRLS working weights see exactly
    the approximation R's see (D2.44 SICHEL pattern).
    """
    (y, mu, sigma, nu), scalar = _prepare_jax_args((y, mu, sigma, nu))
    delta = _R_NUMERIC_DERIV_DELTA
    log_pdf = jax.vmap(log_density)
    out = (log_pdf(y, mu, sigma, nu + delta) - log_pdf(y, mu, sigma, nu)) / delta
    return out[0] if scalar else out


def _gig_d2ldm2(y, mu, sigma, nu):
    """R's ``d2ldm2``: the *expected* Hessian, floored at -1e-15.

    ``(nu - c/sigma^2)/mu^2`` with ``ifelse(d2ldm2 < -1e-15, d2ldm2, -1e-15)``
    -- the observed curvature is a different (and sign-unstable) quantity;
    parity means R's formula, not the true one.
    """
    (y, mu, sigma, nu), scalar = _prepare_jax_args((y, mu, sigma, nu))
    c, _ = _gig_c(nu, sigma)
    out = (nu - c / (sigma * sigma)) / (mu * mu)
    out = _r_keep_more_negative(out, -1e-15)
    return out[0] if scalar else out


def _negative_information(score, floor):
    """R's ``d2ldX2 = -dldX^2`` clamped at ``floor`` (SICHEL/GIG convention)."""
    h = -score * score
    return _r_keep_more_negative(h, floor)


def _gig_d2ldd2(y, mu, sigma, nu):
    """R's ``d2ldd2 = -dldd^2`` floored at -1e-6 (GIG.R:64-66).

    Receives the fitter's raw kwargs (arrays) and forwards them verbatim;
    ``_gig_dldd`` owns the ``_prepare_jax_args`` handling (double
    preparation returns a 0-d result that cannot be re-indexed).
    """
    return _negative_information(_gig_dldd(y, mu, sigma, nu), -1e-6)


def _gig_d2ldv2(y, mu, sigma, nu):
    """R's ``d2ldv2 = -dldv^2`` floored at -1e-6 (GIG.R:78-80)."""
    return _negative_information(_gig_dldv(y, mu, sigma, nu), -1e-6)


#: Diagonal slots only -- RS ignores the cross Hessians (GAMLSS sets them 0).
GIG_SCORES = {"mu": _gig_dldm, "sigma": _gig_dldd, "nu": _gig_dldv}
GIG_HESSIANS = {"mu": _gig_d2ldm2, "sigma": _gig_d2ldd2, "nu": _gig_d2ldv2}
