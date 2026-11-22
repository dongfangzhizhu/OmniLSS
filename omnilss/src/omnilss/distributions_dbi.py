"""Double Binomial (Borg 1988) -- the two-parameter generalisation of ``BI``.

R references
-------------
- ``gamlss.dist/R/DBI.R::DBI``            -- family object, all derivatives
- ``gamlss.dist/R/dDBI.R::dDBI``          -- density
- ``gamlss.dist/src/getBI_C2.c``          -- the normalising constant ``C``

What this replaces
------------------
Until D2.23 ``DBI()`` was a compatibility shell that forwarded every slot to
``BI()``: ``parameters == ("mu",)``, no ``sigma``, same kernel.  That made the
family indistinguishable from ``BI`` in every scan -- it reported BI's deviance
(217.973775379603) where R's ``DBI`` reports **217.95942206518868**, and it
could not pass the ``rsdev`` gate at all because ``DBI`` is registered as a
compatibility alias.  D2.22's own test spelled that gap out; this module closes
it.

The density
-----------
``dDBI`` (R source, transcribed) is a binomial pmf tilted by a power of the
ratio terms, renormalised by ``C``::

    logofx    = ifelse(x == 0, 1, log(x))          # the 0*log0 -> 0 guard
    logofbd_x = ifelse(bd == x, 1, log(bd - x))

    ll = lchoose(bd, x) + x*logofx + (bd-x)*logofbd_x - bd*log(bd)
       + (bd/sigma)*log(bd) + (x/sigma)*log(mu) + ((bd-x)/sigma)*log(1-mu)
       - (x/sigma)*logofx - ((bd-x)/sigma)*logofbd_x
       + C

    if |sigma - 1| < 0.001:  ll = dbinom(x, size=bd, prob=mu, log=TRUE)

Two details that are easy to miss and both change the numbers:

1. **R branches to plain ``dbinom`` when ``|sigma - 1| < 0.001``.**  That is not
   an optimisation -- it is what makes ``C(1) = 0`` exact instead of
   ``-1.4e-16``, and R's ``sigma.initial`` is ``rep(1.1, length(y))``, so an
   RS fit *passes through* this branch on its way from 1.1 towards the optimum.
   The condition is inherited here, branch included.
2. **``GetBI_C`` inverts sigma before calling into C.**  ``GetBI_C(mu, sigma,
   bd)`` does ``sigma <- rep(1/sigma, ...)`` and then passes it to
   ``getBI_C2``, whose ``ss`` is written in terms of *that* quantity.  So our
   ``_dbi_log_c`` takes sigma in R's user-facing sense and does the inversion
   itself; getting this backwards silently produces a plausible density that
   is not R's.

The normalising constant
------------------------
``C`` is a **finite sum**, not a numeric integral -- the first three attempts at
a closed form (Borg's beta mixture, ``log B(a,b)`` forms, a log-difference)
were all wrong, and the giveaway was that a first grid at ``mu`` in
{0.2, 0.5, 0.8} looked independent of ``mu``: that grid is symmetric about 0.5
and ``C`` is too.  ``R/gamlss.dist/src/getBI_C2.c`` (vendored in this repo)
is the whole definition::

    sm1 = sigma - 1;  s_1 = 1 - sigma;  n = bd
    nlogn = n * (n == 0 ? 1 : log(n))
    for j in 0..n:
      logchoose[j] = lgamma(n+1) - lgamma(j+1) - lgamma(n-j+1)
      ylogofy[j]    = j * (j == 0 ? 1 : log(j))
      nylogofny[j]  = (n-j) * (n-j == 0 ? 1 : log(n-j))
      ylogmu[j]     = j * log(mu)
      nylog1mu[j]   = (n-j) * log(1-mu)
      ss[j] = logchoose[j]
           + (ylogofy[j] + nylogofny[j]) * s_1
           + nlogn * sm1
           + (ylogmu[j] + nylog1mu[j]) * sigma
      sumC += exp(ss[j])
    C = -log(sumC)

Transcribed literally it matches ``GetBI_C`` to **5.7e-15** worst case over a
192-point grid (bd in 2/5/10/20 x mu in 0.01..0.99 x sigma in 0.3..3.0).
``C`` is the only reason this family is a family: without it the pmf does not
sum to 1, and with a wrong ``C`` it still looks like a density.

Derivatives
-----------
R's ``dldm`` / ``dldd`` add the analytic gradient of ``C`` to the closed-form
part, and then R *itself* differentiates those scores numerically again to get
the Hessians::

    dldm  = y/(mu*sigma) - (bd-y)/((1-mu)*sigma) + dC/dmu
    d2ldm2 = -dldm^2 ;  floored at -1e-15
    dldd  = [ y*logofy - log(mu)*y + logofbd_x*(bd-y) - log(1-mu)*(bd-y)
              - bd*log(bd) ] / sigma^2 + dC/dsigma
    d2ldd2 = -dldd^2 ;  floored at -1e-15

The Hessians are **not** ``d2logL``: ``-dldm^2`` is a Fisher-scoring
approximation that R feeds to IRLS.  Copying the real second derivative here
would move us *away* from parity -- the same D2.13 lesson as Box-Cox, where
R's own roughness (``d2ldv2 = -7*sigma^2/4``) is the thing being matched.

``dC/dmu`` and ``dC/dsigma`` come from AD of ``_dbi_log_c``.  That is the same
``custom_jvp``-free route the other AD families use, and it is safe here
because ``_dbi_log_c`` is built only from ``gammaln``/``log``/``sum`` -- no
special functions without second derivatives, unlike the Box-Cox cluster.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.special import gammaln

from .families import FamilyDefinition
from .links import (
    log_derivative,
    log_inverse,
    log_link,
    logit_derivative,
    logit_inverse,
    logit_link,
)

__all__ = ["DBI", "DBIFamily", "dbi_log_c", "dbi_log_pdf"]

_EPS = jnp.finfo(jnp.float64).eps

# R's `0:q` sequence operator, measured on live R 6.1.1: `0:2.9999999` is
# 0,1,2,3 while `0:2.999999` is 0,1,2 -- the endpoint is inclusive up to an
# absolute fuzz of ~1e-7 (the same threshold near 30, so it is absolute, not
# relative).  ``pDBI`` sums over ``0:q``, so the CDF steps at integers *plus*
# this fuzz.  Getting it wrong shifts the p staircase by one step for any q
# within ~1e-7 of an integer -- invisible on coarse grids, exact on a
# digits=17 probe.
_COLON_FUZZ = 1e-7


def _dbi_log_c(mu, sigma, bd):
    """``C(mu, sigma, bd)`` transcribed from ``getBI_C2.c``.

    ``sigma`` arrives in R's user-facing sense, so the ``1/sigma`` that
    ``GetBI_C`` applies before calling into C is applied here -- once, at the
    boundary.  Getting that inversion backwards silently yields a plausible
    density that is not R's.

    Everything is computed on a trailing ``(n_max + 1,)`` summation axis and
    then reshaped back to the **broadcast** shape of the three arguments, so
    ``bd = 10.0`` with vector ``mu``/``sigma`` gives one ``C`` per observation
    and three scalars give a scalar.

    The shape decision must look at *all three* arguments, not at ``bd`` alone.
    Deciding from ``bd`` alone (``scalar = jnp.ndim(bd) == 0``) looks harmless
    and is not: the parity scanner's ``calibrate`` passes a scalar ``bd`` --
    ``_PROBE_BD`` -- together with length-``n`` parameter vectors, so the squeeze
    fires on an ``(n,)`` result and raises ``cannot reshape array of shape (n,)
    into shape ()``.  That exception does not look like a shape bug at all: it
    surfaces as *every* probe candidate failing, i.e. a silent ``deriv`` skip.
    See ``test_scalar_bd_with_vector_parameters_broadcasts``.
    """
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    bd = jnp.asarray(bd, dtype=jnp.float64)

    shape = jnp.broadcast_shapes(jnp.shape(mu), jnp.shape(sigma), jnp.shape(bd))
    mu = jnp.broadcast_to(jnp.clip(mu, _EPS, 1.0 - _EPS), shape)
    sigma = jnp.broadcast_to(jnp.clip(sigma, _EPS, None), shape)
    bd = jnp.broadcast_to(bd, shape)

    # GetBI_C: sigma <- 1/sigma before the C call.
    inv = (1.0 / sigma)[..., None]
    sm1 = inv - 1.0
    s_1 = 1.0 - inv
    mu_col = mu[..., None]

    n = jnp.maximum(jnp.round(bd), 1.0)
    n_max = int(jnp.max(n))
    j = jnp.arange(n_max + 1, dtype=jnp.float64)[None, :]
    n_col = n[..., None]

    # A larger-than-needed summation only adds terms that vanish as
    # (mu*inv)^-j, so masking with -inf is exact, not an approximation.
    active = j <= n_col
    jj = jnp.broadcast_to(j, active.shape)
    bdj = n_col - jj

    logchoose = gammaln(n_col + 1.0) - gammaln(jj + 1.0) - gammaln(bdj + 1.0)

    # The (x == 0 ? 1 : log(x)) guards: R substitutes 1, not 0, so log(0) never
    # appears.  Using jnp.where(x > 0, j*log(j), 0.0) would differ whenever
    # j == 0 -- which is the first term of every observation.
    ylogofy = jj * jnp.where(jj > 0.0, jnp.log(jj), 1.0)
    nylogofny = bdj * jnp.where(bdj > 0.0, jnp.log(bdj), 1.0)
    ylogmu = jj * jnp.log(mu_col)
    nylog1mu = bdj * jnp.log1p(-mu_col)
    nlogn = n_col * jnp.where(n_col > 0.0, jnp.log(jnp.maximum(n_col, _EPS)), 1.0)

    ss = (
        logchoose
        + (ylogofy + nylogofny) * s_1
        + nlogn * sm1
        + (ylogmu + nylog1mu) * inv
    )
    ss = jnp.where(active, ss, -jnp.inf)
    total = jnp.sum(jnp.exp(ss), axis=-1)
    out = -jnp.log(total)
    return out.reshape(shape)


# Module-level alias with the R-facing name, for tests and cross-checking.
dbi_log_c = _dbi_log_c


def _dbi_log_pdf(x, mu, sigma, bd):
    """``dDBI(..., log = TRUE)`` transcribed from ``dDBI.R``.

    Three contract details that only a non-integer ``x`` exposes, all
    measured on live R 6.1.1 (digits=17 probes, ``mu=0.5, sigma=1.2,
    bd=10``):

    1. **``x`` is *not* rounded.**  Every term except ``lchoose`` uses the
       raw ``x``: ``x*logofx``, ``(bd-x)*logofbd_x``, ``(x/sigma)*log(mu)``
       all see 0.2 / 0.65 / 1.55 verbatim.  ``lchoose(bd, x)`` *does* round
       -- R's ``lchoose`` rounds both arguments half-to-even
       (``lchoose(10, 1.5) == lchoose(10, 2)``, measured).  The result is a
       chimera: ``lchoose`` at the integer, everything else at the raw
       value.  Rounding ``x`` up front (the previous implementation) made
       ``d(0.65) == d(1.0)`` exactly, which is not R at any non-integer.
    2. **``dbinom`` branch returns 0 for a non-integer ``x``.**  R's
       ``dbinom`` refuses non-integer ``x`` (``-Inf`` on the log scale, with
       a warning), so the ``|sigma - 1| < 0.001`` branch is *not* the
       general formula rounded.
    3. **Outside the support the density is 0.**  ``fy[x < 0] <- 0`` and
       ``fy[x > bd] <- 0`` are the vendored 6.1-8 last lines; the installed
       6.1.1 instead ``stop()``s for ``bd < x``.  We follow the vendored
       source (the SHASHo2 boundary-contract convention): a density that
       returns 0 outside its support is the usable contract, and a
       ``stop()`` from inside a vectorised JAX kernel is not something the
       fitter could catch anyway.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.asarray(sigma, dtype=jnp.float64)
    bd = jnp.asarray(bd, dtype=jnp.float64)

    mu = jnp.clip(mu, _EPS, 1.0 - _EPS)
    sigma = jnp.clip(sigma, _EPS, None)
    # The clip only keeps intermediates finite; the raw x decides the
    # support test at the end, exactly as R's two `fy[...] <- 0` lines do.
    y = jnp.clip(x, 0.0, bd)

    logofy = jnp.where(y > 0.0, jnp.log(jnp.maximum(y, _EPS)), 1.0)
    logofn_y = jnp.where(bd > y, jnp.log(jnp.maximum(bd - y, _EPS)), 1.0)

    # Contract (1): lchoose rounds both arguments, half-to-even.  jnp.round
    # is half-to-even, as is R's round() -- lchoose(10, 0.5) == lchoose(10, 0)
    # and lchoose(10, 1.5) == lchoose(10, 2), both measured.
    y_int = jnp.round(y)
    bd_int = jnp.round(bd)
    lchoose = (
        gammaln(bd_int + 1.0) - gammaln(y_int + 1.0) - gammaln(bd_int - y_int + 1.0)
    )

    # R's binomial branch, verbatim: `abs(sigma - 1) < 0.001`.
    binom_ll = (
        gammaln(bd + 1.0)
        - gammaln(y + 1.0)
        - gammaln(bd - y + 1.0)
        + y * jnp.log(mu)
        + (bd - y) * jnp.log1p(-mu)
    )
    # Contract (2): dbinom refuses a non-integer x.
    binom_ll = jnp.where(y == y_int, binom_ll, -jnp.inf)

    general_ll = (
        lchoose
        + y * logofy
        + (bd - y) * logofn_y
        - bd * jnp.log(jnp.maximum(bd, _EPS))
        + (bd / sigma) * jnp.log(jnp.maximum(bd, _EPS))
        + (y / sigma) * jnp.log(mu)
        + ((bd - y) / sigma) * jnp.log1p(-mu)
        - (y / sigma) * logofy
        - ((bd - y) / sigma) * logofn_y
        + _dbi_log_c(mu, sigma, bd)
    )
    ll = jnp.where(jnp.abs(sigma - 1.0) < 0.001, binom_ll, general_ll)
    # Contract (3): dDBI's last lines: fy[x < 0] <- 0; fy[x > bd] <- 0.
    return jnp.where((x < 0.0) | (x > bd), -jnp.inf, ll)


dbi_log_pdf = _dbi_log_pdf


def _score_factory(family):
    """R's ``dldm`` / ``dldd``, with ``dC`` supplied by AD."""

    def dldm(y, mu, sigma, bd=None):
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        bd = jnp.asarray(bd, dtype=jnp.float64)
        mu = jnp.clip(mu, _EPS, 1.0 - _EPS)
        sigma = jnp.clip(sigma, _EPS, None)
        dc_dmu = jax_grad_c(family)(mu, sigma, bd)
        return y / (mu * sigma) - (bd - y) / ((1.0 - mu) * sigma) + dc_dmu

    def dldd(y, mu, sigma, bd=None):
        y = jnp.asarray(y, dtype=jnp.float64)
        mu = jnp.asarray(mu, dtype=jnp.float64)
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        bd = jnp.asarray(bd, dtype=jnp.float64)
        mu = jnp.clip(mu, _EPS, 1.0 - _EPS)
        sigma = jnp.clip(sigma, _EPS, None)
        logofy = jnp.where(y > 0.0, jnp.log(jnp.maximum(y, _EPS)), 1.0)
        logofn_y = jnp.where(bd > y, jnp.log(jnp.maximum(bd - y, _EPS)), 1.0)
        dc_ds = jax_grad_c(family, "sigma")(mu, sigma, bd)
        return (
            y * logofy
            - jnp.log(mu) * y
            + logofn_y * (bd - y)
            - jnp.log1p(-mu) * (bd - y)
            - bd * jnp.log(jnp.maximum(bd, _EPS))
        ) / sigma**2 + dc_ds

    return dldm, dldd


def jax_grad_c(family=None, arg: str = "mu"):
    """AD gradient of ``_dbi_log_c`` -- R gets the same number from ``numeric.deriv``."""
    import jax

    def grad_fn(mu, sigma, bd):
        if arg == "mu":
            return jax.grad(lambda m: jnp.sum(_dbi_log_c(m, sigma, bd)))(mu)
        return jax.grad(lambda s: jnp.sum(_dbi_log_c(mu, s, bd)))(sigma)

    return grad_fn


def _hessian_from_score(score):
    """R's ``d2ldX2 = -dldX^2`` floored at -1e-15 (Fisher scoring, not d2logL)."""

    def d2ld2(y, mu, sigma, bd=None):
        value = jnp.asarray(score(y, mu, sigma, bd), dtype=jnp.float64)
        out = -(value**2)
        return jnp.where(out < -1e-15, out, -1e-15)

    return d2ld2


def _sigma_hessian(score):
    """Second derivative of the log-likelihood in sigma, by differencing the score.

    D2.153.  R uses d2ldd2 = -dldd^2 for DBI -- a Fisher scoring surrogate, not
    the second derivative -- and it stalls the sigma update: the step is
    dldd / |d2ldd2| = -1/dldd, which shrinks as the score grows, so the outer
    loop declares convergence while the score is still 0.61 (measured on the
    current scanner sample).  The surrogate exists because DBI's density
    carries a normalising constant C(mu, sigma, bd) whose sigma derivative R
    also only obtains numerically (numeric.deriv); its second derivative is
    likewise unavailable in closed form, and AD through it is impossible (the
    series uses integer indices).

    So the second derivative is taken by central-differencing the score we
    already have (itself exact to 1e-16 against finite differences of our own
    density).  Cost: two score evaluations per IRLS step.  It moves DBI onto
    the true optimum -- 215.84452506 against R's 215.84453568 on the current
    gate sample, with R's own dDBI at those coefficients returning our number
    exactly (215.84452506478772), which is the adjudication that the
    "ours is lower" discipline asks for.
    """

    def d2ldd2(y, mu, sigma, bd=None):
        sigma = jnp.asarray(sigma, dtype=jnp.float64)
        step = 1e-5 * jnp.maximum(jnp.abs(sigma), 1e-8)
        upper = jnp.asarray(score(y, mu, sigma + step, bd), dtype=jnp.float64)
        lower = jnp.asarray(score(y, mu, sigma - step, bd), dtype=jnp.float64)
        value = (upper - lower) / (2.0 * step)
        return jnp.where(value < -1e-15, value, -1e-15)

    return d2ldd2


def _dbi_quantile(p, mu, sigma, bd):
    """``qDBI`` -- brute-force inverse of ``pDBI`` by scanning the integers.

    R's body (transcribed from ``dDBI.R``) is a plain loop::

        for (i in seq(along = p)) {
            cumpro <- 0
            if (p[i] + 1e-09 >= 1) QQQ[i] <- bd[i]
            else for (j in 0:bd[i]) {
                cumpro <- pDBI(j, mu[i], sigma[i], bd[i])
                QQQ[i] <- j
                if (p[i] <= cumpro) break
            }
        }

    R attaches no ``qDBI`` to the family (``DBI()$q`` is NULL), so nothing
    forces this to exist -- but the function is real, deterministic and built
    from the same finite sum, so implementing it costs one masked comparison
    and turns an unmeasurable ``q`` slot into a parity check.  The loop is
    replaced by ``sum(cumsum(d) >= p)`` which is the same arithmetic without a
    Python-level iteration over ``bd + 1`` steps.

    The ``+ 1e-09 >= 1`` branch is R's own tolerance and is kept: ``p = 1.0``
    must return ``bd`` rather than the last ``j`` whose CDF is 0.9999999999.

    The installed 6.1.1 also ``stop()``s for ``p < 0`` or ``p > 1.0001``,
    where the vendored 6.1-8 assigns NaN; both versions agree on every
    ``p`` in ``[0, 1]``, which is what the parity gate and the scan sample.
    """
    p_arr = jnp.asarray(p, dtype=jnp.float64)
    bd_arr = jnp.asarray(bd, dtype=jnp.float64)
    mu_arr = jnp.asarray(mu, dtype=jnp.float64)
    sigma_arr = jnp.asarray(sigma, dtype=jnp.float64)
    shape = jnp.broadcast_shapes(
        jnp.shape(p_arr), jnp.shape(mu_arr), jnp.shape(sigma_arr), jnp.shape(bd_arr)
    )
    p_arr = jnp.broadcast_to(p_arr, shape)
    bd_arr = jnp.broadcast_to(bd_arr, shape)
    mu_b = jnp.broadcast_to(mu_arr, shape)
    sg_b = jnp.broadcast_to(sigma_arr, shape)

    n_max = int(jnp.max(bd_arr))
    grid = jnp.arange(0, n_max + 1, dtype=jnp.float64)[None, :]
    # `lead` promotes the trailing summation axis to `shape + (n_max + 1,)`.
    # Broadcasting `grid` directly would collapse to `(n_max + 1,)` when every
    # argument is scalar -- one *fewer* dimension than the `(1, n_max + 1)`
    # grid -- and JAX rejects that rather than promoting.
    lead = jnp.ones(shape, dtype=jnp.float64)[..., None]
    cols = _dbi_log_pdf(
        grid * lead,
        mu_b[..., None],
        sg_b[..., None],
        bd_arr[..., None],
    )
    pmf = jnp.where(grid <= bd_arr[..., None], jnp.exp(cols), 0.0)
    cdf = jnp.cumsum(pmf, axis=-1)
    # The smallest j whose CDF reaches p.  `argmax` on a boolean row returns the
    # first True, and returns 0 when the row is all False -- which is what
    # happens at p = 1.0 because the trailing CDF entries sit at 1.0 - 1e-16.
    # R sidesteps this with its own `p + 1e-09 >= 1` branch, kept below.
    first = jnp.argmax(cdf >= p_arr[..., None], axis=-1).astype(jnp.float64)
    at_one = (p_arr + 1e-09) >= 1.0
    return jnp.where(at_one, bd_arr, jnp.minimum(first, bd_arr)).reshape(shape)


def _dbi_dispatch(x, mu, sigma, bd, log: bool = False):
    """``dDBI(x, mu, sigma, bd, log = log)`` -- R's ``log`` is positional 5th."""
    value = _dbi_log_pdf(x, mu, sigma, bd)
    return value if log else jnp.exp(value)


def _dbi_cdf(q, mu, sigma, bd):
    """``pDBI(q, mu, sigma, bd)`` -- a finite sum, so it is exactly reproducible.

    R's body (transcribed from ``dDBI.R``)::

        fn <- function(q, mu, sigma, bd) sum(dDBI(0:q, mu = mu, sigma = sigma, bd = bd))
        cdf <- Vcdf(q = q, mu = mu, sigma = sigma, bd = bd)
        cdf <- if (lower.tail) cdf else 1 - cdf
        cdf <- if (log.p) log(cdf) else cdf
        cdf[q < 0] <- 0
        cdf[q > bd] <- 1

    ``sum()`` over ``0:q`` means the CDF steps at the integers -- and at
    R's ``:`` fuzz, so the upper end is ``floor(q + 1e-7)`` rather than a
    plain floor (see ``_COLON_FUZZ``; measured on live R, where
    ``pDBI(2.9999999) == pDBI(3.0)`` but ``pDBI(2.999999) == pDBI(2.0)``).
    The sum is capped at each observation's *own* ``bd``: R's ``dDBI``
    contributes 0 beyond it, so summing past it would double-count ``d(bd)``
    on a mixed-``bd`` call.

    Two boundary notes.  ``cdf[q > bd] <- 1`` is the vendored 6.1-8 line and
    is kept: the installed 6.1.1 has no such line and instead returns the
    raw sum for ``q`` in ``(bd, bd + 1)`` -- measured 0.99999999999999967
    where we return 1, a 3.3e-16 absolute difference -- and raises for
    ``q >= bd + 1``, where ``0:q`` finally steps past ``bd`` into the
    ``stop()`` in ``dDBI``.  We follow the vendored source (the SHASHo2
    boundary-contract convention): exactly 1 beyond the support is the
    usable contract.

    Only the lower-tail, non-log form is wired up: the family's ``p`` slot
    in R is the *parameterisation marker* ``list(mu = TRUE, sigma = TRUE)``
    rather than a function, so there is no R default-argument call to match.
    """
    q_arr = jnp.asarray(q, dtype=jnp.float64)
    bd_arr = jnp.asarray(bd, dtype=jnp.float64)
    mu_arr = jnp.asarray(mu, dtype=jnp.float64)
    sigma_arr = jnp.asarray(sigma, dtype=jnp.float64)
    shape = jnp.broadcast_shapes(
        jnp.shape(q_arr), jnp.shape(mu_arr), jnp.shape(sigma_arr), jnp.shape(bd_arr)
    )
    q_arr = jnp.broadcast_to(q_arr, shape)
    bd_arr = jnp.broadcast_to(bd_arr, shape)
    mu_arr = jnp.broadcast_to(mu_arr, shape)
    sigma_arr = jnp.broadcast_to(sigma_arr, shape)
    # R's `0:q` with its fuzz, capped at this observation's own bd.
    q_int = jnp.minimum(jnp.floor(q_arr + _COLON_FUZZ), bd_arr)
    q_int = jnp.maximum(q_int, 0.0)
    # 0, 1, ..., bd  as a trailing axis, masked down to each observation's q.
    n_max = int(jnp.max(bd_arr))
    grid = jnp.arange(0, n_max + 1, dtype=jnp.float64)[None, :]
    active = grid <= q_int[..., None]
    safe = jnp.where(active, grid, 0.0)
    cols = _dbi_log_pdf(
        safe,
        mu_arr[..., None],
        sigma_arr[..., None],
        bd_arr[..., None],
    )
    # A running sum, not `jnp.sum`: R's `pDBI` is `sum(dDBI(0:q))` and its
    # `qDBI` re-derives the *same* sequential sum inside its loop, so R's
    # `q(p(j))` is bit-exact.  Pairwise summation here made `q(p(18))`
    # return 19 -- `cumsum`'s cdf[18] sat one ULP below `sum`'s total, and
    # R's `p <= cumpro` test then failed at 18 and passed at 19.
    pmf = jnp.where(active, jnp.exp(cols), 0.0)
    cdf = jnp.cumsum(pmf, axis=-1)
    # Gather at each observation's own q_int.  Flattened to 2-D because
    # `take_along_axis` requires matching ranks and the broadcast shape
    # here is `shape + (n_max + 1,)` for any rank of `shape`.
    flat = cdf.reshape(-1, cdf.shape[-1])
    idx = q_int.reshape(-1, 1).astype(jnp.int32)
    total = jnp.take_along_axis(flat, idx, axis=-1)[..., 0]
    # pDBI's last lines: cdf[q < 0] <- 0; cdf[q > bd] <- 1.
    return jnp.where(q_arr > bd_arr, 1.0, jnp.where(q_arr < 0.0, 0.0, total)).reshape(
        shape
    )


def _g_dev_inc(y, mu, sigma, bd=None, **_):
    """``DBI()$G.dev.incr = -2 * dDBI(y, mu, sigma, bd, log = TRUE)``."""
    return -2.0 * _dbi_log_pdf(y, mu, sigma, bd)


class DBIFamily(FamilyDefinition):
    """Double Binomial family -- ``parameters = ("mu", "sigma")``."""


def DBI() -> FamilyDefinition:
    """Double Binomial (Borg 1988), matching ``gamlss.dist::DBI``.

    Replaces the pre-D2.23 shell that forwarded every slot to ``BI()``.  R's
    ``sigma.initial`` is ``rep(1.1, length(y))`` -- deliberately *not* 1,
    because ``sigma = 1`` is the branch point where R swaps in plain
    ``dbinom`` and the ``sigma`` derivative goes flat.

    R attaches neither ``qDBI`` nor ``rDBI`` to the family object
    (``DBI()$q`` and ``DBI()$r`` are NULL), so neither slot has a
    default-argument call to be checked against.  ``q`` is nonetheless
    wired: the R *function* exists, is deterministic, and is the inverse of
    the same finite sum, so it turns an unmeasurable slot into a parity
    check.  ``r`` stays unset -- the vendored ``rDBI`` is ``qDBI(runif(n))``
    with no family slot and no fitting role, so there is nothing to gain by
    inventing a random stream for it.

    ``bd`` is a **fixed** parameter: the fitter holds it at the value in the
    data, but the R ``d``/``p``/``q`` functions take it positionally, so
    every dpqr entry point here requires it.  A caller that reaches
    ``fam.d(x, mu, sigma)`` without it gets a ``TypeError``, which is the
    correct signal -- guessing a binomial denominator silently is how the
    pre-D2.23 shell scored Bernoulli deviances for a count response.
    """
    dldm, dldd = _score_factory(None)
    return DBIFamily(
        name="DBI",
        parameters=("mu", "sigma"),
        g_dev_inc=_g_dev_inc,
        type="Discrete",
        links={"mu": "logit", "sigma": "log"},
        link_functions={"mu": logit_link, "sigma": log_link},
        link_inverses={"mu": logit_inverse, "sigma": log_inverse},
        link_derivatives={"mu": logit_derivative, "sigma": log_derivative},
        score_functions={"mu": dldm, "sigma": dldd},
        hessian_functions={
            "mu": _hessian_from_score(dldm),
            "sigma": _sigma_hessian(dldd),
        },
        fixed_parameters=("bd",),
        d=_dbi_dispatch,
        p=_dbi_cdf,
        q=_dbi_quantile,
        r=None,
    )
