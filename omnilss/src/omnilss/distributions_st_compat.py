"""Compatibility constructors for skew-t families ST1/ST2/ST3/ST4.

R reference: `gamlss.dist` exports `ST1`, `ST2`, `ST3`, `ST4`.

- ST1: Azzalini type 1: loglik = log(2) + pt(nu*z, df=tau) + dt(z, df=tau) - log(sigma)
- ST2: Azzalini type 2: same with w = nu*sqrt((tau+1)/(tau+z^2))*z, pt(w, df=tau+1)
- ST3: two-piece skew-t with reciprocal scales on the two sides.
- ST4: two-piece skew-t with separate left/right degrees of freedom.
"""

from __future__ import annotations

from dataclasses import dataclass

from .families import FamilyDefinition
from .links import (
    identity_derivative,
    identity_inverse,
    identity_link,
    log_derivative,
    log_inverse,
    log_link,
)


@dataclass(frozen=True)
class ST1Family(FamilyDefinition):
    """Compatibility shell for ST1 family."""


@dataclass(frozen=True)
class ST2Family(FamilyDefinition):
    """Compatibility shell for ST2 family."""


@dataclass(frozen=True)
class ST3Family(FamilyDefinition):
    """Compatibility shell for ST3 family."""


@dataclass(frozen=True)
class ST4Family(FamilyDefinition):
    """Compatibility shell for ST4 family."""


def _t_cdf_jax(t, df):
    """Student-t CDF via regularized incomplete beta (JAX-compatible).

    For t >= 0: P(T <= t) = 1 - 0.5 * I_{df/(df+t^2)}(df/2, 1/2)
    For t <  0: P(T <= t) = 0.5 * I_{df/(df+t^2)}(df/2, 1/2)
    """
    import jax.numpy as jnp
    from jax.scipy.special import betainc

    df_safe = jnp.maximum(jnp.asarray(df, dtype=jnp.float64), 1e-6)
    t_val = jnp.asarray(t, dtype=jnp.float64)
    x = df_safe / (df_safe + t_val * t_val)
    ibeta = betainc(df_safe / 2.0, 0.5, x)
    cdf = jnp.where(t_val > 0, 1.0 - 0.5 * ibeta, 0.5 * ibeta)
    return jnp.where(t_val == 0, 0.5, cdf)


def _norm_logpdf(x):
    """Standard-normal log density -- R's ``dNO(x, mu=0, sigma=1, log=TRUE)``.

    Spelled out rather than delegated to `jax.scipy.stats.norm` because a
    `scipy.stats.*` call anywhere inside a density is a landmine: `fitting_cg`
    differentiates `family.d(..., log=True)` inside `jit`, and scipy answers a
    tracer with `__array__`, i.e. `TracerArrayConversionError`, once per CG
    cycle.  Plain `jnp` primitives are two-sided differentiable.
    """
    import jax.numpy as jnp

    x_val = jnp.asarray(x, dtype=jnp.float64)
    return -0.5 * x_val * x_val - 0.5 * jnp.log(2.0 * jnp.pi)


def _norm_logcdf(x):
    """Standard-normal log CDF -- R's ``pNO(x, mu=0, sigma=1, log.p=TRUE)``.

    `jax.scipy.special.log_ndtr` is exactly what R's `pNO(log.p=TRUE)` computes
    (both go through `log_ndtr`), and unlike a hand-rolled `log(0.5*erfc(...))`
    it stays accurate in the far left tail -- at x = -3 the naive form is off
    by 3.9 in the log, which no tolerance can hide.

    The reason this was not the first choice: `log_ndtr` has to be
    differentiable through `fitting_cg`'s `jit`-compiled Hessian.  Checked
    directly -- `grad` and `grad(grad)` of `log_ndtr` are both finite, so it is
    safe here.  (The `scipy.stats.*` hazard noted in `_norm_logpdf` does not
    apply: `jax.scipy` is pure JAX.)
    """
    import jax.numpy as jnp
    from jax.scipy.special import log_ndtr

    return log_ndtr(jnp.asarray(x, dtype=jnp.float64))


def _st1_log_pdf(y, mu, sigma, nu, tau):
    """R ST1.R dST1, including its normal-limit branch.

    R:
        loglik1 <- pt(w, df=tau, log.p=TRUE) + dt(z, df=tau, log=TRUE) + log(2) - log(sigma)
        loglik2 <- pNO(w, 0, 1, log.p=TRUE)   + dNO(z, 0, 1, log=TRUE)   + log(2) - log(sigma)
        loglik  <- ifelse(tau < 1e+06, loglik1, loglik2)

    The branch is not cosmetic.  R's own ST1 optimum on the parity sample sits
    at tau = 1.4e56, i.e. deep inside the normal limit -- and omitting the
    branch makes our density wrong *at R's solution*, not merely at the
    extremes.  ST3/ST4 carry the same `tau < 1e+06` switch; ST1/ST2 were the
    two holdouts.
    """
    import jax.numpy as jnp
    from jax.scipy.stats import t as t_dist

    z = (
        jnp.asarray(y, dtype=jnp.float64) - jnp.asarray(mu, dtype=jnp.float64)
    ) / jnp.asarray(sigma, dtype=jnp.float64)
    tau_v = jnp.asarray(tau, dtype=jnp.float64)
    tau_safe = jnp.maximum(tau_v, 1e-6)
    w = jnp.asarray(nu, dtype=jnp.float64) * z

    loglik1 = (
        jnp.log(2.0)
        + jnp.log(_t_cdf_jax(w, tau_safe))
        + t_dist.logpdf(z, tau_safe)
        - jnp.log(jnp.asarray(sigma, dtype=jnp.float64))
    )
    loglik2 = (
        jnp.log(2.0)
        + _norm_logcdf(w)
        + _norm_logpdf(z)
        - jnp.log(jnp.asarray(sigma, dtype=jnp.float64))
    )
    return jnp.where(tau_v < _GAUSS_LIMIT, loglik1, loglik2)


def _st2_log_pdf(y, mu, sigma, nu, tau):
    """R ST2.R dST2, including its normal-limit branch.

    R's dST2 mirrors dST1: the t kernel uses `df = tau + 1` for the CDF and
    `df = tau` for the density, and everything switches to the normal limit at
    `tau >= 1e+06` (with `lam` pinned to 1 there).
    """
    import jax.numpy as jnp
    from jax.scipy.stats import t as t_dist

    z = (
        jnp.asarray(y, dtype=jnp.float64) - jnp.asarray(mu, dtype=jnp.float64)
    ) / jnp.asarray(sigma, dtype=jnp.float64)
    tau_v = jnp.asarray(tau, dtype=jnp.float64)
    tau_safe = jnp.maximum(tau_v, 1e-6)
    z2 = z * z
    lam = jnp.where(tau_v < _GAUSS_LIMIT, (tau_safe + 1.0) / (tau_safe + z2), 1.0)
    nu_v = jnp.asarray(nu, dtype=jnp.float64)
    w = nu_v * jnp.sqrt(lam) * z

    loglik1 = (
        jnp.log(2.0)
        + jnp.log(_t_cdf_jax(w, tau_safe + 1.0))
        + t_dist.logpdf(z, tau_safe)
        - jnp.log(jnp.asarray(sigma, dtype=jnp.float64))
    )
    loglik2 = (
        jnp.log(2.0)
        + _norm_logcdf(w)
        + _norm_logpdf(z)
        - jnp.log(jnp.asarray(sigma, dtype=jnp.float64))
    )
    return jnp.where(tau_v < _GAUSS_LIMIT, loglik1, loglik2)


# ---------------------------------------------------------------------------
# R's closed-form scores / diagonal second derivatives (D2.18)
# ---------------------------------------------------------------------------
#
# Why these are transcribed instead of auto-differentiated
# --------------------------------------------------------
# `_t_cdf_jax` reaches `betainc(df/2, 0.5, x)`, and `betainc` has no
# derivative in `a`/`b`.  The first derivative fails too, so this is *worse*
# than the Box-Cox cluster of D2.17: there, D2.13's `custom_jvp` patched the
# score and only the Hessian re-entered `betainc`.  Here the score itself
# raises, so every RS cycle failed with
# `Betainc gradient with respect to a and b not supported` and neither family
# could be fitted at all.  `custom_jvp` cannot help: it is applied *after*
# the derivative is taken, so there is nothing to attach it to.
#
# Transcribing R's formulas is the right move for parity reasons too.  Every
# second derivative below is `-dldX^2` clamped at -1e-15 -- a rough observed
# approximation, not a true Hessian.  D2.15 established that for this project
# the target is *what R feeds the IRLS*, so a "better" second derivative makes
# parity worse.  Keep the roughness.
#
# The `dldt` forward difference with its hard-coded step is likewise
# deliberate and must stay bit-comparable: R uses 0.001 on `tau` (ST1) and
# 0.0005 on `df = tau+1` (ST2), and those are different step sizes for what
# is nominally the same derivative.  Widening either one changes the number.

_EPS_ST = 1e-15
_GAUSS_LIMIT = 1e6  # R's `ifelse(tau < 1e+06, <t kernel>, <normal>)` cutoff


def _st_pieces(y, mu, sigma, nu, tau, is_st2):
    """Shared kernel quantities for ST1/ST2, transcribed from R.

    Returns ``(z, lam, w, df, dwdz)`` where ``w`` is R's skewed-t argument.

    ST1 and ST2 differ in more than the argument: ST2's ``w`` carries
    ``sqrt(lam)`` and the density uses ``df = tau + 1`` throughout, while ST1
    uses ``w = nu*z`` with ``df = tau``.  R's ``lam`` shows up in *both*
    families' ``dldm``/``dldd`` even though only ST2's ``w`` needs it -- that
    is R's actual code, so it is reproduced rather than tidied up.
    """
    import jax.numpy as jnp
    from jax.scipy.stats import t as t_dist

    y = jnp.asarray(y, dtype=jnp.float64)
    mu = jnp.asarray(mu, dtype=jnp.float64)
    sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), 1e-12)
    nu = jnp.asarray(nu, dtype=jnp.float64)
    tau = jnp.maximum(jnp.asarray(tau, dtype=jnp.float64), 1e-6)

    z = (y - mu) / sigma
    z2 = z * z
    # R: lam <- ifelse(tau < 1e+06, (tau+1)/(tau+z^2), 1)
    lam = jnp.where(tau < _GAUSS_LIMIT, (tau + 1.0) / (tau + z2), 1.0)
    if is_st2:
        w = nu * jnp.sqrt(lam) * z
        df = tau + 1.0
        # R: dwdz <- nu*(lam^0.5)*(1 - lam*(z^2)/(tau+1))
        dwdz = nu * jnp.sqrt(lam) * (1.0 - lam * z2 / (tau + 1.0))
    else:
        w = nu * z
        df = tau
        dwdz = jnp.zeros_like(z)
    return z, lam, w, df, dwdz, t_dist


def _st_hazard(w, df, t_dist):
    """R's ``dt(w, df=tau) / pt(w, df=tau)`` -- the hazard ratio both families use."""
    import jax.numpy as jnp

    return t_dist.pdf(w, df) / jnp.maximum(_t_cdf_jax(w, df), 1e-300)


def _st_dldt_fd(w, df, t_dist, step):
    """R's forward difference of ``pt(w, df, log.p=TRUE)`` with respect to df.

    ST1 differences on ``tau`` with step 0.001; ST2 differences on
    ``df = tau + 1`` with step 0.0005.  Both are transcribed as written --
    R's own value is a finite-difference approximation, and matching it is
    the parity target.
    """
    import jax.numpy as jnp

    return (jnp.log(_t_cdf_jax(w, df + step)) - jnp.log(_t_cdf_jax(w, df - step))) / (
        2.0 * step
    )


def _st_score_and_hessian(y, mu, sigma, nu, tau, is_st2, want_hessian=False):
    """Return R's four scores and, optionally, four diagonal second derivatives."""
    import jax.numpy as jnp
    from jax.scipy.special import digamma

    z, lam, w, df, dwdz, t_dist = _st_pieces(y, mu, sigma, nu, tau, is_st2)
    z2 = z * z
    tau_v = jnp.maximum(jnp.asarray(tau, dtype=jnp.float64), 1e-6)
    hz = _st_hazard(w, df, t_dist)

    # --- ST1: w = nu*z, so dw/dz = nu.  ST2: dw/dz as R defines it. ---
    dw_dz = nu if not is_st2 else dwdz

    dldm = -hz * dw_dz / sigma + lam * z / sigma
    dldd = -hz * z * dw_dz / sigma + (lam * z2 - 1.0) / sigma
    # R: dwdv <- w/nu, then dldv <- (dt/pt)*dwdv.  At nu = 0 that is 0/0, and
    # R does return NaN there -- `ST1(nu=0)$dldv` is NaN and `d2ldv2` is NA.
    # No guard is added: "fixing" it would diverge from the reference, and R's
    # own `nu.initial` is 0.1, so the optimiser does not sit on that boundary.
    dldv = hz * (w / nu)
    j_fd = _st_dldt_fd(w, df, t_dist, 0.001 if not is_st2 else 0.0005)
    dldt = j_fd + 0.5 * (
        digamma((tau_v + 1.0) / 2.0)
        - digamma(tau_v / 2.0)
        - 1.0 / tau_v
        - jnp.log(1.0 + z2 / tau_v)
        + lam * z2 / tau_v
    )

    scores = {"mu": dldm, "sigma": dldd, "nu": dldv, "tau": dldt}
    if not want_hessian:
        return scores, None
    # R: d2ldX2 <- -dldX*dldX, then ifelse(< -1e-15, ., -1e-15)
    #
    # The NaN propagation is not incidental.  At nu = 0 `dldv` is NaN (w/nu with
    # w = 0), and R's `ifelse(NA, x, y)` returns **NA**, so `d2ldv2` is NA too
    # (verified: `is.nan(ST1(nu=0)$d2ldv2)` is FALSE, the value is NA).  A plain
    # `jnp.where(cond, a, b)` would return -1e-15 there instead, because the
    # comparison against NaN yields False.  The explicit NaN check reproduces R.
    hess = {}
    for key, val in scores.items():
        sq = -val * val
        floored = jnp.where(sq < -_EPS_ST, sq, -_EPS_ST)
        hess[key] = jnp.where(jnp.isnan(sq), jnp.nan, floored)
    return scores, hess


def _make_st_derivative_tables(is_st2):
    """Build the eight R-derivative callables for one family."""

    def _table(y, mu, sigma, nu, tau):
        return _st_score_and_hessian(y, mu, sigma, nu, tau, is_st2, want_hessian=True)

    scores, hess = {}, {}
    for key in ("mu", "sigma", "nu", "tau"):
        scores[key] = lambda y, mu, sigma, nu, tau, _k=key: _table(
            y, mu, sigma, nu, tau
        )[0][_k]
        hess[key] = lambda y, mu, sigma, nu, tau, _k=key: _table(y, mu, sigma, nu, tau)[
            1
        ][_k]
    return scores, hess


def _build_st1_st2(name, family_class, log_pdf_func):
    from .ad import build_ad_family

    # Standardized (mu=0, sigma=1) density in pure scipy/numpy.  R's pST1/pST2
    # integrate dST1(x, 0, 1, nu, tau) with integrate(); we do the same with quad
    # but avoid per-evaluation JAX dispatch (much faster).
    import scipy.stats as _sps
    import numpy as _np

    def _std_log_density(z, ni, ti, is_st2):
        z_arr = _np.asarray(z, dtype=_np.float64)
        ti_safe = max(float(ti), 1e-6)
        if is_st2:
            # ST2: w = nu*sqrt((tau+1)/(tau+z^2))*z ; df = tau+1
            lam = (ti_safe + 1.0) / (ti_safe + z_arr * z_arr)
            w = ni * _np.sqrt(lam) * z_arr
            lcdf = _sps.t.logcdf(w, ti_safe + 1.0)
        else:
            # ST1: w = nu*z ; df = tau
            w = ni * z_arr
            lcdf = _sps.t.logcdf(w, ti_safe)
        lpdf = _sps.t.logpdf(z_arr, ti_safe)
        return _np.log(2.0) + lcdf + lpdf  # standardized sigma=1

    def _density(z, _ni, _ti):
        return _np.exp(_std_log_density(z, _ni, _ti, name == "ST2"))

    def _st_p(q, mu=0.0, sigma=1.0, nu=0.0, tau=2.0, lower_tail=True, log_p=False):
        import numpy as np
        import jax.numpy as jnp
        from scipy.integrate import quad

        q_arr = np.atleast_1d(np.asarray(q, dtype=np.float64))
        mu_arr = np.atleast_1d(np.asarray(mu, dtype=np.float64))
        sigma_arr = np.atleast_1d(np.asarray(sigma, dtype=np.float64))
        nu_arr = np.atleast_1d(np.asarray(nu, dtype=np.float64))
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=np.float64))
        n = max(len(q_arr), len(mu_arr), len(sigma_arr), len(nu_arr), len(tau_arr))

        out = []
        for i in range(n):
            qi = float(q_arr[i % len(q_arr)])
            mui = float(mu_arr[i % len(mu_arr)])
            si = float(sigma_arr[i % len(sigma_arr)])
            ni = float(nu_arr[i % len(nu_arr)])
            ti = float(tau_arr[i % len(tau_arr)])
            z_q = (qi - mui) / si

            # R 的 integrate() 从 -Inf 积分无截断；重尾（tau 接近 1）时
            # z 在 -60..-600 区间密度仍非零（如 ST1 z=-123 时 p~0.002）。
            if z_q <= -600.0:
                val = 0.0
            elif z_q >= 600.0:
                val = 1.0
            else:
                val, _ = quad(
                    lambda z: _density(z, ni, ti),
                    -np.inf,
                    z_q,
                    limit=400,
                    epsabs=1e-12,
                    epsrel=1e-12,
                )
            out.append(float(min(max(val, 0.0), 1.0)))

        cdf = _np.array(out)
        if not lower_tail:
            cdf = 1.0 - cdf
        if log_p:
            cdf = _np.log(cdf)
        return jnp.asarray(cdf, dtype=jnp.float64)

    def _st_q(p, mu=0.0, sigma=1.0, nu=0.0, tau=2.0, lower_tail=True, log_p=False):
        import numpy as np
        from scipy.optimize import brentq

        p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64))
        mu_arr = np.atleast_1d(np.asarray(mu, dtype=np.float64))
        sigma_arr = np.atleast_1d(np.asarray(sigma, dtype=np.float64))
        nu_arr = np.atleast_1d(np.asarray(nu, dtype=np.float64))
        tau_arr = np.atleast_1d(np.asarray(tau, dtype=np.float64))
        if log_p:
            p_arr = np.exp(p_arr)
        if not lower_tail:
            p_arr = 1.0 - p_arr
        p_arr = np.clip(p_arr, 1e-10, 1 - 1e-10)
        n = max(len(p_arr), len(mu_arr), len(sigma_arr), len(nu_arr), len(tau_arr))
        out = []
        for i in range(n):
            pv = float(p_arr[i % len(p_arr)])
            mui = float(mu_arr[i % len(mu_arr)])
            si = float(sigma_arr[i % len(sigma_arr)])
            ni = float(nu_arr[i % len(nu_arr)])
            ti = float(tau_arr[i % len(tau_arr)])
            if si <= 0 or ti <= 0:
                out.append(np.nan)
                continue

            def obj(x, _pv=pv, _mui=mui, _si=si, _ni=ni, _ti=ti):
                return (
                    float(
                        np.asarray(_st_p(np.array([x]), _mui, _si, _ni, _ti)).reshape(
                            -1
                        )[0]
                    )
                    - _pv
                )

            # R ST1.R/ST2.R strategy: start from mu and expand one sigma at a
            # time until the bracket contains the root (mirrors uniroot usage).
            try:
                h_mu = obj(mui)
            except Exception:
                out.append(np.nan)
                continue
            found = False
            if h_mu < 0.0:
                # root is to the right of mu
                lo, hi = mui, mui + si
                j = 2
                while True:
                    try:
                        h_hi = obj(hi)
                    except Exception:
                        h_hi = np.nan
                    if h_hi >= 0.0 or (not np.isfinite(h_hi)):
                        found = np.isfinite(h_hi)
                        break
                    hi = mui + j * si
                    j += 1
                    if j > 20000:
                        break
            else:
                # root is to the left of mu
                lo, hi = mui - si, mui
                j = 2
                while True:
                    try:
                        h_lo = obj(lo)
                    except Exception:
                        h_lo = np.nan
                    if h_lo <= 0.0 or (not np.isfinite(h_lo)):
                        found = np.isfinite(h_lo)
                        break
                    lo = mui - j * si
                    j += 1
                    if j > 20000:
                        break
            if found:
                try:
                    out.append(brentq(obj, lo, hi, xtol=1e-10))
                    continue
                except Exception:
                    pass
            out.append(np.nan)
        res = np.array(out)
        if np.ndim(p) == 0:
            return res[0]
        return res

    def _st_r(key, n, mu=0.0, sigma=1.0, nu=0.0, tau=2.0):
        import jax.numpy as _jnp
        import jax.random as _jr

        u = _jr.uniform(key, shape=(n,), dtype=_jnp.float64)
        return _st_q(u, mu, sigma, nu, tau)

    # R's closed-form scores / diagonal second derivatives.  Without these the
    # AD path raises inside `betainc` on the *first* derivative (worse than the
    # Box-Cox cluster in D2.17, where a `custom_jvp` at least fixed the score).
    # See the block comment above `_st_score_and_hessian`.
    _scores, _hess = _make_st_derivative_tables(name == "ST2")

    return build_ad_family(
        family_class=family_class,
        name=name,
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=log_pdf_func,
        type_="Continuous",
        score_overrides=_scores,
        hessian_overrides=_hess,
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
        p=_st_p,
        q=_st_q,
        r=_st_r,
    )


def _clone_with_name(base: FamilyDefinition, name: str, family_class):
    return family_class(
        name=name,
        parameters=base.parameters,
        g_dev_inc=base.g_dev_inc,
        type=base.type,
        links=base.links,
        link_functions=base.link_functions,
        link_inverses=base.link_inverses,
        link_derivatives=base.link_derivatives,
        score_functions=base.score_functions,
        hessian_functions=base.hessian_functions,
        fixed_parameters=base.fixed_parameters,
        d=base.d,
        p=base.p,
        q=base.q,
        r=base.r,
    )


def ST1() -> FamilyDefinition:
    """Skew t Type 1 (Azzalini type 1): real density, not an ST3C clone."""
    return _build_st1_st2("ST1", ST1Family, _st1_log_pdf)


def ST2() -> FamilyDefinition:
    """Skew t Type 2 (Azzalini type 2): real density, not an ST3C clone."""
    return _build_st1_st2("ST2", ST2Family, _st2_log_pdf)


def ST3() -> FamilyDefinition:
    from .ad import build_ad_family

    return build_ad_family(
        family_class=ST3Family,
        name="ST3",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_st3_log_pdf,
        type_="Continuous",
        p=_st3_p,
        q=_st3_q,
        r=_st3_r,
        links={"mu": "identity", "sigma": "log", "nu": "log", "tau": "log"},
        link_functions={
            "mu": identity_link,
            "sigma": log_link,
            "nu": log_link,
            "tau": log_link,
        },
        link_inverses={
            "mu": identity_inverse,
            "sigma": log_inverse,
            "nu": log_inverse,
            "tau": log_inverse,
        },
        link_derivatives={
            "mu": identity_derivative,
            "sigma": log_derivative,
            "nu": log_derivative,
            "tau": log_derivative,
        },
    )


def _st3_log_pdf(y, mu, sigma, nu, tau):
    """R ST3.R dST3: reciprocal scales left/right of ``mu``."""
    import jax.numpy as _jnp
    from jax.scipy.stats import t as _jt

    sigma = _jnp.maximum(
        _jnp.asarray(sigma, dtype=_jnp.float64), _jnp.finfo(_jnp.float64).eps
    )
    nu = _jnp.maximum(
        _jnp.asarray(nu, dtype=_jnp.float64), _jnp.finfo(_jnp.float64).eps
    )
    tau = _jnp.maximum(
        _jnp.asarray(tau, dtype=_jnp.float64), _jnp.finfo(_jnp.float64).eps
    )
    y = _jnp.asarray(y, dtype=_jnp.float64)
    mu = _jnp.asarray(mu, dtype=_jnp.float64)
    z = (y - mu) / sigma
    log_norm = _jnp.log(2.0 * nu / (1.0 + nu * nu)) - _jnp.log(sigma)
    return _jnp.where(
        y < mu, _jt.logpdf(nu * z, tau) + log_norm, _jt.logpdf(z / nu, tau) + log_norm
    )


def _st3_p(q, mu=0.0, sigma=1.0, nu=1.0, tau=10.0, lower_tail=True, log_p=False):
    import numpy as _np
    import jax.numpy as jnp
    from scipy.stats import t as _spt

    q_arr, mu_arr, sigma_arr, nu_arr, tau_arr = [
        _np.atleast_1d(_np.asarray(v, dtype=_np.float64))
        for v in (q, mu, sigma, nu, tau)
    ]
    n = max(map(len, (q_arr, mu_arr, sigma_arr, nu_arr, tau_arr)))
    out = []
    for i in range(n):
        qi, mi, si, ni, ti = [
            float(v[i % len(v)]) for v in (q_arr, mu_arr, sigma_arr, nu_arr, tau_arr)
        ]
        z = (qi - mi) / si
        if qi < mi:
            value = 2.0 * _spt.cdf(ni * z, ti) / (1.0 + ni * ni)
        else:
            value = (1.0 + 2.0 * ni * ni * (_spt.cdf(z / ni, ti) - 0.5)) / (
                1.0 + ni * ni
            )
        out.append(value)
    result = _np.asarray(out)
    if not lower_tail:
        result = 1.0 - result
    if log_p:
        result = _np.log(result)
    return jnp.asarray(result, dtype=jnp.float64)


def _st3_q(p, mu=0.0, sigma=1.0, nu=1.0, tau=10.0, lower_tail=True, log_p=False):
    import numpy as _np
    import jax.numpy as jnp
    from scipy.stats import t as _spt

    p_arr, mu_arr, sigma_arr, nu_arr, tau_arr = [
        _np.atleast_1d(_np.asarray(v, dtype=_np.float64))
        for v in (p, mu, sigma, nu, tau)
    ]
    if log_p:
        p_arr = _np.exp(p_arr)
    if not lower_tail:
        p_arr = 1.0 - p_arr
    n = max(map(len, (p_arr, mu_arr, sigma_arr, nu_arr, tau_arr)))
    out = []
    for i in range(n):
        pi, mi, si, ni, ti = [
            float(v[i % len(v)]) for v in (p_arr, mu_arr, sigma_arr, nu_arr, tau_arr)
        ]
        if pi == 0.0:
            value = -_np.inf
        elif pi == 1.0:
            value = _np.inf
        elif not _np.isfinite(pi) or pi < 0.0 or pi > 1.0:
            value = _np.nan
        elif pi < 1.0 / (1.0 + ni * ni):
            value = mi + (si / ni) * _spt.ppf(pi * (1.0 + ni * ni) / 2.0, ti)
        else:
            value = mi + (si * ni) * _spt.ppf(
                (pi * (1.0 + ni * ni) - 1.0) / (2.0 * ni * ni) + 0.5, ti
            )
        out.append(value)
    return jnp.asarray(out, dtype=jnp.float64)


def _st3_r(key, n, mu=0.0, sigma=1.0, nu=1.0, tau=10.0):
    import jax.numpy as jnp
    import jax.random as _jr

    return _st3_q(_jr.uniform(key, shape=(n,), dtype=jnp.float64), mu, sigma, nu, tau)


def _st4_log_pdf(y, mu, sigma, nu, tau):
    """R ST4.R dST4: two-piece t with degrees of freedom nu (left) and tau (right).

    k = C(tau)/C(nu), C(df)=Gamma((df+1)/2)/(sqrt(df*pi)*Gamma(df/2)).
    loglik = dt(z, df=nu)        if y < mu
           = dt(z, df=tau) + lk  if y >= mu   (lk = log(k))
    + log(2/(1+k)) - log(sigma)
    """
    import jax.numpy as _jnp
    from jax.scipy.stats import t as _jt
    from jax.scipy.special import gammaln as _gl

    eps = _jnp.finfo(_jnp.float64).eps
    sigma = _jnp.maximum(sigma, eps)
    nu = _jnp.maximum(nu, eps)
    tau = _jnp.maximum(tau, eps)
    z = (y - mu) / sigma

    lk1 = _gl(0.5) + _gl(nu / 2.0) - _gl((nu + 1.0) / 2.0) + 0.5 * _jnp.log(nu)
    lk1 = _jnp.where(nu < 1e6, lk1, 0.5 * _jnp.log(2.0 * _jnp.pi))
    lk2 = _gl(0.5) + _gl(tau / 2.0) - _gl((tau + 1.0) / 2.0) + 0.5 * _jnp.log(tau)
    lk2 = _jnp.where(tau < 1e6, lk2, 0.5 * _jnp.log(2.0 * _jnp.pi))
    lk = lk2 - lk1
    k = _jnp.exp(lk)

    loglikb = -0.5 * _jnp.log(2.0 * _jnp.pi) - 0.5 * z * z
    loglik1 = _jt.logpdf(z, nu)
    loglik1 = _jnp.where(nu < 1e6, loglik1, loglikb)
    loglik2 = _jt.logpdf(z, tau)
    loglik2 = _jnp.where(tau < 1e6, loglik2, loglikb)

    loglik = _jnp.where(y < mu, loglik1, loglik2 + lk)
    loglik = loglik + _jnp.log(2.0 / (1.0 + k)) - _jnp.log(sigma)
    return loglik


def _st4_p(q, mu=0.0, sigma=1.0, nu=1.0, tau=10.0, lower_tail=True, log_p=False):
    """R ST4.R pST4: cdf = ifelse(q<mu, 2*pt(z,nu), 1+2k*(pt(z,tau)-0.5)) / (1+k)."""
    import numpy as _np
    import jax.numpy as jnp
    from scipy.stats import t as _spt
    from scipy.special import gammaln as _gl

    q_arr = _np.atleast_1d(_np.asarray(q, dtype=_np.float64))
    mu_arr = _np.atleast_1d(_np.asarray(mu, dtype=_np.float64))
    sigma_arr = _np.atleast_1d(_np.asarray(sigma, dtype=_np.float64))
    nu_arr = _np.atleast_1d(_np.asarray(nu, dtype=_np.float64))
    tau_arr = _np.atleast_1d(_np.asarray(tau, dtype=_np.float64))
    n = max(len(q_arr), len(mu_arr), len(sigma_arr), len(nu_arr), len(tau_arr))

    out = []
    for i in range(n):
        qi = float(q_arr[i % len(q_arr)])
        mui = float(mu_arr[i % len(mu_arr)])
        si = float(sigma_arr[i % len(sigma_arr)])
        ni = float(nu_arr[i % len(nu_arr)])
        ti = float(tau_arr[i % len(tau_arr)])
        lk1 = _gl(0.5) + _gl(ni / 2.0) - _gl((ni + 1.0) / 2.0) + 0.5 * _np.log(ni)
        lk1 = lk1 if ni < 1e6 else 0.5 * _np.log(2.0 * _np.pi)
        lk2 = _gl(0.5) + _gl(ti / 2.0) - _gl((ti + 1.0) / 2.0) + 0.5 * _np.log(ti)
        lk2 = lk2 if ti < 1e6 else 0.5 * _np.log(2.0 * _np.pi)
        k = _np.exp(lk2 - lk1)
        z = (qi - mui) / si
        cdf1 = 2.0 * _spt.cdf(z, ni)
        cdf2 = 1.0 + 2.0 * k * (_spt.cdf(z, ti) - 0.5)
        cdf = cdf1 if qi < mui else cdf2
        cdf = cdf / (1.0 + k)
        out.append(float(min(max(cdf, 0.0), 1.0)))

    cdf = _np.array(out)
    if not lower_tail:
        cdf = 1.0 - cdf
    if log_p:
        cdf = _np.log(cdf)
    return jnp.asarray(cdf, dtype=jnp.float64)


def _st4_q(p, mu=0.0, sigma=1.0, nu=1.0, tau=10.0, lower_tail=True, log_p=False):
    """R ST4.R qST4: q = ifelse(p<1/(1+k), mu+sigma*qt(p(1+k)/2,nu), mu+sigma*qt((p(1+k)-1)/(2k)+0.5,tau))."""
    import numpy as _np
    import jax.numpy as jnp
    from scipy.stats import t as _spt
    from scipy.special import gammaln as _gl

    p_arr = _np.atleast_1d(_np.asarray(p, dtype=_np.float64))
    if log_p:
        p_arr = _np.exp(p_arr)
    if not lower_tail:
        p_arr = 1.0 - p_arr
    mu_arr = _np.atleast_1d(_np.asarray(mu, dtype=_np.float64))
    sigma_arr = _np.atleast_1d(_np.asarray(sigma, dtype=_np.float64))
    nu_arr = _np.atleast_1d(_np.asarray(nu, dtype=_np.float64))
    tau_arr = _np.atleast_1d(_np.asarray(tau, dtype=_np.float64))
    n = max(len(p_arr), len(mu_arr), len(sigma_arr), len(nu_arr), len(tau_arr))

    out = []
    for i in range(n):
        pi = float(p_arr[i % len(p_arr)])
        mui = float(mu_arr[i % len(mu_arr)])
        si = float(sigma_arr[i % len(sigma_arr)])
        ni = float(nu_arr[i % len(nu_arr)])
        ti = float(tau_arr[i % len(tau_arr)])
        lk1 = _gl(0.5) + _gl(ni / 2.0) - _gl((ni + 1.0) / 2.0) + 0.5 * _np.log(ni)
        lk1 = lk1 if ni < 1e6 else 0.5 * _np.log(2.0 * _np.pi)
        lk2 = _gl(0.5) + _gl(ti / 2.0) - _gl((ti + 1.0) / 2.0) + 0.5 * _np.log(ti)
        lk2 = lk2 if ti < 1e6 else 0.5 * _np.log(2.0 * _np.pi)
        k = _np.exp(lk2 - lk1)
        if pi == 0.0:
            q = -_np.inf
        elif pi == 1.0:
            q = _np.inf
        elif not _np.isfinite(pi) or pi < 0.0 or pi > 1.0:
            q = _np.nan
        elif pi < 1.0 / (1.0 + k):
            q = mui + si * _spt.ppf(pi * (1.0 + k) / 2.0, ni)
        else:
            q = mui + si * _spt.ppf((pi * (1.0 + k) - 1.0) / (2.0 * k) + 0.5, ti)
        out.append(float(q))

    res = _np.array(out)
    return jnp.asarray(res, dtype=jnp.float64)


def _st4_r(key, n, mu=0.0, sigma=1.0, nu=1.0, tau=10.0):
    """R ST4.R rST4: inverse-CDF sampling."""
    import jax.numpy as jnp
    import jax.random as _jr

    u = _jr.uniform(key, shape=(n,), dtype=jnp.float64)
    return _st4_q(u, mu, sigma, nu, tau)


def ST4() -> FamilyDefinition:
    """Skew t Type 4 (two-piece t with separate left/right df nu/tau)."""
    from .ad import build_ad_family

    return build_ad_family(
        family_class=ST4Family,
        name="ST4",
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=_st4_log_pdf,
        type_="Continuous",
        p=_st4_p,
        q=_st4_q,
        r=_st4_r,
        links={"mu": "identity", "sigma": "log", "nu": "log", "tau": "log"},
        link_functions={
            "mu": identity_link,
            "sigma": log_link,
            "nu": log_link,
            "tau": log_link,
        },
        link_inverses={
            "mu": identity_inverse,
            "sigma": log_inverse,
            "nu": log_inverse,
            "tau": log_inverse,
        },
        link_derivatives={
            "mu": identity_derivative,
            "sigma": log_derivative,
            "nu": log_derivative,
            "tau": log_derivative,
        },
    )
