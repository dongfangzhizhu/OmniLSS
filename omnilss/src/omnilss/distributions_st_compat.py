"""Compatibility constructors for skew-t families ST1/ST2/ST3/ST4.

R reference: `gamlss.dist` exports `ST1`, `ST2`, `ST3`, `ST4`.

- ST1: Azzalini type 1: loglik = log(2) + pt(nu*z, df=tau) + dt(z, df=tau) - log(sigma)
- ST2: Azzalini type 2: same with w = nu*sqrt((tau+1)/(tau+z^2))*z, pt(w, df=tau+1)
- ST3/ST4: routed to the ST3C numeric kernel (distinct names preserved).
"""

from __future__ import annotations

from dataclasses import dataclass

from .distributions_b15 import ST3C
from .families import FamilyDefinition
from .links import identity_derivative, identity_inverse, identity_link, log_derivative, log_inverse, log_link


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


def _st1_log_pdf(y, mu, sigma, nu, tau):
    """R ST1.R dST1: log(2) + pt(nu*z, df=tau) + dt(z, df=tau) - log(sigma)."""
    import jax.numpy as jnp
    from jax.scipy.stats import t as t_dist

    z = (jnp.asarray(y, dtype=jnp.float64) - jnp.asarray(mu, dtype=jnp.float64)) / jnp.asarray(
        sigma, dtype=jnp.float64
    )
    tau_safe = jnp.maximum(jnp.asarray(tau, dtype=jnp.float64), 1e-6)
    loglik = (
        jnp.log(2.0)
        + jnp.log(_t_cdf_jax(jnp.asarray(nu, dtype=jnp.float64) * z, tau_safe))
        + t_dist.logpdf(z, tau_safe)
        - jnp.log(jnp.asarray(sigma, dtype=jnp.float64))
    )
    return loglik


def _st2_log_pdf(y, mu, sigma, nu, tau):
    """R ST2.R dST2: w = nu*sqrt((tau+1)/(tau+z^2))*z; pt(w, df=tau+1)."""
    import jax.numpy as jnp
    from jax.scipy.stats import t as t_dist

    z = (jnp.asarray(y, dtype=jnp.float64) - jnp.asarray(mu, dtype=jnp.float64)) / jnp.asarray(
        sigma, dtype=jnp.float64
    )
    tau_safe = jnp.maximum(jnp.asarray(tau, dtype=jnp.float64), 1e-6)
    lam = (tau_safe + 1.0) / (tau_safe + z * z)
    w = jnp.asarray(nu, dtype=jnp.float64) * jnp.sqrt(lam) * z
    loglik = (
        jnp.log(2.0)
        + jnp.log(_t_cdf_jax(w, tau_safe + 1.0))
        + t_dist.logpdf(z, tau_safe)
        - jnp.log(jnp.asarray(sigma, dtype=jnp.float64))
    )
    return loglik


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
                    lambda z: _density(z, ni, ti), -np.inf, z_q,
                    limit=400, epsabs=1e-12, epsrel=1e-12,
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
                return float(np.asarray(_st_p(np.array([x]), _mui, _si, _ni, _ti)).reshape(-1)[0]) - _pv

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

    return build_ad_family(
        family_class=family_class,
        name=name,
        parameters=("mu", "sigma", "nu", "tau"),
        log_pdf_func=log_pdf_func,
        type_="Continuous",
        links={"mu": "identity", "sigma": "log", "nu": "identity", "tau": "log"},
        link_functions={"mu": identity_link, "sigma": log_link, "nu": identity_link, "tau": log_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse, "nu": identity_inverse, "tau": log_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative, "nu": identity_derivative, "tau": log_derivative},
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
    return _clone_with_name(ST3C(), "ST3", ST3Family)


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
    p_arr = _np.clip(p_arr, 1e-16, 1.0 - 1e-16)
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
        if pi < 1.0 / (1.0 + k):
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
        link_functions={"mu": identity_link, "sigma": log_link, "nu": log_link, "tau": log_link},
        link_inverses={"mu": identity_inverse, "sigma": log_inverse, "nu": log_inverse, "tau": log_inverse},
        link_derivatives={"mu": identity_derivative, "sigma": log_derivative, "nu": log_derivative, "tau": log_derivative},
    )
