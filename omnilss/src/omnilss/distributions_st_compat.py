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

    # p via numerical integration of the density (R ST1.R/ST2.R use integrate too).
    def _st_p(q, mu=0.0, sigma=1.0, nu=0.0, tau=2.0, lower_tail=True, log_p=False):
        import jax.numpy as jnp

        q_arr = jnp.asarray(q, dtype=jnp.float64)
        mu_j = jnp.asarray(mu, dtype=jnp.float64)
        sigma_j = jnp.asarray(sigma, dtype=jnp.float64)
        nu_j = jnp.asarray(nu, dtype=jnp.float64)
        tau_j = jnp.asarray(tau, dtype=jnp.float64)

        # Shared grid over the real line (transform -10..10 in z-space).
        z_grid = jnp.linspace(-12.0, 12.0, 8192)
        x_grid = mu_j + sigma_j * z_grid
        d_grid = jnp.exp(log_pdf_func(x_grid, mu_j, sigma_j, nu_j, tau_j))
        dz = 24.0 / (len(z_grid) - 1)
        mass = (d_grid[:-1] + d_grid[1:]) * 0.5 * dz
        cum = jnp.concatenate([jnp.zeros((1,)), jnp.cumsum(mass)], axis=0)

        # Map q to z and interpolate the cumulative mass.
        z_q = (q_arr - mu_j) / sigma_j
        z_clip = jnp.clip(z_q, -12.0, 12.0)
        idx = jnp.searchsorted(z_grid, z_clip, side="right") - 1
        idx = jnp.clip(idx, 0, len(z_grid) - 2)
        z0 = z_grid[idx]
        frac = (z_clip - z0) / jnp.maximum(z_grid[idx + 1] - z0, 1e-12)
        cdf = cum[idx] + frac * (cum[idx + 1] - cum[idx])
        cdf = jnp.clip(cdf, 0.0, 1.0)
        if not lower_tail:
            cdf = 1.0 - cdf
        if log_p:
            cdf = jnp.log(cdf)
        return cdf

    def _st_q(p, mu=0.0, sigma=1.0, nu=0.0, tau=2.0, lower_tail=True, log_p=False):
        import numpy as np
        from scipy.optimize import brentq

        p_arr = np.atleast_1d(np.asarray(p, dtype=np.float64))
        if log_p:
            p_arr = np.exp(p_arr)
        if not lower_tail:
            p_arr = 1.0 - p_arr
        p_arr = np.clip(p_arr, 1e-10, 1 - 1e-10)
        out = []
        for pv in p_arr:
            def obj(x, _pv=pv):
                return float(np.asarray(_st_p(np.array([x]), mu, sigma, nu, tau)).reshape(-1)[0]) - _pv
            # Adaptive bracket: expand until h changes sign (R ST1.R does the same).
            lo, hi = mu - 12.0 * sigma, mu + 12.0 * sigma
            found = False
            for _ in range(6):
                try:
                    if obj(lo) * obj(hi) < 0.0:
                        found = True
                        break
                except Exception:
                    pass
                lo, hi = mu - 40.0 * (hi - mu), mu + 40.0 * (mu - lo)
            if not found:
                try:
                    lo, hi = mu - 1000.0 * sigma, mu + 1000.0 * sigma
                    if obj(lo) * obj(hi) < 0.0:
                        found = True
                except Exception:
                    pass
            if found:
                try:
                    out.append(brentq(obj, lo, hi, xtol=1e-8))
                    continue
                except Exception:
                    pass
            out.append(np.nan)
        res = np.array(out)
        if np.ndim(p) == 0:
            return res[0]
        return res

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


def ST4() -> FamilyDefinition:
    return _clone_with_name(ST3C(), "ST4", ST4Family)
