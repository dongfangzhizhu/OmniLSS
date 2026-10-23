"""Bessel function implementations for JAX.

This module provides modified Bessel functions I_ν(x) and K_ν(x)
using scipy via jax.pure_callback for exact computation with custom gradients.

For cases where scipy is not available, we fall back to approximations.
"""

import jax.numpy as jnp
from jax.scipy.special import i0e, i1e, gammaln
import jax
import math

# Try to import scipy for exact Bessel functions
try:
    from scipy.special import iv, kv, ivp, kvp
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


def _log_bessel_iv_scipy(nu, x):
    """Compute log(I_ν(x)) using scipy's exponentially-scaled form.

    ive(v, x) = I_v(x) * exp(-x) does not overflow, so
    log(I) = log(ive) + x is exact across the whole range -- the raw iv
    reaches inf near x ~ 700 and the previous clamp/asymptotic branch
    lost the (4*nu^2-1)/(8x) correction there.
    """
    import numpy as np
    from scipy.special import ive

    nu_np = np.asarray(nu, dtype=np.float64)
    x_np = np.asarray(x, dtype=np.float64)
    scaled = ive(nu_np, x_np)
    return np.log(np.maximum(scaled, np.finfo(np.float64).tiny)) + x_np


def _log_bessel_kv_scipy(nu, x):
    """Compute log(K_ν(x)) using scipy's exponentially-scaled form.

    K_ν(x) itself drops below double eps for x >~ 35 (K_2(50) = 3.4e-23),
    so the previous "safety" clamp maximum(kv, eps) silently replaced every
    large-argument log(K) with log(eps) = -36.04 and zeroed the normalizer
    differences the Sichel-family densities depend on
    (log K_ν(α) - log K_ν(1/σ)).  Measured consequence: SI's density came
    out 1.04e+04 times too large at mu=10, sigma=0.02, nu=2 (D2.53).
    kve(v, x) = K_v(x) * exp(x) does not underflow, so
    log(K) = log(kve) - x is exact across the whole range (verified against
    log(kv) where kv itself is representable, and against the known
    asymptotic at x = 700 where kv underflows to 0).
    """
    import numpy as np
    from scipy.special import kve

    nu_np = np.asarray(nu, dtype=np.float64)
    x_np = np.asarray(x, dtype=np.float64)
    scaled = kve(nu_np, x_np)
    return np.log(np.maximum(scaled, np.finfo(np.float64).tiny)) - x_np


def _log_bessel_iv_jvp(primals, tangents):
    """Custom JVP rule for log(I_ν(x)).

    The previous rule used the asymptotic tangent ``1 - nu/(x + |nu| + 1)``
    and ignored the ``nu`` tangent; both are replaced here (D2.47, same
    defect as the K rule).

    d/dx log(I_ν(x)) = I'_ν(x) / I_ν(x) = I_{ν+1}(x)/I_ν(x) + ν/x
        (from I'_ν = (I_{ν-1} + I_{ν+1})/2 and I_{ν-1} - I_{ν+1} = 2ν I_ν/x)

    d/dν log(I_ν(x)) has no closed form here, so it is a central difference
    of the exact primal with ``_ORDER_TANGENT_STEP`` -- see the K rule for
    the measured accuracy.
    """
    nu, x = primals
    nu_dot, x_dot = tangents

    primal_out = log_bessel_iv(nu, x)

    eps = jnp.finfo(jnp.float64).eps
    x_safe = jnp.maximum(x, eps)
    log_ratio = log_bessel_iv(nu + 1.0, x_safe) - log_bessel_iv(nu, x_safe)
    ratio = jnp.exp(jnp.clip(log_ratio, -745.0, 709.0))
    d_dx = ratio + nu / x_safe
    d_dnu = _log_bessel_order_tangent(log_bessel_iv, nu, x_safe)

    tangent_out = nu_dot * d_dnu + x_dot * d_dx
    return primal_out, tangent_out


#: Central-difference step for the *order* (nu) tangent of ``log K_nu`` /
#: ``log I_nu``.  scipy exposes no derivative with respect to the order, so
#: the tangent is ``(f(nu+h) - f(nu-h)) / 2h`` on the exact primal.  With
#: h = 1e-5 the truncation error is O(h^2) ~ 1e-10 relative and the roundoff
#: contribution eps/h ~ 2e-11 -- measured against a Richardson-extrapolated
#: reference below 3e-9 on the (nu, x) grid the families use.
_ORDER_TANGENT_STEP = 1e-5


def _log_bessel_order_tangent(primal_fn, nu, x):
    """d/dnu of a log-Bessel function, as a central difference."""
    h = _ORDER_TANGENT_STEP
    return (primal_fn(nu + h, x) - primal_fn(nu - h, x)) / (2.0 * h)


def _log_bessel_kv_jvp(primals, tangents):
    """Custom JVP rule for log(K_ν(x)).

    The previous rule substituted the *asymptotic* tangent
    ``-1 - nu/(x + |nu| + 1)`` for the true derivative and dropped the
    ``nu`` tangent entirely.  The primal is exact (scipy), so every density
    was right while every AD score was wrong -- measured by gate 2 of
    ``tools/rs_family_parity_scan.py`` as mu off by 9.3e-01 (SICHEL),
    4.98e-01 (GIG) and 6.42e-01 (SI) relative (D2.44/D2.47).

    d/dx log(K_ν(x)) = K'_ν(x)/K_ν(x) = -K_{ν+1}(x)/K_ν(x) + ν/x
        (K'_ν = -(K_{ν-1} + K_{ν+1})/2 and K_{ν-1} - K_{ν+1} = -2ν K_ν/x,
        so K'_ν = -K_{ν+1} + (ν/x)K_ν)

    d/dν log(K_ν(x)) has no closed form in scipy; it is a central difference
    of the exact primal (``_ORDER_TANGENT_STEP``).
    """
    nu, x = primals
    nu_dot, x_dot = tangents

    primal_out = log_bessel_kv(nu, x)

    eps = jnp.finfo(jnp.float64).eps
    x_safe = jnp.maximum(x, eps)
    # K_{nu+1}/K_nu evaluated in log space; the clip only binds where the
    # ratio would overflow/underflow double anyway (x -> 0 or huge |nu|).
    log_ratio = log_bessel_kv(nu + 1.0, x_safe) - log_bessel_kv(nu, x_safe)
    ratio = jnp.exp(jnp.clip(log_ratio, -745.0, 709.0))
    d_dx = -ratio + nu / x_safe
    d_dnu = _log_bessel_order_tangent(log_bessel_kv, nu, x_safe)

    tangent_out = nu_dot * d_dnu + x_dot * d_dx
    return primal_out, tangent_out


if SCIPY_AVAILABLE:
    @jax.custom_jvp
    def log_bessel_iv(nu, x):
        """Logarithm of modified Bessel function log(I_ν(x)) using scipy.
        
        Args:
            nu: Order (scalar or array)
            x: Argument (must be positive)
        
        Returns:
            log(I_ν(x))
        """
        # Use pure_callback to call scipy with sequential vmap
        result_shape = jax.ShapeDtypeStruct(jnp.shape(x), jnp.float64)
        return jax.pure_callback(
            _log_bessel_iv_scipy,
            result_shape,
            nu,
            x,
            vmap_method='sequential'
        )
    
    log_bessel_iv.defjvp(_log_bessel_iv_jvp)
    
    @jax.custom_jvp
    def log_bessel_kv(nu, x):
        """Logarithm of modified Bessel function log(K_ν(x)) using scipy.
        
        Args:
            nu: Order (scalar or array)
            x: Argument (must be positive)
        
        Returns:
            log(K_ν(x))
        """
        # Use pure_callback to call scipy with sequential vmap
        result_shape = jax.ShapeDtypeStruct(jnp.shape(x), jnp.float64)
        return jax.pure_callback(
            _log_bessel_kv_scipy,
            result_shape,
            nu,
            x,
            vmap_method='sequential'
        )
    
    log_bessel_kv.defjvp(_log_bessel_kv_jvp)

else:
    # Fallback implementations when scipy is not available
    def log_bessel_iv(nu, x):
        """Logarithm of modified Bessel function log(I_ν(x)).
        
        Fallback implementation using approximations.
        """
        eps = jnp.finfo(jnp.float64).eps
        x_safe = jnp.maximum(x, eps)
        nu_abs = jnp.abs(nu)
        
        # Asymptotic for large x
        log_asymptotic = (
            x_safe 
            - 0.5 * jnp.log(2.0 * math.pi * x_safe)
            - (4.0 * nu**2 - 1.0) / (8.0 * x_safe)
        )
        
        # Small x approximation
        log_small_x_pos = nu_abs * jnp.log(x_safe / 2.0) - gammaln(nu_abs + 1.0)
        
        # Use i0e/i1e for nu ≈ 0, 1
        log_i0 = jnp.log(jnp.maximum(i0e(x_safe), eps)) + x_safe
        log_i1 = jnp.log(jnp.maximum(i1e(x_safe), eps)) + x_safe
        
        use_asymptotic = x_safe > 15.0
        
        result = jnp.where(
            use_asymptotic,
            log_asymptotic,
            jnp.where(
                nu_abs < 0.3,
                log_i0,
                jnp.where(
                    jnp.abs(nu_abs - 1.0) < 0.3,
                    log_i1,
                    jnp.where(
                        x_safe < 2.0,
                        log_small_x_pos,
                        log_asymptotic
                    )
                )
            )
        )
        
        return result
    
    def log_bessel_kv(nu, x):
        """Logarithm of modified Bessel function log(K_ν(x)).
        
        Fallback implementation using approximations.
        """
        eps = jnp.finfo(jnp.float64).eps
        x_safe = jnp.maximum(x, eps)
        nu_abs = jnp.abs(nu)
        
        # Asymptotic for large x
        log_asymptotic = (
            0.5 * jnp.log(math.pi / (2.0 * x_safe)) 
            - x_safe
            + (4.0 * nu**2 - 1.0) / (8.0 * x_safe)
        )
        
        # Small x approximations
        euler_gamma = 0.5772156649015329
        k0_approx = -jnp.log(x_safe / 2.0) - euler_gamma
        log_k0_small = jnp.log(jnp.maximum(k0_approx, eps))
        
        log_knu_small = gammaln(nu_abs) - jnp.log(2.0) + nu_abs * jnp.log(2.0 / x_safe)
        
        crossover = 5.0 + 2.0 * nu_abs
        
        result = jnp.where(
            x_safe > crossover,
            log_asymptotic,
            jnp.where(
                nu_abs < 0.1,
                jnp.where(
                    x_safe < 2.0,
                    log_k0_small,
                    log_asymptotic
                ),
                jnp.where(
                    x_safe < 2.0,
                    log_knu_small,
                    log_asymptotic
                )
            )
        )
        
        return result
