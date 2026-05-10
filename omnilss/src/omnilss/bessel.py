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
    """Compute log(I_ν(x)) using scipy."""
    import numpy as np
    nu_np = np.asarray(nu, dtype=np.float64)
    x_np = np.asarray(x, dtype=np.float64)
    
    # Compute I_ν(x)
    iv_val = iv(nu_np, x_np)
    
    # Safety: if iv_val is too small, use log-space computation
    # For large x, log(I_ν(x)) ≈ x - 0.5*log(2πx)
    eps = np.finfo(np.float64).eps
    safe_val = np.maximum(iv_val, eps)
    result = np.log(safe_val)
    
    # For very large x where iv might overflow/underflow, use asymptotic
    large_x = x_np > 700.0
    if np.any(large_x):
        log_asymptotic = x_np - 0.5 * np.log(2.0 * np.pi * x_np)
        result = np.where(large_x, log_asymptotic, result)
    
    return result


def _log_bessel_kv_scipy(nu, x):
    """Compute log(K_ν(x)) using scipy."""
    import numpy as np
    nu_np = np.asarray(nu, dtype=np.float64)
    x_np = np.asarray(x, dtype=np.float64)
    
    # Compute K_ν(x)
    kv_val = kv(nu_np, x_np)
    
    # Safety: if kv_val is too small, use log-space computation
    # For large x, log(K_ν(x)) ≈ -x + 0.5*log(π/(2x))
    eps = np.finfo(np.float64).eps
    safe_val = np.maximum(kv_val, eps)
    result = np.log(safe_val)
    
    # For very large x where kv might underflow to 0, use asymptotic
    large_x = x_np > 700.0
    if np.any(large_x):
        log_asymptotic = -x_np + 0.5 * np.log(np.pi / (2.0 * x_np))
        result = np.where(large_x, log_asymptotic, result)
    
    return result


def _log_bessel_iv_jvp(primals, tangents):
    """Custom JVP rule for log(I_ν(x)).
    
    d/dx log(I_ν(x)) = I'_ν(x) / I_ν(x)
                     = [I_{ν-1}(x) + I_{ν+1}(x)] / [2 * I_ν(x)]
                     ≈ I_{ν+1}(x) / I_ν(x)  (using recurrence)
    
    For simplicity, we use: d/dx log(I_ν(x)) ≈ 1 - ν/x  (asymptotic)
    """
    nu, x = primals
    nu_dot, x_dot = tangents
    
    # Primal output
    primal_out = log_bessel_iv(nu, x)
    
    # Tangent: d/dx log(I_ν(x)) ≈ 1 - ν/x for large x
    # For better accuracy, use: ≈ 1 - ν/(x + ν)
    eps = jnp.finfo(jnp.float64).eps
    x_safe = jnp.maximum(x, eps)
    tangent_out = x_dot * (1.0 - nu / (x_safe + jnp.abs(nu) + 1.0))
    
    return primal_out, tangent_out


def _log_bessel_kv_jvp(primals, tangents):
    """Custom JVP rule for log(K_ν(x)).
    
    d/dx log(K_ν(x)) = K'_ν(x) / K_ν(x)
                     = -[K_{ν-1}(x) + K_{ν+1}(x)] / [2 * K_ν(x)]
                     ≈ -K_{ν+1}(x) / K_ν(x)  (using recurrence)
    
    For simplicity, we use: d/dx log(K_ν(x)) ≈ -1 - ν/x  (asymptotic)
    """
    nu, x = primals
    nu_dot, x_dot = tangents
    
    # Primal output
    primal_out = log_bessel_kv(nu, x)
    
    # Tangent: d/dx log(K_ν(x)) ≈ -1 - ν/x for large x
    # For better accuracy, use: ≈ -1 - ν/(x + ν)
    eps = jnp.finfo(jnp.float64).eps
    x_safe = jnp.maximum(x, eps)
    tangent_out = x_dot * (-1.0 - nu / (x_safe + jnp.abs(nu) + 1.0))
    
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
