"""JAX Auto-Differentiation support for GAMLSS Families."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import jax
import jax.numpy as jnp

from .families import FamilyDefinition


def _prepare_jax_args(args: tuple[Any, ...]) -> tuple[list[jnp.ndarray], bool]:
    """Broadcast scalar parameters to match vector inputs for vmapped calls.
    
    Optimized version with reduced overhead.
    """
    # Fast path for common case: all arrays of same size
    arrays = [jnp.asarray(arg, dtype=jnp.float64) for arg in args]
    
    # Quick check if all same size
    sizes = [int(arr.size) if arr.ndim > 0 else 1 for arr in arrays]
    target_size = max(sizes)
    
    if all(s == target_size for s in sizes):
        # All same size - fast path
        scalar_output = target_size == 1
        prepared = [jnp.reshape(arr, (-1,)) if arr.ndim > 0 else jnp.reshape(arr, (1,)) for arr in arrays]
        return prepared, scalar_output
    
    # Slow path: need broadcasting
    scalar_output = target_size == 1 and all(arr.ndim == 0 for arr in arrays)
    prepared: list[jnp.ndarray] = []
    
    for arr in arrays:
        flat = jnp.reshape(arr, (-1,)) if arr.ndim > 0 else jnp.reshape(arr, (1,))
        if flat.size == target_size:
            prepared.append(flat)
        elif flat.size == 1:
            prepared.append(jnp.repeat(flat, target_size))
        else:
            raise ValueError(
                "Arguments must be broadcastable to a common length; "
                f"got sizes {sizes}"
            )
    
    return prepared, scalar_output


def build_ad_family(
    family_class: type[FamilyDefinition],
    name: str,
    parameters: tuple[str, ...],
    log_pdf_func: Callable[..., jnp.ndarray],
    type_: str = "Continuous",
    links: Mapping[str, str] | None = None,
    link_functions: Mapping[str, Callable[..., Any]] | None = None,
    link_inverses: Mapping[str, Callable[..., Any]] | None = None,
    link_derivatives: Mapping[str, Callable[..., Any]] | None = None,
    fixed_parameters: tuple[str, ...] | None = None,
    d: Callable[..., Any] | None = None,
    p: Callable[..., Any] | None = None,
    q: Callable[..., Any] | None = None,
    r: Callable[..., Any] | None = None,
) -> FamilyDefinition:
    """Build a FamilyDefinition dynamically using JAX auto-differentiation (AD).
    
    This factory massively accelerates adding new families to OmniLSS. 
    By just providing the raw mathematical scalar `log_pdf_func`, JAX AD provides
    exact analytic derivatives for score and hessian functions mapping over arrays.
    
    Parameters
    ----------
    d, p, q, r : callable, optional
        Explicit dpqr functions. If provided, they override the auto-generated
        placeholders and are JIT-compiled for performance.
    """
    # Pre-compile the vmapped log_pdf for reuse across g_dev_inc and _d
    _vmap_log_pdf = jax.jit(jax.vmap(log_pdf_func))

    # Pre-compile vmapped gradients and hessians for each parameter
    _vmap_grads = {}
    _vmap_hess = {}
    for i, par in enumerate(parameters):
        idx = i + 1  # argnums=i+1 because y is arg 0
        _vmap_grads[par] = jax.jit(jax.vmap(jax.grad(log_pdf_func, argnums=idx)))
        _vmap_hess[par] = jax.jit(jax.vmap(
            jax.grad(jax.grad(log_pdf_func, argnums=idx), argnums=idx)
        ))
    
    # Warm-up compilation with dummy data to avoid first-call overhead
    # This significantly improves performance for small to medium datasets
    try:
        dummy_size = 10
        dummy_args = [jnp.ones(dummy_size, dtype=jnp.float64) * 0.5 for _ in range(len(parameters) + 1)]
        _ = _vmap_log_pdf(*dummy_args)
        for par in parameters:
            _ = _vmap_grads[par](*dummy_args)
            _ = _vmap_hess[par](*dummy_args)
    except Exception:
        # If warm-up fails, continue without it (compilation will happen on first real call)
        pass

    def _g_dev_inc(*args, **kwargs) -> jnp.ndarray:
        # Support both positional and keyword arguments
        if kwargs:
            args = [kwargs.get('y')]
            for param in parameters:
                args.append(kwargs.get(param))
        
        args_1d, scalar_output = _prepare_jax_args(tuple(args))
        res = _vmap_log_pdf(*args_1d)
        if scalar_output:
            return -2.0 * res[0]
        return -2.0 * res

    score_functions = {}
    hessian_functions = {}
    
    for par in parameters:
        def make_grad(p_name):
            vmap_grad = _vmap_grads[p_name]
            def grad_func(*args, **kwargs):
                if kwargs:
                    args = [kwargs.get('y')]
                    for param in parameters:
                        args.append(kwargs.get(param))
                args_1d, scalar_output = _prepare_jax_args(tuple(args))
                res = vmap_grad(*args_1d)
                if scalar_output:
                    return res[0]
                return res
            return grad_func
            
        def make_hess(p_name):
            vmap_h = _vmap_hess[p_name]
            def hessian_func(*args, **kwargs):
                if kwargs:
                    args = [kwargs.get('y')]
                    for param in parameters:
                        args.append(kwargs.get(param))
                args_1d, scalar_output = _prepare_jax_args(tuple(args))
                res = vmap_h(*args_1d)
                if scalar_output:
                    return res[0]
                return res
            return hessian_func
            
        score_functions[par] = make_grad(par)
        hessian_functions[par] = make_hess(par)

    # We also inject `pdf` into the class dynamically 
    # since FamilyDefinition lacks it by default but tests/plot tools need it
    def _pdf(self, *args):
        args_1d, scalar_output = _prepare_jax_args(tuple(args))
        res = _vmap_log_pdf(*args_1d)
        if scalar_output:
            return jnp.exp(res[0])
        return jnp.exp(res)

    def _d_auto(*args, log=False):
        """Density/PMF function (auto-generated from log_pdf_func)."""
        args_1d, scalar_output = _prepare_jax_args(tuple(args))
        res = _vmap_log_pdf(*args_1d)
        if scalar_output:
            return res[0] if log else jnp.exp(res[0])
        return res if log else jnp.exp(res)

    def _p_placeholder(*args, lower_tail=True, log_p=False):
        """CDF function (placeholder - returns NaN for distributions without analytical CDF)."""
        q_val = jnp.asarray(args[0] if args else 0.0, dtype=jnp.float64)
        return jnp.full_like(q_val, jnp.nan)
    
    def _q_placeholder(*args, lower_tail=True, log_p=False):
        """Quantile function (placeholder - returns NaN for distributions without analytical quantile)."""
        p_val = jnp.asarray(args[0] if args else 0.5, dtype=jnp.float64)
        return jnp.full_like(p_val, jnp.nan)
    
    def _r_placeholder(key, n, *args):
        """Random generation (placeholder - returns NaN for distributions without implementation)."""
        return jnp.full((n,), jnp.nan, dtype=jnp.float64)

    # Use provided dpqr functions if available, otherwise use auto-generated ones
    _d_final = d if d is not None else _d_auto
    _p_final = p if p is not None else _p_placeholder
    _q_final = q if q is not None else _q_placeholder
    _r_final = r if r is not None else _r_placeholder
    
    # Python lets us patch methods dynamically but since family_class is a subclass
    # we can just attach it. Because frozen dataclasses don't allow modifying instance dict,
    # we modify the class once.
    if not hasattr(family_class, "pdf"):
        setattr(family_class, "pdf", _pdf)
    if not hasattr(family_class, "d"):
        setattr(family_class, "d", _d_final)
    if not hasattr(family_class, "p"):
        setattr(family_class, "p", _p_final)
    if not hasattr(family_class, "q"):
        setattr(family_class, "q", _q_final)
    if not hasattr(family_class, "r"):
        setattr(family_class, "r", _r_final)

    return family_class(
        name=name,
        parameters=parameters,
        g_dev_inc=_g_dev_inc,
        type=type_,
        links=links,
        link_functions=link_functions,
        link_inverses=link_inverses,
        link_derivatives=link_derivatives,
        score_functions=score_functions,
        hessian_functions=hessian_functions,
        fixed_parameters=fixed_parameters,
        d=_d_final,  # Add d function
        p=_p_final,  # Add p function
        q=_q_final,  # Add q function
        r=_r_final,  # Add r function
    )
