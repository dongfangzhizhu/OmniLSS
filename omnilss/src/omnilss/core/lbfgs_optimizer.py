"""L-BFGS optimizer for OmniLSS.

This module implements the Limited-memory BFGS (L-BFGS) algorithm for
joint optimization of distribution parameters. L-BFGS is a quasi-Newton
method that approximates the Hessian using a limited history of gradients,
making it memory-efficient for large-scale problems.

Key Features:
- Memory-efficient Hessian approximation
- Fast convergence for smooth problems
- Line search for step size selection
- Suitable for large-scale optimization

Examples:
    Basic usage:
    
    >>> from omnilss.core import LBFGSOptimizer
    >>> optimizer = LBFGSOptimizer(max_iter=100, history_size=10)
    >>> result = optimizer.optimize(loss_fn, init_params, data)
    
    With custom settings:
    
    >>> optimizer = LBFGSOptimizer(
    ...     max_iter=200,
    ...     history_size=20,
    ...     learning_rate=1.0,
    ...     tol=1e-8
    ... )
    >>> result = optimizer.optimize(loss_fn, init_params, data, verbose=True)

References:
    - Nocedal, J., & Wright, S. (2006). Numerical optimization.
      Springer Science & Business Media.
    - Liu, D. C., & Nocedal, J. (1989). On the limited memory BFGS method
      for large scale optimization. Mathematical programming, 45(1-3), 503-528.
"""

from __future__ import annotations

from typing import Callable, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import warnings

import jax
import jax.numpy as jnp
import optax

from .optimizer import Optimizer, OptimizationResult, Params, LossFn


class LBFGSOptimizer:
    """L-BFGS optimizer for joint parameter optimization.
    
    L-BFGS (Limited-memory Broyden-Fletcher-Goldfarb-Shanno) is a quasi-Newton
    method that approximates the inverse Hessian using a limited history of
    gradient differences. This makes it memory-efficient while maintaining
    fast convergence properties.
    
    The algorithm maintains a history of the last `history_size` gradient
    differences and parameter updates to approximate the Hessian. This is
    particularly useful for problems with many parameters where storing
    the full Hessian would be prohibitive.
    
    Args:
        max_iter: Maximum number of iterations
        history_size: Number of gradient/parameter pairs to store (typically 5-20)
        learning_rate: Initial step size (typically 1.0 for L-BFGS)
        tol: Convergence tolerance for loss change
        grad_tol: Convergence tolerance for gradient norm
        line_search: Line search method (currently disabled for API simplicity)
        c1: Armijo condition parameter (reserved for future use)
        c2: Curvature condition parameter (reserved for future use)
        max_ls_iter: Maximum line search iterations (reserved for future use)
        
    Attributes:
        optimizer: Optax L-BFGS gradient transformation
        
    Examples:
        >>> # Simple quadratic problem
        >>> def loss_fn(params, data):
        ...     return jnp.sum(params["x"] ** 2)
        >>> 
        >>> init_params = {"x": jnp.array([1.0, 2.0, 3.0])}
        >>> optimizer = LBFGSOptimizer(max_iter=50, history_size=10)
        >>> result = optimizer.optimize(loss_fn, init_params, {})
        >>> print(f"Converged: {result.converged}")
        >>> print(f"Final loss: {result.loss:.6f}")
        
        >>> # GAMLSS-style problem
        >>> def gamlss_loss(params, data):
        ...     mu = params["mu"]
        ...     log_sigma = params["log_sigma"]
        ...     sigma = jnp.exp(log_sigma)
        ...     residuals = (data["y"] - mu) / sigma
        ...     nll = 0.5 * jnp.log(2 * jnp.pi) + log_sigma + 0.5 * residuals ** 2
        ...     return jnp.mean(nll)
        >>> 
        >>> init_params = {"mu": jnp.array(0.0), "log_sigma": jnp.array(0.0)}
        >>> optimizer = LBFGSOptimizer(max_iter=100)
        >>> result = optimizer.optimize(gamlss_loss, init_params, data, verbose=True)
    """
    
    def __init__(
        self,
        max_iter: int = 100,
        history_size: int = 10,
        learning_rate: float = 1.0,
        tol: float = 1e-6,
        grad_tol: float = 1e-5,
        line_search: Optional[str] = "zoom",
        c1: float = 1e-4,
        c2: float = 0.9,
        max_ls_iter: int = 20,
    ):
        self.max_iter = max_iter
        self.history_size = history_size
        self.learning_rate = learning_rate
        self.tol = tol
        self.grad_tol = grad_tol
        self.line_search = line_search
        self.c1 = c1
        self.c2 = c2
        self.max_ls_iter = max_ls_iter
        
        # Create Optax L-BFGS optimizer without line search
        # Note: We use a simpler approach without zoom line search to avoid
        # the complex API requirements of GradientTransformationExtraArgs
        self.optimizer = optax.lbfgs(
            learning_rate=learning_rate,
            memory_size=history_size,
            linesearch=None,  # Disable line search for simpler API
        )
    
    def optimize(
        self,
        loss_fn: LossFn,
        init_params: Params,
        data: Any,
        verbose: bool = False,
        print_every: int = 10
    ) -> OptimizationResult:
        """Execute L-BFGS optimization.
        
        Args:
            loss_fn: Loss function with signature loss_fn(params, data) -> loss
            init_params: Initial parameters
            data: Data to pass to loss function
            verbose: Whether to print progress
            print_every: Print frequency (if verbose=True)
            
        Returns:
            OptimizationResult containing final parameters and statistics
            
        Notes:
            L-BFGS typically converges faster than first-order methods like
            Adam or SGD, especially for smooth optimization problems. However,
            it may require more memory and computation per iteration.
            
        Examples:
            >>> def loss_fn(params, data):
            ...     return jnp.sum((params["x"] - data["target"]) ** 2)
            >>> 
            >>> init_params = {"x": jnp.array([0.0, 0.0])}
            >>> data = {"target": jnp.array([1.0, 2.0])}
            >>> 
            >>> optimizer = LBFGSOptimizer(max_iter=50)
            >>> result = optimizer.optimize(loss_fn, init_params, data, verbose=True)
            >>> 
            >>> print(f"Converged in {result.n_iter} iterations")
            >>> print(f"Final loss: {result.loss:.6f}")
        """
        # Initialize
        params = init_params
        opt_state = self.optimizer.init(params)
        loss_history = []
        grad_norms = []
        
        # JIT-compiled update function
        @jax.jit
        def update_step(params, opt_state, data):
            loss, grads = jax.value_and_grad(loss_fn)(params, data)
            updates, opt_state = self.optimizer.update(
                grads, opt_state, params
            )
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, grads
        
        if verbose:
            print(f"Starting L-BFGS optimization")
            print(f"History size: {self.history_size}")
            print(f"Max iterations: {self.max_iter}")
            print(f"Tolerance: {self.tol}")
            print(f"Gradient tolerance: {self.grad_tol}")
            print("-" * 60)
        
        # Optimization loop
        for i in range(self.max_iter):
            params, opt_state, loss, grads = update_step(
                params, opt_state, data
            )
            loss_val = float(loss)
            loss_history.append(loss_val)
            
            # Compute gradient norm
            grad_norm = self._compute_grad_norm(grads)
            grad_norms.append(float(grad_norm))
            
            # Print progress
            if verbose and i % print_every == 0:
                print(f"Iter {i:4d}: loss = {loss_val:.6f}, ||grad|| = {grad_norm:.6f}")
            
            # Check convergence
            if i > 0:
                # Loss change convergence
                loss_change = abs(loss_history[-1] - loss_history[-2])
                if loss_change < self.tol:
                    if verbose:
                        print("-" * 60)
                        print(f"Converged at iteration {i} (loss change < {self.tol})")
                    return OptimizationResult(
                        params=params,
                        loss=loss_val,
                        n_iter=i,
                        converged=True,
                        loss_history=loss_history,
                        grad_norms=grad_norms
                    )
                
                # Gradient norm convergence
                if grad_norm < self.grad_tol:
                    if verbose:
                        print("-" * 60)
                        print(f"Converged at iteration {i} (grad norm < {self.grad_tol})")
                    return OptimizationResult(
                        params=params,
                        loss=loss_val,
                        n_iter=i,
                        converged=True,
                        loss_history=loss_history,
                        grad_norms=grad_norms
                    )
        
        # Did not converge
        if verbose:
            print("-" * 60)
            print(f"Did not converge after {self.max_iter} iterations")
            print(f"Final loss: {loss_val:.6f}")
            print(f"Final grad norm: {grad_norm:.6f}")
        
        warnings.warn(
            f"L-BFGS did not converge after {self.max_iter} iterations. "
            f"Consider increasing max_iter or adjusting tolerance.",
            RuntimeWarning
        )
        
        return OptimizationResult(
            params=params,
            loss=loss_val,
            n_iter=self.max_iter,
            converged=False,
            loss_history=loss_history,
            grad_norms=grad_norms
        )
    
    @staticmethod
    def _compute_grad_norm(grads: Params) -> jnp.ndarray:
        """Compute L2 norm of gradients.
        
        Args:
            grads: Gradient dictionary
            
        Returns:
            L2 norm of all gradients
        """
        return jnp.sqrt(sum(
            jnp.sum(g ** 2) for g in jax.tree_util.tree_leaves(grads)
        ))
    
    def __repr__(self) -> str:
        return (
            f"LBFGSOptimizer(max_iter={self.max_iter}, "
            f"history_size={self.history_size}, "
            f"learning_rate={self.learning_rate})"
        )


# Convenience function for L-BFGS optimization
def lbfgs_optimize(
    loss_fn: LossFn,
    init_params: Params,
    data: Any,
    max_iter: int = 100,
    history_size: int = 10,
    verbose: bool = False,
    **kwargs
) -> OptimizationResult:
    """Convenience function for L-BFGS optimization.
    
    This function creates an L-BFGS optimizer and runs optimization in one call.
    Useful for quick experiments and simple use cases.
    
    Args:
        loss_fn: Loss function
        init_params: Initial parameters
        data: Data
        max_iter: Maximum iterations
        history_size: L-BFGS history size
        verbose: Print progress
        **kwargs: Additional optimizer arguments
        
    Returns:
        OptimizationResult
        
    Examples:
        >>> from omnilss.core.lbfgs_optimizer import lbfgs_optimize
        >>> 
        >>> result = lbfgs_optimize(
        ...     loss_fn=my_loss,
        ...     init_params=init_params,
        ...     data=data,
        ...     max_iter=100,
        ...     verbose=True
        ... )
        >>> print(f"Converged: {result.converged}")
    """
    optimizer = LBFGSOptimizer(
        max_iter=max_iter,
        history_size=history_size,
        **kwargs
    )
    return optimizer.optimize(loss_fn, init_params, data, verbose=verbose)
