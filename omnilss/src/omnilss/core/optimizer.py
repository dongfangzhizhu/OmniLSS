"""Joint optimization module for OmniLSS.

This module implements joint optimization of all distribution parameters,
replacing the traditional RS (Rigby-Stasinopoulos) algorithm with modern
gradient-based optimizers from Optax.

Key Features:
- Joint optimization of all parameters (μ, σ, ν, τ)
- Multiple optimizer backends (Adam, SGD, RMSprop, Adagrad)
- Automatic convergence detection
- JIT compilation for performance
- Full type annotations

Examples:
    Basic usage with Adam optimizer:
    
    >>> from omnilss.core import JointOptimizer
    >>> optimizer = JointOptimizer(method="adam", learning_rate=0.01)
    >>> result = optimizer.optimize(loss_fn, init_params, data)
    >>> print(f"Final loss: {result.loss:.6f}")
    >>> print(f"Converged: {result.converged}")
    
    Using different optimizers:
    
    >>> # SGD
    >>> opt_sgd = JointOptimizer(method="sgd", learning_rate=0.1)
    >>> 
    >>> # RMSprop
    >>> opt_rms = JointOptimizer(method="rmsprop", learning_rate=0.001)
    >>> 
    >>> # Adagrad
    >>> opt_ada = JointOptimizer(method="adagrad", learning_rate=0.01)

References:
    - Kingma, D. P., & Ba, J. (2014). Adam: A method for stochastic optimization.
      arXiv preprint arXiv:1412.6980.
    - Tieleman, T., & Hinton, G. (2012). Lecture 6.5-rmsprop: Divide the gradient
      by a running average of its recent magnitude. COURSERA: Neural networks for
      machine learning, 4(2), 26-31.
"""

from __future__ import annotations

from typing import Protocol, Callable, Dict, Any, Tuple, Optional
from dataclasses import dataclass, field
import warnings

import jax
import jax.numpy as jnp
import optax

# Type aliases
Params = Dict[str, jnp.ndarray]
LossFn = Callable[[Params, Any], float]
GradFn = Callable[[Params, Any], Tuple[float, Params]]


class Optimizer(Protocol):
    """Protocol for optimizer implementations.
    
    This protocol defines the interface that all optimizers must implement.
    It allows for easy extension with custom optimizers.
    """
    
    def step(
        self, 
        params: Params, 
        grads: Params
    ) -> Tuple[Params, Any]:
        """Execute one optimization step.
        
        Args:
            params: Current parameters
            grads: Gradients of the loss with respect to parameters
            
        Returns:
            Tuple of (new_params, opt_state)
        """
        ...
    
    def init(self, params: Params) -> Any:
        """Initialize optimizer state.
        
        Args:
            params: Initial parameters
            
        Returns:
            Initial optimizer state
        """
        ...


@dataclass
class OptimizationResult:
    """Result of an optimization run.
    
    Attributes:
        params: Final optimized parameters
        loss: Final loss value
        n_iter: Number of iterations performed
        converged: Whether the optimization converged
        loss_history: History of loss values at each iteration
        grad_norms: History of gradient norms (if tracked)
    """
    params: Params
    loss: float
    n_iter: int
    converged: bool
    loss_history: list = field(default_factory=list)
    grad_norms: Optional[list] = None
    
    def __repr__(self) -> str:
        status = "converged" if self.converged else "not converged"
        return (
            f"OptimizationResult(loss={self.loss:.6f}, "
            f"n_iter={self.n_iter}, {status})"
        )


class JointOptimizer:
    """Joint optimizer for all distribution parameters.
    
    This optimizer simultaneously optimizes all parameters (μ, σ, ν, τ) using
    gradient-based methods from Optax. This is in contrast to the traditional
    RS algorithm which updates parameters in a coordinate descent fashion.
    
    Supported optimizers:
        - adam: Adaptive Moment Estimation (default, recommended)
        - sgd: Stochastic Gradient Descent
        - rmsprop: Root Mean Square Propagation
        - adagrad: Adaptive Gradient Algorithm
    
    Args:
        method: Optimization method ("adam", "sgd", "rmsprop", "adagrad")
        learning_rate: Learning rate (step size)
        max_iter: Maximum number of iterations
        tol: Convergence tolerance (for loss change)
        grad_tol: Gradient norm tolerance for convergence
        track_grad_norms: Whether to track gradient norms
        **kwargs: Additional arguments passed to the Optax optimizer
    
    Examples:
        >>> optimizer = JointOptimizer(
        ...     method="adam",
        ...     learning_rate=0.01,
        ...     max_iter=1000,
        ...     tol=1e-6
        ... )
        >>> 
        >>> def loss_fn(params, data):
        ...     # Compute negative log-likelihood
        ...     return -log_likelihood(params, data)
        >>> 
        >>> result = optimizer.optimize(loss_fn, init_params, data, verbose=True)
        >>> print(f"Converged in {result.n_iter} iterations")
    """
    
    def __init__(
        self,
        method: str = "adam",
        learning_rate: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
        grad_tol: float = 1e-5,
        track_grad_norms: bool = False,
        **kwargs
    ):
        self.method = method.lower()
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.grad_tol = grad_tol
        self.track_grad_norms = track_grad_norms
        self.kwargs = kwargs
        
        # Create optimizer
        self.optimizer = self._create_optimizer()
    
    def _create_optimizer(self) -> optax.GradientTransformation:
        """Create an Optax optimizer.
        
        Returns:
            Optax gradient transformation
            
        Raises:
            ValueError: If method is not recognized
        """
        if self.method == "adam":
            return optax.adam(self.learning_rate, **self.kwargs)
        elif self.method == "sgd":
            return optax.sgd(self.learning_rate, **self.kwargs)
        elif self.method == "rmsprop":
            return optax.rmsprop(self.learning_rate, **self.kwargs)
        elif self.method == "adagrad":
            return optax.adagrad(self.learning_rate, **self.kwargs)
        else:
            raise ValueError(
                f"Unknown optimization method: {self.method}. "
                f"Supported methods: adam, sgd, rmsprop, adagrad"
            )
    
    def optimize(
        self,
        loss_fn: LossFn,
        init_params: Params,
        data: Any,
        verbose: bool = False,
        print_every: int = 10
    ) -> OptimizationResult:
        """Execute optimization.
        
        Args:
            loss_fn: Loss function with signature loss_fn(params, data) -> loss
            init_params: Initial parameters
            data: Data to pass to loss function
            verbose: Whether to print progress
            print_every: Print frequency (if verbose=True)
            
        Returns:
            OptimizationResult containing final parameters and statistics
            
        Examples:
            >>> def loss_fn(params, data):
            ...     pred = params["mu"] + params["sigma"] * data["x"]
            ...     return jnp.mean((data["y"] - pred) ** 2)
            >>> 
            >>> init_params = {"mu": jnp.array(0.0), "sigma": jnp.array(1.0)}
            >>> data = {"x": jnp.array([1, 2, 3]), "y": jnp.array([2, 4, 6])}
            >>> 
            >>> optimizer = JointOptimizer(method="adam", learning_rate=0.1)
            >>> result = optimizer.optimize(loss_fn, init_params, data, verbose=True)
        """
        # Initialize
        params = init_params
        opt_state = self.optimizer.init(params)
        loss_history = []
        grad_norms = [] if self.track_grad_norms else None
        
        # Create JIT-compiled update function
        @jax.jit
        def update_step(params, opt_state, data):
            loss, grads = jax.value_and_grad(loss_fn)(params, data)
            updates, opt_state = self.optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state, loss, grads
        
        if verbose:
            print(f"Starting optimization with {self.method.upper()}")
            print(f"Learning rate: {self.learning_rate}")
            print(f"Max iterations: {self.max_iter}")
            print(f"Tolerance: {self.tol}")
            print("-" * 60)
        
        # Optimization loop
        for i in range(self.max_iter):
            params, opt_state, loss, grads = update_step(params, opt_state, data)
            loss_val = float(loss)
            loss_history.append(loss_val)
            
            # Track gradient norms if requested
            if self.track_grad_norms:
                grad_norm = self._compute_grad_norm(grads)
                grad_norms.append(float(grad_norm))
            
            # Print progress
            if verbose and i % print_every == 0:
                msg = f"Iter {i:4d}: loss = {loss_val:.6f}"
                if self.track_grad_norms:
                    msg += f", ||grad|| = {grad_norms[-1]:.6f}"
                print(msg)
            
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
                if self.track_grad_norms and grad_norms[-1] < self.grad_tol:
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
        
        warnings.warn(
            f"Optimization did not converge after {self.max_iter} iterations. "
            f"Consider increasing max_iter or adjusting learning_rate.",
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


def create_optimizer(
    method: str = "adam",
    **kwargs
) -> JointOptimizer:
    """Factory function to create an optimizer.
    
    This is a convenience function that creates a JointOptimizer with
    the specified method and parameters. In the future, this can be
    extended to support other optimizer types (e.g., L-BFGS).
    
    Args:
        method: Optimization method ("adam", "sgd", "rmsprop", "adagrad", "lbfgs")
        **kwargs: Additional arguments passed to the optimizer
        
    Returns:
        Optimizer instance
        
    Examples:
        >>> # Create Adam optimizer
        >>> opt = create_optimizer("adam", learning_rate=0.01)
        >>> 
        >>> # Create SGD optimizer with momentum
        >>> opt = create_optimizer("sgd", learning_rate=0.1, momentum=0.9)
        >>> 
        >>> # Create RMSprop optimizer
        >>> opt = create_optimizer("rmsprop", learning_rate=0.001)
    """
    if method.lower() == "lbfgs":
        # Import L-BFGS optimizer when implemented
        from .lbfgs_optimizer import LBFGSOptimizer
        return LBFGSOptimizer(**kwargs)
    else:
        return JointOptimizer(method=method, **kwargs)


# Convenience function for backward compatibility
def joint_optimize(
    loss_fn: LossFn,
    init_params: Params,
    data: Any,
    method: str = "adam",
    learning_rate: float = 0.01,
    max_iter: int = 1000,
    verbose: bool = False,
    **kwargs
) -> OptimizationResult:
    """Convenience function for joint optimization.
    
    This function creates an optimizer and runs optimization in one call.
    Useful for quick experiments and simple use cases.
    
    Args:
        loss_fn: Loss function
        init_params: Initial parameters
        data: Data
        method: Optimization method
        learning_rate: Learning rate
        max_iter: Maximum iterations
        verbose: Print progress
        **kwargs: Additional optimizer arguments
        
    Returns:
        OptimizationResult
        
    Examples:
        >>> from omnilss.core import joint_optimize
        >>> 
        >>> result = joint_optimize(
        ...     loss_fn=my_loss,
        ...     init_params=init_params,
        ...     data=data,
        ...     method="adam",
        ...     learning_rate=0.01,
        ...     verbose=True
        ... )
    """
    optimizer = create_optimizer(
        method=method,
        learning_rate=learning_rate,
        max_iter=max_iter,
        **kwargs
    )
    return optimizer.optimize(loss_fn, init_params, data, verbose=verbose)
