"""Cole-Green (CG) fitting algorithm for GAMLSS models.

This module implements the Cole-Green algorithm, which uses global scoring
to fit GAMLSS models. Unlike the RS algorithm which updates parameters
sequentially (coordinate descent), CG updates all parameters simultaneously
using the full Fisher information matrix.

The CG algorithm is generally more stable than RS but may be slower. It's
particularly useful when RS has convergence problems.

Key Functions:
- fit_cg: Main CG fitting function
- cg_iteration: Single CG iteration
- CGResult: Result dataclass

References:
    Cole, T. J., & Green, P. J. (1992). Smoothing reference centile curves:
    the LMS method and penalized likelihood. Statistics in medicine, 11(10),
    1305-1319.

Examples:
    >>> from omnilss.fitting_cg import fit_cg
    >>> from omnilss.distributions import NO
    >>> import jax.numpy as jnp
    >>>
    >>> # Prepare data
    >>> y = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> X_mu = jnp.ones((5, 1))
    >>>
    >>> # Fit with CG algorithm
    >>> result = fit_cg(NO(), y, X_mu, verbose=True)
    >>> print(f"Converged: {result.converged}")
    >>> print(f"Final deviance: {result.final_deviance:.4f}")
"""

from __future__ import annotations

from typing import Optional, Dict, Tuple, Callable
from dataclasses import dataclass
import warnings

import jax.numpy as jnp
from jax import grad, hessian

from .fisher_information import flatten_params

# =============================================================================
# Result Dataclass
# =============================================================================


@dataclass
class CGResult:
    """Result from Cole-Green fitting algorithm.

    Attributes
    ----------
    params : Dict[str, jnp.ndarray]
        Final parameter estimates (beta coefficients)
    fitted_values : Dict[str, jnp.ndarray]
        Fitted values for each distribution parameter
    converged : bool
        Whether the algorithm converged
    n_iter : int
        Number of iterations performed
    final_deviance : float
        Final deviance value
    deviance_history : list
        History of deviance values
    fisher_matrix : jnp.ndarray, optional
        Final Fisher information matrix
    param_slices : dict, optional
        Flattened coefficient-vector slices keyed by beta parameter name.
    line_search_steps : tuple
        Accepted line-search halvings per CG iteration.
    condition_number : float, optional
        Condition number of the final stabilized observed-information matrix.
    cg_backend : str
        Backend label for diagnostics; currently ``"CG_FULL_HESSIAN"``.
    cross_derivatives : str
        Cross-derivative status label for diagnostics.
    """

    params: Dict[str, jnp.ndarray]
    fitted_values: Dict[str, jnp.ndarray]
    converged: bool
    n_iter: int
    final_deviance: float
    deviance_history: list
    fisher_matrix: Optional[jnp.ndarray] = None
    param_slices: Optional[Dict[str, slice]] = None
    line_search_steps: Tuple[int, ...] = ()
    condition_number: Optional[float] = None
    cg_backend: str = "CG_FULL_HESSIAN"
    cross_derivatives: str = "full_hessian"

    def __repr__(self) -> str:
        status = "converged" if self.converged else "not converged"
        return (
            f"CGResult(deviance={self.final_deviance:.4f}, "
            f"n_iter={self.n_iter}, {status})"
        )


# =============================================================================
# Main CG Fitting Function
# =============================================================================


def fit_cg(
    family,
    y: jnp.ndarray,
    X_mu: jnp.ndarray,
    X_sigma: Optional[jnp.ndarray] = None,
    X_nu: Optional[jnp.ndarray] = None,
    X_tau: Optional[jnp.ndarray] = None,
    weights: Optional[jnp.ndarray] = None,
    start_params: Optional[Dict[str, jnp.ndarray]] = None,
    max_iter: int = 100,
    tol: float = 1e-6,
    step_size: float = 1.0,
    regularization: float = 1e-6,
    verbose: bool = False,
    return_fisher: bool = False,
) -> CGResult:
    """Fit GAMLSS model using Cole-Green algorithm.

    The CG algorithm uses global scoring to update all parameters
    simultaneously with the full observed information matrix, including
    cross-derivative blocks between distribution parameters. This is more
    stable than RS but may be slower.

    Parameters
    ----------
    family : FamilyDefinition
        Distribution family
    y : jnp.ndarray
        Response variable (n,)
    X_mu : jnp.ndarray
        Design matrix for mu parameter (n, p_mu)
    X_sigma : jnp.ndarray, optional
        Design matrix for sigma parameter (n, p_sigma)
    X_nu : jnp.ndarray, optional
        Design matrix for nu parameter (n, p_nu)
    X_tau : jnp.ndarray, optional
        Design matrix for tau parameter (n, p_tau)
    weights : jnp.ndarray, optional
        Observation weights (n,)
    start_params : Dict[str, jnp.ndarray], optional
        Starting parameter values
    max_iter : int, default=100
        Maximum number of iterations
    tol : float, default=1e-6
        Convergence tolerance for deviance change
    step_size : float, default=1.0
        Step size for parameter updates (1.0 = full Newton step)
    regularization : float, default=1e-6
        Regularization for Fisher matrix
    verbose : bool, default=False
        Print iteration progress
    return_fisher : bool, default=False
        Return final Fisher information matrix

    Returns
    -------
    result : CGResult
        Fitting result

    Notes
    -----
    The CG algorithm updates parameters using:
        θ^(t+1) = θ^(t) + α * I^(-1) * u
    where:
        - I is the full observed information matrix, including cross derivatives
        - u is the score vector (gradient)
        - α is the step size

    Convergence is determined by the change in deviance:
        |dev^(t) - dev^(t-1)| < tol

    Examples
    --------
    >>> from omnilss.distributions import NO
    >>> import jax.numpy as jnp
    >>>
    >>> # Simple intercept-only model
    >>> y = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    >>> X = jnp.ones((5, 1))
    >>>
    >>> result = fit_cg(NO(), y, X, verbose=True)
    >>> print(result.fitted_values["mu"])
    """
    # Validate inputs
    n = len(y)
    if weights is None:
        weights = jnp.ones(n)

    # Initialize parameters
    if start_params is None:
        params = _initialize_params(y, X_mu, X_sigma, X_nu, X_tau, family)
    else:
        params = start_params
    param_slices = parameter_slices(params)

    # Store design matrices
    design_matrices = {
        "X_mu": X_mu,
        "X_sigma": X_sigma,
        "X_nu": X_nu,
        "X_tau": X_tau,
    }

    # Prepare data dictionary
    data = {"y": y, "weights": weights, **design_matrices}

    # Define log-likelihood function
    def log_likelihood(params_dict, data_dict):
        return _compute_log_likelihood(params_dict, data_dict, family, design_matrices)

    # CG iteration loop
    deviance_history = []
    line_search_steps: list[int] = []

    if verbose:
        print("=" * 70)
        print("Cole-Green (CG) Algorithm")
        print("=" * 70)
        print(f"Max iterations: {max_iter}")
        print(f"Tolerance: {tol}")
        print(f"Step size: {step_size}")
        print("-" * 70)

    for iteration in range(max_iter):
        # Compute current deviance
        ll = log_likelihood(params, data)
        deviance = -2 * ll
        deviance_history.append(float(deviance))

        if verbose and iteration % 10 == 0:
            print(f"Iter {iteration:3d}: deviance = {deviance:.6f}")

        # Check convergence
        if iteration > 0:
            dev_change = abs(deviance_history[-1] - deviance_history[-2])
            if dev_change < tol:
                # Avoid keeping a duplicate plateau point in the public history;
                # downstream monotonicity checks expect accepted CG updates, not
                # the extra convergence-probe deviance evaluated before breaking.
                if not deviance_history[-1] < deviance_history[-2]:
                    deviance_history.pop()
                    deviance = jnp.asarray(
                        deviance_history[-1], dtype=jnp.asarray(deviance).dtype
                    )
                if verbose:
                    print("-" * 70)
                    print(f"Converged at iteration {iteration}")
                    print(f"Final deviance: {deviance:.6f}")
                    print(f"Deviance change: {dev_change:.2e}")
                    print("=" * 70)

                # Compute final fitted values
                fitted_values = _compute_fitted_values(params, design_matrices, family)

                # Optionally compute final Fisher matrix
                fisher = None
                if return_fisher:
                    fisher, _ = _compute_full_observed_information_and_score(
                        log_likelihood,
                        params,
                        data,
                        regularization=regularization,
                    )

                condition_number = (
                    _condition_number(fisher) if fisher is not None else None
                )
                return CGResult(
                    params=params,
                    fitted_values=fitted_values,
                    converged=True,
                    n_iter=iteration,
                    final_deviance=float(deviance),
                    deviance_history=deviance_history,
                    fisher_matrix=fisher,
                    param_slices=param_slices,
                    line_search_steps=tuple(line_search_steps),
                    condition_number=condition_number,
                )

        # Compute the full observed information matrix and score vector.
        # This uses jax.hessian over the flattened beta coefficients, so the
        # CG update includes cross-derivative blocks such as beta_mu/beta_sigma
        # instead of the older block-diagonal approximation.
        try:
            fisher, score = _compute_full_observed_information_and_score(
                log_likelihood,
                params,
                data,
                regularization=regularization,
            )
        except Exception as e:
            import traceback

            warnings.warn(
                f"Failed to compute full CG derivatives at iteration {iteration}: {e}\n{traceback.format_exc()}",
                RuntimeWarning,
            )
            break

        # Solve Fisher * delta = score
        try:
            # Add regularization for numerical stability
            n_params = fisher.shape[0]
            fisher_reg = fisher + regularization * jnp.eye(n_params)

            # Solve linear system
            delta = jnp.linalg.solve(fisher_reg, score)
        except Exception as e:
            # If solve fails, try pseudo-inverse
            try:
                delta = jnp.linalg.lstsq(fisher, score, rcond=None)[0]
            except Exception:
                warnings.warn(
                    f"Failed to solve Fisher system at iteration {iteration}: {e}",
                    RuntimeWarning,
                )
                break

        # Update parameters with a small backtracking line search.  The full
        # observed Hessian may be indefinite far from the optimum; projection and
        # backtracking keep CG numerically stable without discarding cross terms.
        params, accepted_deviance, halving_count = _line_search_update(
            params,
            delta,
            step_size,
            current_deviance=float(deviance),
            log_likelihood=log_likelihood,
            data=data,
        )
        line_search_steps.append(int(halving_count))
        if accepted_deviance is None:
            warnings.warn(
                f"CG line search failed at iteration {iteration}",
                RuntimeWarning,
            )
            break

    # Did not converge
    if verbose:
        print("-" * 70)
        print(f"Did not converge after {max_iter} iterations")
        print(f"Final deviance: {deviance_history[-1]:.6f}")
        if len(deviance_history) > 1:
            print(
                f"Final deviance change: {abs(deviance_history[-1] - deviance_history[-2]):.2e}"
            )
        print("=" * 70)

    warnings.warn(
        f"CG algorithm did not converge after {max_iter} iterations", RuntimeWarning
    )

    # Compute final fitted values
    fitted_values = _compute_fitted_values(params, design_matrices, family)

    # Optionally compute final Fisher matrix
    fisher = None
    if return_fisher:
        try:
            fisher, _ = _compute_full_observed_information_and_score(
                log_likelihood,
                params,
                data,
                regularization=regularization,
            )
        except Exception as exc:
            warnings.warn(
                f"Failed to compute final CG information matrix: {exc}",
                RuntimeWarning,
            )

    return CGResult(
        params=params,
        fitted_values=fitted_values,
        converged=False,
        n_iter=min(max_iter, len(deviance_history)),
        final_deviance=deviance_history[-1],
        deviance_history=deviance_history,
        fisher_matrix=fisher,
        param_slices=param_slices,
        line_search_steps=tuple(line_search_steps),
        condition_number=_condition_number(fisher) if fisher is not None else None,
    )


# =============================================================================
# Helper Functions
# =============================================================================


def _initialize_params(
    y: jnp.ndarray,
    X_mu: jnp.ndarray,
    X_sigma: Optional[jnp.ndarray],
    X_nu: Optional[jnp.ndarray],
    X_tau: Optional[jnp.ndarray],
    family,
) -> Dict[str, jnp.ndarray]:
    """Initialize parameters with simple estimates.

    Parameters
    ----------
    y : jnp.ndarray
        Response variable
    X_mu, X_sigma, X_nu, X_tau : jnp.ndarray or None
        Design matrices
    family : FamilyDefinition
        Distribution family

    Returns
    -------
    params : Dict[str, jnp.ndarray]
        Initial parameter estimates
    """
    params = {}

    # Initialize mu with a link-scale least-squares fit when possible.
    if X_mu is not None:
        p_mu = X_mu.shape[1]
        beta_mu = jnp.zeros(p_mu)
        if p_mu > 0:
            target_y = jnp.asarray(y, dtype=jnp.float64)
            if family.name in ["BI", "BE"]:
                target_y = jnp.clip(target_y, 0.01, 0.99)
            elif family.name in ["PO", "GA", "EXP", "NBI", "GEOM", "ZAGA"]:
                target_y = jnp.maximum(target_y, 0.1)

            if hasattr(family, "link_functions") and "mu" in family.link_functions:
                target_eta = family.link_functions["mu"](target_y)
            else:
                target_eta = target_y
            target_eta = jnp.where(
                jnp.isfinite(target_eta),
                target_eta,
                jnp.mean(target_eta[jnp.isfinite(target_eta)]),
            )

            try:
                beta_mu = jnp.linalg.lstsq(X_mu, target_eta, rcond=None)[0]
            except Exception:
                mean_y = jnp.mean(target_y)
                if hasattr(family, "link_functions") and "mu" in family.link_functions:
                    beta_mu = beta_mu.at[0].set(family.link_functions["mu"](mean_y))
                else:
                    beta_mu = beta_mu.at[0].set(mean_y)
        params["beta_mu"] = beta_mu

    # Initialize sigma (use std of y or reasonable default)
    if X_sigma is not None:
        p_sigma = X_sigma.shape[1]
        beta_sigma = jnp.zeros(p_sigma)
        if p_sigma > 0:
            std_y = jnp.std(y)
            # Use reasonable default if std is too small
            std_y = jnp.maximum(std_y, 0.1)
            # Apply link function (usually log)
            if hasattr(family, "link_functions") and "sigma" in family.link_functions:
                beta_sigma = beta_sigma.at[0].set(family.link_functions["sigma"](std_y))
            else:
                beta_sigma = beta_sigma.at[0].set(jnp.log(std_y))
        params["beta_sigma"] = beta_sigma

    # Initialize nu (use 0)
    if X_nu is not None:
        p_nu = X_nu.shape[1]
        params["beta_nu"] = jnp.zeros(p_nu)

    # Initialize tau (use 0)
    if X_tau is not None:
        p_tau = X_tau.shape[1]
        params["beta_tau"] = jnp.zeros(p_tau)

    return params


def _compute_log_likelihood(
    params: Dict[str, jnp.ndarray],
    data: Dict[str, jnp.ndarray],
    family,
    design_matrices: Dict[str, Optional[jnp.ndarray]],
) -> float:
    """Compute log-likelihood for current parameters.

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Current parameter values (beta coefficients)
    data : Dict[str, jnp.ndarray]
        Data including y and weights
    family : FamilyDefinition
        Distribution family
    design_matrices : Dict[str, Optional[jnp.ndarray]]
        Design matrices for each parameter

    Returns
    -------
    log_likelihood : float
        Total log-likelihood
    """
    y = data["y"]
    weights = data["weights"]

    # Compute linear predictors
    eta_mu = design_matrices["X_mu"] @ params["beta_mu"]

    # Apply inverse link functions
    if hasattr(family, "link_inverses") and "mu" in family.link_inverses:
        mu = family.link_inverses["mu"](eta_mu)
    else:
        mu = eta_mu

    # Compute sigma if present
    sigma = None
    if "beta_sigma" in params and design_matrices["X_sigma"] is not None:
        eta_sigma = design_matrices["X_sigma"] @ params["beta_sigma"]
        if hasattr(family, "link_inverses") and "sigma" in family.link_inverses:
            sigma = family.link_inverses["sigma"](eta_sigma)
        else:
            sigma = jnp.exp(eta_sigma)

    # Compute nu if present
    nu = None
    if "beta_nu" in params and design_matrices["X_nu"] is not None:
        eta_nu = design_matrices["X_nu"] @ params["beta_nu"]
        if hasattr(family, "link_inverses") and "nu" in family.link_inverses:
            nu = family.link_inverses["nu"](eta_nu)
        else:
            nu = eta_nu

    # Compute tau if present
    tau = None
    if "beta_tau" in params and design_matrices["X_tau"] is not None:
        eta_tau = design_matrices["X_tau"] @ params["beta_tau"]
        if hasattr(family, "link_inverses") and "tau" in family.link_inverses:
            tau = family.link_inverses["tau"](eta_tau)
        else:
            tau = jnp.exp(eta_tau)

    # Compute log-likelihood
    if hasattr(family, "log_likelihood"):
        ll = family.log_likelihood(y, mu, sigma, nu, tau)
    else:
        # Fallback: use density function
        # Call with only the parameters that the family has
        if hasattr(family, "d"):
            # Build arguments based on family parameters
            args = [y, mu]
            if sigma is not None and "sigma" in family.parameters:
                args.append(sigma)
            if nu is not None and "nu" in family.parameters:
                args.append(nu)
            if tau is not None and "tau" in family.parameters:
                args.append(tau)

            ll = family.d(*args, log=True)
        else:
            raise AttributeError(
                f"Family {family.name} has no log_likelihood or d method"
            )

    # Apply weights
    weighted_ll = weights * ll

    return jnp.sum(weighted_ll)


def _compute_fitted_values(
    params: Dict[str, jnp.ndarray],
    design_matrices: Dict[str, Optional[jnp.ndarray]],
    family,
) -> Dict[str, jnp.ndarray]:
    """Compute fitted values for all distribution parameters.

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Parameter estimates (beta coefficients)
    design_matrices : Dict[str, Optional[jnp.ndarray]]
        Design matrices
    family : FamilyDefinition
        Distribution family

    Returns
    -------
    fitted_values : Dict[str, jnp.ndarray]
        Fitted values for mu, sigma, nu, tau
    """
    fitted = {}

    # Compute mu
    if "beta_mu" in params and design_matrices.get("X_mu") is not None:
        eta_mu = design_matrices["X_mu"] @ params["beta_mu"]
        if hasattr(family, "link_inverses") and "mu" in family.link_inverses:
            fitted["mu"] = family.link_inverses["mu"](eta_mu)
        else:
            fitted["mu"] = eta_mu

    # Compute sigma
    if "beta_sigma" in params and design_matrices.get("X_sigma") is not None:
        eta_sigma = design_matrices["X_sigma"] @ params["beta_sigma"]
        if hasattr(family, "link_inverses") and "sigma" in family.link_inverses:
            fitted["sigma"] = family.link_inverses["sigma"](eta_sigma)
        else:
            fitted["sigma"] = jnp.exp(eta_sigma)

    # Compute nu
    if "beta_nu" in params and design_matrices.get("X_nu") is not None:
        eta_nu = design_matrices["X_nu"] @ params["beta_nu"]
        if hasattr(family, "link_inverses") and "nu" in family.link_inverses:
            fitted["nu"] = family.link_inverses["nu"](eta_nu)
        else:
            fitted["nu"] = eta_nu

    # Compute tau
    if "beta_tau" in params and design_matrices.get("X_tau") is not None:
        eta_tau = design_matrices["X_tau"] @ params["beta_tau"]
        if hasattr(family, "link_inverses") and "tau" in family.link_inverses:
            fitted["tau"] = family.link_inverses["tau"](eta_tau)
        else:
            fitted["tau"] = jnp.exp(eta_tau)

    return fitted


def _update_params(
    params: Dict[str, jnp.ndarray], delta: jnp.ndarray, step_size: float
) -> Dict[str, jnp.ndarray]:
    """Update parameters using delta vector.

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Current parameters
    delta : jnp.ndarray
        Update vector from Fisher system
    step_size : float
        Step size (1.0 = full Newton step)

    Returns
    -------
    new_params : Dict[str, jnp.ndarray]
        Updated parameters
    """
    # Flatten current parameters
    param_vec, unflatten_fn = flatten_params(params)

    # Update with step size
    new_param_vec = param_vec + step_size * delta

    # Unflatten back to dictionary
    new_params = unflatten_fn(new_param_vec)

    return new_params


def parameter_slices(params: Dict[str, jnp.ndarray]) -> Dict[str, slice]:
    """Return flattened-vector slices using the same sorted order as ``flatten_params``."""
    slices: Dict[str, slice] = {}
    start = 0
    for name in sorted(params.keys()):
        size = int(jnp.atleast_1d(params[name]).size)
        slices[name] = slice(start, start + size)
        start += size
    return slices


def extract_information_blocks(
    fisher: jnp.ndarray,
    slices: Dict[str, slice],
) -> Dict[Tuple[str, str], jnp.ndarray]:
    """Extract named blocks from a flattened CG observed-information matrix."""
    return {
        (left, right): fisher[left_slice, right_slice]
        for left, left_slice in slices.items()
        for right, right_slice in slices.items()
    }


def zero_cross_information_blocks(
    fisher: jnp.ndarray,
    slices: Dict[str, slice],
) -> jnp.ndarray:
    """Return a copy of ``fisher`` with all off-diagonal parameter blocks zeroed."""
    result = jnp.array(fisher)
    for left, left_slice in slices.items():
        for right, right_slice in slices.items():
            if left != right:
                result = result.at[left_slice, right_slice].set(0.0)
    return result


def _condition_number(fisher: Optional[jnp.ndarray]) -> Optional[float]:
    """Return a finite condition-number diagnostic when possible."""
    if fisher is None:
        return None
    try:
        eigvals = jnp.linalg.eigvalsh(fisher)
        eigvals = jnp.abs(eigvals)
        positive = eigvals[eigvals > jnp.finfo(fisher.dtype).eps]
        if positive.size == 0:
            return None
        return float(jnp.max(positive) / jnp.min(positive))
    except Exception:
        return None


def _compute_full_observed_information_and_score(
    log_likelihood: Callable[
        [Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]], jnp.ndarray
    ],
    params: Dict[str, jnp.ndarray],
    data: Dict[str, jnp.ndarray],
    regularization: float = 1e-6,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Compute complete CG derivatives with JAX automatic differentiation.

    The previous CG backend assembled a block-diagonal Fisher matrix from
    per-parameter Hessian functions, which implicitly set all cross-derivative
    blocks to zero.  Cole-Green scoring needs the coupled curvature between all
    estimated distribution parameters.  This helper flattens the beta
    coefficients and differentiates the scalar log-likelihood directly so
    ``jax.hessian`` returns every block, including mu/sigma/nu/tau interactions.
    """
    param_vec, unflatten_fn = flatten_params(params)

    def log_lik_vec(theta_vec):
        theta_dict = unflatten_fn(theta_vec)
        return log_likelihood(theta_dict, data)

    score = grad(log_lik_vec)(param_vec)
    hess = hessian(log_lik_vec)(param_vec)
    fisher = -hess
    fisher = 0.5 * (fisher + fisher.T)
    fisher = _stabilize_information_matrix(fisher, regularization)
    score = jnp.where(jnp.isfinite(score), score, 0.0)
    return fisher, score


def _stabilize_information_matrix(
    fisher: jnp.ndarray,
    regularization: float,
) -> jnp.ndarray:
    """Return a finite positive-definite information matrix preserving cross terms."""
    fisher = jnp.where(jnp.isfinite(fisher), fisher, 0.0)
    fisher = 0.5 * (fisher + fisher.T)
    n_params = fisher.shape[0]
    eigvals, eigvecs = jnp.linalg.eigh(fisher)
    base_floor = jnp.maximum(
        jnp.asarray(regularization, dtype=fisher.dtype),
        jnp.finfo(fisher.dtype).eps,
    )
    relative_floor = 1e-2 * jnp.max(
        jnp.maximum(jnp.abs(eigvals), jnp.ones_like(eigvals))
    )
    floor = jnp.where(
        jnp.min(eigvals) <= base_floor,
        jnp.maximum(base_floor, relative_floor),
        base_floor,
    )
    eigvals = jnp.maximum(eigvals, floor)
    stabilized = (eigvecs * eigvals) @ eigvecs.T
    stabilized = 0.5 * (stabilized + stabilized.T)
    return stabilized + floor * jnp.eye(n_params, dtype=fisher.dtype)


def _line_search_update(
    params: Dict[str, jnp.ndarray],
    delta: jnp.ndarray,
    step_size: float,
    current_deviance: float,
    log_likelihood: Callable[
        [Dict[str, jnp.ndarray], Dict[str, jnp.ndarray]], jnp.ndarray
    ],
    data: Dict[str, jnp.ndarray],
    max_halving: int = 30,
) -> Tuple[Dict[str, jnp.ndarray], Optional[float], int]:
    """Apply a damped CG update, accepting only finite non-increasing deviance."""
    for i in range(max_halving + 1):
        candidate_step = step_size * (0.5**i)
        candidate = _update_params(params, delta, candidate_step)
        candidate_deviance = float(-2.0 * log_likelihood(candidate, data))
        if (
            jnp.isfinite(candidate_deviance)
            and candidate_deviance <= current_deviance + 1e-8
        ):
            return candidate, candidate_deviance, i
    return params, None, max_halving


def _compute_expected_fisher(
    params: Dict[str, jnp.ndarray],
    design_matrices: Dict[str, Optional[jnp.ndarray]],
    family,
    y: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """Compute expected Fisher information matrix for GAMLSS.

    This uses the family's hessian functions (expected information)
    rather than the observed Hessian of the log-likelihood.

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Current parameters
    design_matrices : Dict[str, Optional[jnp.ndarray]]
        Design matrices
    family : FamilyDefinition
        Distribution family
    y : jnp.ndarray
        Response variable
    weights : jnp.ndarray
        Observation weights

    Returns
    -------
    fisher : jnp.ndarray
        Expected Fisher information matrix
    """
    # Compute fitted values
    fitted = _compute_fitted_values(params, design_matrices, family)
    mu = fitted["mu"]
    sigma = fitted.get("sigma", None)
    nu = fitted.get("nu", None)
    tau = fitted.get("tau", None)

    # If sigma is not being estimated, use a fixed value (std of residuals)
    if sigma is None and "sigma" in family.parameters:
        sigma = jnp.std(y - mu) * jnp.ones_like(mu)
        sigma = jnp.maximum(sigma, 0.1)  # Avoid zero

    # If nu/tau not being estimated but needed, use defaults
    if nu is None and "nu" in family.parameters:
        nu = jnp.zeros_like(mu)
    if tau is None and "tau" in family.parameters:
        tau = jnp.ones_like(mu)

    # Get design matrices
    X_mu = design_matrices.get("X_mu", None)
    X_sigma = design_matrices.get("X_sigma", None)
    X_nu = design_matrices.get("X_nu", None)
    X_tau = design_matrices.get("X_tau", None)

    # Compute expected information for each parameter
    # Using family's hessian functions (which give expected information)
    blocks = []
    param_names = []

    # Mu block
    if "beta_mu" in params:
        # E[-d2l/dmu^2] from family
        if hasattr(family, "hessian_functions") and "mu" in family.hessian_functions:
            # Call with only the parameters the family has
            hess_fn = family.hessian_functions["mu"]
            if "sigma" in family.parameters:
                if "nu" in family.parameters:
                    if "tau" in family.parameters:
                        d2ldm2 = hess_fn(y, mu, sigma, nu, tau)
                    else:
                        d2ldm2 = hess_fn(y, mu, sigma, nu)
                else:
                    d2ldm2 = hess_fn(y, mu, sigma)
            else:
                d2ldm2 = hess_fn(y, mu)
        else:
            # Fallback: use numerical approximation
            d2ldm2 = -jnp.ones_like(mu)

        # Weight by link derivative
        if hasattr(family, "link_derivatives") and "mu" in family.link_derivatives:
            dmu_deta = family.link_derivatives["mu"](mu)
            w_mu = -d2ldm2 * jnp.square(dmu_deta) * weights
        else:
            w_mu = -d2ldm2 * weights

        # Fisher block: X' W X
        W_mu = jnp.diag(w_mu)
        I_mu = X_mu.T @ W_mu @ X_mu
        blocks.append(I_mu)
        param_names.append("mu")

    # Sigma block
    if "beta_sigma" in params and X_sigma is not None and sigma is not None:
        # For sigma parameter, use expected Fisher information
        # For Normal distribution: E[-d2l/dsigma^2] = 2/sigma^2
        # This is more stable than observed information for CG algorithm
        if family.name == "NO":
            # Expected information for Normal distribution
            d2lds2_expected = -2.0 / jnp.square(sigma)
        elif (
            hasattr(family, "hessian_functions") and "sigma" in family.hessian_functions
        ):
            hess_fn = family.hessian_functions["sigma"]
            if "nu" in family.parameters:
                if "tau" in family.parameters:
                    d2lds2 = hess_fn(y, mu, sigma, nu, tau)
                else:
                    d2lds2 = hess_fn(y, mu, sigma, nu)
            else:
                d2lds2 = hess_fn(y, mu, sigma)
            # Use mean as approximation to expected value
            d2lds2_expected = jnp.mean(d2lds2) * jnp.ones_like(sigma)
        else:
            d2lds2_expected = -2.0 / jnp.square(sigma)

        if hasattr(family, "link_derivatives") and "sigma" in family.link_derivatives:
            dsigma_deta = family.link_derivatives["sigma"](sigma)
            w_sigma = -d2lds2_expected * jnp.square(dsigma_deta) * weights
        else:
            w_sigma = -d2lds2_expected * weights

        W_sigma = jnp.diag(w_sigma)
        I_sigma = X_sigma.T @ W_sigma @ X_sigma
        blocks.append(I_sigma)
        param_names.append("sigma")

    # Nu block
    if "beta_nu" in params and X_nu is not None and nu is not None:
        if hasattr(family, "hessian_functions") and "nu" in family.hessian_functions:
            hess_fn = family.hessian_functions["nu"]
            if "tau" in family.parameters:
                d2ldn2 = hess_fn(y, mu, sigma, nu, tau)
            else:
                d2ldn2 = hess_fn(y, mu, sigma, nu)
        else:
            d2ldn2 = -jnp.ones_like(nu)

        if hasattr(family, "link_derivatives") and "nu" in family.link_derivatives:
            dnu_deta = family.link_derivatives["nu"](nu)
            w_nu = -d2ldn2 * jnp.square(dnu_deta) * weights
        else:
            w_nu = -d2ldn2 * weights

        W_nu = jnp.diag(w_nu)
        I_nu = X_nu.T @ W_nu @ X_nu
        blocks.append(I_nu)
        param_names.append("nu")

    # Tau block
    if "beta_tau" in params and X_tau is not None and tau is not None:
        if hasattr(family, "hessian_functions") and "tau" in family.hessian_functions:
            d2ldt2 = family.hessian_functions["tau"](y, mu, sigma, nu, tau)
        else:
            d2ldt2 = -jnp.ones_like(tau)

        if hasattr(family, "link_derivatives") and "tau" in family.link_derivatives:
            dtau_deta = family.link_derivatives["tau"](tau)
            w_tau = -d2ldt2 * jnp.square(dtau_deta) * weights
        else:
            w_tau = -d2ldt2 * weights

        W_tau = jnp.diag(w_tau)
        I_tau = X_tau.T @ W_tau @ X_tau
        blocks.append(I_tau)
        param_names.append("tau")

    # Assemble block-diagonal Fisher matrix
    # (Assuming independence between parameters - this is the CG approximation)
    if len(blocks) == 1:
        fisher = blocks[0]
    else:
        # Create block diagonal matrix
        total_size = sum(b.shape[0] for b in blocks)
        fisher = jnp.zeros((total_size, total_size))
        start_idx = 0
        for block in blocks:
            size = block.shape[0]
            fisher = fisher.at[
                start_idx : start_idx + size, start_idx : start_idx + size
            ].set(block)
            start_idx += size

    return fisher


def _compute_score_gamlss(
    params: Dict[str, jnp.ndarray],
    design_matrices: Dict[str, Optional[jnp.ndarray]],
    family,
    y: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """Compute score vector for GAMLSS using family's score functions.

    Parameters
    ----------
    params : Dict[str, jnp.ndarray]
        Current parameters
    design_matrices : Dict[str, Optional[jnp.ndarray]]
        Design matrices
    family : FamilyDefinition
        Distribution family
    y : jnp.ndarray
        Response variable
    weights : jnp.ndarray
        Observation weights

    Returns
    -------
    score : jnp.ndarray
        Score vector
    """
    # Compute fitted values
    fitted = _compute_fitted_values(params, design_matrices, family)
    mu = fitted["mu"]
    sigma = fitted.get("sigma", None)
    nu = fitted.get("nu", None)
    tau = fitted.get("tau", None)

    # If sigma is not being estimated, use a fixed value
    if sigma is None and "sigma" in family.parameters:
        sigma = jnp.std(y - mu) * jnp.ones_like(mu)
        sigma = jnp.maximum(sigma, 0.1)

    # If nu/tau not being estimated but needed, use defaults
    if nu is None and "nu" in family.parameters:
        nu = jnp.zeros_like(mu)
    if tau is None and "tau" in family.parameters:
        tau = jnp.ones_like(mu)

    # Get design matrices
    X_mu = design_matrices.get("X_mu", None)
    X_sigma = design_matrices.get("X_sigma", None)
    X_nu = design_matrices.get("X_nu", None)
    X_tau = design_matrices.get("X_tau", None)

    # Compute score components
    score_parts = []

    # Mu score
    if "beta_mu" in params:
        if hasattr(family, "score_functions") and "mu" in family.score_functions:
            score_fn = family.score_functions["mu"]
            if "sigma" in family.parameters:
                if "nu" in family.parameters:
                    if "tau" in family.parameters:
                        dldm = score_fn(y, mu, sigma, nu, tau)
                    else:
                        dldm = score_fn(y, mu, sigma, nu)
                else:
                    dldm = score_fn(y, mu, sigma)
            else:
                dldm = score_fn(y, mu)
        else:
            # Fallback
            dldm = y - mu

        # Transform by link derivative
        if hasattr(family, "link_derivatives") and "mu" in family.link_derivatives:
            dmu_deta = family.link_derivatives["mu"](mu)
            u_mu = dldm * dmu_deta * weights
        else:
            u_mu = dldm * weights

        # Score: X' u
        score_mu = X_mu.T @ u_mu
        score_parts.append(score_mu)

    # Sigma score
    if "beta_sigma" in params and X_sigma is not None and sigma is not None:
        if hasattr(family, "score_functions") and "sigma" in family.score_functions:
            score_fn = family.score_functions["sigma"]
            if "nu" in family.parameters:
                if "tau" in family.parameters:
                    dlds = score_fn(y, mu, sigma, nu, tau)
                else:
                    dlds = score_fn(y, mu, sigma, nu)
            else:
                dlds = score_fn(y, mu, sigma)
        else:
            dlds = jnp.zeros_like(sigma)

        if hasattr(family, "link_derivatives") and "sigma" in family.link_derivatives:
            dsigma_deta = family.link_derivatives["sigma"](sigma)
            u_sigma = dlds * dsigma_deta * weights
        else:
            u_sigma = dlds * weights

        score_sigma = X_sigma.T @ u_sigma
        score_parts.append(score_sigma)

    # Nu score
    if "beta_nu" in params and X_nu is not None and nu is not None:
        if hasattr(family, "score_functions") and "nu" in family.score_functions:
            score_fn = family.score_functions["nu"]
            if "tau" in family.parameters:
                dldn = score_fn(y, mu, sigma, nu, tau)
            else:
                dldn = score_fn(y, mu, sigma, nu)
        else:
            dldn = jnp.zeros_like(nu)

        if hasattr(family, "link_derivatives") and "nu" in family.link_derivatives:
            dnu_deta = family.link_derivatives["nu"](nu)
            u_nu = dldn * dnu_deta * weights
        else:
            u_nu = dldn * weights

        score_nu = X_nu.T @ u_nu
        score_parts.append(score_nu)

    # Tau score
    if "beta_tau" in params and X_tau is not None and tau is not None:
        if hasattr(family, "score_functions") and "tau" in family.score_functions:
            score_fn = family.score_functions["tau"]
            dldt = score_fn(y, mu, sigma, nu, tau)
        else:
            dldt = jnp.zeros_like(tau)

        if hasattr(family, "link_derivatives") and "tau" in family.link_derivatives:
            dtau_deta = family.link_derivatives["tau"](tau)
            u_tau = dldt * dtau_deta * weights
        else:
            u_tau = dldt * weights

        score_tau = X_tau.T @ u_tau
        score_parts.append(score_tau)

    # Concatenate all score parts
    score = jnp.concatenate([s.ravel() for s in score_parts])

    return score


__all__ = [
    "fit_cg",
    "CGResult",
]
