"""R-compatible mixed fitting and the historical hybrid dispatcher.

``mixed_fit`` implements the R ``mixed(n1, n2)`` sequence (RS then CG).
``hybrid_fit`` preserves OmniLSS's data-dependent choice between a single RS
or L-BFGS backend.

References:
    Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized additive models 
    for location, scale and shape. Journal of the Royal Statistical Society: 
    Series C, 54(3), 507-554.
"""

from numbers import Integral
from typing import Dict, Optional, Literal
import time
import numpy as np
import warnings

from .cg_algorithm_v2 import cg_fit_v2
from .rs_algorithm import rs_fit
from .lbfgs_algorithm import joint_lbfgs_fit as cg_fit
from ..model import GAMLSSModel
from ..families import FamilyDefinition
from ..formula_parser import has_smooth_terms


def _contains_smooth_terms(*formulas: str | None) -> bool:
    """Return True when any formula contains a known smooth-term marker."""
    return any(has_smooth_terms(formula) for formula in formulas)


def _auto_select_algorithm(family_name: str, n: int, has_smooth: bool) -> str:
    """Auto-select the fitting algorithm for mixed fitting.

    Rules:
    - smooth terms require RS because L-BFGS/CG do not support smooth terms;
    - numerically unstable high-parameter families stay on RS;
    - small datasets prefer RS;
    - larger non-smooth datasets may use the L-BFGS compatibility backend.
    """
    unstable_families = {"BCPE", "BCT"}
    family_key = family_name.upper()
    if has_smooth or family_key in unstable_families or n < 500:
        return "rs"
    return "lbfgs"


def hybrid_fit(
    formula: str,
    sigma_formula: str = "~ 1",
    nu_formula: Optional[str] = None,
    tau_formula: Optional[str] = None,
    family: str | FamilyDefinition = "NO",
    data: Optional[Dict[str, np.ndarray]] = None,
    algorithm: Literal["auto", "rs", "cg", "lbfgs"] = "auto",
    max_iter: int = 20,
    tol: float = 1e-4,
    rtol: float = 1e-6,
    verbose: bool = False,
    *,
    weights=None,
    parameter_formulas=None,
) -> GAMLSSModel:
    """Fit a GAMLSS model using the historical hybrid dispatcher.

    The hybrid dispatcher selects one backend; it is not R ``mixed()``:
    1. Automatically select RS or L-BFGS from data characteristics
    2. Force RS for comparison or smooth-term support
    3. Force the L-BFGS compatibility backend for experiments
    
    Args:
        formula: Formula for mu parameter
        sigma_formula: Formula for sigma parameter
        nu_formula: Formula for nu parameter (if applicable)
        tau_formula: Formula for tau parameter (if applicable)
        family: Distribution family name or configured family instance
        weights: Optional observation weights, forwarded to either backend.
        parameter_formulas: Parameter formula mapping. Entries take precedence
            over the legacy nu_formula/tau_formula arguments; input is copied.
        data: Dictionary of data arrays
        algorithm: Algorithm selection strategy:
            - "auto": Automatically select best algorithm (currently uses RS)
            - "rs": Use RS algorithm
            - "cg": Use CG algorithm
        max_iter: Maximum iterations
        tol: Convergence tolerance
        verbose: Whether to print iteration info
        
    Returns:
        Fitted GAMLSSModel
        
    Example:
        >>> data = {"y": y, "x": x}
        >>> # Automatic selection (uses RS)
        >>> model = hybrid_fit("y ~ x", "~ x", family="NO", data=data)
        >>> 
        >>> # Explicit RS
        >>> model = hybrid_fit("y ~ x", "~ x", family="NO", data=data, algorithm="rs")
        >>> 
        >>> # Explicit CG
        >>> model = mixed_fit("y ~ x", "~ x", family="NO", data=data, algorithm="cg")
    """
    if data is None:
        raise ValueError("data must be provided")
    
    if verbose:
        print("=" * 70)
        print("Hybrid Dispatcher")
        print("=" * 70)
    
    formulas = dict(parameter_formulas or {})
    if nu_formula is not None:
        formulas.setdefault("nu", nu_formula)
    if tau_formula is not None:
        formulas.setdefault("tau", tau_formula)
    response_name = formula.split("~", 1)[0].strip()
    n = len(np.asarray(data[response_name])) if response_name in data else 0
    has_smooth = _contains_smooth_terms(formula, sigma_formula, *formulas.values())

    # Algorithm selection
    if algorithm == "auto":
        family_name = getattr(family, "name", str(family))
        selected_algorithm = _auto_select_algorithm(family_name, n, has_smooth)
        if verbose:
            reason = "smooth terms" if has_smooth else f"n={n}, family={family}"
            print(f"Auto-selection: Using {selected_algorithm.upper()} algorithm ({reason})")
    else:
        algo_key = str(algorithm).lower()
        aliases = {
            "rs": "rs",
            "cg": "lbfgs",
            "lbfgs": "lbfgs",
            "l-bfgs": "lbfgs",
            "auto": "auto",
        }
        selected_algorithm = aliases.get(algo_key, algorithm)
        if algo_key == "cg" and selected_algorithm == "lbfgs":
            warnings.warn(
                "algorithm='cg' is a deprecated alias for 'lbfgs' (L-BFGS joint optimizer).",
                DeprecationWarning,
                stacklevel=2,
            )
        if verbose:
            print(f"Using {selected_algorithm.upper()} algorithm")
    
    if selected_algorithm == "lbfgs" and has_smooth:
        raise ValueError(
            "The HYBRID lbfgs compatibility backend does not support penalized "
            "smooth terms; use algorithm='rs' or algorithm='auto'."
        )

    # Fit using selected algorithm
    if selected_algorithm == "rs":
        model = rs_fit(
            formula=formula,
            sigma_formula=sigma_formula,
            parameter_formulas=formulas or None,
            weights=weights,
            family=family,
            data=data,
            max_iter=max_iter,
            tol=tol,
            rtol=rtol,
            verbose=verbose
        )
        # Retain the old mixed_* keys for callers of the historical dispatcher.
        model.additional_slots["hybrid_algorithm_used"] = "rs"
        model.additional_slots["hybrid_auto_selected"] = algorithm == "auto"
        model.additional_slots["mixed_algorithm_used"] = "rs"
        model.additional_slots["mixed_auto_selected"] = algorithm == "auto"
        
    elif selected_algorithm == "lbfgs":
        model = cg_fit(
            formula=formula,
            sigma_formula=sigma_formula,
            nu_formula=nu_formula,
            tau_formula=tau_formula,
            parameter_formulas=formulas or None,
            weights=weights,
            family=family,
            data=data,
            max_outer_iter=max_iter,
            outer_tol=tol,
            verbose=verbose
        )
        model.additional_slots["hybrid_algorithm_used"] = "lbfgs"
        model.additional_slots["hybrid_auto_selected"] = algorithm == "auto"
        model.additional_slots["mixed_algorithm_used"] = "lbfgs"
        model.additional_slots["mixed_auto_selected"] = algorithm == "auto"
    
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}. Use 'auto', 'rs', or 'lbfgs'.")
    
    if verbose:
        print("=" * 70)
        print(f"Hybrid dispatcher completed using {selected_algorithm.upper()}")
        print("=" * 70)
    
    return model


def _stage_cycles(name: str, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= 0")
    value = int(value)
    if value < 0:
        raise ValueError(f"{name} must be >= 0")
    return value


def mixed_fit(
    formula: str,
    sigma_formula: str = "~ 1",
    nu_formula: Optional[str] = None,
    tau_formula: Optional[str] = None,
    family: str | FamilyDefinition = "NO",
    data: Optional[Dict[str, np.ndarray]] = None,
    n1: int = 1,
    n2: int = 20,
    tol: float = 1e-4,
    rtol: float = 1e-6,
    verbose: bool = False,
    *,
    weights=None,
    parameter_formulas=None,
    max_iter: int | None = None,
    start_values: dict | None = None,
    contrasts=None,
) -> GAMLSSModel:
    """Fit the R-compatible two-stage ``mixed(n1, n2)`` algorithm.

    The first stage runs RS for at most ``n1`` outer cycles. Every estimable
    parameter coefficient, linear predictor and fitted value is then validated
    and copied into the eta-scale Cole-Green backend for at most ``n2`` cycles.
    Penalized smooth terms remain unsupported because CG cannot yet preserve
    their penalty state; use :func:`hybrid_fit` for the historical automatic
    RS/L-BFGS dispatcher.
    """

    if data is None:
        raise ValueError("data must be provided")
    n1 = _stage_cycles("n1", n1)
    if max_iter is not None:
        warnings.warn(
            "mixed_fit(max_iter=...) is deprecated; use n2=... for the CG stage",
            DeprecationWarning,
            stacklevel=2,
        )
        n2 = max_iter
    n2 = _stage_cycles("n2", n2)

    formulas = dict(parameter_formulas or {})
    if nu_formula is not None:
        formulas.setdefault("nu", nu_formula)
    if tau_formula is not None:
        formulas.setdefault("tau", tau_formula)
    if _contains_smooth_terms(formula, sigma_formula, *formulas.values()):
        raise ValueError(
            "R-compatible MIXED does not yet support penalized smooth terms in "
            "the CG stage; use method='RS' or method='HYBRID'"
        )

    rs_model = rs_fit(
        formula=formula,
        sigma_formula=sigma_formula,
        parameter_formulas=formulas or None,
        weights=weights,
        family=family,
        data=data,
        max_iter=n1,
        tol=tol,
        rtol=rtol,
        verbose=verbose,
        # D2.131: the CG stage continues from the RS state, so the user start
        # values only enter the RS stage -- exactly like R's mixed().
        start_values=start_values,
        contrasts=contrasts,
    )
    rs_history = tuple(
        float(value)
        for value in rs_model.additional_slots.get(
            "deviance_history", (float(rs_model.g_dev),)
        )
    )

    cg_model = cg_fit_v2(
        formula=formula,
        sigma_formula=sigma_formula,
        parameter_formulas=formulas or None,
        weights=weights,
        family=rs_model.family,
        data=dict(data),
        max_iter=n2,
        tol=tol,
        verbose=verbose,
        initial_model=rs_model,
        contrasts=contrasts,
    )
    cg_history = tuple(
        float(value)
        for value in cg_model.additional_slots.get(
            "deviance_history", (float(cg_model.g_dev),)
        )
    )
    boundary_deviance = float(rs_model.g_dev)
    # D2.137: CG starts from X @ beta (the coefficients are the state it
    # continues from), while an RS model stopped after one outer cycle reports
    # the deviance of its entry linear predictor -- the two differ by one inner
    # WLS step.  D2.238: that bookkeeping gap is NOT uniform across families.
    # It is 2e-8 absolute on a normal sample (rel 2.5e-10) but 2.75e-3 on IG
    # (rel 2.9e-5), so the old rtol=1e-7 rejected legitimate MIXED fits for
    # BE and IG.  The bound is now 1e-4 -- still two orders of magnitude below
    # a real state corruption (which is O(1) relative), and above the measured
    # worst legitimate gap across the 22 promoted families.
    if not cg_history or not np.isclose(
        cg_history[0], boundary_deviance, rtol=1e-4, atol=1e-4
    ):
        raise RuntimeError(
            "MIXED RS/CG boundary deviance changed before the first CG update"
        )
    combined_history = rs_history + cg_history[1:]

    slots = dict(cg_model.additional_slots)
    cg_iterations = int(slots.get("cg_iterations", cg_model.iter))
    rs_iterations = int(rs_model.iter)
    slots.update(
        {
            "method": "MIXED",
            "converged": bool(slots.get("cg_converged", False)),
            "cycles": cg_iterations,
            "deviance_history": combined_history,
            "mixed_rs_n1": n1,
            "mixed_cg_n2": n2,
            "mixed_rs_iterations": rs_iterations,
            "mixed_cg_iterations": cg_iterations,
            "mixed_total_iterations": rs_iterations + cg_iterations,
            "mixed_rs_converged": bool(
                rs_model.additional_slots.get("rs_converged", False)
            ),
            "mixed_cg_converged": bool(slots.get("cg_converged", False)),
            "mixed_rs_deviance_history": rs_history,
            "mixed_cg_deviance_history": cg_history,
            "mixed_boundary_deviance": boundary_deviance,
            "mixed_cg_backend": slots.get("cg_backend", "CG_IRLS_CROSS"),
            "mixed_state_parameters": tuple(rs_model.family.estimable_parameters),
        }
    )
    cg_model.additional_slots = slots
    cg_model.iter = cg_iterations
    # Align the first-class convergence fields with the MIXED verdict recorded
    # in the slots (the CG stage decides, exactly as the slot did before), so
    # MIXED models report the same outcome through the unified accessor.
    mixed_converged = bool(slots.get("converged", False))
    cg_model.converged = mixed_converged
    if cg_model.termination_reason is None:
        cg_model.termination_reason = (
            "relative_deviance_converged" if mixed_converged else "cg_stage_not_converged"
        )
    cg_model.call = {
        **dict(cg_model.call or {}),
        "method": "MIXED",
        "mixed_n1": n1,
        "mixed_n2": n2,
    }
    cg_model.control = {
        **dict(cg_model.control or {}),
        "mixed.n1": n1,
        "mixed.n2": n2,
    }
    return cg_model


def compare_algorithms(
    formula: str,
    sigma_formula: str = "~ 1",
    nu_formula: Optional[str] = None,
    tau_formula: Optional[str] = None,
    family: str = "NO",
    data: Optional[Dict[str, np.ndarray]] = None,
    verbose: bool = True
) -> Dict[str, object]:
    """Compare RS and CG algorithms on the same data.
    
    Args:
        formula: Formula for mu parameter
        sigma_formula: Formula for sigma parameter
        nu_formula: Formula for nu parameter (if applicable)
        tau_formula: Formula for tau parameter (if applicable)
        family: Distribution family name
        data: Dictionary of data arrays
        verbose: Whether to print comparison results
        
    Returns:
        Dictionary with fitted models plus metrics: {"rs": model_rs, "cg": model_cg, "metrics": {...}}
        
    Example:
        >>> data = {"y": y, "x": x}
        >>> models = compare_algorithms("y ~ x", "~ x", family="NO", data=data)
        >>> print(f"RS deviance: {models['rs'].g_dev:.4f}")
        >>> print(f"CG deviance: {models['cg'].g_dev:.4f}")
    """
    if data is None:
        raise ValueError("data must be provided")
    
    if verbose:
        print("\n" + "=" * 70)
        print("Algorithm Comparison")
        print("=" * 70)
    
    # Fit with RS
    if verbose:
        print("\nFitting with RS algorithm...")
    
    # Build parameter_formulas for nu and tau
    parameter_formulas_rs = {}
    if nu_formula is not None:
        parameter_formulas_rs["nu"] = nu_formula
    if tau_formula is not None:
        parameter_formulas_rs["tau"] = tau_formula
    
    rs_start = time.perf_counter()
    model_rs = rs_fit(
        formula=formula,
        sigma_formula=sigma_formula,
        parameter_formulas=parameter_formulas_rs if parameter_formulas_rs else None,
        family=family,
        data=data,
        verbose=False
    )
    rs_runtime = time.perf_counter() - rs_start

    # Fit with CG/L-BFGS compatibility backend
    if verbose:
        print("Fitting with CG/L-BFGS algorithm...")
    cg_start = time.perf_counter()
    model_cg = cg_fit(
        formula=formula,
        sigma_formula=sigma_formula,
        nu_formula=nu_formula,
        tau_formula=tau_formula,
        family=family,
        data=data,
        verbose=False
    )
    cg_runtime = time.perf_counter() - cg_start

    metrics = {
        "rs": {
            "deviance": float(model_rs.g_dev),
            "iterations": model_rs.additional_slots.get("rs_iterations"),
            "runtime_seconds": rs_runtime,
            "converged": bool(model_rs.convergence_status()),
        },
        "cg": {
            "deviance": float(model_cg.g_dev),
            "iterations": model_cg.additional_slots.get("cg_iterations"),
            "runtime_seconds": cg_runtime,
            "converged": bool(model_cg.convergence_status()),
        },
    }
    
    # Print comparison
    if verbose:
        print("\n" + "-" * 70)
        print("Results:")
        print("-" * 70)
        print(f"{'Algorithm':<15} {'Deviance':<15} {'Iterations':<15} {'Status':<15}")
        print("-" * 70)
        
        rs_iters = model_rs.additional_slots.get("rs_iterations", "-")
        rs_conv = bool(model_rs.convergence_status())
        print(f"{'RS':<15} {model_rs.g_dev:<15.4f} {rs_iters:<15} "
              f"{'Converged' if rs_conv else 'Not converged':<15}")
        
        cg_iters = model_cg.additional_slots.get("cg_iterations", "-")
        cg_conv = bool(model_cg.convergence_status())
        print(f"{'CG':<15} {model_cg.g_dev:<15.4f} {cg_iters:<15} "
              f"{'Converged' if cg_conv else 'Not converged':<15}")
        
        print("-" * 70)
        print(f"Deviance difference (RS - CG): {model_rs.g_dev - model_cg.g_dev:+.4f}")
        print("=" * 70)
    
    return {"rs": model_rs, "cg": model_cg, "metrics": metrics}
