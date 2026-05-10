"""Formula processing for GAMLSS models with smooth terms.

This module provides formula parsing and design matrix construction,
supporting both linear terms and smooth terms (pb, ps, cs, etc.).

R source: gamlss/R/gamlss-5.R, gamlss/R/add.r
References:
- Wilkinson, G. N., & Rogers, C. E. (1973). Symbolic description of factorial models
  for analysis of variance. Journal of the Royal Statistical Society: Series C, 22(3), 392-399.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, Optional

import jax.numpy as jnp
import numpy as np


@dataclass
class SmoothTerm:
    """Represents a smooth term in a formula.
    
    Attributes
    ----------
    smoother : str
        Type of smoother: "pb", "ps", "cs", "s", "lo"
    variable : str
        Variable name to smooth
    df : float, optional
        Degrees of freedom
    lambda_ : float, optional
        Smoothing parameter
    kwargs : dict
        Additional smoother-specific arguments
        
    Notes
    -----
    Random effects ("random", "re") are not currently implemented.
    """
    smoother: str
    variable: str
    df: Optional[float] = None
    lambda_: Optional[float] = None
    kwargs: dict = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}


@dataclass
class TensorProductTerm:
    """Represents a tensor product smooth term in a formula.
    
    Attributes
    ----------
    smoother : str
        Type of tensor product: "te" or "ti"
    variables : list[str]
        Variable names to smooth (2 or more)
    k : int, optional
        Number of basis functions per dimension
    k_list : list[int], optional
        Number of basis functions for each dimension
    bs : str, optional
        Basis type (default: "ps")
    lambda_ : float, optional
        Smoothing parameter
    kwargs : dict
        Additional arguments
    """
    smoother: str
    variables: list[str]
    k: Optional[int] = None
    k_list: Optional[list[int]] = None
    bs: str = "ps"
    lambda_: Optional[float] = None
    kwargs: dict = None
    
    def __post_init__(self):
        if self.kwargs is None:
            self.kwargs = {}
        if len(self.variables) < 2:
            raise ValueError(f"Tensor product {self.smoother}() requires at least 2 variables")


@dataclass
class LinearTerm:
    """Represents a linear term in a formula.
    
    Attributes
    ----------
    variable : str
        Variable name
    """
    variable: str


@dataclass
class ParsedFormula:
    """Parsed formula with linear and smooth terms.
    
    Attributes
    ----------
    response : str
        Response variable name
    linear_terms : list[LinearTerm]
        Linear terms
    smooth_terms : list[SmoothTerm]
        Smooth terms
    tensor_terms : list[TensorProductTerm]
        Tensor product smooth terms
    has_intercept : bool
        Whether formula includes intercept
    """
    response: str
    linear_terms: list[LinearTerm]
    smooth_terms: list[SmoothTerm]
    tensor_terms: list[TensorProductTerm]
    has_intercept: bool = True


def parse_formula(formula: str) -> ParsedFormula:
    """Parse a GAMLSS formula with smooth terms.
    
    Supports formulas like:
    - "y ~ x1 + x2"  (linear terms only)
    - "y ~ pb(x1) + x2"  (smooth + linear)
    - "y ~ pb(x1, df=5) + pb(x2, lambda=1.0)"  (multiple smooths)
    - "y ~ s(x1) + x2"  (s() function with auto lambda selection)
    - "y ~ s(x1, method='REML') + s(x2, method='GCV')"  (multiple s() with methods)
    - "y ~ te(x1, x2)"  (tensor product smooth)
    - "y ~ ti(x1) + ti(x2) + ti(x1, x2)"  (tensor product interaction)
    - "y ~ x1 + x2 - 1"  (no intercept)
    
    Parameters
    ----------
    formula : str
        Formula string
    
    Returns
    -------
    parsed : ParsedFormula
        Parsed formula object
    
    Examples
    --------
    >>> parsed = parse_formula("y ~ x1 + pb(x2, df=5)")
    >>> parsed.response
    'y'
    >>> len(parsed.linear_terms)
    1
    >>> len(parsed.smooth_terms)
    1
    
    >>> # Using s() function
    >>> parsed = parse_formula("y ~ s(x1) + s(x2, method='REML')")
    >>> len(parsed.smooth_terms)
    2
    >>> parsed.smooth_terms[0].method
    'auto'
    >>> parsed.smooth_terms[1].method
    'REML'
    
    >>> # Using te() function
    >>> parsed = parse_formula("y ~ te(x1, x2)")
    >>> len(parsed.tensor_terms)
    1
    >>> parsed.tensor_terms[0].variables
    ['x1', 'x2']
    """
    if "~" not in formula:
        raise ValueError("Formula must contain '~'")
    
    left, right = formula.split("~", 1)
    response = left.strip()
    
    if not response:
        raise ValueError("Formula must have a response variable")
    
    # Parse right-hand side
    rhs = right.strip()
    
    # Check for intercept removal
    has_intercept = True
    if "- 1" in rhs or "-1" in rhs:
        has_intercept = False
        rhs = rhs.replace("- 1", "").replace("-1", "")
    
    # Split by + but not inside parentheses
    terms = _split_formula_terms(rhs)
    
    linear_terms = []
    smooth_terms = []
    tensor_terms = []
    
    for term in terms:
        term = term.strip()
        if not term or term == "1":
            continue
        
        # Check if it's a smooth term (pb, ps, cs, s, te, ti)
        smooth_match = re.match(r'(\w+)\((.*)\)', term)
        if smooth_match:
            smoother = smooth_match.group(1)
            args_str = smooth_match.group(2)
            
            # Check if it's a tensor product term
            if smoother in ('te', 'ti'):
                # Parse tensor product term
                tensor = _parse_tensor_term(smoother, args_str)
                tensor_terms.append(tensor)
            # Check if it's a recognized smooth term
            elif smoother in ('pb', 'ps', 'cs', 's'):
                # Parse smooth term
                smooth = _parse_smooth_term(smoother, args_str)
                smooth_terms.append(smooth)
            else:
                # Not a smooth term, treat as linear (e.g., log(x))
                linear_terms.append(LinearTerm(variable=term))
        else:
            # Linear term
            linear_terms.append(LinearTerm(variable=term))
    
    return ParsedFormula(
        response=response,
        linear_terms=linear_terms,
        smooth_terms=smooth_terms,
        tensor_terms=tensor_terms,
        has_intercept=has_intercept,
    )


def _split_formula_terms(rhs: str) -> list[str]:
    """Split formula terms by + but not inside parentheses."""
    terms = []
    current = []
    depth = 0
    
    for char in rhs:
        if char == '(':
            depth += 1
            current.append(char)
        elif char == ')':
            depth -= 1
            current.append(char)
        elif char == '+' and depth == 0:
            terms.append(''.join(current))
            current = []
        else:
            current.append(char)
    
    if current:
        terms.append(''.join(current))
    
    return terms


def _parse_smooth_term(smoother: str, args_str: str) -> SmoothTerm:
    """Parse smooth term arguments.
    
    Examples:
    - "x" -> variable="x"
    - "x, df=5" -> variable="x", df=5
    - "x, lambda=1.0, degree=3" -> variable="x", lambda_=1.0, kwargs={"degree": 3}
    - "x, method='REML'" -> variable="x", method="REML"
    - "x, method='auto', smoother='ps'" -> variable="x", method="auto", smoother="ps"
    """
    # Split arguments
    args = [arg.strip() for arg in args_str.split(',')]
    
    if not args or not args[0]:
        raise ValueError(f"Smooth term {smoother}() must have a variable")
    
    variable = args[0]
    df = None
    lambda_ = None
    method = None
    actual_smoother = smoother  # Default to the function name
    kwargs = {}
    
    # Parse keyword arguments
    for arg in args[1:]:
        if '=' not in arg:
            continue
        
        key, value = arg.split('=', 1)
        key = key.strip()
        value = value.strip()
        
        # Remove quotes from string values
        if value.startswith(("'", '"')) and value.endswith(("'", '"')):
            value = value[1:-1]
        
        # Handle special keys
        if key == 'method':
            method = value
        elif key == 'smoother':
            actual_smoother = value
        elif key == 'df':
            try:
                df = float(value)
            except ValueError:
                raise ValueError(f"df must be a number, got {value!r}")
        elif key in ('lambda', 'lambda_'):
            try:
                lambda_ = float(value)
            except ValueError:
                raise ValueError(f"lambda must be a number, got {value!r}")
        else:
            # Try to parse as number
            try:
                kwargs[key] = float(value)
            except ValueError:
                # Keep as string
                kwargs[key] = value
    
    # Create SmoothTerm
    term = SmoothTerm(
        smoother=actual_smoother,
        variable=variable,
        df=df,
        lambda_=lambda_,
        kwargs=kwargs,
    )
    
    # Add method attribute if specified (for s() function)
    # If smoother is 's' and no method specified, default to 'auto'
    if method is not None:
        term.method = method
    elif smoother == 's':
        term.method = 'auto'
    
    return term


def _parse_tensor_term(smoother: str, args_str: str) -> TensorProductTerm:
    """Parse tensor product term arguments.
    
    Examples:
    - "x1, x2" -> variables=["x1", "x2"]
    - "x1, x2, k=10" -> variables=["x1", "x2"], k=10
    - "x1, x2, k_list=[5, 8]" -> variables=["x1", "x2"], k_list=[5, 8]
    - "x1, x2, x3, bs='cr'" -> variables=["x1", "x2", "x3"], bs="cr"
    """
    # First, handle k_list specially because it contains commas
    k_list = None
    if 'k_list=' in args_str:
        # Extract k_list value
        match = re.search(r'k_list\s*=\s*\[([^\]]+)\]', args_str)
        if match:
            k_list_str = match.group(1)
            try:
                k_list = [int(x.strip()) for x in k_list_str.split(',')]
            except ValueError:
                raise ValueError(f"k_list must be a list of integers")
            # Remove k_list from args_str
            args_str = re.sub(r',?\s*k_list\s*=\s*\[[^\]]+\]', '', args_str)
    
    # Split remaining arguments
    args = [arg.strip() for arg in args_str.split(',') if arg.strip()]
    
    if len(args) < 2:
        raise ValueError(f"Tensor product {smoother}() requires at least 2 variables")
    
    # Extract variables (all non-keyword arguments)
    variables = []
    k = None
    bs = "ps"
    lambda_ = None
    kwargs = {}
    
    for arg in args:
        if '=' not in arg:
            # Variable name
            variables.append(arg.strip())
        else:
            # Keyword argument
            key, value = arg.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Remove quotes from string values
            if value.startswith(("'", '"')) and value.endswith(("'", '"')):
                value = value[1:-1]
            
            # Handle special keys
            if key == 'k':
                try:
                    k = int(value)
                except ValueError:
                    raise ValueError(f"k must be an integer, got {value!r}")
            elif key == 'bs':
                bs = value
            elif key in ('lambda', 'lambda_'):
                try:
                    lambda_ = float(value)
                except ValueError:
                    raise ValueError(f"lambda must be a number, got {value!r}")
            else:
                # Try to parse as number
                try:
                    kwargs[key] = float(value)
                except ValueError:
                    # Keep as string
                    kwargs[key] = value
    
    if len(variables) < 2:
        raise ValueError(f"Tensor product {smoother}() requires at least 2 variables")
    
    return TensorProductTerm(
        smoother=smoother,
        variables=variables,
        k=k,
        k_list=k_list,
        bs=bs,
        lambda_=lambda_,
        kwargs=kwargs,
    )


def build_design_matrix(
    parsed: ParsedFormula,
    data: dict[str, Any],
    fit_smooths: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build design matrix from parsed formula.
    
    Parameters
    ----------
    parsed : ParsedFormula
        Parsed formula
    data : dict
        Data dictionary with variable names as keys
    fit_smooths : bool, default=True
        If True, fit smooth terms. If False, only build linear part.
    
    Returns
    -------
    X : np.ndarray
        Design matrix, shape (n, p)
    smooth_info : dict
        Information about fitted smooth terms
    
    Notes
    -----
    The design matrix includes:
    - Intercept column (if has_intercept=True)
    - Linear term columns
    - Smooth term columns (basis functions)
    - Tensor product term columns (tensor basis functions)
    
    smooth_info contains fitted smooth objects for prediction.
    """
    # Get response to determine sample size
    y = np.asarray(data[parsed.response])
    n = len(y)
    
    columns = []
    column_names = []
    smooth_info = {}
    
    # Add intercept
    if parsed.has_intercept:
        columns.append(np.ones(n))
        column_names.append("(Intercept)")
    
    # Add linear terms
    for term in parsed.linear_terms:
        if term.variable not in data:
            raise ValueError(f"Variable '{term.variable}' not found in data")
        
        x = np.asarray(data[term.variable])
        if len(x) != n:
            raise ValueError(f"Variable '{term.variable}' has wrong length")
        
        columns.append(x)
        column_names.append(term.variable)
    
    # Add smooth terms
    if fit_smooths:
        for i, term in enumerate(parsed.smooth_terms):
            if term.variable not in data:
                raise ValueError(f"Variable '{term.variable}' not found in data")
            
            x = np.asarray(data[term.variable])
            if len(x) != n:
                raise ValueError(f"Variable '{term.variable}' has wrong length")
            
            # Build smooth basis
            smooth_result = _build_smooth_basis(term, x)
            
            # Add basis columns
            for j in range(smooth_result['basis'].shape[1]):
                columns.append(smooth_result['basis'][:, j])
                column_names.append(f"{term.smoother}({term.variable})_{j}")
            
            # Store smooth info for prediction
            smooth_info[f"smooth_{i}"] = smooth_result
        
        # Add tensor product terms
        for i, term in enumerate(parsed.tensor_terms):
            # Check all variables exist
            for var in term.variables:
                if var not in data:
                    raise ValueError(f"Variable '{var}' not found in data")
            
            # Extract variable data
            X_vars = [np.asarray(data[var]) for var in term.variables]
            
            # Check lengths
            for var, x in zip(term.variables, X_vars):
                if len(x) != n:
                    raise ValueError(f"Variable '{var}' has wrong length")
            
            # Build tensor product basis
            tensor_result = _build_tensor_basis(term, X_vars)
            
            # Add basis columns
            for j in range(tensor_result['basis'].shape[1]):
                columns.append(tensor_result['basis'][:, j])
                var_str = ','.join(term.variables)
                column_names.append(f"{term.smoother}({var_str})_{j}")
            
            # Store tensor info for prediction
            smooth_info[f"tensor_{i}"] = tensor_result
    
    # Stack columns
    if not columns:
        raise ValueError("No terms in formula")
    
    X = np.column_stack(columns)
    
    return X, smooth_info


def _build_smooth_basis(term: SmoothTerm, x: np.ndarray) -> dict[str, Any]:
    """Build basis matrix for a smooth term.
    
    Returns a dictionary with:
    - basis: basis matrix
    - knots: knot sequence (for prediction)
    - smoother: smoother type
    - variable: variable name
    - other smoother-specific info
    """
    # For s() function, determine actual smoother to use
    # Default to 'pb' if smoother is 's'
    actual_smoother = term.smoother
    if actual_smoother == 's':
        # Check if smoother was specified in kwargs
        if 'smoother' in term.kwargs:
            actual_smoother = term.kwargs['smoother']
        else:
            actual_smoother = 'pb'  # Default to pb
    
    if actual_smoother == "pb":
        from omnilss.smoothers.bsplines import bspline_design_matrix
        
        # Extract parameters
        n_knots = term.kwargs.get('inter', None)
        degree = int(term.kwargs.get('degree', 3))
        
        # Build basis
        basis, knots = bspline_design_matrix(
            x,
            n_knots=n_knots,
            degree=degree,
        )
        
        return {
            'basis': np.array(basis),
            'knots': knots,
            'degree': degree,
            'smoother': 'pb',
            'variable': term.variable,
            'df': term.df,
            'lambda_': term.lambda_,
        }
    
    elif actual_smoother == "ps":
        # P-splines smooth with interval-based knot placement
        from omnilss.smoothers.ps import _create_ps_knots
        from omnilss.smoothers.bsplines import bspline_basis
        import jax.numpy as jnp

        ps_intervals = int(term.kwargs.get('ps_intervals', 20))
        degree = int(term.kwargs.get('degree', 3))

        knots = _create_ps_knots(x, ps_intervals=ps_intervals, degree=degree)
        # Vectorised basis construction
        basis = np.asarray(
            bspline_basis(
                jnp.array(x, dtype=jnp.float64),
                jnp.array(knots, dtype=jnp.float64),
                degree=degree,
            )
        )

        return {
            'basis': basis,
            'knots': knots,
            'degree': degree,
            'ps_intervals': ps_intervals,
            'smoother': 'ps',
            'variable': term.variable,
            'df': term.df,
            'lambda_': term.lambda_,
        }

    elif actual_smoother == "cs":
        # Cubic smoothing spline via scipy UnivariateSpline
        # Returns a "virtual" basis: identity columns (fitted values are stored directly)
        from omnilss.smoothers.cs import fit_cubic_spline

        r = fit_cubic_spline(x, np.zeros(len(x)), df=term.df)  # placeholder fit
        # For cs(), we store the spline object and use it directly at fit time
        # The "basis" is a single column of ones (intercept-like placeholder)
        return {
            'basis': np.ones((len(x), 1)),  # placeholder
            'smoother': 'cs',
            'variable': term.variable,
            'df': term.df,
            'lambda_': term.lambda_,
            'cs_x': x.copy(),  # store x for re-fitting
        }
    
    else:
        raise NotImplementedError(f"Smoother '{actual_smoother}' not yet implemented")


def _build_tensor_basis(term: TensorProductTerm, X_vars: list[np.ndarray]) -> dict[str, Any]:
    """Build tensor product basis matrix.
    
    Parameters
    ----------
    term : TensorProductTerm
        Tensor product term specification
    X_vars : list[np.ndarray]
        List of variable arrays
    
    Returns
    -------
    result : dict
        Dictionary with:
        - basis: tensor product basis matrix
        - penalty: tensor product penalty matrix
        - smoother: "te" or "ti"
        - variables: variable names
        - k: k value (if specified)
        - k_list: k_list (if specified)
        - bs: basis type
    """
    from omnilss.smoothers.tensor_smooth import create_tensor_basis
    
    # Determine k values
    k = term.k if term.k is not None else 10  # Default k=10
    k_list = term.k_list
    
    # Create tensor product basis
    basis, penalty = create_tensor_basis(
        *X_vars,
        k=k,
        k_list=k_list,
        bs=term.bs
    )
    
    return {
        'basis': basis,
        'penalty': penalty,
        'smoother': term.smoother,
        'variables': term.variables,
        'k': k,
        'k_list': k_list,
        'bs': term.bs,
        'lambda_': term.lambda_,
    }


def predict_smooth(
    smooth_info: dict[str, Any],
    x_new: np.ndarray,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Predict smooth term at new x values.
    
    Parameters
    ----------
    smooth_info : dict
        Smooth info from build_design_matrix
    x_new : np.ndarray
        New x values
    coefficients : np.ndarray
        Fitted coefficients for this smooth term
    
    Returns
    -------
    predictions : np.ndarray
        Predicted smooth values
    """
    if smooth_info['smoother'] in ('pb', 'ps', 'cs'):
        from omnilss.smoothers.bsplines import bspline_basis
        
        # Build basis at new points
        basis_new = bspline_basis(
            jnp.array(x_new),
            jnp.array(smooth_info['knots']),
            degree=smooth_info['degree'],
        )
        
        # Predict
        return np.array(basis_new @ jnp.array(coefficients))
    
    else:
        raise NotImplementedError(f"Prediction for '{smooth_info['smoother']}' not implemented")
