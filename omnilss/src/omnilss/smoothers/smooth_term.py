"""Smooth term specification for GAMLSS models.

This module provides the s() function for creating smooth terms with automatic
smoothing parameter selection, similar to mgcv::s() in R.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass
class SmoothTerm:
    """Specification for a smooth term with automatic lambda selection.
    
    This class represents a smooth term that can be used in GAMLSS formulas.
    It encapsulates all the information needed to fit a smooth function,
    including the variable name, smoothing method, and smoother type.
    
    Attributes
    ----------
    variable : str
        Variable name to smooth
    method : str
        Method for lambda selection:
        - "auto": Use GCV (default, recommended)
        - "GCV": Generalized Cross-Validation
        - "REML": Restricted Maximum Likelihood
        - "AIC": Akaike Information Criterion
        - "manual": User-specified df or lambda
    smoother : str
        Type of smoother to use:
        - "pb": P-splines (penalized B-splines)
        - "ps": P-splines with equal-spaced knots
        - "cs": Cubic smoothing splines
    df : float, optional
        Target degrees of freedom (overrides automatic selection)
    lambda_ : float, optional
        Fixed smoothing parameter (overrides automatic selection)
    kwargs : dict
        Additional arguments passed to the smoother
        
    Examples
    --------
    >>> # Automatic lambda selection
    >>> term = SmoothTerm("x", method="auto", smoother="pb")
    
    >>> # Manual df specification
    >>> term = SmoothTerm("x", method="manual", smoother="pb", df=5)
    
    >>> # With additional arguments
    >>> term = SmoothTerm("x", method="REML", smoother="pb", 
    ...                   kwargs={"n_knots": 20, "degree": 3})
    
    Notes
    -----
    This class is typically created using the s() function rather than
    directly instantiating it.
    
    See Also
    --------
    s : Function to create smooth terms
    """
    
    variable: str
    method: str = "auto"
    smoother: str = "pb"
    df: Optional[float] = None
    lambda_: Optional[float] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate the smooth term specification."""
        # Validate variable name
        if not isinstance(self.variable, str) or not self.variable:
            raise ValueError("variable must be a non-empty string")
        
        # Validate method
        valid_methods = ["auto", "GCV", "REML", "AIC", "ML", "GAIC", "manual"]
        if self.method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}, got {self.method!r}"
            )
        
        # Validate smoother
        valid_smoothers = ["pb", "ps", "cs"]
        if self.smoother not in valid_smoothers:
            raise ValueError(
                f"smoother must be one of {valid_smoothers}, got {self.smoother!r}"
            )
        
        # Validate df and lambda_ are not both specified
        if self.df is not None and self.lambda_ is not None:
            raise ValueError("Cannot specify both df and lambda_")
        
        # Validate df is positive
        if self.df is not None and self.df <= 0:
            raise ValueError(f"df must be positive, got {self.df}")
        
        # Validate lambda_ is non-negative
        if self.lambda_ is not None and self.lambda_ < 0:
            raise ValueError(f"lambda_ must be non-negative, got {self.lambda_}")
    
    def __repr__(self) -> str:
        """Return a string representation of the smooth term."""
        parts = [f"s({self.variable!r}"]
        
        if self.method != "auto":
            parts.append(f"method={self.method!r}")
        
        if self.smoother != "pb":
            parts.append(f"smoother={self.smoother!r}")
        
        if self.df is not None:
            parts.append(f"df={self.df}")
        
        if self.lambda_ is not None:
            parts.append(f"lambda_={self.lambda_}")
        
        if self.kwargs:
            for key, value in self.kwargs.items():
                parts.append(f"{key}={value!r}")
        
        return ", ".join(parts) + ")"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert the smooth term to a dictionary.
        
        Returns
        -------
        dict
            Dictionary representation of the smooth term
        """
        return {
            "variable": self.variable,
            "method": self.method,
            "smoother": self.smoother,
            "df": self.df,
            "lambda_": self.lambda_,
            "kwargs": self.kwargs.copy(),
        }


def s(
    variable: str,
    method: Literal["auto", "GCV", "REML", "AIC", "ML", "GAIC"] = "auto",
    smoother: Literal["pb", "ps", "cs"] = "pb",
    df: Optional[float] = None,
    lambda_: Optional[float] = None,
    **kwargs
) -> SmoothTerm:
    """Create a smooth term with automatic lambda selection.
    
    This function creates a smooth term specification that can be used in
    GAMLSS formulas. By default, it uses automatic lambda selection via GCV,
    similar to mgcv::s() in R.
    
    Parameters
    ----------
    variable : str
        Variable name to smooth
    method : {"auto", "GCV", "REML", "AIC", "ML", "GAIC"}, default="auto"
        Method for automatic lambda selection:
        - "auto": Use GCV (default, recommended)
        - "GCV": Generalized Cross-Validation
        - "REML": Restricted Maximum Likelihood (more stable than GCV)
        - "AIC": Akaike Information Criterion
        - "ML": Maximum Likelihood (iterative)
        - "GAIC": Generalized AIC
    smoother : {"pb", "ps", "cs"}, default="pb"
        Type of smoother to use:
        - "pb": P-splines (penalized B-splines) - default, flexible
        - "ps": P-splines with equal-spaced knots - similar to pb
        - "cs": Cubic smoothing splines - uses scipy
    df : float, optional
        Target degrees of freedom (overrides automatic selection)
    lambda_ : float, optional
        Fixed smoothing parameter (overrides automatic selection)
    **kwargs
        Additional arguments passed to the smoother:
        - n_knots : int - Number of knots (for pb)
        - ps_intervals : int - Number of intervals (for ps)
        - degree : int - B-spline degree (default 3)
        - order : int - Penalty order (default 2)
        
    Returns
    -------
    term : SmoothTerm
        Smooth term specification
        
    Examples
    --------
    >>> # Automatic lambda selection (recommended)
    >>> s("x")
    s('x')
    
    >>> # Use REML for lambda selection
    >>> s("x", method="REML")
    s('x', method='REML')
    
    >>> # Manual df specification
    >>> s("x", df=5)
    s('x', df=5)
    
    >>> # Use cubic splines
    >>> s("x", smoother="cs")
    s('x', smoother='cs')
    
    >>> # Specify number of knots
    >>> s("x", n_knots=20)
    s('x', n_knots=20)
    
    >>> # Multiple options
    >>> s("x", method="REML", smoother="ps", ps_intervals=30)
    s('x', method='REML', smoother='ps', ps_intervals=30)
    
    Notes
    -----
    This function is inspired by mgcv::s() in R, providing a unified interface
    for smooth terms with automatic smoothing parameter selection.
    
    The automatic selection methods are:
    - GCV: Fast and reliable, good default choice
    - REML: More stable than GCV, especially for small samples
    - AIC: Information-theoretic approach
    - ML: Iterative maximum likelihood
    - GAIC: Generalized AIC with penalty parameter
    
    When df or lambda_ is specified, automatic selection is disabled and the
    method parameter is ignored.
    
    References
    ----------
    - Wood, S. N. (2017). Generalized Additive Models: An Introduction with R.
      Chapman and Hall/CRC.
    - Rigby, R. A., & Stasinopoulos, D. M. (2005). Generalized additive models
      for location, scale and shape. Journal of the Royal Statistical Society:
      Series C, 54(3), 507-554.
    - Eilers, P. H. C., & Marx, B. D. (1996). Flexible smoothing with B-splines
      and penalties. Statistical Science, 11(2), 89-121.
    
    See Also
    --------
    SmoothTerm : Class representing a smooth term
    fit_pspline : Fit P-splines
    fit_pspline_smooth : Fit P-spline smooth
    fit_cubic_spline : Fit cubic splines
    """
    # If df or lambda_ is specified, override method
    if df is not None or lambda_ is not None:
        actual_method = "manual"
    else:
        actual_method = method
    
    return SmoothTerm(
        variable=variable,
        method=actual_method,
        smoother=smoother,
        df=df,
        lambda_=lambda_,
        kwargs=kwargs
    )
