"""Typed runtime error hierarchy for OmniLSS."""

from __future__ import annotations


class OmniLSSError(Exception):
    """Base runtime error."""


class NumericalError(OmniLSSError):
    """Numerical instability or invalid floating-point state."""


class ConvergenceError(OmniLSSError):
    """Optimizer/solver failed to converge under configured policy."""


class DistributionError(OmniLSSError):
    """Distribution/family domain or parameterization violation."""


class RuntimeExecutionError(OmniLSSError):
    """Runtime execution/environment failure."""
