"""Numerical configuration for R-compatible precision.

R source alignment note:
- The original `gamlss` package relies on double-precision numerics.
- The Python/JAX port therefore enables `float64` globally by default.
"""

from __future__ import annotations

import jax


jax.config.update("jax_enable_x64", True)
