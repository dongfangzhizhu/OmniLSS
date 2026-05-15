"""Numerical configuration for R-compatible precision.

R source alignment note:
- The original `gamlss` package relies on double-precision numerics.
- The Python/JAX port therefore enables `float64` globally by default.
"""

from __future__ import annotations

import jax


# Single source of truth for OmniLSS precision policy.
# Do not call jax.config.update("jax_enable_x64", True) in other modules,
# otherwise importing submodules can mutate global JAX state unpredictably.
jax.config.update("jax_enable_x64", True)
