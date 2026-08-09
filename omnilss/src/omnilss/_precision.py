"""Global numerical precision policy for omnilss.

Enables JAX float64 (x64) at import time so that every ``jnp.asarray(..., dtype=jnp.float64)``
in the codebase actually produces float64 instead of silently truncating to float32.

This module MUST be imported before any other omnilss module touches JAX.
``omnilss/__init__.py`` imports it first; tests and tools that import
``omnilss`` directly inherit the policy.  For isolated use, import this
module before importing anything else from omnilss.
"""

from __future__ import annotations

import jax as _jax

_jax.config.update("jax_enable_x64", True)
