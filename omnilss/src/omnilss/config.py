# SPDX-License-Identifier: GPL-3.0-or-later
"""OmniLSS runtime configuration.

This module controls device-aware automatic method selection for the
``gamlss()`` entry point.

Automatic method selection
--------------------------
When ``method='auto'`` (the default), OmniLSS selects the fitting backend
based on the available device and the number of observations:

    CPU  → always use ``'RS'`` (NumPy IRLS, fastest on CPU)
    GPU  → use ``'RS_JAX'`` when n >= GPU_CROSSOVER_N[family],
           otherwise use ``'RS'``
    TPU  → use ``'RS_JAX'`` when n >= TPU_CROSSOVER_N[family],
           otherwise use ``'RS'``

Benchmark basis
---------------
Crossover values were determined by benchmarking on:

    CPU benchmark (Windows, Intel i7-12700K, 12P/20L cores, 34 GB RAM):
        JAX 0.6.2 CPU backend — JAX RS is ALWAYS slower than NumPy RS
        on CPU, regardless of n.  No crossover found up to n=1,000,000.
        See: docs/benchmarks/jax_rs_<timestamp>.md

    GPU benchmark (WSL2 Debian, NVIDIA RTX 3060 12 GB, CUDA 12):
        JAX 0.10.0 GPU backend — pure JAX core is still slower than
        NumPy RS for the tested n values (up to 500,000) with a small
        design matrix (p=2).  The JAX while_loop + jnp.linalg.lstsq
        has GPU kernel launch overhead that dominates for small p.
        See: docs/benchmarks/gpu_crossover_<timestamp>.md

        IMPORTANT: GPU crossover has NOT been found yet.  The default
        GPU_CROSSOVER_N is set to math.inf (never auto-switch to JAX on
        GPU).  Update this value after benchmarking with your specific
        GPU and workload (larger design matrices, batch fitting, etc.).

    TPU benchmark: NOT YET PERFORMED.
        TPU_CROSSOVER_N defaults to math.inf.

Customisation
-------------
Override crossover thresholds at runtime::

    import omnilss.config as cfg

    # Force JAX on GPU for n >= 50,000 (after your own benchmarking)
    cfg.GPU_CROSSOVER_N["NO"] = 50_000
    cfg.GPU_CROSSOVER_N["default"] = 100_000

    # Disable auto-switching entirely (always use NumPy RS)
    cfg.AUTO_METHOD_ENABLED = False

    # Force JAX everywhere (useful for testing)
    cfg.FORCE_JAX = True

Or via environment variables before importing omnilss::

    OMNILSS_AUTO_METHOD=0          # disable auto-switching
    OMNILSS_FORCE_JAX=1            # force JAX everywhere
    OMNILSS_GPU_CROSSOVER_N=50000  # single value for all families

Per-family crossover values
---------------------------
``GPU_CROSSOVER_N`` and ``TPU_CROSSOVER_N`` are dicts mapping family name
to the minimum n at which JAX is faster.  The special key ``"default"``
applies to any family not explicitly listed.

A value of ``math.inf`` means "never auto-switch to JAX for this family".
"""

from __future__ import annotations

import math
import os
from typing import Union

# ---------------------------------------------------------------------------
# Master switch
# ---------------------------------------------------------------------------

#: Set to False to disable all automatic method selection.
#: When False, ``method='auto'`` behaves like ``method='RS'``.
AUTO_METHOD_ENABLED: bool = os.environ.get("OMNILSS_AUTO_METHOD", "1") != "0"

#: Set to True to force ``method='RS_JAX'`` regardless of device or n.
#: Useful for testing the JAX path on CPU.
FORCE_JAX: bool = os.environ.get("OMNILSS_FORCE_JAX", "0") == "1"

# ---------------------------------------------------------------------------
# CPU crossover thresholds
# ---------------------------------------------------------------------------
# Benchmark result: JAX RS is ALWAYS slower than NumPy RS on CPU.
# No crossover found up to n=1,000,000.
# → CPU always uses NumPy RS.

CPU_CROSSOVER_N: dict[str, float] = {
    "default": math.inf,  # never auto-switch on CPU
}

# ---------------------------------------------------------------------------
# GPU crossover thresholds
# ---------------------------------------------------------------------------
# Benchmark result (RTX 3060, JAX 0.10.0, CUDA 12, small design matrix p=2):
#   Pure JAX core is 1.7–4.3x SLOWER than NumPy RS up to n=500,000.
#   No crossover found.
#
# Hypothesis: GPU advantage requires larger design matrices (p >> 2) or
# batch fitting across many datasets.  With p=2, GPU kernel launch overhead
# dominates.
#
# → Default: never auto-switch on GPU.
# → Update these values after benchmarking with your specific workload.
#
# To update after your own benchmark:
#   import omnilss.config as cfg
#   cfg.GPU_CROSSOVER_N["NO"] = 50_000   # example
#   cfg.GPU_CROSSOVER_N["default"] = 100_000

_gpu_env = os.environ.get("OMNILSS_GPU_CROSSOVER_N")
_gpu_default: float = float(_gpu_env) if _gpu_env else math.inf

GPU_CROSSOVER_N: dict[str, float] = {
    # Per-family thresholds (math.inf = never auto-switch)
    # Update after GPU benchmarking with your workload.
    "NO":  math.inf,
    "GA":  math.inf,
    "PO":  math.inf,
    "BI":  math.inf,
    "WEI": math.inf,
    "TF":  math.inf,
    "default": _gpu_default,
}

# ---------------------------------------------------------------------------
# TPU crossover thresholds
# ---------------------------------------------------------------------------
# NOT YET BENCHMARKED.  Update after TPU testing.
#
# To update:
#   import omnilss.config as cfg
#   cfg.TPU_CROSSOVER_N["NO"] = 10_000   # example
#   cfg.TPU_CROSSOVER_N["default"] = 20_000

_tpu_env = os.environ.get("OMNILSS_TPU_CROSSOVER_N")
_tpu_default: float = float(_tpu_env) if _tpu_env else math.inf

TPU_CROSSOVER_N: dict[str, float] = {
    "default": _tpu_default,
}

# ---------------------------------------------------------------------------
# JAX-supported families
# ---------------------------------------------------------------------------
# Families that have a FamilyJAXSpec implementation.
# Only these can use method='RS_JAX'.

JAX_SUPPORTED_FAMILIES: frozenset[str] = frozenset(
    ["NO", "GA", "PO", "BI", "WEI", "TF"]
)

# ---------------------------------------------------------------------------
# Auto-selection logic
# ---------------------------------------------------------------------------

def _get_crossover(table: dict[str, float], family_name: str) -> float:
    return table.get(family_name, table.get("default", math.inf))


def auto_select_method(family_name: str, n_obs: int) -> str:
    """Return the fitting method to use for the given family and n.

    Called by ``gamlss()`` when ``method='auto'``.

    Parameters
    ----------
    family_name : str
        Distribution family name, e.g. ``"NO"``.
    n_obs : int
        Number of observations.

    Returns
    -------
    str
        One of ``'RS'`` or ``'RS_JAX'``.

    Notes
    -----
    Decision tree::

        FORCE_JAX=True          → 'RS_JAX' (if family supported)
        AUTO_METHOD_ENABLED=False → 'RS'
        family not JAX-supported  → 'RS'
        backend == 'cpu'          → 'RS'  (JAX never faster on CPU)
        backend == 'gpu'          → 'RS_JAX' if n >= GPU_CROSSOVER_N[family]
        backend == 'tpu'          → 'RS_JAX' if n >= TPU_CROSSOVER_N[family]
        else                      → 'RS'
    """
    if FORCE_JAX and family_name in JAX_SUPPORTED_FAMILIES:
        return "RS_JAX"

    if not AUTO_METHOD_ENABLED:
        return "RS"

    if family_name not in JAX_SUPPORTED_FAMILIES:
        return "RS"

    try:
        import jax
        backend = jax.default_backend()
    except Exception:
        return "RS"

    if backend == "cpu":
        # JAX is never faster than NumPy RS on CPU (benchmark result).
        return "RS"

    if backend == "gpu":
        threshold = _get_crossover(GPU_CROSSOVER_N, family_name)
        return "RS_JAX" if n_obs >= threshold else "RS"

    if backend == "tpu":
        threshold = _get_crossover(TPU_CROSSOVER_N, family_name)
        return "RS_JAX" if n_obs >= threshold else "RS"

    return "RS"


def get_config_summary() -> dict:
    """Return a summary of the current configuration for diagnostics."""
    try:
        import jax
        backend = jax.default_backend()
        devices = [str(d) for d in jax.devices()]
    except Exception:
        backend = "unknown"
        devices = []

    return {
        "auto_method_enabled": AUTO_METHOD_ENABLED,
        "force_jax":           FORCE_JAX,
        "jax_backend":         backend,
        "jax_devices":         devices,
        "jax_supported_families": sorted(JAX_SUPPORTED_FAMILIES),
        "cpu_crossover_n":     {k: (v if v != math.inf else "inf")
                                for k, v in CPU_CROSSOVER_N.items()},
        "gpu_crossover_n":     {k: (v if v != math.inf else "inf")
                                for k, v in GPU_CROSSOVER_N.items()},
        "tpu_crossover_n":     {k: (v if v != math.inf else "inf")
                                for k, v in TPU_CROSSOVER_N.items()},
    }
