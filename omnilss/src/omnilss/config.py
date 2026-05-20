# SPDX-License-Identifier: GPL-3.0-or-later
"""OmniLSS runtime configuration.

This module controls device-aware automatic method selection for the
``gamlss()`` entry point.

Automatic method selection
--------------------------
When ``method='auto'``, OmniLSS selects the fitting backend based on the
available device and the number of observations:

    CPU  → always use ``'RS'`` (NumPy IRLS, fastest on CPU)
    GPU  → use ``'RS_JAX'`` when n >= GPU_CROSSOVER_N[family],
           otherwise use ``'RS'``
    TPU  → use ``'RS_JAX'`` when n >= TPU_CROSSOVER_N[family],
           otherwise use ``'RS'``

A threshold value of ``math.inf`` means "never auto-switch to JAX for this
family".  The current default GPU/TPU thresholds are placeholders until the
crossover benchmark suite is run on target hardware.

Customisation
-------------
Override crossover thresholds at runtime::

    import omnilss.config as cfg

    cfg.set_crossover("gpu", n=50_000, family="NO")
    cfg.set_crossover("gpu", n=100_000)  # default for unlisted families

Or via environment variables before importing omnilss::

    OMNILSS_AUTO_METHOD=0
    OMNILSS_FORCE_JAX=1
    OMNILSS_GPU_CROSSOVER_N=NO=50000,default=100000
    OMNILSS_TPU_CROSSOVER_N=NO=10000,GA=20000

Configuration files
-------------------
At import time, OmniLSS loads the first existing configuration file from:

1. ``OMNILSS_CONFIG_FILE``
2. ``./omnilss_config.yaml``
3. ``~/.omnilss/config.yaml``

Environment variables are applied after configuration files and therefore
always take precedence.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import importlib.util
import json
import math
import os
from pathlib import Path
from typing import Any, Literal
import warnings

# ---------------------------------------------------------------------------
# JAX-supported families
# ---------------------------------------------------------------------------
# Families that have a FamilyJAXSpec implementation.
# Only these can use method='RS_JAX'.

JAX_SUPPORTED_FAMILIES: frozenset[str] = frozenset(
    ["NO", "GA", "PO", "BI", "WEI", "TF"]
)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

AUTO_METHOD_ENABLED: bool = True
FORCE_JAX: bool = False

# Benchmark result: JAX RS is ALWAYS slower than NumPy RS on CPU.
CPU_CROSSOVER_N: dict[str, float] = {
    "default": math.inf,
}

# GPU benchmark result (RTX 3060, JAX 0.10.0, CUDA 12, p=2): pure JAX core
# was 1.7–4.3x slower than NumPy RS up to n=500,000.  No crossover found.
# Keep placeholders until benchmarks/gpu_crossover_sweep.py is run on the
# target workload/hardware.
GPU_CROSSOVER_N: dict[str, float] = {
    "NO": math.inf,
    "GA": math.inf,
    "PO": math.inf,
    "BI": math.inf,
    "WEI": math.inf,
    "TF": math.inf,
    "default": math.inf,
}

# TPU benchmark: not yet performed.  Keep the same family coverage as GPU so
# YAML/env overrides can target any current JAX-supported family.
TPU_CROSSOVER_N: dict[str, float] = {
    "NO": math.inf,
    "GA": math.inf,
    "PO": math.inf,
    "BI": math.inf,
    "WEI": math.inf,
    "TF": math.inf,
    "default": math.inf,
}


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------

def _parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"invalid boolean value: {value!r}")


def _parse_float(value: Any) -> float:
    if isinstance(value, str) and value.strip().lower() in {
        "inf",
        "+inf",
        ".inf",
        "infinity",
        "+infinity",
    }:
        return math.inf
    parsed = float(value)
    if parsed < 0:
        raise ValueError("crossover threshold must be non-negative")
    return parsed


def _parse_scalar(text: str) -> Any:
    stripped = text.strip()
    if stripped == "":
        return ""
    lowered = stripped.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"inf", "+inf", ".inf", "infinity", "+infinity"}:
        return math.inf
    if (stripped.startswith('"') and stripped.endswith('"')) or (
        stripped.startswith("'") and stripped.endswith("'")
    ):
        return stripped[1:-1]
    try:
        return int(stripped.replace("_", ""))
    except ValueError:
        try:
            return float(stripped.replace("_", ""))
        except ValueError:
            return stripped


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    """Load the small YAML subset used by OmniLSS config files.

    PyYAML is used when installed.  This fallback intentionally supports only
    top-level scalars and one-level nested mappings, which is sufficient for
    documented OmniLSS config files.
    """
    result: dict[str, Any] = {}
    current_mapping: dict[str, Any] | None = None
    current_key: str | None = None

    for raw_line in text.splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            raise ValueError(f"invalid YAML line: {raw_line!r}")
        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()
        if indent == 0:
            if value == "":
                current_mapping = {}
                result[key] = current_mapping
                current_key = key
            else:
                result[key] = _parse_scalar(value)
                current_mapping = None
                current_key = None
        elif indent == 2 and current_mapping is not None and current_key is not None:
            current_mapping[key] = _parse_scalar(value)
        else:
            raise ValueError(f"unsupported YAML indentation: {raw_line!r}")

    return result


def _load_config_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        loaded = json.loads(text)
    else:
        if importlib.util.find_spec("yaml") is None:
            loaded = _minimal_yaml_load(text)
        else:
            yaml = importlib.import_module("yaml")
            loaded = yaml.safe_load(text) or {}
    if not isinstance(loaded, dict):
        raise ValueError("configuration root must be a mapping")
    return loaded


def _known_family_or_default(family: str) -> bool:
    return family == "default" or family in JAX_SUPPORTED_FAMILIES


def _apply_threshold_mapping(table: dict[str, float], values: Any, *, source: str) -> None:
    if not isinstance(values, dict):
        raise ValueError(f"{source} must be a mapping")
    for family, threshold in values.items():
        family_name = str(family).upper() if str(family).lower() != "default" else "default"
        if not _known_family_or_default(family_name):
            warnings.warn(
                f"Unknown family {family_name!r} in {source}; accepting it for future compatibility.",
                UserWarning,
                stacklevel=3,
            )
        table[family_name] = _parse_float(threshold)


def _apply_config_mapping(config: dict[str, Any], *, source: str = "configuration") -> None:
    """Apply a parsed OmniLSS configuration mapping in-place."""
    global AUTO_METHOD_ENABLED, FORCE_JAX

    if "auto_method_enabled" in config:
        AUTO_METHOD_ENABLED = _parse_bool(config["auto_method_enabled"])
    if "force_jax" in config:
        FORCE_JAX = _parse_bool(config["force_jax"])
    if "gpu_crossover_n" in config:
        _apply_threshold_mapping(GPU_CROSSOVER_N, config["gpu_crossover_n"], source=f"{source}:gpu_crossover_n")
    if "tpu_crossover_n" in config:
        _apply_threshold_mapping(TPU_CROSSOVER_N, config["tpu_crossover_n"], source=f"{source}:tpu_crossover_n")


def _candidate_config_files() -> list[Path]:
    env_path = os.environ.get("OMNILSS_CONFIG_FILE")
    if env_path:
        return [Path(env_path).expanduser()]
    return [Path.cwd() / "omnilss_config.yaml", Path.home() / ".omnilss" / "config.yaml"]


def _load_first_config_file() -> None:
    for path in _candidate_config_files():
        if not path.exists():
            continue
        try:
            _apply_config_mapping(_load_config_mapping(path), source=str(path))
        except Exception as exc:
            warnings.warn(
                f"Could not load OmniLSS config file {path}: {exc}. Using defaults and environment overrides.",
                UserWarning,
                stacklevel=2,
            )
        return


def _parse_threshold_env(value: str) -> dict[str, float]:
    """Parse threshold env vars as either a scalar or family=value pairs."""
    text = value.strip()
    if not text:
        return {}
    if "=" not in text:
        return {"default": _parse_float(text)}

    parsed: dict[str, float] = {}
    for item in text.split(","):
        if not item.strip():
            continue
        if "=" not in item:
            raise ValueError(f"invalid threshold override item: {item!r}")
        family, raw_threshold = item.split("=", 1)
        family_name = family.strip().upper() if family.strip().lower() != "default" else "default"
        parsed[family_name] = _parse_float(raw_threshold.strip())
    return parsed


def _apply_env_overrides() -> None:
    global AUTO_METHOD_ENABLED, FORCE_JAX

    if "OMNILSS_AUTO_METHOD" in os.environ:
        AUTO_METHOD_ENABLED = _parse_bool(os.environ["OMNILSS_AUTO_METHOD"])
    if "OMNILSS_FORCE_JAX" in os.environ:
        FORCE_JAX = _parse_bool(os.environ["OMNILSS_FORCE_JAX"])

    gpu_env = os.environ.get("OMNILSS_GPU_CROSSOVER_N")
    if gpu_env:
        _apply_threshold_mapping(GPU_CROSSOVER_N, _parse_threshold_env(gpu_env), source="OMNILSS_GPU_CROSSOVER_N")

    tpu_env = os.environ.get("OMNILSS_TPU_CROSSOVER_N")
    if tpu_env:
        _apply_threshold_mapping(TPU_CROSSOVER_N, _parse_threshold_env(tpu_env), source="OMNILSS_TPU_CROSSOVER_N")


# ---------------------------------------------------------------------------
# Public configuration API
# ---------------------------------------------------------------------------

def set_crossover(
    device: Literal["gpu", "tpu"],
    n: float,
    family: str = "default",
) -> None:
    """Set the crossover threshold for a device and distribution family.

    Parameters
    ----------
    device : {"gpu", "tpu"}
        Target accelerator device.  CPU thresholds are intentionally not
        configurable because CPU auto-routing always uses NumPy RS.
    n : float
        Observation-count threshold.  Use ``math.inf`` to disable automatic
        JAX routing for the selected family.
    family : str, default "default"
        Distribution family name.  ``"default"`` applies to families without a
        more specific threshold.

    Examples
    --------
    >>> import omnilss.config as cfg
    >>> cfg.set_crossover("gpu", n=50_000, family="NO")
    >>> cfg.set_crossover("gpu", n=100_000)
    >>> cfg.set_crossover("tpu", n=10_000, family="NO")
    """
    device_name = str(device).lower()
    if device_name not in {"gpu", "tpu"}:
        raise ValueError("device must be one of {'gpu', 'tpu'}; CPU never auto-switches")

    threshold = _parse_float(n)
    family_name = str(family).upper() if str(family).lower() != "default" else "default"
    if not _known_family_or_default(family_name):
        warnings.warn(
            f"Unknown family {family_name!r}; accepting it for future compatibility.",
            UserWarning,
            stacklevel=2,
        )

    table = GPU_CROSSOVER_N if device_name == "gpu" else TPU_CROSSOVER_N
    table[family_name] = threshold


@contextmanager
def crossover_config(gpu: dict[str, float] | None = None, tpu: dict[str, float] | None = None):
    """Temporarily override crossover thresholds and restore them on exit.

    Examples
    --------
    >>> import omnilss.config as cfg
    >>> with cfg.crossover_config(gpu={"NO": 50_000, "default": 100_000}):
    ...     cfg.auto_select_method("NO", 75_000)
    'RS'
    """
    original_gpu = dict(GPU_CROSSOVER_N)
    original_tpu = dict(TPU_CROSSOVER_N)
    try:
        if gpu is not None:
            _apply_threshold_mapping(GPU_CROSSOVER_N, gpu, source="crossover_config(gpu)")
        if tpu is not None:
            _apply_threshold_mapping(TPU_CROSSOVER_N, tpu, source="crossover_config(tpu)")
        yield
    finally:
        GPU_CROSSOVER_N.clear()
        GPU_CROSSOVER_N.update(original_gpu)
        TPU_CROSSOVER_N.clear()
        TPU_CROSSOVER_N.update(original_tpu)


def _format_threshold(value: float) -> str:
    return "inf" if value == math.inf else f"{value:g}"


def _current_backend() -> tuple[str, list[str]]:
    if importlib.util.find_spec("jax") is None:
        return "unknown", []
    jax = importlib.import_module("jax")
    return jax.default_backend(), [str(d) for d in jax.devices()]


def crossover_summary(verbose: bool = False) -> None:
    """Print the current automatic method-routing configuration."""
    backend, devices = _current_backend()
    device_label = backend.upper() if backend != "unknown" else "unknown"
    device_detail = f" ({devices[0]})" if verbose and devices else ""

    print("OmniLSS method-routing configuration")
    print("=" * 38)
    print(f"Current device: {device_label}{device_detail}")
    print(f"Automatic routing: {'enabled' if AUTO_METHOD_ENABLED else 'disabled'}")
    print(f"Force JAX: {'yes' if FORCE_JAX else 'no'}")
    print()
    print("GPU crossover thresholds (use RS_JAX when n >= threshold):")
    for family, threshold in GPU_CROSSOVER_N.items():
        suffix = "  (never auto-switch)" if threshold == math.inf else ""
        print(f"  {family:<7}: {_format_threshold(threshold)}{suffix}")
    if all(value == math.inf for value in GPU_CROSSOVER_N.values()):
        print("  [note: all GPU thresholds are placeholders; update after benchmarks/gpu_crossover_sweep.py]")
    print()
    print("TPU crossover thresholds (use RS_JAX when n >= threshold):")
    for family, threshold in TPU_CROSSOVER_N.items():
        suffix = "  (untested)" if threshold == math.inf else ""
        print(f"  {family:<7}: {_format_threshold(threshold)}{suffix}")


# ---------------------------------------------------------------------------
# Auto-selection logic
# ---------------------------------------------------------------------------

def _get_crossover(table: dict[str, float], family_name: str) -> float:
    return table.get(family_name, table.get("default", math.inf))


def auto_select_method(family_name: str, n_obs: int) -> str:
    """Return the fitting method to use for the given family and n.

    Decision tree::

        FORCE_JAX=True            → 'RS_JAX' (if family supported)
        AUTO_METHOD_ENABLED=False → 'RS'
        family not JAX-supported  → 'RS'
        backend == 'cpu'          → 'RS'
        backend == 'gpu'          → 'RS_JAX' if n >= GPU_CROSSOVER_N[family]
        backend == 'tpu'          → 'RS_JAX' if n >= TPU_CROSSOVER_N[family]
        else                      → 'RS'
    """
    return auto_select_method_trace(family_name, n_obs).method


@dataclass(frozen=True)
class MethodRoutingDecision:
    """Structured decision payload for method auto-routing."""

    method: str
    reason: str
    backend: str
    threshold: float | None = None


def auto_select_method_trace(family_name: str, n_obs: int) -> MethodRoutingDecision:
    """Return method auto-selection with structured routing context."""
    family = str(family_name).upper()

    if FORCE_JAX and family in JAX_SUPPORTED_FAMILIES:
        return MethodRoutingDecision(
            method="RS_JAX",
            reason="force_jax_enabled",
            backend=_current_backend()[0],
            threshold=None,
        )
    if not AUTO_METHOD_ENABLED:
        return MethodRoutingDecision(
            method="RS",
            reason="auto_method_disabled",
            backend=_current_backend()[0],
            threshold=None,
        )
    if family not in JAX_SUPPORTED_FAMILIES:
        return MethodRoutingDecision(
            method="RS",
            reason="family_not_jax_supported",
            backend=_current_backend()[0],
            threshold=None,
        )

    backend, _devices = _current_backend()
    if backend == "cpu":
        return MethodRoutingDecision(
            method="RS",
            reason="cpu_backend_prefers_numpy_rs",
            backend=backend,
            threshold=None,
        )
    if backend == "gpu":
        threshold = _get_crossover(GPU_CROSSOVER_N, family)
        return MethodRoutingDecision(
            method="RS_JAX" if n_obs >= threshold else "RS",
            reason="gpu_crossover_reached" if n_obs >= threshold else "gpu_crossover_not_reached",
            backend=backend,
            threshold=threshold,
        )
    if backend == "tpu":
        threshold = _get_crossover(TPU_CROSSOVER_N, family)
        return MethodRoutingDecision(
            method="RS_JAX" if n_obs >= threshold else "RS",
            reason="tpu_crossover_reached" if n_obs >= threshold else "tpu_crossover_not_reached",
            backend=backend,
            threshold=threshold,
        )
    return MethodRoutingDecision(
        method="RS",
        reason="unknown_backend_fallback",
        backend=backend,
        threshold=None,
    )




def describe_method_routing_reason(reason: str) -> str:
    """Return a user-facing explanation for a routing reason code."""
    mapping = {
        "force_jax_enabled": "FORCE_JAX is enabled, so RS_JAX is selected when supported.",
        "auto_method_disabled": "Automatic method routing is disabled; using NumPy RS.",
        "family_not_jax_supported": "This family is not JAX-enabled yet; using NumPy RS.",
        "cpu_backend_prefers_numpy_rs": "CPU backend defaults to NumPy RS for stability and overhead control.",
        "gpu_crossover_reached": "Sample size reached the configured GPU crossover threshold.",
        "gpu_crossover_not_reached": "Sample size is below the configured GPU crossover threshold.",
        "tpu_crossover_reached": "Sample size reached the configured TPU crossover threshold.",
        "tpu_crossover_not_reached": "Sample size is below the configured TPU crossover threshold.",
        "unknown_backend_fallback": "Backend could not be classified; falling back to NumPy RS.",
        "explicit_method_requested": "User explicitly requested this method; auto-routing crossover logic was bypassed.",
    }
    return mapping.get(reason, "No explanation registered for this routing reason code.")

def get_config_summary() -> dict[str, Any]:
    """Return a summary of the current configuration for diagnostics."""
    backend, devices = _current_backend()

    return {
        "auto_method_enabled": AUTO_METHOD_ENABLED,
        "force_jax": FORCE_JAX,
        "jax_backend": backend,
        "jax_devices": devices,
        "jax_supported_families": sorted(JAX_SUPPORTED_FAMILIES),
        "cpu_crossover_n": {k: (v if v != math.inf else "inf") for k, v in CPU_CROSSOVER_N.items()},
        "gpu_crossover_n": {k: (v if v != math.inf else "inf") for k, v in GPU_CROSSOVER_N.items()},
        "tpu_crossover_n": {k: (v if v != math.inf else "inf") for k, v in TPU_CROSSOVER_N.items()},
    }


_load_first_config_file()
_apply_env_overrides()
