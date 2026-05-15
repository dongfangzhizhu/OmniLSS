"""Model serialization helpers for GAMLSSModel."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


def _to_numpy_safe(value: Any) -> Any:
    """Recursively convert JAX arrays/array-likes to numpy arrays for portability."""
    if isinstance(value, dict):
        return {k: _to_numpy_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [_to_numpy_safe(v) for v in value]
        return type(value)(converted)
    try:
        import jax.numpy as jnp  # local import
        if isinstance(value, jnp.ndarray):
            return np.asarray(value)
    except Exception:
        pass
    if hasattr(value, "shape") and hasattr(value, "dtype") and not isinstance(value, np.ndarray):
        try:
            return np.asarray(value)
        except Exception:
            return value
    return value


def save_model(model: Any, path: str | Path) -> None:
    """Save a fitted GAMLSS model using cloudpickle.

    Notes
    -----
    cloudpickle is used to support closure fields like `rqres`.
    """
    try:
        import cloudpickle  # type: ignore
    except ImportError as exc:
        raise ImportError("Model serialization requires cloudpickle. Install with: pip install cloudpickle") from exc

    payload = _to_numpy_safe(model)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        cloudpickle.dump(payload, f)


def load_model(path: str | Path) -> Any:
    """Load a serialized GAMLSS model."""
    try:
        import cloudpickle  # type: ignore
    except ImportError as exc:
        raise ImportError("Model serialization requires cloudpickle. Install with: pip install cloudpickle") from exc

    with Path(path).open("rb") as f:
        return cloudpickle.load(f)
