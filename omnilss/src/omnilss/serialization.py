"""Model serialization helpers for GAMLSSModel."""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

OMNILSS_MODEL_VERSION = "0.3.0"


def _json_safe(value: Any) -> Any:
    value = _to_numpy_safe(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _to_numpy_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _to_numpy_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        converted = [_to_numpy_safe(v) for v in value]
        return type(value)(converted)
    try:
        import jax.numpy as jnp

        if isinstance(value, jnp.ndarray):
            return np.asarray(value)
    except Exception:
        pass
    if (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and not isinstance(value, np.ndarray)
    ):
        try:
            return np.asarray(value)
        except Exception:
            return value
    return value



def _design_matrix_schema(model: Any) -> dict[str, Any]:
    """Build a minimal serializable design-matrix schema for prediction audits."""

    schema: dict[str, Any] = {"version": 1, "parameters": {}}
    formulas = dict(getattr(model, "formulas", {}) or {})
    terms = dict(getattr(model, "terms", {}) or {})
    design_matrices = dict(getattr(model, "design_matrices", {}) or {})
    coefficients = dict(getattr(model, "coefficients", {}) or {})

    for parameter in getattr(model, "parameters", ()):  # tuple/list on GAMLSSModel
        term_info = terms.get(parameter, {}) or {}
        design = design_matrices.get(parameter)
        coef = coefficients.get(parameter)
        n_columns = None
        if design is not None:
            arr = np.asarray(design)
            if arr.ndim == 2:
                n_columns = int(arr.shape[1])
        coefficient_count = None
        if coef is not None:
            coefficient_count = int(np.asarray(coef).size)

        schema["parameters"][parameter] = {
            "formula": str(formulas.get(parameter, "")),
            "term_labels": list(term_info.get("term_labels", []) or []),
            "has_intercept": bool(term_info.get("intercept", True)),
            "n_columns": n_columns,
            "coefficient_count": coefficient_count,
        }
    return schema

def save_model_pickle(model: Any, path: str | Path) -> None:
    try:
        import cloudpickle  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Model serialization requires cloudpickle. Install with: pip install cloudpickle"
        ) from exc

    payload = _to_numpy_safe(model)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        cloudpickle.dump(payload, f)


def load_model_pickle(path: str | Path) -> Any:
    try:
        import cloudpickle  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Model serialization requires cloudpickle. Install with: pip install cloudpickle"
        ) from exc

    with Path(path).open("rb") as f:
        return cloudpickle.load(f)


def save_model_json(model: Any, path: str | Path) -> None:
    from .model import GAMLSSModel

    if not isinstance(model, GAMLSSModel):
        raise TypeError("save_model_json currently supports GAMLSSModel instances only")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        arrays: dict[str, np.ndarray] = {}
        arrays["y"] = np.asarray(model.y)
        for k, v in model.coefficients.items():
            arrays[f"coef__{k}"] = np.asarray(v)
        for k, v in model.linear_predictors.items():
            arrays[f"eta__{k}"] = np.asarray(v)
        for k, v in model.fitted_values.items():
            arrays[f"fit__{k}"] = np.asarray(v)

        import io

        buf = io.BytesIO()
        np.savez(buf, **arrays)
        zf.writestr("arrays.npz", buf.getvalue())

        meta = {
            "omnilss_version": OMNILSS_MODEL_VERSION,
            "family": getattr(model.family, "name", str(model.family)),
            "parameters": list(model.parameters),
            "formulas": {k: str(v) for k, v in dict(model.formulas).items()},
            "n": int(model.n),
            "g_dev": float(model.g_dev),
            "iter": int(model.iter),
            "type": str(model.type),
            "call": _json_safe(dict(model.call or {})),
            "design_matrix_schema": _json_safe(_design_matrix_schema(model)),
        }
        zf.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))


def load_model_json(path: str | Path):
    from .distributions import resolve_family
    from .model import GAMLSSModel

    with zipfile.ZipFile(Path(path), "r") as zf:
        meta = json.loads(zf.read("meta.json").decode("utf-8"))
        design_matrix_schema = meta.get("design_matrix_schema", {})
        version = meta.get("omnilss_version", "")
        if not str(version).startswith("0.3."):
            raise ValueError(f"Incompatible model version: {version}")

        import io

        npz = np.load(io.BytesIO(zf.read("arrays.npz")))
        params = tuple(meta["parameters"])
        fitted = {p: npz[f"fit__{p}"] for p in params if f"fit__{p}" in npz}
        coeffs = {p: npz[f"coef__{p}"] for p in params if f"coef__{p}" in npz}
        etas = {p: npz[f"eta__{p}"] for p in params if f"eta__{p}" in npz}
        y = npz["y"]

    fam = resolve_family(meta["family"])
    n = int(meta["n"])
    return GAMLSSModel(
        par=params,
        family=fam,
        df_fit=float(
            sum(len(np.ravel(coeffs.get(p, np.array([0.0])))) for p in params)
        ),
        g_dev=float(meta["g_dev"]),
        n=n,
        y=y,
        fitted_values=fitted,
        coefficients=coeffs,
        linear_predictors=etas,
        formulas=meta.get("formulas", {}),
        terms={},
        design_matrices={
            p: np.zeros((n, max(int(np.size(coeffs.get(p, np.array([0.0])))), 1)))
            for p in params
        },
        weights=np.ones(n),
        residuals=np.zeros(n),
        iter=int(meta.get("iter", 0)),
        type=meta.get("type", "Continuous"),
        parameters=params,
        call=meta.get("call", {}),
        additional_slots={
            "loaded_from_json": True,
            "omnilss_version": version,
            "design_matrix_schema": design_matrix_schema,
        },
    )


# Backward-compatible aliases
save_model = save_model_pickle
load_model = load_model_pickle
