"""Model serialization helpers for GAMLSSModel."""

from __future__ import annotations

import ast
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

OMNILSS_MODEL_VERSION = "0.3.0"
ARTIFACT_VALIDATION_REPORT_VERSION = 1
SUPPORTED_DESIGN_SCHEMA_VERSIONS = (2,)
SUPPORTED_ARTIFACT_SCHEMA_VERSIONS = (2,)
CURRENT_DESIGN_SCHEMA_VERSION = max(SUPPORTED_DESIGN_SCHEMA_VERSIONS)
CURRENT_ARTIFACT_SCHEMA_VERSION = max(SUPPORTED_ARTIFACT_SCHEMA_VERSIONS)


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


def _model_call_metadata(model: Any, include_training_data: bool) -> dict[str, Any]:
    """Return JSON-safe call metadata without training arrays by default."""

    call = dict(getattr(model, "call", {}) or {})
    if not include_training_data:
        call.pop("data", None)
        call["training_data_omitted"] = True
    return _json_safe(call)


def _artifact_diagnostics(model: Any) -> dict[str, Any]:
    """Return stable scalar diagnostics for JSON artifacts."""

    slots = dict(getattr(model, "additional_slots", {}) or {})
    keys = (
        "method",
        "cg_backend",
        "rs_converged",
        "cg_converged",
        "converged",
        "aic",
        "sbc",
        "df.residual",
        "df_residual",
        "smooth_edf",
        "family_capability",
    )
    return _json_safe({key: slots[key] for key in keys if key in slots})


def _smooth_metadata_snapshot(model: Any) -> dict[str, list[dict[str, Any]]]:
    """Return JSON-safe smooth basis metadata needed for prediction."""

    slots = dict(getattr(model, "additional_slots", {}) or {})
    smooth_infos = slots.get("smooth_infos", {})
    if not isinstance(smooth_infos, dict):
        return {}

    snapshot: dict[str, list[dict[str, Any]]] = {}
    for parameter, info in smooth_infos.items():
        smooth_fits = getattr(info, "smooth_fits", None)
        if smooth_fits is None and isinstance(info, list):
            smooth_fits = info
        if smooth_fits is None and isinstance(info, dict):
            smooth_fits = [info] if "variable" in info else list(info.values())
        if not smooth_fits:
            continue
        entries: list[dict[str, Any]] = []
        for smooth in smooth_fits:
            if isinstance(smooth, dict):
                entries.append(dict(smooth))
                continue
            knots = getattr(smooth, "knots", None)
            entries.append(
                {
                    "term_index": int(getattr(smooth, "term_index", -1)),
                    "variable": str(getattr(smooth, "variable", "")),
                    "smoother": str(getattr(smooth, "smoother", "")),
                    "basis_smoother": str(
                        getattr(smooth, "basis_smoother", None)
                        or getattr(smooth, "smoother", "")
                    ),
                    "lambda_": float(getattr(smooth, "lambda_", 0.0)),
                    "edf": float(getattr(smooth, "edf", 0.0)),
                    "basis_columns": list(getattr(smooth, "basis_columns", (0, 0))),
                    "selection_method": getattr(smooth, "selection_method", None),
                    "criterion_value": getattr(smooth, "criterion_value", None),
                    "knots": (
                        np.asarray(knots, dtype=np.float64).tolist()
                        if knots is not None and np.asarray(knots).size > 0
                        else None
                    ),
                    "degree": getattr(smooth, "degree", None),
                    "order": getattr(smooth, "order", None),
                }
            )
        snapshot[str(parameter)] = entries
    return snapshot


def _family_capability_snapshot(model: Any) -> dict[str, Any]:
    """Build a JSON-safe capability snapshot for the model family."""

    from .family_capabilities import get_family_capability

    family_obj = getattr(model, "family", None)
    family_name = getattr(family_obj, "name", str(family_obj))
    return get_family_capability(str(family_name)).as_dict()


def _artifact_issue(
    code: str, path: str, message: str, *, severity: str = "error"
) -> dict[str, str]:
    """Return a stable machine-readable artifact validation issue."""

    return {
        "type": "artifact_validation_issue",
        "severity": severity,
        "code": code,
        "path": path,
        "message": message,
    }


def _artifact_report(
    artifact: Path, errors: list[dict[str, str]], warnings: list[dict[str, str]]
) -> dict[str, Any]:
    """Return the versioned artifact validation report envelope."""

    return {
        "type": "artifact_validation_report",
        "version": ARTIFACT_VALIDATION_REPORT_VERSION,
        "artifact": str(artifact),
        "ok": not errors,
        "error_count": len(errors),
        "warning_count": len(warnings),
        "schema_policy": artifact_schema_policy(),
        "errors": errors,
        "warnings": warnings,
    }



def artifact_schema_policy() -> dict[str, Any]:
    """Return the JSON artifact schema compatibility and migration policy."""

    return {
        "type": "artifact_schema_policy",
        "omnilss_model_version": OMNILSS_MODEL_VERSION,
        "validation_report_version": ARTIFACT_VALIDATION_REPORT_VERSION,
        "current_design_schema_version": CURRENT_DESIGN_SCHEMA_VERSION,
        "current_artifact_schema_version": CURRENT_ARTIFACT_SCHEMA_VERSION,
        "supported_design_schema_versions": list(SUPPORTED_DESIGN_SCHEMA_VERSIONS),
        "supported_artifact_schema_versions": list(SUPPORTED_ARTIFACT_SCHEMA_VERSIONS),
        "migration_policy": {
            "older_schema": (
                "Artifacts with schema versions older than the supported set must "
                "be re-saved from a compatible OmniLSS runtime before production use."
            ),
            "newer_schema": (
                "Artifacts with schema versions newer than this runtime are rejected "
                "until the runtime is upgraded."
            ),
            "training_data": (
                "Training arrays remain omitted by default; validators warn when "
                "they are intentionally embedded."
            ),
        },
    }


def _schema_version_issue(
    *, schema_name: str, version: Any, path: str, expected: int
) -> dict[str, str] | None:
    try:
        parsed = int(version)
    except (TypeError, ValueError):
        return _artifact_issue(
            f"invalid_{schema_name}_schema_version",
            path,
            f"{schema_name.replace('_', ' ').title()} schema version must be an integer",
        )
    if parsed == expected:
        return None
    if parsed < expected:
        return _artifact_issue(
            f"{schema_name}_schema_migration_required",
            path,
            f"{schema_name.replace('_', ' ').title()} schema version {parsed} is older than supported version {expected}; re-save the artifact with a compatible OmniLSS runtime",
        )
    return _artifact_issue(
        f"unsupported_future_{schema_name}_schema_version",
        path,
        f"{schema_name.replace('_', ' ').title()} schema version {parsed} is newer than this runtime supports ({expected}); upgrade OmniLSS before loading",
    )


def _factor_variable(term: str) -> str | None:
    if term.startswith("factor(") and term.endswith(")"):
        variable = term[len("factor(") : -1].strip()
        return variable or None
    return None


def _smooth_function(term: str) -> str | None:
    if "(" not in term or not term.endswith(")"):
        return None
    function = term.split("(", 1)[0].strip()
    return function if function in {"pb", "ps", "cs", "s", "lo", "te", "ti"} else None


def _expected_numeric_transform_ast(term: str) -> str | None:
    if _factor_variable(term) is not None or _smooth_function(term) is not None:
        return None
    if not any(ch in term for ch in "()+-*/"):
        return None
    try:
        return ast.dump(ast.parse(term, mode="eval"), include_attributes=False)
    except SyntaxError:
        return ""


def validate_model_json(path: str | Path) -> dict[str, Any]:
    """Validate a JSON model artifact without loading executable model state.

    The validator is intentionally schema-focused: it checks archive structure,
    metadata versions, parameter/schema consistency, coefficient dimensions, and
    smooth metadata availability for schema-safe prediction.
    """

    errors: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []
    p = Path(path)
    if not p.exists():
        return _artifact_report(
            p,
            [
                _artifact_issue(
                    "artifact_missing", str(p), f"Artifact does not exist: {p}"
                )
            ],
            [],
        )

    try:
        with zipfile.ZipFile(p, "r") as zf:
            names = set(zf.namelist())
            if "meta.json" not in names:
                errors.append(
                    _artifact_issue("missing_meta", "meta.json", "Missing meta.json")
                )
                meta: dict[str, Any] = {}
            else:
                try:
                    meta = json.loads(zf.read("meta.json").decode("utf-8"))
                except Exception as exc:
                    errors.append(
                        _artifact_issue(
                            "invalid_meta", "meta.json", f"Invalid meta.json: {exc}"
                        )
                    )
                    meta = {}

            arrays: dict[str, Any] = {}
            if "arrays.npz" not in names:
                errors.append(
                    _artifact_issue(
                        "missing_arrays", "arrays.npz", "Missing arrays.npz"
                    )
                )
            else:
                try:
                    import io

                    npz = np.load(io.BytesIO(zf.read("arrays.npz")))
                    arrays = {name: npz[name] for name in npz.files}
                except Exception as exc:
                    errors.append(
                        _artifact_issue(
                            "invalid_arrays", "arrays.npz", f"Invalid arrays.npz: {exc}"
                        )
                    )
    except zipfile.BadZipFile as exc:
        return _artifact_report(
            p,
            [_artifact_issue("invalid_zip", str(p), f"Invalid zip artifact: {exc}")],
            [],
        )

    if not errors or meta:
        version = str(meta.get("omnilss_version", ""))
        if not version.startswith("0.3."):
            errors.append(
                _artifact_issue(
                    "unsupported_version",
                    "meta.omnilss_version",
                    f"Unsupported model version: {version!r}",
                )
            )

        parameters = [str(param) for param in meta.get("parameters", [])]
        if not parameters:
            errors.append(
                _artifact_issue(
                    "missing_parameters",
                    "meta.parameters",
                    "Artifact must list model parameters",
                )
            )

        schema = meta.get("design_matrix_schema", {})
        if not isinstance(schema, dict):
            errors.append(
                _artifact_issue(
                    "invalid_design_schema",
                    "meta.design_matrix_schema",
                    "Design matrix schema must be an object",
                )
            )
            schema_parameters: dict[str, Any] = {}
        else:
            design_version_issue = _schema_version_issue(
                schema_name="design_matrix",
                version=schema.get("version"),
                path="meta.design_matrix_schema.version",
                expected=CURRENT_DESIGN_SCHEMA_VERSION,
            )
            if design_version_issue is not None:
                errors.append(design_version_issue)
            artifact_version_issue = _schema_version_issue(
                schema_name="artifact",
                version=schema.get("artifact_version"),
                path="meta.design_matrix_schema.artifact_version",
                expected=CURRENT_ARTIFACT_SCHEMA_VERSION,
            )
            if artifact_version_issue is not None:
                errors.append(artifact_version_issue)
            schema_parameters = schema.get("parameters", {}) or {}
            if not isinstance(schema_parameters, dict):
                errors.append(
                    _artifact_issue(
                        "invalid_schema_parameters",
                        "meta.design_matrix_schema.parameters",
                        "Schema parameters must be an object",
                    )
                )
                schema_parameters = {}

        smooth_infos = meta.get("smooth_infos", {}) or {}
        if not isinstance(smooth_infos, dict):
            errors.append(
                _artifact_issue(
                    "invalid_smooth_infos",
                    "meta.smooth_infos",
                    "Smooth metadata must be an object",
                )
            )
            smooth_infos = {}

        for parameter in parameters:
            coef_key = f"coef__{parameter}"
            param_path = f"meta.design_matrix_schema.parameters.{parameter}"
            param_schema = schema_parameters.get(parameter)
            if not isinstance(param_schema, dict):
                errors.append(
                    _artifact_issue(
                        "missing_parameter_schema",
                        param_path,
                        f"Missing design schema for parameter {parameter!r}",
                    )
                )
                continue

            if not param_schema.get("formula"):
                errors.append(
                    _artifact_issue(
                        "missing_parameter_formula",
                        f"{param_path}.formula",
                        f"Missing formula for parameter {parameter!r}",
                    )
                )
            term_order = param_schema.get("term_order", [])
            if not isinstance(term_order, list):
                errors.append(
                    _artifact_issue(
                        "invalid_term_order",
                        f"{param_path}.term_order",
                        "term_order must be a list",
                    )
                )
                term_order = []

            factor_levels = param_schema.get("factor_levels", {}) or {}
            if not isinstance(factor_levels, dict):
                errors.append(
                    _artifact_issue(
                        "invalid_factor_levels",
                        f"{param_path}.factor_levels",
                        "factor_levels must be an object keyed by variable name",
                    )
                )
                factor_levels = {}
            numeric_transform_ast = param_schema.get("numeric_transform_ast", {}) or {}
            if not isinstance(numeric_transform_ast, dict):
                errors.append(
                    _artifact_issue(
                        "invalid_numeric_transform_ast",
                        f"{param_path}.numeric_transform_ast",
                        "numeric_transform_ast must be an object keyed by term",
                    )
                )
                numeric_transform_ast = {}

            for term_index, term in enumerate(term_order):
                term_text = str(term)
                factor_variable = _factor_variable(term_text)
                if factor_variable is not None:
                    levels = factor_levels.get(factor_variable)
                    if not isinstance(levels, list) or not levels:
                        errors.append(
                            _artifact_issue(
                                "missing_factor_levels",
                                f"{param_path}.factor_levels.{factor_variable}",
                                f"Factor term {term_text!r} requires non-empty factor levels",
                            )
                        )
                    continue

                expected_ast = _expected_numeric_transform_ast(term_text)
                if expected_ast == "":
                    errors.append(
                        _artifact_issue(
                            "invalid_numeric_transform_term",
                            f"{param_path}.term_order.{term_index}",
                            f"Numeric transform term {term_text!r} is not parseable",
                        )
                    )
                elif expected_ast is not None:
                    stored_ast = numeric_transform_ast.get(term_text)
                    if not isinstance(stored_ast, str) or not stored_ast:
                        errors.append(
                            _artifact_issue(
                                "missing_numeric_transform_ast",
                                f"{param_path}.numeric_transform_ast.{term_text}",
                                f"Numeric transform term {term_text!r} requires AST metadata",
                            )
                        )
                    elif stored_ast != expected_ast:
                        errors.append(
                            _artifact_issue(
                                "numeric_transform_ast_mismatch",
                                f"{param_path}.numeric_transform_ast.{term_text}",
                                f"Numeric transform AST for term {term_text!r} does not match the parsed term",
                            )
                        )

            expected_columns = param_schema.get("n_columns")
            if coef_key in arrays and expected_columns is not None:
                coef_count = int(np.asarray(arrays[coef_key]).size)
                if coef_count != int(expected_columns):
                    errors.append(
                        _artifact_issue(
                            "coefficient_schema_mismatch",
                            f"arrays.{coef_key}",
                            f"Coefficient count {coef_count} does not match schema n_columns {expected_columns}",
                        )
                    )

            if param_schema.get("smooth_metadata_required"):
                schema_smooth = param_schema.get("smooth_basis_metadata") or []
                artifact_smooth = smooth_infos.get(parameter) or []
                if not schema_smooth and not artifact_smooth:
                    errors.append(
                        _artifact_issue(
                            "missing_smooth_metadata",
                            f"{param_path}.smooth_basis_metadata",
                            f"Parameter {parameter!r} requires smooth metadata",
                        )
                    )
                for idx, smooth in enumerate(artifact_smooth):
                    if not isinstance(smooth, dict):
                        errors.append(
                            _artifact_issue(
                                "invalid_smooth_metadata_entry",
                                f"meta.smooth_infos.{parameter}.{idx}",
                                "Smooth metadata entries must be objects",
                            )
                        )
                        continue
                    if smooth.get("smoother") in {"pb", "ps", "s"} and not smooth.get(
                        "knots"
                    ):
                        errors.append(
                            _artifact_issue(
                                "missing_smooth_knots",
                                f"meta.smooth_infos.{parameter}.{idx}.knots",
                                "B-spline smooth metadata requires knots",
                            )
                        )

        if bool(meta.get("training_data_included", False)):
            warnings.append(
                _artifact_issue(
                    "training_data_included",
                    "meta.training_data_included",
                    "Artifact includes training data; use only when intentional",
                    severity="warning",
                )
            )

    return _artifact_report(p, errors, warnings)


def _capability_snapshot_from_artifact(path: str | Path) -> dict[str, Any]:
    """Read the saved family capability snapshot from an artifact metadata file."""

    with zipfile.ZipFile(Path(path), "r") as zf:
        meta = json.loads(zf.read("meta.json").decode("utf-8"))
    return dict(meta.get("family_capability", {}) or {})


def compare_model_capability_snapshot(model_or_path: Any) -> dict[str, Any]:
    """Compare a saved model capability snapshot with the current registry.

    Parameters
    ----------
    model_or_path:
        Either a loaded ``GAMLSSModel``-like object or a JSON artifact path.

    Returns
    -------
    dict
        A JSON-friendly compatibility report with ``ok=True`` when the saved
        snapshot matches the current runtime capability registry.
    """

    from .family_capabilities import FEATURES, get_family_capability

    if isinstance(model_or_path, (str, Path)):
        artifact_capability = _capability_snapshot_from_artifact(model_or_path)
    else:
        slots = dict(getattr(model_or_path, "additional_slots", {}) or {})
        artifact_capability = dict(slots.get("family_capability", {}) or {})

    family_name = artifact_capability.get("name")
    if not family_name and not isinstance(model_or_path, (str, Path)):
        family_obj = getattr(model_or_path, "family", None)
        family_name = (
            getattr(family_obj, "name", str(family_obj))
            if family_obj is not None
            else None
        )

    if not family_name:
        return {
            "ok": False,
            "family": None,
            "artifact_capability": artifact_capability,
            "runtime_capability": None,
            "changes": [
                {
                    "code": "missing_artifact_capability",
                    "message": "Artifact does not contain a family capability snapshot",
                }
            ],
        }

    runtime_capability = get_family_capability(str(family_name)).as_dict()
    artifact_features = dict(artifact_capability.get("features", {}) or {})
    runtime_features = dict(runtime_capability.get("features", {}) or {})
    changes: list[dict[str, Any]] = []
    for feature in FEATURES:
        saved_status = artifact_features.get(feature)
        runtime_status = runtime_features.get(feature)
        if saved_status != runtime_status:
            changes.append(
                {
                    "code": "capability_status_changed",
                    "feature": feature,
                    "artifact_status": saved_status,
                    "runtime_status": runtime_status,
                }
            )

    return {
        "ok": not changes,
        "family": str(family_name),
        "artifact_capability": artifact_capability,
        "runtime_capability": runtime_capability,
        "changes": changes,
    }

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


def save_model_json(
    model: Any, path: str | Path, *, include_training_data: bool = False
) -> None:
    from .design_schema import ensure_model_design_schema
    from .model import GAMLSSModel

    if not isinstance(model, GAMLSSModel):
        raise TypeError("save_model_json currently supports GAMLSSModel instances only")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(p, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        arrays: dict[str, np.ndarray] = {}
        if include_training_data:
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
            "terms": _json_safe(dict(getattr(model, "terms", {}) or {})),
            "n": int(model.n),
            "df_fit": float(model.df_fit),
            "g_dev": float(model.g_dev),
            "iter": int(model.iter),
            "type": str(model.type),
            "call": _model_call_metadata(model, include_training_data),
            "training_data_included": bool(include_training_data),
            "design_matrix_schema": _json_safe(ensure_model_design_schema(model)),
            "family_capability": _json_safe(_family_capability_snapshot(model)),
            "smooth_infos": _json_safe(_smooth_metadata_snapshot(model)),
            "diagnostics": _artifact_diagnostics(model),
        }
        zf.writestr("meta.json", json.dumps(meta, ensure_ascii=False, indent=2))


def load_model_json(path: str | Path):
    from .distributions import resolve_family
    from .model import GAMLSSModel

    with zipfile.ZipFile(Path(path), "r") as zf:
        meta = json.loads(zf.read("meta.json").decode("utf-8"))
        design_matrix_schema = meta.get("design_matrix_schema", {})
        if not isinstance(design_matrix_schema, dict):
            raise ValueError("Artifact is missing a valid design_matrix_schema object")
        for issue in (
            _schema_version_issue(
                schema_name="design_matrix",
                version=design_matrix_schema.get("version"),
                path="meta.design_matrix_schema.version",
                expected=CURRENT_DESIGN_SCHEMA_VERSION,
            ),
            _schema_version_issue(
                schema_name="artifact",
                version=design_matrix_schema.get("artifact_version"),
                path="meta.design_matrix_schema.artifact_version",
                expected=CURRENT_ARTIFACT_SCHEMA_VERSION,
            ),
        ):
            if issue is not None:
                raise ValueError(issue["message"])
        family_capability = meta.get("family_capability", {})
        smooth_infos = meta.get("smooth_infos", {})
        version = meta.get("omnilss_version", "")
        if not str(version).startswith("0.3."):
            raise ValueError(f"Incompatible model version: {version}")

        import io

        npz = np.load(io.BytesIO(zf.read("arrays.npz")))
        params = tuple(meta["parameters"])
        fitted = {p: npz[f"fit__{p}"] for p in params if f"fit__{p}" in npz}
        coeffs = {p: npz[f"coef__{p}"] for p in params if f"coef__{p}" in npz}
        etas = {p: npz[f"eta__{p}"] for p in params if f"eta__{p}" in npz}
        y = npz["y"] if "y" in npz.files else np.array([], dtype=np.float64)

    fam = resolve_family(meta["family"])
    n = int(meta["n"])
    return GAMLSSModel(
        par=params,
        family=fam,
        df_fit=float(
            meta.get(
                "df_fit",
                sum(len(np.ravel(coeffs.get(p, np.array([0.0])))) for p in params),
            )
        ),
        g_dev=float(meta["g_dev"]),
        n=n,
        y=y,
        fitted_values=fitted,
        coefficients=coeffs,
        linear_predictors=etas,
        formulas=meta.get("formulas", {}),
        terms=meta.get("terms", {}),
        design_matrices={},
        weights=np.ones(n),
        residuals=np.zeros(n),
        iter=int(meta.get("iter", 0)),
        type=meta.get("type", "Continuous"),
        parameters=params,
        call=meta.get("call", {}),
        additional_slots={
            "loaded_from_json": True,
            "omnilss_version": version,
            "training_data_included": bool(meta.get("training_data_included", True)),
            "design_matrix_schema": design_matrix_schema,
            "family_capability": family_capability,
            "smooth_infos": smooth_infos,
            "artifact_diagnostics": meta.get("diagnostics", {}),
        },
    )


# Backward-compatible aliases
save_model = save_model_pickle
load_model = load_model_pickle
