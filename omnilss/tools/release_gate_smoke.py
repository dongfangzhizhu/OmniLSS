"""Run offline Month 1 release-gate smoke checks.

The smoke checks exercise the core trust path required by the Month 1 plan:
fit -> predict -> serialize -> validate -> load -> predict, plus representative
schema-safe prediction errors.

Usage
-----
PYTHONPATH=omnilss/src python omnilss/tools/release_gate_smoke.py
"""

from __future__ import annotations

import argparse
import json
import tempfile
import warnings
from pathlib import Path
from typing import Any, Sequence


def _issue(code: str, message: str, *, path: str = "$") -> dict[str, str]:
    return {"severity": "error", "code": code, "path": path, "message": message}


def _warning_summary(captured: Sequence[warnings.WarningMessage]) -> dict[str, int]:
    summary: dict[str, int] = {}
    for warning in captured:
        key = warning.category.__name__
        summary[key] = summary.get(key, 0) + 1
    return summary


def run_release_gate_smoke(workdir: str | Path | None = None) -> dict[str, Any]:
    """Run deterministic offline release-gate smoke checks and return a report."""

    checks: list[dict[str, Any]] = []
    issues: list[dict[str, str]] = []

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")

        import numpy as np

        from omnilss import gamlss
        from omnilss.distributions import resolve_family
        from omnilss.model import GAMLSSModel
        from omnilss.prediction import PredictionSchemaError
        from omnilss.serialization import load_model_json, save_model_json, validate_model_json

        def add_check(name: str, ok: bool, **details: Any) -> None:
            checks.append({"name": name, "ok": bool(ok), **details})

        rng = np.random.default_rng(20260518)
        n = 48
        x = rng.normal(size=n)
        y = 1.0 + 0.3 * x + rng.normal(scale=0.2, size=n)
        newdata = {"x": np.array([-1.0, 0.0, 1.0])}

        tmp_ctx = tempfile.TemporaryDirectory() if workdir is None else None
        tmp_path = Path(tmp_ctx.name if tmp_ctx is not None else workdir)
        tmp_path.mkdir(parents=True, exist_ok=True)
        try:
            try:
                model = gamlss(
                    "y ~ x",
                    family="NO",
                    data={"y": y, "x": x},
                    max_iter=5,
                    verbose=False,
                )
                original = model.predict_params(newdata)
                artifact = tmp_path / "month1-release-gate-smoke.omnilss"
                save_model_json(model, artifact)
                validation = validate_model_json(artifact)
                loaded = load_model_json(artifact)
                restored = loaded.predict_params(newdata)
                max_abs_error = max(
                    float(
                        np.max(
                            np.abs(
                                np.asarray(restored[param])
                                - np.asarray(original[param])
                            )
                        )
                    )
                    for param in original
                )
                add_check(
                    "linear_fit_predict_json_roundtrip",
                    max_abs_error <= 1e-10,
                    max_abs_error=max_abs_error,
                    tolerance=1e-10,
                )
                if max_abs_error > 1e-10:
                    issues.append(
                        _issue(
                            "linear_roundtrip_prediction_mismatch",
                            f"roundtrip prediction max_abs_error={max_abs_error}",
                            path="$.checks.linear_fit_predict_json_roundtrip",
                        )
                    )
                add_check(
                    "model_artifact_validator",
                    bool(validation.get("ok")),
                    errors=validation.get("errors", []),
                    warnings=validation.get("warnings", []),
                )
                if not validation.get("ok"):
                    issues.append(
                        _issue(
                            "model_artifact_validator_failed",
                            "validate_model_json reported a non-OK artifact",
                            path="$.checks.model_artifact_validator",
                        )
                    )
            except Exception as exc:  # noqa: BLE001 - report all smoke failures.
                add_check("linear_fit_predict_json_roundtrip", False, error=str(exc))
                issues.append(
                    _issue(
                        "linear_roundtrip_smoke_exception",
                        str(exc),
                        path="$.checks.linear_fit_predict_json_roundtrip",
                    )
                )

            try:
                loaded.predict_params({"z": np.array([0.0, 1.0])})
            except PredictionSchemaError as exc:
                ok = exc.code in {"missing_prediction_variable", "term_evaluation_failed"}
                add_check(
                    "missing_variable_prediction_error",
                    ok,
                    code=exc.code,
                    parameter=exc.parameter,
                )
                if not ok:
                    issues.append(
                        _issue(
                            "unexpected_missing_variable_error_code",
                            f"unexpected PredictionSchemaError code {exc.code!r}",
                            path="$.checks.missing_variable_prediction_error",
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                add_check("missing_variable_prediction_error", False, error=str(exc))
                issues.append(
                    _issue(
                        "missing_variable_error_not_structured",
                        str(exc),
                        path="$.checks.missing_variable_prediction_error",
                    )
                )
            else:
                add_check("missing_variable_prediction_error", False)
                issues.append(
                    _issue(
                        "missing_variable_error_not_raised",
                        "prediction unexpectedly accepted missing variable input",
                        path="$.checks.missing_variable_prediction_error",
                    )
                )

            try:
                family = resolve_family("NO")
                factor_model = GAMLSSModel(
                    par=family.parameters,
                    family=family,
                    df_fit=2.0,
                    g_dev=0.0,
                    n=4,
                    y=np.array([], dtype=np.float64),
                    coefficients={"mu": np.array([0.0, 1.0])},
                    formulas={"mu": "y ~ factor(grp)"},
                    parameters=family.parameters,
                    additional_slots={
                        "design_matrix_schema": {
                            "version": 2,
                            "artifact_version": 2,
                            "parameters": {
                                "mu": {
                                    "parameter": "mu",
                                    "formula": "y ~ factor(grp)",
                                    "term_order": ["factor(grp)"],
                                    "has_intercept": True,
                                    "factor_levels": {"grp": ["a", "b"]},
                                    "n_columns": 2,
                                    "coefficient_count": 2,
                                }
                            },
                        }
                    },
                )
                factor_model.predict_params({"grp": np.array(["c", "a"])})
            except PredictionSchemaError as exc:
                ok = exc.code == "unseen_factor_levels"
                add_check(
                    "unseen_factor_prediction_error",
                    ok,
                    code=exc.code,
                    parameter=exc.parameter,
                )
                if not ok:
                    issues.append(
                        _issue(
                            "unexpected_unseen_factor_error_code",
                            f"unexpected PredictionSchemaError code {exc.code!r}",
                            path="$.checks.unseen_factor_prediction_error",
                        )
                    )
            except Exception as exc:  # noqa: BLE001
                add_check("unseen_factor_prediction_error", False, error=str(exc))
                issues.append(
                    _issue(
                        "unseen_factor_error_not_structured",
                        str(exc),
                        path="$.checks.unseen_factor_prediction_error",
                    )
                )
            else:
                add_check("unseen_factor_prediction_error", False)
                issues.append(
                    _issue(
                        "unseen_factor_error_not_raised",
                        "prediction unexpectedly accepted unseen factor levels",
                        path="$.checks.unseen_factor_prediction_error",
                    )
                )
        finally:
            if tmp_ctx is not None:
                tmp_ctx.cleanup()

    return {
        "ok": not issues,
        "checks": checks,
        "issues": issues,
        "warning_summary": _warning_summary(captured),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workdir",
        type=Path,
        default=None,
        help="Optional directory for temporary smoke artifacts.",
    )
    args = parser.parse_args(argv)
    report = run_release_gate_smoke(args.workdir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
