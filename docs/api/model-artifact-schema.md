# OmniLSS Model Artifact Schema and Validation

[中文版本](model-artifact-schema_cn.md)

This document describes the production-facing JSON model artifact boundary introduced for the Week 1/Week 2 trustworthy-core work in the [six-month execution plan](../development/six-month-execution-plan-2026-05-17.md).

## Artifact Layout

A `.omnilss` JSON artifact is a ZIP archive with two required members:

- `meta.json`: JSON metadata, formulas, schema snapshots, diagnostics, and capability snapshots.
- `arrays.npz`: NumPy arrays for coefficients, fitted values, and linear predictors. Training response data is omitted by default.

## Required Metadata

`meta.json` should include:

- `omnilss_version`, currently compatible with `0.3.x` loaders.
- `parameters`, listing model parameters such as `mu` and `sigma`.
- `design_matrix_schema.version == 2` and `design_matrix_schema.artifact_version == 2`.
- `design_matrix_schema.parameters.<parameter>` entries with formula, term order, intercept state, column count, factor levels, numeric transform AST metadata, and smooth basis metadata where required.
- `family_capability`, a compact family capability snapshot.
- `smooth_infos`, when smooth prediction reconstruction is required.
- `diagnostics`, including compact scalar model diagnostics.

## Training Data Policy

`save_model_json(model, path)` omits full training data by default. Use `save_model_json(model, path, include_training_data=True)` only when training arrays are intentionally needed for a controlled workflow. Validators report `training_data_included` as a warning rather than an error.

## Programmatic Validation

```python
from omnilss import validate_model_json

report = validate_model_json("model.omnilss")
if not report["ok"]:
    for error in report["errors"]:
        print(error["code"], error["path"], error["message"])
```

The same validation is available as a development CLI:

```bash
PYTHONPATH=src python tools/validate_model_artifact.py model.omnilss
PYTHONPATH=src python tools/validate_model_artifact.py model.omnilss --fail-on-warning
```

## Structured Validation Issues

Validation reports use a versioned envelope so clients can route on stable issue fields instead of parsing human-readable text:

```json
{
  "type": "artifact_validation_report",
  "version": 1,
  "artifact": "model.omnilss",
  "ok": false,
  "error_count": 1,
  "warning_count": 0,
  "errors": [
    {
      "type": "artifact_validation_issue",
      "severity": "error",
      "code": "coefficient_schema_mismatch",
      "path": "arrays.coef__mu",
      "message": "Coefficient count 2 does not match schema n_columns 99"
    }
  ],
  "warnings": []
}
```

Important issue codes include:

- `missing_meta`, `invalid_meta`, `missing_arrays`, `invalid_arrays`, `invalid_zip`.
- `unsupported_version`, `invalid_design_matrix_schema_version`, `design_matrix_schema_migration_required`, `unsupported_future_design_matrix_schema_version`, `invalid_artifact_schema_version`, `artifact_schema_migration_required`, `unsupported_future_artifact_schema_version`.
- `missing_parameter_schema`, `missing_parameter_formula`, `invalid_term_order`.
- `missing_factor_levels`, `invalid_factor_levels`.
- `missing_numeric_transform_ast`, `invalid_numeric_transform_ast`, `invalid_numeric_transform_term`, `numeric_transform_ast_mismatch`.
- `coefficient_schema_mismatch`.
- `missing_smooth_metadata`, `invalid_smooth_metadata_entry`, `missing_smooth_knots`.
- `training_data_included` as a warning with `severity == "warning"`.


## Artifact Schema Migration Policy

`artifact_schema_policy()` exposes the runtime compatibility contract for JSON artifacts. The current runtime supports design-matrix schema version `2` and artifact schema version `2`. Older schema versions are rejected with `*_schema_migration_required` issue codes and should be re-saved from a compatible OmniLSS runtime before production use. Newer schema versions are rejected with `unsupported_future_*_schema_version` issue codes until the serving runtime is upgraded.

## Artifact and Prediction Error Example

The following abbreviated `meta.json` fragment shows the minimum schema fields a categorical predictor needs for schema-safe prediction:

```json
{
  "omnilss_version": "0.3.0",
  "parameters": ["mu"],
  "design_matrix_schema": {
    "version": 2,
    "artifact_version": 2,
    "parameters": {
      "mu": {
        "formula": "y ~ factor(grp)",
        "term_order": ["factor(grp)"],
        "has_intercept": true,
        "factor_levels": {"grp": ["a", "b"]},
        "n_columns": 2,
        "coefficient_count": 2
      }
    }
  }
}
```

If a client predicts with an unseen level through either the default `model.predict_params()` surface or the legacy R-aligned `predict()` / `predict_all()` surfaces, the runtime raises the same structured envelope:

```python
from omnilss.prediction import PredictionSchemaError
from omnilss.predict_gamlss_23_12_21 import predict

try:
    predict(model, what="mu", newdata={"grp": ["c", "a"]})
except PredictionSchemaError as exc:
    print(exc.to_dict())
```

Example output:

```json
{
  "code": "unseen_factor_levels",
  "parameter": "mu",
  "term": "factor(grp)",
  "reason": "unseen factor levels ['c']",
  "message": "Factor term 'factor(grp)' contains unseen levels ['c']"
}
```

The validator CLI remains the pre-runtime artifact gate. A valid categorical artifact should report no errors:

```bash
PYTHONPATH=src python tools/validate_model_artifact.py categorical.omnilss
```

```json
{
  "ok": true,
  "errors": [],
  "warnings": []
}
```


## Prediction Error Boundary

Runtime prediction schema failures raise `PredictionSchemaError` with `code`, `parameter`, `term`, and `reason`. Clients should route on `code`, not on human-readable exception messages.
