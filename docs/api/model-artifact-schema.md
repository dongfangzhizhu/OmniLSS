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

Validation reports use this shape:

```json
{
  "ok": false,
  "errors": [
    {
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
- `unsupported_version`, `unsupported_schema_version`, `unsupported_artifact_schema_version`.
- `missing_parameter_schema`, `missing_parameter_formula`, `invalid_term_order`.
- `coefficient_schema_mismatch`.
- `missing_smooth_metadata`, `invalid_smooth_metadata_entry`, `missing_smooth_knots`.
- `training_data_included` as a warning.

## Prediction Error Boundary

Runtime prediction schema failures raise `PredictionSchemaError` with `code`, `parameter`, `term`, and `reason`. Clients should route on `code`, not on human-readable exception messages.
