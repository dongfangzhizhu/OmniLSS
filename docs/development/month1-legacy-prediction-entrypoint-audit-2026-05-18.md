# Month 1 Legacy Prediction Entry-Point Audit Progress (2026-05-18)

[中文版本](month1-legacy-prediction-entrypoint-audit-2026-05-18_cn.md)

This note advances Week 2 of the [six-month weekly implementation checklist](six-month-weekly-implementation-checklist-2026-05-17.md) and the Month 1 trustworthy-core stream in the [six-month execution plan](six-month-execution-plan-2026-05-17.md).

## Scope

The audit covers the legacy R-aligned prediction entry points that remained outside the schema-safe production path:

- `omnilss.predict_gamlss_23_12_21.predict()`;
- `omnilss.predictAll_22_08_22.predict_all()` through its use of the same single-parameter prediction helper.

## Completed Changes

- Legacy `newdata` prediction now delegates to the same schema-checked design-matrix builder used by `GAMLSSModel.predict_params()`.
- Structured `PredictionSchemaError` failures now propagate through legacy `predict()` and `predict_all()` for authoritative model artifacts instead of falling back to term-label reconstruction.
- A bounded fallback remains only for partial, hand-built legacy objects that do not carry a complete schema contract. The fallback is blocked when the saved parameter schema declares an expected column count.
- Dot-formula schema materialization now prefers expanded term metadata when available, because literal `.` formulas are not portable after serialization without the original model-frame expansion.

## Evidence Added

- A legacy `predict()` regression test verifies that unseen factor levels raise the structured schema error from the shared prediction builder.
- A legacy `predict_all()` regression test verifies that missing smooth metadata raises the structured schema error from a loaded JSON artifact.
- Indirect-wrapper regression tests verify that `prodist_data()` and `get_pef_data()` continue to propagate the same structured schema error instead of swallowing or rewriting prediction failures.

## Remaining Week 2 Follow-Up

- Continue auditing plotting/report helpers that accept `newdata` or build prediction grids to ensure they preserve the `PredictionSchemaError` envelope.
- A public example has been added to [Model Artifact Schema and Validation](../api/model-artifact-schema.md#artifact-and-prediction-error-example) showing the legacy-entrypoint error envelope alongside validator CLI output.
