# Month 1 Prediction Schema and Parser Checklist Progress (2026-05-17)

[中文版本](month1-prediction-schema-parser-checklist-2026-05-17_cn.md)

## Plan Checklist Reference

This checklist advances Month 1 Workstreams D1 and D2 in the [six-month execution plan](six-month-execution-plan-2026-05-17.md): artifact schema v2, hard prediction schema errors, formula parser hardening, and malicious/edge-case parser tests.

## Checklist Status

| Plan item | Status | Evidence |
|---|---|---|
| Missing prediction variables fail with structured schema errors | In progress / strengthened | `PredictionSchemaError` now carries `code`, `parameter`, `term`, and `reason` fields for schema consumers. |
| Unseen factor levels fail explicitly | Added | Regression coverage verifies `unseen_factor_levels` metadata. |
| Missing smooth metadata fails explicitly | Added | Regression coverage verifies `missing_smooth_metadata` metadata. |
| Column-count mismatches fail explicitly | Added | Regression coverage verifies `schema_column_mismatch` metadata. |
| Smooth/tensor argument parsing avoids naive comma splitting | Added | Formula parsing now splits at top-level delimiters and respects nested brackets and quoted strings. |
| Parser edge cases covered by tests | Added | Tests cover nested `k_list=[5, 8]`, quoted comma arguments, and unbalanced delimiter rejection. |
| Production prediction avoids arbitrary Python execution | Maintained | Numeric expression evaluation remains delegated to the AST allowlist evaluator. |

## Remaining Month 1 Follow-ups

1. Extend structured schema errors to every legacy prediction entry point that bypasses `omnilss.prediction`.
2. Add public artifact-schema examples showing the structured error shape and recommended client handling.
3. Continue replacing legacy string splitting in non-core helper modules as they become production prediction paths.
