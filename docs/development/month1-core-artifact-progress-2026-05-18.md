# Month 1 Core Artifact Progress (2026-05-18)

> Chinese version: [month1-core-artifact-progress-2026-05-18_cn.md](month1-core-artifact-progress-2026-05-18_cn.md)
>
> Parent plan: [six-month-execution-plan-2026-05-17.md](six-month-execution-plan-2026-05-17.md)

## Scope

This note records the next implementation step against Month 1 / Workstream D1: **Model Artifact and Design-Matrix Schema v2**.

## Implemented Progress

- JSON model artifacts now omit full training data by default.
- The training response array is written only when `include_training_data=True` is passed to `save_model_json()`.
- Serialized call metadata marks `training_data_omitted=true` when training data is redacted.
- `df_fit` is persisted in JSON metadata and restored on load instead of being recomputed from nominal coefficient counts.
- Stable scalar diagnostics are captured in artifact metadata for downstream audit/reporting paths.
- Artifact validation reports now use a versioned `artifact_validation_report` envelope with typed `artifact_validation_issue` entries, explicit severity, and error/warning counts.
- Artifact validation now checks categorical factor-level metadata and numeric-transform AST metadata required for schema-safe prediction.
- `artifact_schema_policy()` now documents supported schema versions and validation/load behavior for legacy or future artifacts.
- Smooth-aware fit degrees of freedom are now computed through a shared algorithm helper to keep RS and CG accounting consistent.

## Remaining D1 Work

- Persist enough smooth-basis metadata for schema-safe smooth prediction roundtrips where supported.
- Continue expanding structured prediction/artifact error coverage into future service fit/sample boundaries.
- Expand schema validation to cover remaining interaction and unsupported smoother boundaries more comprehensively.
- Keep migration policy aligned with future schema revisions as they are introduced.
