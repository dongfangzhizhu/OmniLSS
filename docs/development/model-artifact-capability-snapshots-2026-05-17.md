# Model Artifact Capability Snapshots (2026-05-17)

[中文版本](model-artifact-capability-snapshots-2026-05-17_cn.md)

This note advances the artifact work in the [six-month execution plan](six-month-execution-plan-2026-05-17.md) and completes the second follow-up from the [family capability registry note](family-capability-registry-2026-05-17.md): JSON model artifacts now persist a snapshot of the fitted family's capability metadata.

## Runtime behavior

`save_model_json()` now writes a `family_capability` object into `meta.json`. The object is generated from the runtime capability registry and includes:

- family name;
- feature statuses such as `rs_fit`, `rs_jax_fit`, `prediction`, `r_consistency`, and `production_safe`;
- human-readable registry notes.

`load_model_json()` restores the same object into `model.additional_slots["family_capability"]` so downstream prediction, reporting, service APIs, and audit tools can inspect the evidence tier that was true when the artifact was written.

## Why this matters

- Model artifacts become more auditable: consumers can see whether the family was validated, experimental, or unsupported for key features.
- Service runtimes can expose capability snapshots in model reports without recomputing them.
- Future artifact migrations can detect when runtime capability policy changes after a model was trained.

## Follow-up work

1. Add an artifact compatibility report that compares the saved capability snapshot with the current runtime registry.
2. Add service endpoints that expose model artifact capability snapshots.
3. Include capability snapshots in generated calibration and governance reports.
4. Promote features from `experimental` to `validated` only through documented validation reports.
