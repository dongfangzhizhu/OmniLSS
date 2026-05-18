# Family Capability Registry Implementation Note (2026-05-17)

[中文版本](family-capability-registry-2026-05-17_cn.md)

This note records the first concrete development step from the [six-month execution plan](six-month-execution-plan-2026-05-17.md): creating a runtime family capability registry and marking every registered family by feature as `validated`, `experimental`, or `unsupported`.

## What changed

- Added `omnilss.family_capabilities` as the runtime capability registry.
- Covered every family listed by the existing distribution registry.
- Added feature-level statuses for:
  - `rs_fit`;
  - `rs_jax_fit`;
  - `cg_fit`;
  - `prediction`;
  - `sampling`;
  - `smooth_terms`;
  - `r_consistency`;
  - `ad_hessian`;
  - `production_safe`.
- Exported helper functions from the package root:
  - `get_family_capability()`;
  - `list_family_capabilities()`;
  - `family_supports()`;
  - `require_family_capability()`.
- Added tests that verify full registry coverage, feature completeness, unsupported-route errors, experimental opt-in behavior, and clear unknown-family/unknown-feature failures.
- Generated capability matrices include the fitting-method routing map (`method_capability_features`) and strict-mode policy flags so docs, service responses, and runtime gates share the same contract.

## Evidence tiers

| Status | Meaning | Runtime implication |
|---|---|---|
| `validated` | The repository currently has enough evidence for that specific feature tier. | May be used without explicit experimental opt-in. |
| `experimental` | The feature exists or is expected to work for exploratory use, but it is not yet production-safe. | `require_family_capability(..., allow_experimental=False)` rejects it. |
| `unsupported` | The current implementation does not advertise this feature for the family. | Always rejected by `require_family_capability()`. |

## Current policy

- `NO` is the only family marked `production_safe=validated` in this first registry iteration.
- Families with repository R-consistency test coverage are marked `r_consistency=validated` for that feature only.
- `RS_JAX` is marked experimental only for `NO`, `GA`, `PO`, `BI`, `WEI`, and `TF`; all other families are marked `rs_jax_fit=unsupported`.
- `AD/Hessian` is marked experimental for the same narrow JAX core-family set and unsupported elsewhere.
- Broad RS, CG, prediction, sampling, and smooth-term capabilities remain experimental until stronger fit/predict/artifact validation gates are completed.

## Example usage

```python
from omnilss.family_capabilities import (
    FamilyCapabilityError,
    family_supports,
    get_family_capability,
    require_family_capability,
)

capability = get_family_capability("NO")
assert capability.is_production_safe

if family_supports("GA", "rs_fit"):
    # Includes experimental support by default.
    ...

try:
    require_family_capability("GB2", "rs_jax_fit", allow_experimental=True)
except FamilyCapabilityError:
    # GB2 is not advertised for RS_JAX.
    ...
```

## Follow-up work

1. Completed in [Method Routing Capability Gates](method-routing-capability-gates-2026-05-17.md): `gamlss()` checks method/family capabilities before backend fitting starts.
2. Completed in [Model Artifact Capability Snapshots](model-artifact-capability-snapshots-2026-05-17.md): JSON model artifacts save and restore capability snapshots.
3. Completed in [Month 1 Capability Matrix Progress](month1-capability-matrix-progress-2026-05-18.md): generate a machine-readable capability matrix artifact for documentation and service APIs, including the method-routing contract.
4. Promote family features from `experimental` to `validated` only through documented validation reports.
5. Add service endpoints that expose capability data for UI and AutoML candidate filtering.
