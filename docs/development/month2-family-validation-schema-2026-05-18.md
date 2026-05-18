# Month 2 Family Validation Schema Seed (2026-05-18)

[中文版本](month2-family-validation-schema-2026-05-18_cn.md)

> Parent plan: [six-month execution plan](six-month-execution-plan-2026-05-17.md)
>
> Weekly checklist: [six-month weekly implementation checklist](six-month-weekly-implementation-checklist-2026-05-17.md)
>
> Machine-readable seed: [core-family-validation-plan-2026-05-18.json](core-family-validation-plan-2026-05-18.json)

## Scope

This note starts the next planned workstream after the Month 1 release-gate preflight: **D4 validation matrix**.

The current output is a schema seed and prioritized-family JSON plan. It does not mark statistical checks as passed yet; it defines the shape of evidence that Week 5 implementation should fill.

## Prioritized Families

The initial validation plan follows the six-month execution plan and prioritizes:

`NO`, `GA`, `PO`, `BI`, `NBI`, `BE`, `WEI`, `TF`, `LOGNO`, and `ZAGA`.

## Required Check Types

Each family row in the JSON plan has placeholders for:

- density or PMF reference values;
- CDF monotonicity;
- quantile/CDF inverse consistency;
- random sample moments where applicable;
- score versus finite-difference checks;
- Hessian versus finite-difference or AD checks;
- edge cases and invalid parameter handling.

## Failure Classification

Future validation outputs should classify failures as one of:

- `implementation_bug`;
- `numerical_tolerance_issue`;
- `unsupported_domain`;
- `reference_mismatch`;
- `environment_unavailable`.

## Next Implementation Step

Add a generator/runner that fills the JSON plan with concrete check results for `NO` first, then expands through the remaining prioritized families in order.
