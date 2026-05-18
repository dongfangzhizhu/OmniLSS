# Phase 3 Progress: Numerical-Contract Validation Baseline

中文版本: [phase3-numerical-contract-validation-progress-2026-05-18_cn.md](phase3-numerical-contract-validation-progress-2026-05-18_cn.md)

## Completed in this step

- Added an independent numerical-contract test module for core distribution behavior.
- Introduced `NO` dpqr contract checks validating quantile monotonicity and `p(q(p))` roundtrip consistency.
- Added score/Hessian contract checks for `NO`, covering expected-score near-zero behavior, negative Hessian sign constraints, and an empirical Fisher identity check `E[s^2] ≈ -E[H]`.

## Notes

This establishes a reproducible baseline for Phase 3 Task 3.4. Follow-up work can extend the same contract matrix to additional families (`GA`, `PO`, `WEI`, `TF`) with family-specific tolerances and boundary-condition grids.
