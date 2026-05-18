# Phase 2 Progress: CG/L-BFGS Naming Cleanup

中文版本: [phase2-cg-lbfgs-renaming-progress-2026-05-18_cn.md](phase2-cg-lbfgs-renaming-progress-2026-05-18_cn.md)

## Completed in this step

- Moved the historical `cg_algorithm.py` implementation to `lbfgs_algorithm.py` to make the optimizer identity explicit.
- Added a compatibility shim at `cg_algorithm.py` for old imports.
- Added `lbfgs_fit` as a clear alias for `joint_lbfgs_fit`.
- Kept formula-based `cg_fit` and `cg_fit_lbfgs` compatibility wrappers with `DeprecationWarning` so existing callers keep working while new code can migrate to clearer names.
- Updated mixed-algorithm internals to import the L-BFGS backend from `lbfgs_algorithm.py` instead of the historical CG module name.
- Implemented the initial `algorithm="auto"` selection helper for mixed fitting, including smooth-term, unstable-family, and small-dataset RS rules.
- Extended `compare_algorithms` with deviance, iteration, convergence, and runtime metrics while preserving the existing `{"rs": model, "cg": model}` access pattern.

## Remaining Phase 2 work

- Fully retire or merge `cg_algorithm_v2.py` after its direct test coverage and helper usage are migrated.
- Expand `fitting_cg.py` singular-Hessian fallback coverage.
- Continue tuning `algorithm="auto"` thresholds with benchmark data.
