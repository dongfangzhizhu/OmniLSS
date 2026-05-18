# Phase 2 Progress: `cg_fit` Uses the Real Cole-Green Backend

中文版本: [phase2-cg-fit-real-cg-progress-2026-05-18_cn.md](phase2-cg-fit-real-cg-progress-2026-05-18_cn.md)

## Completed in this step

- Updated the package-level `omnilss.algorithms.cg_fit` wrapper so it routes formula-based calls to `gamlss(method="CG")`, the full-Hessian Cole-Green backend.
- Kept historical L-BFGS compatibility available through `cg_fit_lbfgs`, `joint_lbfgs_fit`, and `lbfgs_fit`.
- Continued accepting legacy step-size keyword arguments on `cg_fit` for backward-compatible call signatures; the full-Hessian CG backend ignores those values and uses its own damped line search.
- Added regression coverage confirming package-level `cg_fit` no longer emits the old L-BFGS deprecation warning.

## Notes

The lower-level `omnilss.algorithms.cg_algorithm.cg_fit` module shim remains a deprecated compatibility alias for old imports from the historical CG-named L-BFGS module. New package-level code should use `omnilss.algorithms.cg_fit` for Cole-Green and `joint_lbfgs_fit`/`lbfgs_fit` for L-BFGS.
