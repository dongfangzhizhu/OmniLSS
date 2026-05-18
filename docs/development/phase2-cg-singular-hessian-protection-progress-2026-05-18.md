# Phase 2 Progress: CG Singular-Hessian Protection Validation

中文版本: [phase2-cg-singular-hessian-protection-progress-2026-05-18_cn.md](phase2-cg-singular-hessian-protection-progress-2026-05-18_cn.md)

## Completed in this step

- Added a regression test for `fit_cg` using a near-singular (perfectly collinear) design matrix in the `mu` block.
- Verified that with explicit regularization enabled, `fit_cg` returns finite deviance and finite Fisher information instead of crashing on the linear solve.
- Fixed a duplicated `warnings.warn(...)` call in the non-converged branch of `fitting_cg.py` to keep warning behavior clean and deterministic.

## Notes

This step strengthens Phase 2 Task 2.4 coverage by validating the numerical-contract expectation that CG can survive ill-conditioned design matrices when regularization is supplied.
