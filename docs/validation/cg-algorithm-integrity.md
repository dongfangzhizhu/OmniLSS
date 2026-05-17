# CG Algorithm Integrity Validation

## Status

CG is implemented as a complete observed-information update path for the
formula-level `gamlss(method="CG")` API and for the lower-level
`omnilss.fitting_cg.fit_cg()` helper. The implementation computes the score with
`jax.grad` and the full coefficient Hessian with `jax.hessian`, so cross-parameter
blocks such as `mu`/`sigma`, `mu`/`nu`, and `sigma`/`nu` are preserved instead of
being dropped by a block-diagonal approximation.

## Numerical safeguards

- The observed-information matrix is symmetrized before solving.
- Non-finite matrix entries are replaced before factorization.
- Eigenvalues are floored to keep the solve positive-definite while retaining
  cross-derivative structure.
- A backtracking line search accepts only finite, non-increasing deviance updates.

## Regression coverage

- `omnilss/tests/test_fitting_cg.py` checks CG deviance behavior and direct
  lower-level fitting invariants.
- `omnilss/tests/test_cg_cross_derivatives.py` checks that a heteroscedastic
  Normal model has a non-zero `beta_mu`/`beta_sigma` cross block, verifies block
  symmetry, proves that zeroing off-diagonal information changes the CG update
  direction, validates both `eta_score_hessian()` and the `eta_cross_hessian()`
  convenience alias, asserts that `gamlss(method="CG")` records
  `CG_FULL_HESSIAN` backend diagnostics, checks the opt-in `CG_IRLS_CROSS`
  backend records eta-correction diagnostics, covers weighted IRLS-cross fitting
  plus non-zero offset preservation in `cg_outer_step()`, and covers the
  `GA`/`BCCG`/`BCT`/`SHASH` validation matrix.
- `omnilss/tests/test_cg_algorithm.py` exercises the public `cg_fit()` wrapper,
  which now delegates to the complete formula-level CG backend rather than to an
  legacy RS-only fallback.

## Checked-in reference artifact

- `docs/benchmarks/cg-cross-derivative-reference-2026-05-17.md` and `.json`
  record a deterministic smoke comparison for `RS`, `CG_FULL_HESSIAN`, and
  `CG_IRLS_CROSS`. This is not a performance claim; it is a reproducible
  backend/trajectory reference.

## Benchmark scripts reviewed

- `benchmarks/comprehensive_r_consistency_test.py` now times Python fitting after
  an untimed warm-up, synchronizes JAX arrays before stopping timers, and passes
  algorithm selection through `method=` so RS/CG/Mixed are actually exercised.
- `benchmarks/comprehensive_performance_test.py` now obtains R timings from one
  R process per benchmark case and uses the in-process elapsed timings reported
  by R, excluding `Rscript` startup, package loading, and CSV parsing from speed
  ratios.

## Required release gate

Before publishing a performance or R-equivalence claim, run:

```bash
python benchmarks/comprehensive_r_consistency_test.py --quick --require-r
python benchmarks/comprehensive_performance_test.py --quick --require-r --n-repeats 3
```

Python-only benchmark runs are smoke checks only and must not be cited as
R-equivalence evidence.
