# Architecture Stabilization Code Audit — 2026-05-15

This audit records the code-level issues found while continuing the 30-day
architecture stabilization plan. It is intentionally scoped to architecture,
testing, and documentation hygiene; it does not authorize new feature work during
the freeze.

## Findings

### 1. Distribution protocol conformance was too permissive

`DistributionProtocol` includes a `parameters()` method, but the runtime
conformance list did not require it. As a result, an object could pass
`assert_distribution_protocol()` while missing the canonical parameter metadata
needed by the migration layer.

Action taken: `parameters` is now part of `REQUIRED_DISTRIBUTION_METHODS`, and
the architecture contract test verifies every required method is callable.

### 2. Fixed data parameters leaked into canonical parameter metadata

Some legacy families include fixed data parameters in `family.parameters` because
the density function needs them. The beta-binomial family is the concrete case:
`bd` is the binomial denominator and must be supplied with the data, but it is
not an estimable model parameter.

Before this patch, `FamilyDistributionAdapter.parameters()` attempted to map all
legacy parameter names to canonical `Parameter` objects. That failed for fixed
non-canonical names such as `bd` and also risked exposing fixed data columns as
optimizable parameters.

Action taken: the adapter now uses `family.estimable_parameters` for canonical
parameter metadata, links, constraints, scores, Hessians, and initialization.
Fixed data parameters are still accepted in the likelihood `params` mapping.

### 3. Legacy d/p/q/r call styles are inconsistent

Hand-written d/p/q/r functions generally accept keyword parameters, while some
AD-generated fallback functions accept positional parameters only. The protocol
adapter now supports both call styles so legacy families can participate in the
canonical protocol during migration.

### 4. Script-style tests were not collectable by pytest

`tests/test_bb_dpqr.py` executed checks at import time and did not define pytest
test functions, so direct pytest collection exited with no collected tests.

Action taken: the file now contains explicit pytest tests for BB PMF
normalization, CDF monotonicity, quantile/CDF consistency, and random-sample
support.

### 5. README benchmark and test claims were too static

The root README and package README contained fixed test counts and fixed speedup
claims. Those claims conflicted with the benchmark policy, which requires
hardware/backend/R-availability context and generated artifacts before publishing
performance or R-equivalence statements.

Action taken: static speedup and pass-count claims were replaced with the current
architecture policy and validation-gate guidance.

## Remaining Risks

- RS, CG, Mixed, and L-BFGS still need protocol wrappers before Task 4 can move
  beyond Partial.
- Individual families still need protocol conformance coverage beyond the
  representative fixed-parameter adapter case.
- The parameter system still needs a complete mapping strategy for non-standard
  legacy data parameters and future distribution-specific metadata.
- The gRPC boundary remains a placeholder; protobuf/API design is still outside
  the current P0/P1 stabilization scope.
- Benchmark artifacts were not regenerated in this patch; no new performance or
  R-equivalence claims should be inferred from it.

## Verification Added

- `tests/test_core_architecture_contracts.py` now covers the fixed-data-parameter
  boundary using `BB()` and asserts that `bd` is accepted for likelihood
  evaluation but excluded from canonical estimable parameter metadata.
- `tests/test_bb_dpqr.py` is now pytest-collectable and verifies core BB d/p/q/r
  behavior.


## Follow-up Compatibility Work

- The R bridge now verifies `jsonlite`, `gamlss`, and `gamlss.dist` availability
  during setup, loads `gamlss.dist` in the evaluation script, and returns
  structured JSON errors from R.
- `tests/test_dpqr_r_consistency.py` now includes BB d/p/q cases so the fixed
  `bd` parameter boundary is checked against R when an R-enabled environment is
  available.
- Detailed R-backed d/p/q coverage is tracked in
  [`../testing/r-consistency-coverage.md`](../testing/r-consistency-coverage.md).
