# OmniLSS Test Matrix

## Layers

1. **Numerical consistency**: compare Python results with R `gamlss` references
   using `1e-5` to `1e-6` tolerances where numerically appropriate.
2. **Gradient tests**: validate `jax.grad` against finite differences or trusted
   analytic scores.
3. **Stability tests**: cover NaN, overflow, underflow, and singular Hessian
   scenarios.
4. **Architecture contracts**: verify canonical protocol required methods,
   estimable-parameter metadata, and legacy-adapter edge cases such as fixed
   data parameters.
5. **Regression snapshots**: protect refactors from silently changing known
   fitted values, likelihoods, and predictions.
6. **Benchmarks**: separate compile/setup time, cold runtime, warm runtime, setup-excluded R elapsed time, memory,
   GPU runtime, and batch scaling.

## Freeze Rule

Test additions should validate architecture and numerical stability. They should
not accompany new distributions, formula syntax, optimizers, wrappers, or
smoothing methods during the 30-day freeze.

## Current Architecture Contract Focus

`tests/test_core_architecture_contracts.py` is the smoke suite for P0
architecture boundaries. It currently covers canonical parameter PyTree behavior,
standard GAMLSS parameter lookup, distribution protocol required methods, fixed
data parameter exclusion in the legacy distribution adapter, and the Optax
optimizer adapter's non-mutating step boundary.

`tests/test_bb_dpqr.py` also now uses collectable pytest functions rather than
import-time script checks, so BB d/p/q/r behavior contributes to normal pytest
status reporting.

R-backed d/p/q coverage is tracked in [`r-consistency-coverage.md`](r-consistency-coverage.md). CG algorithm integrity is tracked in [`../validation/cg-algorithm-integrity.md`](../validation/cg-algorithm-integrity.md).
