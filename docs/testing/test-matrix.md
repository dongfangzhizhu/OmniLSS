# OmniLSS Test Matrix

## Layers

1. **Numerical consistency**: compare Python results with R `gamlss` references
   using `1e-5` to `1e-6` tolerances where numerically appropriate.
2. **Gradient tests**: validate `jax.grad` against finite differences or trusted
   analytic scores.
3. **Stability tests**: cover NaN, overflow, underflow, and singular Hessian
   scenarios.
4. **Regression snapshots**: protect refactors from silently changing known
   fitted values, likelihoods, and predictions.
5. **Benchmarks**: separate compile time, cold runtime, warm runtime, memory,
   GPU runtime, and batch scaling.

## Freeze Rule

Test additions should validate architecture and numerical stability. They should
not accompany new distributions, formula syntax, optimizers, wrappers, or
smoothing methods during the 30-day freeze.
