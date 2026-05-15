# ADR-002: Optimizer Interface

# Context

RS, CG, mixed algorithms, Adam, and L-BFGS are implemented through different
entry points and state conventions. This creates optimizer explosion and makes
JAX transforms difficult to reason about.

# Decision

The canonical optimizer boundary is:

```python
class OptimizerProtocol:
    def init(self, params): ...
    def step(self, loss_fn, params, state): ...
```

Optimizers must return new params/state and must not mutate model internals.
Adapters such as `OptaxOptimizer` can expose existing libraries through this
boundary.

# Alternatives

- Preserve separate optimizer-specific APIs.
- Let optimizers mutate fitted model objects directly.
- Make every optimizer subclass a large framework base class.

# Consequences

- Optimizers become composable with `jit`, `grad`, `vmap`, and eventually `pmap`.
- Existing algorithms can be wrapped before being rewritten.
- Optimizer behavior becomes testable independently of model objects.
