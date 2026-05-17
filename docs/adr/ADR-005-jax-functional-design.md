# ADR-005: JAX Functional Design

[中文版本](ADR-005-jax-functional-design_cn.md)

# Context

The long-term direction is a JAX-native distributional modeling framework, not a
line-by-line port of an R statistical package. Mutable caches and hidden model
state block `jit`, `grad`, `vmap`, and parallel execution.

# Decision

New core abstractions must use pure functions and explicit state:

```python
params = init_fn(data)
loss = loglikelihood(params, data)
```

Mutable distribution state, cached gradients, and optimizers that rewrite model
internals are forbidden in new architecture code.

# Alternatives

- Preserve R-style mutable fitting objects as the core design.
- Use OOP mutation for convenience and wrap it later.
- Restrict JAX support to isolated kernels.

# Consequences

- Code becomes easier to transform with JAX.
- Tests can compare inputs/outputs without hidden state.
- Legacy APIs need adapters until they can be rewritten.
