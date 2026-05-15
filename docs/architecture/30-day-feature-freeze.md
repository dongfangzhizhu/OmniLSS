# 30-Day Architecture Freeze

## Objective

OmniLSS is in a complexity-control phase. For the next 30 days, the project goal
is **architecture stabilization**, not feature expansion.

## Freeze Scope

Do not add:

- new distributions
- new optimizers
- new formula syntax
- new wrappers
- new smoothing methods
- GUI, SaaS, AutoML, or broad platform features

Allowed work:

- refactor existing architecture
- unify protocols
- improve tests
- document decisions
- reduce coupling and mutation

## Dependency Direction

New code must follow a single directional graph:

```text
api -> core
smooth -> core
deep -> core
```

`core` must not import internals from `api`, `smooth`, or `deep`.

## Transitional Package Boundaries

The target namespace is being introduced incrementally to avoid breaking current
users:

```text
omnilss.core.distributions
omnilss.core.likelihood
omnilss.core.optimization
omnilss.core.params
omnilss.core.links
omnilss.core.constraints
omnilss.core.losses
omnilss.smooth.spline
omnilss.smooth.tensor_spline
omnilss.smooth.penalties
omnilss.api.formula
omnilss.api.sklearn
omnilss.api.grpc
```

Legacy modules remain importable until migration tests prove parity.
