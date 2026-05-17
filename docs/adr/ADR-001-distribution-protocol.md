# ADR-001: Distribution Protocol

[中文版本](ADR-001-distribution-protocol_cn.md)

# Context

OmniLSS contains many distribution families whose interfaces are currently
spread across legacy GAMLSS-style `d/p/q/r` functions, score functions, Hessian
helpers, and ad hoc initialization logic. This increases the risk of silent API
drift and makes JAX-native composition difficult.

# Decision

All distributions will converge on a stateless `DistributionProtocol` with:

- `logpdf`
- `cdf`
- `ppf`
- `sample`
- `score`
- `hessian`
- `init_params`
- `parameter_constraints`
- `links`

Implementations must be pure functional, JAX-compatible, and PyTree-compatible.
Legacy `FamilyDefinition` objects may use a migration adapter while files are
moved into the new namespace.

# Alternatives

- Keep the current `FamilyDefinition` shape indefinitely.
- Introduce an object hierarchy with mutable cached gradients.
- Migrate all distributions in one breaking commit.

# Consequences

- New code has one protocol target.
- Existing distributions can migrate incrementally.
- Distribution-specific parameter hacks become unacceptable.
- Tests can assert protocol conformance before implementation details are moved.
