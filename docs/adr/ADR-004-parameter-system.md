# ADR-004: Parameter System

# Context

Parameters such as `mu`, `sigma`, `nu`, and `tau` require links, constraints,
initialization, batching, and validation. Keeping these rules in individual
distributions produces duplicated hacks and inconsistent behavior.

# Decision

OmniLSS will use canonical `Parameter` definitions:

```python
Parameter(name="sigma", link=LogLink(), constraint=Positive())
```

Each parameter supports transform, inverse transform, constraint validation,
initialization, and batching through pure JAX-compatible operations.

# Alternatives

- Keep parameter logic inside each distribution.
- Use plain strings for parameter names and links.
- Add mutable parameter objects attached to models.

# Consequences

- Distribution implementations become smaller and more uniform.
- Constraints and links are testable independently.
- Migration requires mapping legacy family parameter names to canonical objects.
