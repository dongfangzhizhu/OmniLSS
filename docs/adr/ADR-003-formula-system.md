# ADR-003: Formula System Freeze

# Context

Formula DSLs tend to become hard-to-maintain parser projects. OmniLSS already
has enough modeling surface area and must avoid adding random-effects, tensor,
nonlinear, or custom syntax during the architecture freeze.

# Decision

For the freeze window, supported formula work is limited to:

- `y ~ x1 + x2`
- existing `pb(x)` smooth terms

Future richer modeling should prefer `formulaic`, `patsy`, or a tensor graph API
over extending a custom parser.

# Alternatives

- Continue expanding custom parser syntax.
- Remove formula support entirely.
- Add random effects and tensor syntax now.

# Consequences

- Parser complexity is capped.
- Refactors can focus on stable model construction boundaries.
- Future formula extensions require an explicit ADR.
