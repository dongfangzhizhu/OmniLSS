# OmniLSS Documentation

OmniLSS is a Python/JAX implementation of generalized additive models for
location, scale, and shape (GAMLSS). This documentation set focuses on the
maintained project references that are present in this repository: algorithm
routing, device-aware configuration, validation gates, benchmark methodology,
architecture decisions, and development environment setup.

## Start here

- [Algorithm API](api/algorithms.md) documents `method="RS"`, `method="RS_JAX"`,
  `method="CG"`, and `method="auto"` routing.
- [Configuration API](api/config.md) documents device-aware routing thresholds
  and configuration-file overrides.
- [CG algorithm integrity validation](validation/cg-algorithm-integrity.md)
  records the current cross-derivative implementation status and release gates.
- [Device method selection](benchmarks/device-method-selection.md) explains how
  benchmark evidence is used before enabling automatic JAX routing thresholds.

## Documentation maintenance policy

The `docs/` tree should contain maintained references, validation reports, and
architecture records. One-off sprint notes, dated closure checklists, and
placeholder pages should be kept out of the published documentation once their
content has been consolidated into the maintained references above.
