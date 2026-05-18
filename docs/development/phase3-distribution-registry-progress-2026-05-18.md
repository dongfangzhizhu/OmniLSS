# Phase 3 Progress: Authoritative Distribution Registry

中文版本: [phase3-distribution-registry-progress-2026-05-18_cn.md](phase3-distribution-registry-progress-2026-05-18_cn.md)

## Completed in this step

- Reworked `distribution_registry.py` around a single uppercase dictionary of family factories.
- Added public `register(name, factory)`, `resolve(name_or_family)`, and `list_families()` functions.
- Preserved the existing `DistributionRegistry`, `create_default_registry()`, and `get_default_registry()` compatibility APIs as snapshots of the authoritative registry.
- Delegated `distributions.resolve_family()` to `distribution_registry.resolve()` while retaining `None -> NO()` behavior.
- Added tests for case-insensitive lookup, already-instantiated family passthrough, dynamic registration, sorted family listing, and unknown-family diagnostics.

## Notes

The registry is now the public lookup surface. Built-in families are still bootstrapped from the legacy resolver table so this step is non-breaking; future new families should register their factory directly with `register()` after definition.
