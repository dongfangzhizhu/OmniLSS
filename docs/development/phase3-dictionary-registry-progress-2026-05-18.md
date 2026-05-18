# Phase 3 Progress: Dictionary-Based Built-In Registry

中文版本: [phase3-dictionary-registry-progress-2026-05-18_cn.md](phase3-dictionary-registry-progress-2026-05-18_cn.md)

## Completed in this step

- Replaced the registry bootstrap path that depended on the legacy resolver with a single `_BUILTIN_FAMILY_FACTORIES` dictionary.
- Mapped each currently registered built-in family name to its module and zero-argument factory attribute.
- Derived `_REGISTERED_FAMILIES` from that dictionary so capability matrices, registry snapshots, and family listing cannot drift from the built-in registry table.
- Added regression coverage for the dictionary-backed registry contract, including mixed-case factory names such as `EXGAUS` -> `exGAUS`.

## Notes

`distributions.resolve_family()` still preserves the public function name and delegates to the authoritative registry. The old legacy resolver remains in `distributions.py` only for compatibility with any direct private imports; registry bootstrap no longer uses it.
