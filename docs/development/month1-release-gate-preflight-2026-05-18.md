# Month 1 Release Gate Preflight (2026-05-18)

[中文版本](month1-release-gate-preflight-2026-05-18_cn.md)

> Parent checklist: [six-month weekly implementation checklist](six-month-weekly-implementation-checklist-2026-05-17.md)
>
> Related capability closure: [Month 1 Capability Matrix Progress](month1-capability-matrix-progress-2026-05-18.md)

## Scope

This note starts Week 4: **Month 1 release gate / Core trust checkpoint with reproducible test evidence**.

The preflight gate is intentionally offline-friendly. It validates release-critical metadata that should pass before optional packaging tools, networked checks, R consistency jobs, or GPU validation are attempted.

## Implemented Preflight Gate

`omnilss/tools/release_check.py` now supports:

```bash
PYTHONPATH=omnilss/src python omnilss/tools/release_check.py --preflight-only
```

The preflight path currently checks:

1. bilingual documentation localization and cross-link policy;
2. generated capability matrix schema/version/route-alias/family coverage through `tools/validate_capability_matrix.py`.

The full release check still runs packaging checks after preflight unless `--preflight-only` is supplied.

## Current Evidence

| Check | Command | Current result | Release blocker if failing |
|---|---|---|---|
| Capability matrix validator tests | `PYTHONPATH=omnilss/src pytest -q omnilss/tests/test_generate_capability_matrix_tool.py` | Pass | Yes |
| Release preflight tests | `PYTHONPATH=omnilss/src pytest -q omnilss/tests/test_release_check.py` | Pass | Yes |
| Offline release preflight | `PYTHONPATH=omnilss/src python omnilss/tools/release_check.py --preflight-only` | Pass | Yes |

## Week 4 Remaining Gate Work

- Add fit → predict → serialize → load → predict smoke evidence to the preflight or a dedicated gate command.
- Add schema-safe prediction error smoke checks for missing variables and unseen factor levels.
- Run broader package/build checks where `build` and `twine` are available.
- Decide whether Month 1 can close with known JAX float64 environment warnings documented as non-blocking, or whether the warning policy needs to change before release.
