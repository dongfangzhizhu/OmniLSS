# R Consistency Verification & Golden Artifacts

[中文版本](tutorial-r-consistency_cn.md)

OmniLSS backs numerical claims with **per-family, per-symbol, artifact-checked
evidence** rather than blanket assertions. Scope and parity counts are tracked
automatically in `docs/development/r-symbol-coverage.json` and ratcheted by
`docs/development/symbol-coverage-baseline.json`. `gamlss2` is explicitly out
of scope (`tools/coverage_scope.py`) and is excluded from 1.0 delivery targets.

## Evidence Layers

1. **Per-symbol R surface & parity matrix (`tools/generate_symbol_coverage.py`)**:
   1,036 / 1,036 (100%) in-scope R symbols across 11 packages matched, with 0
   unverified in-scope symbols (`evidence_gap = 0`, `placeholder = 0`).
2. **Fixed R golden fixtures (`omnilss/tests/fixtures/r_golden/regenerated/`)**:
   22 admitted families with frozen R deviance, coefficients, `newdata`
   predictions, and convergence metadata, allowing CI regression checks without
   installing R.
3. **Milestone & admission gates (`tools/m1_assessment.py`, `m2_assessment.py`,
   `m3_assessment.py`, `m3_admission_evidence.py`, `layered_parity.py`,
   `p2_defect_classification.py`, `m4_scope_inventory.py`)**:
   Every capability claim (`production_safe`, `rs_fit`, `cg_fit`, `prediction`,
   `sampling`, `smooth_terms`) is fail-closed audited against committed
   artifacts in CI.

## Running Verification Locally

```bash
# Run R-consistency tests (requires local R + gamlss + jsonlite)
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=omnilss/src \
  .venv/Scripts/python.exe -m pytest -p no:cacheprovider omnilss/tests/test_r_consistency*.py

# Run R-free golden regression tests
JAX_PLATFORMS=cpu JAX_ENABLE_X64=true PYTHONPATH=omnilss/src \
  .venv/Scripts/python.exe -m pytest -p no:cacheprovider omnilss/tests/test_r_golden*.py

# Verify all checked-in governance artifacts match the source tree
python tools/generate_symbol_coverage.py --check
python tools/m1_assessment.py --check
python tools/m2_assessment.py --check
python tools/m3_assessment.py --check
python tools/m3_admission_evidence.py --check
python tools/m4_scope_inventory.py --check
```

## Regenerating Golden Fixtures

```bash
Rscript omnilss/tests/fixtures/r_golden/generate_golden.R
```

Review numerical diffs carefully before updating frozen fixtures or baseline
ratchets.

## Next Steps

- [gRPC & HTTP Serving](tutorial-grpc-serving.md)
