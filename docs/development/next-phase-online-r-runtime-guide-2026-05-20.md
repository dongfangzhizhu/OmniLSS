# Next-Phase Development Guide (Online + R Runtime Required)

> Date: 2026-05-20  
> Scope: Follow-up work for unfinished items in `four-week-execution-plan-2026-05-20.md` that cannot be fully closed in the current offline / R-less environment.

## 1) Current Checklist Status Snapshot

Based on the active execution checklist:

- **Week 2 CG full-loop implementation and validation**: **Blocked** (R-alignment execution pending in an R-enabled environment).
- **Week 3 warm-start decoupling + benchmark repair**: completed in current repository state.
- **Week 4 integration/release preparation**: open and depends on completion evidence from Week 2 gate and full environment validation.

This guide defines the concrete next actions needed to close the remaining open items.

---

## 2) Environment Requirements (Must-Have)

## 2.1 System/runtime

- Linux/macOS (preferred for CI parity).
- Python runtime matching project support matrix.
- R runtime with `Rscript` on `PATH`.
- Network access enabled for package installation and optional benchmark dataset pulls.

## 2.2 R dependencies

Install and verify:

- `gamlss`
- `gamlss.dist`
- Any companion packages required by the local `tests/rbus/r_bridge.py` flow.

Recommended verification commands:

```bash
Rscript --version
Rscript -e "library(gamlss); library(gamlss.dist); cat('OK\n')"
```

## 2.3 Python dependencies

Ensure full test and benchmark extras are installed, including JAX backend dependencies for the target hardware class.

---

## 3) Week 2 Closure Plan (Priority 1)

Objective: close **"Complete Week 2 CG full-loop implementation and validation"** with auditable R-alignment evidence.

## 3.1 Preflight checks

1. Confirm R bridge discoverability from Python.
2. Confirm `Rscript` executable is found and runnable.
3. Confirm skip reason no longer reports R unavailability.

Suggested command:

```bash
pytest -q omnilss/tests/test_cg_algorithm_full_r_alignment.py -rs
```

Success criterion:
- No blanket skip due to R bridge availability.

## 3.2 Execute Week 2 validation suite

Run:

```bash
pytest -q \
  omnilss/tests/test_cg_algorithm_full.py \
  omnilss/tests/test_cg_algorithm_full_validation.py \
  omnilss/tests/test_cg_algorithm_full_r_alignment.py
```

Success criteria:

- Core and validation tests pass.
- R-alignment tests execute (not skipped for missing runtime).
- Deviance tolerance expectations hold for NO/GA/WEI/NBI.

## 3.3 Document closure artifacts

Update both:

- `docs/reports/CG_FULL_VERIFICATION_2026_05_19.md`
- `docs/development/four-week-execution-plan-2026-05-20.md`

Required evidence to add:

- Exact command lines.
- Date/time (UTC) and environment summary.
- Pass/fail counts (including explicit skip counts if any remain for non-runtime reasons).
- If failures occur: family-specific diagnostics and reproducer snippets.

## 3.4 Checklist state update

Only after successful R-alignment execution:

- Mark Week 2 checklist item as completed.

---

## 4) Week 4 Integration/Release Preparation Plan (Priority 2)

Week 4 should proceed only after Week 2 is closed or formally exception-approved.

## 4.1 Integration validation matrix

Execute integrated regression matrix covering:

- CG algorithm full-loop tests.
- RS/RS_JAX routing sanity tests.
- Serialization / capability snapshot consistency tests.
- Benchmark helper contract tests.

Record exact command bundle and outcomes.

## 4.2 Release-readiness checks

- Update progress docs with final gate statuses.
- Verify no contradictory checklist states remain.
- Ensure English-default documentation is consistent for newly added notes.

## 4.3 Optional CI hardening for online stage

If CI supports R jobs:

- Add/enable CI lane that runs `test_cg_algorithm_full_r_alignment.py` with R dependencies preinstalled.
- Preserve explicit skip-reason behavior for non-R lanes, but require pass on R-enabled lane.

---

## 5) Operational Runbook (Recommended Order)

1. Provision online + R environment.
2. Run R preflight commands.
3. Run Week 2 full validation command bundle.
4. Update Week 2 docs/report/checklist.
5. Run Week 4 integration matrix.
6. Update release-preparation records and checklist.

---

## 6) Definition of Done for Remaining Open Items

## Week 2 done when:

- R-alignment tests execute in R-enabled environment and pass acceptance tolerance.
- Week 2 checklist item is marked complete with evidence links.

## Week 4 done when:

- Integration/release preparation checklist item is marked complete with reproducible command evidence.
- Documentation reflects final, non-contradictory status across development and report artifacts.

