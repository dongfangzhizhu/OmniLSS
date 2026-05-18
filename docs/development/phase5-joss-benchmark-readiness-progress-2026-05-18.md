# Phase 5 Progress: JOSS benchmark and publication readiness — 2026-05-18

中文版本: [phase5-joss-benchmark-readiness-progress-2026-05-18_cn.md](phase5-joss-benchmark-readiness-progress-2026-05-18_cn.md)

## Completed work

- Split the JAX RS benchmark entry point into explicit `no-r`, `optional-r`, and `optional-gpu` suites so CI, release validation, and accelerator exploration no longer imply the same dependency set.
- Added a no-R/no-GPU benchmark smoke job to CI. This job verifies that the benchmark harness executes but does not make R-consistency or GPU-performance claims.
- Extended JAX RS benchmark artifacts with repeated warm timing samples and 95% confidence intervals while keeping cold compilation timing separate.
- Removed stale benchmark wording that described a NumPy RS warm-start in the JAX RS path; the publication-facing text now states that JAX RS uses data-aware cold-start initialization.
- Expanded the JOSS paper draft with the statement of need, algorithm description, validation and benchmark methodology, conservative limitations, and complete cross-linked Chinese documentation.

## Reporting rules confirmed

- Cold and warm runtime numbers must be reported separately.
- Python-only benchmark smoke runs are health checks, not equivalence evidence.
- Optional-R artifacts are required before publishing R comparison claims.
- Optional-GPU artifacts must state whether a JAX GPU device was available.
