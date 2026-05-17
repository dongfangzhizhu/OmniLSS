# OmniLSS Six-Month Roadmap Development Plan (2026-05-18)

> Chinese version: [six-month-roadmap-development-plan-2026-05-18_cn.md](six-month-roadmap-development-plan-2026-05-18_cn.md)
>
> Source roadmap: six-month OmniLSS roadmap for version 0.3.0-dev, organized by the priority sequence **technical usability → completeness → accuracy → commercialization**.

## Development Language and Documentation Policy

- English is the default language for code, comments, commit messages, issue titles, and internal development coordination.
- User-facing planning documentation is maintained in both English and Chinese.
- Chinese documents use the same basename as the English document with `_cn` appended before `.md`.
- Each English/Chinese pair must cross-reference the other version near the top of the document.

## Priority Model

| Priority | Meaning | Release gate |
|---|---|---|
| P0 | Defects that can immediately cause user loss, including wrong results or crashes. | Must be fixed before any expansion work. |
| P1 | Features claimed to exist but not working reliably. | Must be fixed before public confidence-building work. |
| P2 | Working but inaccurate or incomplete functionality. | Must be addressed before broad academic validation. |
| P3 | Completeness extensions that increase academic or commercial value. | Planned after correctness foundations are stable. |
| P4 | Commercialization infrastructure. | Planned after core reliability and validation gates pass. |

## Workstream Breakdown

### Phase 1, Weeks 1-2: Emergency Stability Baseline

**Goal:** eliminate P0 defects and the highest-risk P1 credibility issues.

1. **Correct df_fit/AIC/SBC when smooth terms are present.**
   - Replace nominal smooth basis column counts with effective degrees of freedom (EDF) in standalone RS and CG backend model assembly.
   - Acceptance: smooth models report lower effective df than unpenalized basis counts where penalties apply, and AIC/SBC use the corrected df.
2. **Use distribution-aware residuals in the CG cross-derivative backend.**
   - Reuse the common randomized quantile residual and fallback residual helpers.
   - Acceptance: non-Normal families no longer return hard-coded raw `y - mu` residuals from CG.
3. **Make gRPC server dependencies and generated-stub workflow explicit.**
   - Align the `grpc` optional extra with protobuf v4+ and keep tooling in the extra.
   - Add a repeatable protobuf generation command.
   - Acceptance: contributors can regenerate stubs without guessing commands.
4. **Limit gRPC model registry growth.**
   - Add TTL and maximum-entry eviction for model artifacts.
   - Acceptance: saving more than the configured cap evicts old artifacts and expired artifacts are removed.
5. **Rename the historical L-BFGS backend in public planning.**
   - Keep backward compatibility, but document that this is not the standard Cole-Green algorithm.

### Phase 2, Weeks 3-4: Standard Cole-Green Implementation

**Goal:** provide an eta-scale Cole-Green implementation that is mathematically aligned with the literature and testable against R `gamlss(method="CG")`.

1. Introduce a dedicated `cole_green.py` backend based on eta-scale scores and Hessian cross-derivative corrections.
2. Route `gamlss(method="CG")` to the standard Cole-Green backend by default, while retaining explicit experimental backend names for comparison.
3. Add R consistency tests for NO, GA, NBI, BCCG, and TF where R is available.
4. Add a CG vs RS benchmark report for convergence and deviance comparison.

### Phase 3, Weeks 5-6: Numerical Validation and Test Infrastructure

**Goal:** make numerical credibility reproducible.

1. Add CI for R vs Python consistency tests.
2. Delay smoothing-parameter updates until coefficients have stabilized for several warm-up iterations.
3. Introduce a capability registry that records algorithm support, discreteness, zero-inflation, and derivative availability for each distribution.

### Phase 4, Weeks 7-10: Completeness and Robustness

**Goal:** remove incomplete main-path functionality and make routing more reliable.

1. Implement capability-aware Mixed algorithm routing.
2. Implement BCa bootstrap intervals.
3. Add TPS/B-spline REML smoothing selection integration.
4. Support higher-order B-spline derivatives.

### Phase 5, Weeks 11-16: Deep GAMLSS and Academic Readiness

**Goal:** strengthen OmniLSS's differentiating capabilities beyond R `gamlss`.

1. Add deployment-grade Deep GAMLSS prediction APIs, quantile prediction, CRPS support, and cross-validation.
2. Add semiparametric Deep GAMLSS support combining interpretable linear terms with neural nonlinear components.
3. Prepare and maintain the JOSS paper, examples, benchmarks, and consistency evidence.
4. Upgrade serialization toward safe JSON-based artifacts for production-facing paths.

### Phase 6, Weeks 17-26: Commercialization Infrastructure

**Goal:** prepare production-friendly APIs and release readiness.

1. Complete production-grade gRPC transport, including streamed progress and Arrow payloads.
2. Upgrade model storage to support bounded local persistence and optional Redis-backed persistence.
3. Implement Pro-layer AutoGAMLSS workflows and batch prediction APIs.
4. Calibrate GPU crossover thresholds on representative devices.
5. Complete PyPI release checks for v0.3.0.

## Immediate Execution Plan for This Change Set

This change set starts Phase 1 by delivering:

- bilingual development plan documentation;
- corrected EDF-based `df_fit` handling in standalone RS and CG cross-derivative backend assembly;
- distribution-aware CG residuals;
- bounded gRPC model registry cleanup;
- clearer gRPC optional dependencies and the existing stub regeneration workflow.

## Tracking and Reporting

- Each roadmap task should be tracked with the phase, task number, priority, owner, target date, and current status.
- Test results should record the exact command, environment limitation if any, and whether the result is a release blocker.
- Any deviation from the roadmap must explain whether it protects P0/P1 reliability, validation accuracy, or release timing.
