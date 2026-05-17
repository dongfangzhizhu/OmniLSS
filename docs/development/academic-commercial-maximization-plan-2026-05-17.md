# OmniLSS Academic and Commercial Value Maximization Plan (2026-05-17)

[中文版本](academic-commercial-maximization-plan-2026-05-17_cn.md)

> Source of truth: this plan is based on the 2026-05-17 implementation-level audit of OmniLSS. Runtime code, tests, configuration, and service boundaries are treated as facts; marketing claims and README-style descriptions are not.
>
> Objective: move OmniLSS from a research-oriented Python/JAX GAMLSS implementation into a distributional-regression platform that is simultaneously production-grade, research-credible, and enterprise-ready.

---

## 0. North Star Goals

### 0.1 Technical North Star

Within 90 days, deliver a stable `omnilss-core` that satisfies the following conditions:

- fitting, prediction, serialization, and deserialization are driven by the same design-matrix schema;
- production entry points reject silent fallbacks, unsafe formula execution, and insecure IPC defaults;
- RS, CG, and JAX routes have explicit capability matrices, fallback rules, and reproducible benchmarks;
- core families have golden tests for d/p/q/r functions, scores, Hessians, fitting, prediction, and artifact roundtrips.

### 0.2 Academic North Star

Within 180 days, produce an evidence package suitable for a JOSS / Journal of Statistical Software / applied-statistics software paper:

- define the consistency boundary against R `gamlss` precisely;
- publish reproducible R/Python numerical-consistency reports for core families;
- validate and document the JAX eta-scale Hessian / CG backend with derivations, finite-difference checks, and ablation studies;
- evaluate distributional prediction quality with PIT, CRPS, coverage, and quantile-loss metrics.

### 0.3 Commercial North Star

Within 180 days, deliver an enterprise distributional-modeling MVP:

- keep Core open-source while Pro/Enterprise owns model registry, batch fitting, monitoring, calibration reports, and multi-tenant APIs;
- build the first vertical demo for one of insurance, healthcare, or supply-chain risk;
- sell conditional distributions, quantiles, risk intervals, and auditable calibration reports rather than just point predictions.

---

## 1. Current Code Assets and Risk Profile

### 1.1 Highest-Value Assets

| Asset | Implementation fact | Academic value | Commercial value | Strengthening action |
|---|---|---:|---:|---|
| Unified `FamilyDefinition` contract | Families expose parameters, links, scores, Hessians, and d/p/q/r functions through one runtime contract. | High | High | Add a family capability registry and a golden validation matrix. |
| Broad family coverage | The package exports many continuous, discrete, zero-inflated, heavy-tail, and skewed families. | Medium-high | High | Label every family as fit-ready, predict-ready, sample-ready, validated, or experimental. |
| RS fitting route | `gamlss(method="RS")` routes into the RS implementation. | Medium | Medium-high | Remove hard-coded inner-loop controls and emit rigorous diagnostics. |
| JAX eta-scale Hessian / CG derivative backend | JAX `grad`/`hessian` kernels produce per-observation Hessian tensors. | High | Medium-high | Treat this as the main technical novelty for papers and high-performance enterprise editions. |
| HTTP/gRPC service boundary | HTTP and gRPC prototypes exist. | Low | Medium | Rebuild around auth, TLS, artifact storage, asynchronous jobs, and rate limits. |
| Pro client and AutoML prototype | Pro calls Core through gRPC and currently selects families mostly by deviance. | Low | Medium | Upgrade to model selection, audit reports, calibration, monitoring, and governance workflows. |

### 1.2 Blocking Risks

| Risk | Impact | Priority | Removal criterion |
|---|---|---:|---|
| Lost prediction/serialization metadata | Reloaded models can produce untrustworthy predictions. | P0 | Roundtrip prediction matches the original model; complex formulas and smooth terms do not degrade. |
| Formula `eval()` usage | Unsafe in multi-tenant services. | P0 | All formula expressions pass through an AST whitelist evaluator. |
| Insecure HTTP/gRPC defaults | Enterprise deployment blocker. | P0 | TLS/mTLS or token auth, payload limits, timeouts, and audit logs are available. |
| Silent fallbacks | Incorrect results can masquerade as valid predictions. | P0 | Schema mismatch, missing metadata, and unsupported family capabilities raise structured errors. |
| Weak R/AD validation coverage | Limits academic credibility. | P1 | Core families have reproducible consistency reports. |
| Numerical derivatives in complex families | Can destabilize convergence and inference. | P1 | Numerical-derivative families are labeled and progressively replaced by AD or analytic derivatives. |

---

## 2. Phased Roadmap

## Phase 0: Trustworthy Delivery Foundation (Weeks 1-2)

**Goal:** stop silent failures and make fit → predict → serialize → load → predict a trustworthy closed loop.

### P0-T01 Design-Matrix Schema and Model Artifact v2

**Owner profile:** core architecture / statistical runtime.

**Scope:** `formula_parser.py`, `fitting.py`, `prediction.py`, `serialization.py`, and related tests.

**Tasks:**

- [ ] Define `DesignMatrixSchema` with:
  - parameter name;
  - raw formula;
  - parsed term order;
  - column names;
  - intercept state;
  - factor levels;
  - numeric-transform AST;
  - smooth/tensor basis metadata;
  - training-column checksum.
- [ ] Store schema when `gamlss()` / `rs_fit()` returns a model.
- [ ] Persist schema, terms, smooth metadata, family capabilities, and artifact version in `save_model_json()`.
- [ ] Restore schema in `load_model_json()` and stop fabricating zero design matrices.
- [ ] Make `predict_params()` build design matrices only from schema.
- [ ] Raise `PredictionSchemaError` on column mismatches, missing variables, or missing smooth metadata.
- [ ] Add roundtrip tests for linear formulas, interactions, factors, pb/ps smooths, and multi-parameter families.

**Acceptance criteria:**

- [ ] `save → load → predict_params` matches the original model within `rtol <= 1e-10` for linear models and `rtol <= 1e-7` for smooth models.
- [ ] The previous “dimension mismatch → intercept-only fallback” behavior is removed.
- [ ] Artifact version is bumped, with either a clear migration path or a clear incompatibility message for older v0.3 artifacts.

### P0-T02 Safe Formula Parsing Instead of `eval()`

**Owner profile:** security / formula system.

**Scope:** `_fitting_utils.py`, `formula_parser.py`, and the prediction schema builder.

**Tasks:**

- [ ] Implement an AST whitelist evaluator that permits only:
  - variable names;
  - numeric constants;
  - `+ - * / **`;
  - parentheses;
  - allowlisted functions: `log`, `exp`, `sqrt`, `abs`, `sin`, `cos`, and `I`.
- [ ] Reject attribute access, subscript access, lambdas, comprehensions, imports, and calls to non-allowlisted objects.
- [ ] Return structured errors that include the offending term and the rejection reason.
- [ ] Add hostile-input tests for `__class__`, `np.__dict__`, `open`, `__import__`, and excessively deep ASTs.

**Acceptance criteria:**

- [ ] No production formula path contains `eval(`.
- [ ] Current formula tests still pass.
- [ ] Malicious expressions fail deterministically and without side effects.

### P0-T03 Minimum Production Security Baseline for Services

**Owner profile:** platform / backend security.

**Scope:** `omnilss-server`, `omnilss/src/omnilss/api/grpc/server.py`, `omnilss-pro/client.py`, and Docker configuration.

**Tasks:**

- [ ] Add request-size limits, timeouts, and family/method allowlists for HTTP and gRPC.
- [ ] Support TLS for gRPC; local demos may allow insecure mode, but production mode must reject insecure defaults.
- [ ] Add API-token middleware/interceptors.
- [ ] Convert `/fit` into an asynchronous job or at least add timeout and concurrency controls.
- [ ] Abstract the model registry behind local-filesystem, object-store, and in-memory testing backends.
- [ ] Stop treating `/tmp/{uuid}` paths as production-ready multi-tenant storage.

**Acceptance criteria:**

- [ ] Production mode fails closed when auth is not configured.
- [ ] Oversized payloads, unsupported families, and unsupported methods are rejected.
- [ ] Service errors return structured `code`, `message`, and `request_id` fields.

---

## Phase 1: Academic Credibility Upgrade (Weeks 3-6)

**Goal:** make core distributions and optimizers backed by reproducible, reviewer-grade evidence.

### P1-T01 Family Golden Validation Matrix

**Scope:** `distributions*.py`, `dpqr_functions.py`, `tests/consistency`, and `benchmarks`.

**First core-family batch:** `NO`, `GA`, `PO`, `BI`, `NBI`, `BE`, `WEI`, `TF`, `LOGNO`, `ZIP`.

**Validation dimensions:**

- [ ] d/p/q/r axioms:
  - finite densities;
  - monotone CDFs;
  - `q(p(x)) ≈ x`;
  - random-sample moment sanity checks.
- [ ] scores and Hessians:
  - analytic derivatives vs finite differences;
  - AD derivatives vs finite differences;
  - Hessian symmetry and negative-definiteness / negative-semidefiniteness where applicable.
- [ ] fitting:
  - intercept-only;
  - linear predictor;
  - multi-parameter formula;
  - weighted fitting;
  - extreme but valid inputs.
- [ ] R/GAMLSS consistency:
  - coefficients;
  - fitted values;
  - deviance;
  - AIC/BIC;
  - convergence flag.

**Acceptance criteria:**

- [ ] Each core family has one generated validation-report page.
- [ ] CI always runs Python-only golden tests.
- [ ] R consistency runs in a dedicated CI job and cannot silently skip while reporting success.

### P1-T02 RS Algorithm Rigor

**Scope:** `algorithms/rs_algorithm.py`, `controls.py`, and diagnostics.

**Tasks:**

- [ ] Connect the `rs_step()` inner loop to `GLIMControl`; remove hard-coded `max_iter=20` and `tol=1e-4`.
- [ ] Record structured diagnostics for non-monotone deviance, step-halving failure, and high WLS condition numbers.
- [ ] Distinguish convergence, numerical stabilization, forced clipping, and lambda-update failure.
- [ ] Record parameter-level gradient norms, condition numbers, and step sizes per outer iteration.
- [ ] Add tests and notes explaining behavior differences between RS and `gamlss_ml`.

**Acceptance criteria:**

- [ ] RS convergence reports are reproducible.
- [ ] Results that required aggressive numerical repair are not labeled as ordinary clean convergence.

### P1-T03 CG/JAX Academic Novelty Validation

**Scope:** `cg_derivatives.py`, `algorithms/cg_algorithm_v2.py`, `fitting_cg.py`, and benchmark scripts.

**Tasks:**

- [ ] Freeze the mathematical definition and code interface of the eta-scale Hessian.
- [ ] Validate Hessian cross terms with finite differences.
- [ ] Compare RS vs CG vs CG-v2 on:
  - iteration count;
  - final deviance;
  - runtime;
  - failure rate;
  - condition number.
- [ ] Select three families that best demonstrate CG value, such as `BCT`, `BCPE`, and `SHASH`/`JSU`.
- [ ] Write `docs/benchmarks/cg-validation-YYYY-MM-DD.json` and a Markdown report.

**Acceptance criteria:**

- [ ] CG has explicit success cases and known failure boundaries.
- [ ] A paper-ready RS/CG convergence-comparison figure can be generated from the benchmark artifacts.

---

## Phase 2: Commercial MVP (Weeks 7-10)

**Goal:** wrap the statistical engine as a credible enterprise workflow rather than a bare Python package.

### P2-T01 Enterprise Fit Job API

**Product definition:** enterprise users submit data and formulas, then receive auditable model artifacts and reports.

**Tasks:**

- [ ] `POST /jobs/fit`: submit asynchronous fitting jobs.
- [ ] `GET /jobs/{id}`: inspect status and logs.
- [ ] `GET /models/{id}`: retrieve model metadata.
- [ ] `POST /models/{id}/predict`: request parameter, quantile, and interval predictions.
- [ ] `GET /models/{id}/report`: retrieve calibration, convergence, family-capability, and data-schema reports.
- [ ] Add `request_id`, `tenant_id`, `model_id`, and `artifact_version` everywhere.

**Acceptance criteria:**

- [ ] A single-node demo can train 50 consecutive models without leaking artifacts.
- [ ] Errors can be attributed to data, formula, family, optimizer, or service resources.

### P2-T02 Calibration and Risk Report

**Product value:** the commercial asset is not the model object; it is the auditable distributional-risk decision package.

**Report contents:**

- [ ] PIT histogram;
- [ ] quantile coverage table;
- [ ] CRPS / negative log-likelihood;
- [ ] deviance/AIC/BIC/GAIC;
- [ ] residual diagnostics;
- [ ] tail-risk metrics: P90/P95/P99 and expected shortfall where applicable;
- [ ] model card: family, formula, training-data schema, limitations, and risks.

**Acceptance criteria:**

- [ ] One command generates HTML, Markdown, and JSON reports.
- [ ] Reports are suitable as sales demos and audit attachments.

### P2-T03 Upgrade Pro AutoML from Deviance Wrapper to Model Selection System

**Scope:** `omnilss-pro/automl.py` and Core API.

**Tasks:**

- [ ] Add a family capability filter that only considers families compatible with the response type and prediction goal.
- [ ] Support train/validation splits and cross-validation.
- [ ] Rank candidates by AIC, BIC, GAIC, NLL, CRPS, and coverage.
- [ ] Record structured failure reasons for failed candidate families.
- [ ] Emit a model-selection report.

**Acceptance criteria:**

- [ ] Pro no longer selects solely by training deviance.
- [ ] The report explains why the winning family was selected and why other families were excluded.

---

## Phase 3: Paper and Market Launch (Weeks 11-18)

**Goal:** use academic credibility to support commercial credibility, and use commercial demos to amplify the paper’s practical impact.

### P3-T01 Paper Evidence Package

- [ ] Draft a software paper under `paper/`.
- [ ] Generate benchmark artifacts: JSON, plots, and reproducibility scripts.
- [ ] Publish the validation matrix: family coverage, R consistency, and AD consistency.
- [ ] State the novelty clearly: JAX eta-Hessian CG backend, Python-native GAMLSS artifact schema, and enterprise distributional-modeling API.

### P3-T02 Vertical Demo

Choose one first:

1. **Insurance pricing / claim severity:** high commercial value because heavy tails, quantiles, and tail risk matter directly.
2. **Healthcare growth curves / reference intervals:** traditional GAMLSS strength with strong academic credibility.
3. **Supply-chain demand risk intervals:** enterprise ROI is easy to explain through inventory and service-level decisions.

Each demo must include:

- [ ] data cleaning;
- [ ] family selection;
- [ ] calibration report;
- [ ] quantile/risk decision;
- [ ] API deployment;
- [ ] ROI narrative: reduced risk, lower inventory, better pricing, or improved coverage.

### P3-T03 Open-Core Packaging

- [ ] Core: base families, RS fitting, base prediction, and base reports.
- [ ] Pro: AutoML, model registry, enterprise API, calibration/risk reports, batch jobs, and monitoring.
- [ ] Enterprise: multi-tenancy, SSO, audit logs, private deployment, GPU backend, and SLA.

---

## 3. Documentation Localization Policy

All new and materially changed documentation must follow the bilingual split below:

- the default file is English, for example `topic.md`;
- the Chinese counterpart uses the same stem with `_cn`, for example `topic_cn.md`;
- both versions must include a language-switch link near the top;
- MkDocs navigation should list the English version as the default entry and may list the Chinese counterpart directly below it when the page is a major roadmap, policy, or user-facing guide;
- future documentation PRs should update both versions in the same commit whenever the change affects user-facing content or project governance.

---

## 4. Execution Priority Board

### P0: Immediate

| ID | Task | Value | Estimated effort | Blocker |
|---|---|---:|---:|---|
| P0-T01 | Model artifact v2 + prediction schema | Commercial-delivery foundation | 5-7 days | None |
| P0-T02 | AST-based safe formula parsing | Security red line | 2-4 days | Coordinate with P0-T01 schema |
| P0-T03 | Service security baseline | Enterprise entry requirement | 4-6 days | Artifact-store abstraction |

### P1: Academic Credibility Core

| ID | Task | Value | Estimated effort | Blocker |
|---|---|---:|---:|---|
| P1-T01 | Family golden tests | Paper / trust | 2-4 weeks | Stable P0 schema |
| P1-T02 | RS diagnostics rigor | Statistical reliability | 1 week | None |
| P1-T03 | CG/JAX validation | Main novelty | 2-3 weeks | Partial derivative validation from P1-T01 |

### P2: Commercial MVP

| ID | Task | Value | Estimated effort | Blocker |
|---|---|---:|---:|---|
| P2-T01 | Fit Job API | SaaS foundation | 2 weeks | P0-T03 |
| P2-T02 | Calibration & Risk Report | Direct sales asset | 2 weeks | P1-T01 |
| P2-T03 | Pro AutoML upgrade | Product differentiation | 1-2 weeks | P2-T01 |

---

## 5. Engineering Standards and Definition of Done

### 5.1 Every Core Change Must Satisfy

- [ ] no silent fallback in the changed code path;
- [ ] new failures are represented by structured exceptions or structured API responses;
- [ ] every new capability has tests and an updated `docs/development` progress note;
- [ ] every benchmark fixes seeds, records the environment, and saves JSON artifacts;
- [ ] every statistically meaningful PR includes at least one numerical-consistency test;
- [ ] every commercial API PR includes a security-boundary test.

### 5.2 Prohibited Patterns

- [ ] accepting uncontrolled pickle artifacts on the server;
- [ ] adding production `eval()` paths;
- [ ] falling back to intercept-only prediction after schema mismatch;
- [ ] making an experimental backend the default production backend;
- [ ] reporting only the fastest benchmark run.

---

## 6. Metrics

### 6.1 Technical Metrics

| Metric | Target |
|---|---:|
| Linear-formula roundtrip prediction error | `rtol <= 1e-10` |
| Smooth-formula roundtrip prediction error | `rtol <= 1e-7` |
| Core-family golden-test coverage | 10 families / 6 weeks |
| Production API unauthenticated access | 0 |
| Silent fallback count | 0 |
| Fit-job structured error coverage | 100% known failure modes |

### 6.2 Academic Metrics

| Metric | Target |
|---|---:|
| R-consistency families | 10 core + 10 extended / 180 days |
| AD/finite-difference derivative reports | 10 core families / 90 days |
| Calibration metrics in benchmark report | PIT + CRPS + coverage + NLL |
| Reproducible benchmark scripts | 100% paper figures |

### 6.3 Commercial Metrics

| Metric | Target |
|---|---:|
| Vertical demo | 1 complete / 120 days |
| Enterprise API endpoints | fit job, predict, report, registry |
| Model report generation time | < 10s for n <= 100k metadata/report after fit |
| Sales artifacts | 1 technical whitepaper + 1 demo notebook + 1 API walkthrough |

---

## 7. Near-Term Sprint Plan

### Sprint 1 (2026-05-17 to 2026-05-24)

**Theme:** stop untrustworthy prediction and remove the immediate formula-security red line.

- [ ] P0-T01-1: define the `DesignMatrixSchema` data structure.
- [ ] P0-T01-2: make linear-formula fit/predict use schema.
- [ ] P0-T01-3: persist and restore schema in JSON artifacts.
- [ ] P0-T01-4: remove prediction dimension-mismatch fallback to intercept-only.
- [ ] P0-T02-1: implement the AST evaluator prototype.
- [ ] P0-T02-2: replace `eval()` in `_eval_linear_term()`.
- [ ] Tests: `test_prediction_schema.py`, `test_serialization_json.py`, and malicious-formula tests.

### Sprint 2 (2026-05-25 to 2026-05-31)

**Theme:** smooth-term roundtrip and service security baseline.

- [ ] P0-T01-5: persist and restore pb/ps smooth metadata for prediction.
- [ ] P0-T01-6: persist and restore factor/interactions schema.
- [ ] P0-T03-1: add gRPC/HTTP request limits and token auth.
- [ ] P0-T03-2: introduce a model-registry interface.
- [ ] P0-T03-3: add structured error responses.

### Sprint 3 (2026-06-01 to 2026-06-07)

**Theme:** first golden-test batch and RS diagnostics.

- [ ] P1-T01-1: add golden tests for `NO`, `GA`, `PO`, `BI`, and `NBI`.
- [ ] P1-T02-1: connect the RS inner loop to `GLIMControl`.
- [ ] P1-T02-2: record condition number, gradient norm, and step-halving diagnostics.
- [ ] P1-T01-2: generate the first validation report.

### Sprint 4 (2026-06-08 to 2026-06-14)

**Theme:** verifiable CG/JAX novelty.

- [ ] P1-T03-1: validate eta Hessian with finite differences.
- [ ] P1-T03-2: benchmark RS vs CG ablations.
- [ ] P1-T03-3: emit Markdown and JSON benchmark artifacts.

---

## 8. Decision Principles

1. **Trust beats feature count:** if a feature cannot be validated, roundtripped, or explained on failure, it does not enter the commercial path.
2. **Evidence beats marketing:** performance, accuracy, and convergence advantages must have reproducible benchmark artifacts.
3. **Reports beat raw APIs:** enterprises buy auditable decisions, not isolated prediction functions.
4. **Secure by default:** demos may be insecure, but production must require auth, TLS, and limits.
5. **Clear Open-Core boundary:** Core is the trusted statistical engine; Pro owns workflow, governance, monitoring, and reporting.

---

## 9. Next Development Entry Points

Use the following PR sequence to keep review scope small:

1. `schema-core`: add `DesignMatrixSchema` and linear-formula prediction schema;
2. `safe-formula-evaluator`: replace `eval()` with the AST evaluator;
3. `serialization-v2`: persist and restore schema in JSON artifacts;
4. `prediction-no-silent-fallback`: remove intercept-only fallback and add explicit error types;
5. `service-security-baseline`: add API tokens, payload limits, and structured errors;
6. `family-golden-tests-phase1`: add the first golden tests for `NO`, `GA`, `PO`, `BI`, and `NBI`;
7. `rs-diagnostics-phase1`: connect RS diagnostics and `GLIMControl`;
8. `cg-validation-phase1`: validate eta Hessian and publish CG benchmark reports.
