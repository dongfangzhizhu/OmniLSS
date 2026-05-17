# OmniLSS Six-Month Execution Plan (2026-05-17 to 2026-11-17)

[中文版本](six-month-execution-plan-2026-05-17_cn.md)

> Planning basis: this plan is derived from the implementation-level audit of OmniLSS. It treats runtime code, tests, packaging metadata, and service boundaries as the source of truth. README claims, docstrings, and promotional language are not used as delivery evidence.
>
> Default operating language: English. A synchronized Chinese version is provided for bilingual execution and stakeholder communication.

---

## 1. Executive Objective

Over the next six months, OmniLSS should move from a broad research-oriented Python/JAX GAMLSS implementation into a credible open-core distributional modeling platform.

The plan has three synchronized tracks:

1. **Development:** harden the statistical core, prediction artifacts, validation matrix, and service runtime.
2. **Marketing and promotion:** build trust through transparent benchmarks, technical content, bilingual education, and vertical demos.
3. **Business:** validate a commercial wedge, define Pro/Enterprise boundaries, and prepare first paid pilots.

### Six-Month Target State

By **2026-11-17**, OmniLSS should have:

- a stable Core release candidate with schema-safe fit/predict/serialize/load roundtrips;
- a published validation evidence package for core distributions and fitting routes;
- a secure service MVP with asynchronous jobs, model registry, authentication, and usage logging;
- one vertical demo with measurable business value, preferably insurance risk, healthcare growth/risk curves, or financial tail-risk modeling;
- a Pro/Enterprise product definition with pilot pricing, support boundaries, and a deployment playbook.

---

## 2. Operating Principles

1. **Implementation-first truth.** Ship only what is validated by code, tests, and reproducible artifacts.
2. **No silent statistical degradation.** Unsupported formulas, smooth terms, distribution capabilities, or artifact schemas must raise structured errors.
3. **Core remains credible; Pro becomes useful.** Open-source Core should maximize trust; Pro/Enterprise should sell governance, automation, operations, and vertical workflows.
4. **Academic credibility supports sales.** Benchmarks, R consistency reports, and calibration metrics are not side quests; they are marketing and enterprise risk-reduction assets.
5. **Bilingual-by-default documentation.** Any strategic public-facing plan, release note, tutorial, or benchmark summary should have English and Chinese versions with cross-links.

---

## 3. Six-Month Roadmap Overview

| Month | Theme | Development milestone | Marketing milestone | Business milestone |
|---|---|---|---|---|
| Month 1 | Trustworthy Core | Artifact schema v2, safe prediction, formula safety, test gates | Publish “Trustworthy Distributional Modeling” audit summary | Select primary vertical and pilot persona |
| Month 2 | Statistical Validation | Family capability registry, R/Python consistency reports, AD/Hessian validation | Release benchmark and validation dashboards | Define Pro/Enterprise package boundaries |
| Month 3 | Production Service MVP | Secure API, async fit jobs, persistent model registry, resource limits | Launch bilingual tutorials and demo notebooks | Start design-partner outreach |
| Month 4 | Vertical Productization | Vertical workflow templates and calibration reports | Publish industry demo and case-study content | Run 2-3 design-partner pilots |
| Month 5 | Scale and Reliability | Performance backend improvements, observability, deployment automation | Release performance report and comparison series | Convert pilots into paid beta offers |
| Month 6 | Commercial Readiness | Release candidate, documentation freeze, support playbook | Public launch campaign and webinar | Close first paid pilots or enterprise LOIs |

### Calendar Windows

| Window | Dates | Primary checkpoint |
|---|---|---|
| Month 1 | 2026-05-17 to 2026-06-16 | Core trust foundation checkpoint |
| Month 2 | 2026-06-17 to 2026-07-16 | Validation credibility checkpoint |
| Month 3 | 2026-07-17 to 2026-08-16 | Internal service MVP checkpoint |
| Month 4 | 2026-08-17 to 2026-09-16 | Vertical pilot-readiness checkpoint |
| Month 5 | 2026-09-17 to 2026-10-16 | Paid beta readiness checkpoint |
| Month 6 | 2026-10-17 to 2026-11-17 | Release candidate and commercial readiness checkpoint |

---

## 4. Development Plan

### 4.1 Month 1: Trustworthy Core Foundation

**Goal:** make fit → predict → serialize → load → predict safe and reproducible.

#### Workstream D1: Model Artifact and Design-Matrix Schema v2

**Tasks**

- Define a versioned `DesignMatrixSchema` containing:
  - parameter name;
  - raw formula;
  - parsed term order;
  - intercept state;
  - column names;
  - coefficient count;
  - factor levels;
  - numeric transform AST metadata;
  - smooth basis metadata;
  - family capability metadata.
- Persist schema in model artifacts.
- Restore schema on load without fabricating incomplete design matrices.
- Make prediction depend on saved schema rather than reinterpreting loose formulas.
- Add hard errors for missing variables, unseen factor levels, missing smooth metadata, and column-count mismatches.

**Acceptance criteria**

- Linear roundtrip prediction matches original model within `rtol <= 1e-10`.
- Smooth roundtrip prediction matches within `rtol <= 1e-7` or raises a documented unsupported-feature error.
- Serialized artifacts no longer persist full training data by default.
- All prediction schema failures use structured exception types.

#### Workstream D2: Formula Safety and Parser Hardening

**Tasks**

- Ensure all numeric formula expressions pass through an AST whitelist evaluator.
- Reject attribute access, imports, comprehensions, lambdas, indexing, unsafe calls, and overly deep expressions.
- Replace fragile string splitting for smooth/tensor arguments with a robust parser or constrained grammar.
- Add malicious formula tests.

**Acceptance criteria**

- Formula injection tests pass.
- Formula parsing errors include term name and rejection reason.
- No production prediction path executes arbitrary Python code.

#### Workstream D3: RS/CG/JAX Capability Matrix

**Tasks**

- Create a capability table for every family and method route:
  - fit-ready;
  - predict-ready;
  - sample-ready;
  - smooth-ready;
  - R-consistency-validated;
  - AD/Hessian-validated;
  - production-safe;
  - experimental.
- Use the matrix at runtime to prevent unsupported method/family combinations.
- Document fallback behavior explicitly.

**Acceptance criteria**

- Unsupported routes raise structured errors instead of silently falling back.
- Public docs and runtime capability data match.

### 4.2 Month 2: Statistical Validation and Research Evidence

**Goal:** turn correctness into a reusable scientific evidence package.

#### Workstream D4: Family Validation Matrix

**Tasks**

- Prioritize 10 core families: `NO`, `GA`, `PO`, `BI`, `NBI`, `BE`, `WEI`, `TF`, `LOGNO`, `ZAGA`.
- For each prioritized family, validate:
  - density/PMF values;
  - CDF monotonicity;
  - quantile/CDF inverse consistency;
  - random sample moments where applicable;
  - score vs finite difference;
  - Hessian vs finite difference or AD;
  - edge cases and invalid parameter handling.

**Acceptance criteria**

- Each core family has a machine-readable validation JSON and a human-readable report.
- Failures are classified as implementation bug, numerical tolerance issue, unsupported domain, or reference mismatch.

#### Workstream D5: R/Python Consistency Harness

**Tasks**

- Stabilize the R bridge as an optional but reproducible validation environment.
- Add snapshot datasets for reproducible comparison.
- Report coefficients, fitted values, deviance, AIC/BIC, residual summaries, and convergence flags.
- Separate tests that require R from default CI while keeping scheduled validation jobs.

**Acceptance criteria**

- A reproducible consistency report exists for each prioritized family.
- CI clearly distinguishes “unit pass”, “R unavailable skip”, and “R consistency failure”.

#### Workstream D6: Academic Metrics

**Tasks**

- Add PIT, CRPS, quantile loss, interval coverage, calibration slope, and residual diagnostics.
- Build a reproducible benchmark runner for synthetic and real-like datasets.
- Draft a paper outline centered on Python/JAX GAMLSS validation, capability boundaries, and differentiable fitting routes.

**Acceptance criteria**

- At least three public benchmark scenarios are reproducible from one command.
- Validation outputs can be included in a technical paper or whitepaper.

### 4.3 Month 3: Production Service MVP

**Goal:** convert service prototypes into a safe internal platform MVP.

#### Workstream D7: Secure API and Job Runtime

**Tasks**

- Replace synchronous long-running fit calls with asynchronous jobs.
- Add model registry with persistent storage.
- Add authentication, request IDs, audit logs, timeouts, payload limits, and rate limits.
- Add structured error codes for fit, predict, sample, and artifact operations.
- Add TLS/mTLS deployment option for gRPC and HTTP.

**Acceptance criteria**

- A model survives service restart and can be predicted from a new process.
- Large or invalid payloads fail safely.
- Every request has traceable job/model/user metadata.

#### Workstream D8: Deployment and Observability

**Tasks**

- Provide Docker Compose deployment for Core API, registry, object storage, and metrics.
- Add Prometheus-compatible metrics:
  - fit duration;
  - predict latency;
  - failed jobs;
  - model count;
  - artifact size;
  - memory usage;
  - numerical warning counts.
- Add structured logs with redaction.

**Acceptance criteria**

- Internal demo deployment can be started with one documented command.
- Operational dashboard shows job health and model health.

### 4.4 Month 4: Vertical Productization

**Goal:** package the platform around a domain use case rather than generic modeling.

#### Recommended Primary Vertical: Insurance Risk Modeling

Insurance is the strongest first wedge because GAMLSS naturally supports claim severity, claim frequency, zero inflation, tail risk, quantiles, and regulatory explainability.

**Tasks**

- Build an insurance workflow template:
  - claim frequency model;
  - claim severity model;
  - zero-inflated or heavy-tail alternative;
  - quantile/risk interval report;
  - model comparison report;
  - calibration report.
- Add synthetic insurance dataset generator with known ground truth.
- Add notebook and API demo.
- Add exportable PDF/HTML report.

**Acceptance criteria**

- Demo can answer: “Which customers have high expected claims and high tail risk?”
- Demo includes distributional metrics, not just point prediction metrics.
- Demo can be run locally and through the service API.

### 4.5 Month 5: Scale, Reliability, and Enterprise Fit

**Goal:** make the platform credible for beta customers.

#### Workstream D9: Performance and Backend Improvements

**Tasks**

- Optimize WLS backend choices: dense NumPy, sparse SciPy, JAX dense, and batch prediction.
- Vectorize slow discrete residual/CDF paths.
- Add hardware-aware benchmark reports.
- Define GPU acceleration only where evidence supports it.

**Acceptance criteria**

- Performance report includes hardware, backend, dataset shape, family, and confidence intervals.
- At least one bottleneck is improved by 2x on a benchmarked workload.

#### Workstream D10: Enterprise Reliability

**Tasks**

- Add backup/restore for model registry.
- Add model versioning and rollback.
- Add compatibility tests for artifact versions.
- Add role-based access boundaries for Pro/Enterprise API.
- Add support runbook.

**Acceptance criteria**

- A failed deployment can be rolled back without losing registered models.
- Support can diagnose a failed fit from job metadata and logs.

### 4.6 Month 6: Release Candidate and Commercial Readiness

**Goal:** freeze scope, package the offer, and convert pilots.

**Tasks**

- Cut a Core release candidate.
- Freeze public API changes for the release window.
- Publish capability matrix, validation reports, benchmark reports, and deployment guide.
- Finalize Pro/Enterprise packaging and pilot contracts.
- Run a launch webinar and technical deep-dive.

**Acceptance criteria**

- Release candidate passes all required unit, integration, artifact, service, and validation gates.
- At least two design partners have completed technical evaluation.
- At least one paid pilot, LOI, or procurement process is active.

---

## 5. Marketing and Promotion Plan

### 5.1 Positioning

**Primary message:** OmniLSS provides distributional modeling for teams that need calibrated uncertainty, quantiles, tail risk, and interpretable statistical structure—not just mean predictions.

**Differentiators**

- Python/JAX implementation of GAMLSS-style distributional regression.
- Broad distribution family coverage.
- R consistency and mathematical validation as trust assets.
- Open-source Core plus enterprise workflow layer.
- Bilingual technical education for English and Chinese-speaking users.

### 5.2 Audience Segments

| Segment | Pain | Message | Conversion goal |
|---|---|---|---|
| Statisticians and researchers | Need Python-native distributional regression | Reproducible GAMLSS-style modeling with validation reports | GitHub stars, citations, feedback |
| ML engineers | Need uncertainty beyond point prediction | Quantiles, intervals, and calibration in deployable APIs | Trial service deployment |
| Insurance/finance analysts | Need tail-risk and claims modeling | Model full conditional distributions and risk intervals | Design-partner pilot |
| Healthcare analytics teams | Need growth/risk curves and calibrated ranges | Distributional curves with explainable parameters | Applied demo meeting |
| Enterprise platform teams | Need governance and model operations | Registry, audit logs, monitoring, and support | Paid beta |

### 5.3 Content Calendar

| Month | Content | Channel | Success metric |
|---|---|---|---|
| 1 | Implementation audit summary and roadmap | GitHub docs, blog, LinkedIn/X | 500+ page views, 20+ GitHub interactions |
| 1 | “Why distributional regression matters” explainer | Blog + Chinese translation | 10 qualified inbound conversations |
| 2 | R/Python consistency report | GitHub release notes, technical blog | 5 external technical reviews |
| 2 | Family capability matrix | Docs + issue templates | Reduced support confusion |
| 3 | Secure service MVP demo | Video + demo repo | 5 trial deployments |
| 3 | Tutorial series: fit, predict, quantiles, calibration | Docs + notebooks | 100 notebook launches/downloads |
| 4 | Insurance risk modeling vertical demo | Webinar + case study | 3 design-partner calls |
| 5 | Performance and benchmark report | Blog + benchmarks docs | 2 external benchmark replications |
| 6 | Launch webinar and release candidate announcement | Webinar, GitHub, newsletters | 1 paid pilot or LOI, 10 serious leads |

### 5.4 Community and Credibility Motions

- Publish validation failures honestly with status labels.
- Add “good first validation issue” tasks for family testing.
- Invite R `gamlss` users to compare outputs and report mismatch cases.
- Create bilingual examples for insurance, healthcare, and finance.
- Submit a JOSS pre-submission checklist by Month 6.

---

## 6. Business Plan

### 6.1 Commercial Thesis

Core statistical modeling should remain open to maximize trust and adoption. Revenue should come from enterprise-grade operations, automation, governance, vertical workflows, and support.

### 6.2 Product Packaging

| Package | Target user | Contents | Pricing hypothesis |
|---|---|---|---|
| Core OSS | Researchers, developers | Python package, local fitting, validation docs | Free |
| Pro Developer | Small teams | Hosted or self-hosted API, model registry, reports, batch jobs | $299-$999/month |
| Enterprise | Regulated teams | SSO, RBAC, audit logs, private deployment, support SLA, custom validation | $25k-$150k/year |
| Vertical Solution | Insurance/healthcare/finance teams | Templates, reports, dashboards, consulting onboarding | Pilot fee + annual license |

### 6.3 First Wedge: Insurance

**Why insurance first**

- Clear need for frequency/severity modeling.
- Natural demand for quantiles, intervals, and tail-risk estimates.
- Existing statistical culture makes GAMLSS easier to explain.
- Reports and calibration can justify premium enterprise pricing.

**Pilot offer**

- 4-6 week technical pilot.
- Scope: one dataset, one workflow, benchmark against current GLM/GAM/baseline model.
- Deliverables:
  - fitted distributional models;
  - quantile and tail-risk report;
  - calibration report;
  - deployment/API demo;
  - recommendation memo.
- Pilot fee target: $5k-$15k, credited toward annual contract.

### 6.4 Sales Motion

1. **Design partner identification**
   - Actuarial analytics teams.
   - Risk modeling teams.
   - Health analytics teams with growth/risk curve needs.
2. **Technical discovery**
   - Current models and pain points.
   - Required distributions and metrics.
   - Deployment constraints.
   - Compliance and audit needs.
3. **Pilot proposal**
   - Fixed scope.
   - Clear success metric.
   - Data handling terms.
   - Paid pilot or LOI.
4. **Conversion**
   - Convert pilot report into production roadmap.
   - Offer annual Pro/Enterprise license with onboarding.

### 6.5 Business KPIs

| Time | KPI |
|---|---|
| End of Month 1 | 1 selected vertical, 20 target accounts, 5 discovery calls scheduled |
| End of Month 2 | Product one-pager, pricing hypothesis, 2 warm design-partner conversations |
| End of Month 3 | 5 trial users or deployments, 3 serious pilot leads |
| End of Month 4 | 2-3 active design-partner pilots |
| End of Month 5 | 1 paid beta proposal or LOI in negotiation |
| End of Month 6 | 1 paid pilot/LOI/procurement process active; repeatable pilot playbook |

---

## 7. Team and Governance Plan

### 7.1 Minimum Team Shape

| Role | Responsibility | Month 1-2 | Month 3-4 | Month 5-6 |
|---|---|---:|---:|---:|
| Core statistical engineer | Families, RS/CG, validation | Full-time | Full-time | Full-time |
| Platform/backend engineer | API, jobs, registry, deployment | Part-time | Full-time | Full-time |
| Research lead | R consistency, metrics, paper package | Full-time | Part-time | Part-time |
| Developer advocate | Tutorials, demos, bilingual docs | Part-time | Full-time | Full-time |
| Commercial lead | Pilots, pricing, partnerships | Part-time | Full-time | Full-time |

### 7.2 Weekly Cadence

- Monday: planning and blocker triage.
- Wednesday: validation/build review.
- Friday: demo, metrics review, and documentation update.
- Every month: release checkpoint with go/no-go criteria.

### 7.3 Required Dashboards

- Engineering burndown and release gates.
- Family capability and validation matrix.
- Benchmark history.
- Service health and job metrics.
- Marketing funnel.
- Pilot pipeline.

---

## 8. Risk Register

| Risk | Probability | Impact | Mitigation |
|---|---:|---:|---|
| Validation reveals major family bugs | High | High | Prioritize top 10 families; label others experimental; publish known limitations |
| Service MVP consumes too much time | Medium | High | Keep Month 3 scope internal-MVP only; avoid enterprise feature creep |
| GPL limits embedded commercial adoption | Medium | High | Sell hosted/self-hosted service and enterprise operations, not closed-source embedding |
| JAX acceleration underperforms | Medium | Medium | Market validation and distributional metrics first; enable GPU only where benchmarked |
| No design partners convert | Medium | High | Start outreach in Month 1; focus on insurance wedge; use paid diagnostic pilot |
| Bilingual docs drift | Medium | Medium | Require cross-links and update checklist for every strategic doc |

---

## 9. Go/No-Go Gates

### End of Month 2: Core Credibility Gate

Proceed only if:

- artifact roundtrip is reliable for core formulas;
- top 10 family validation matrix is underway;
- unsupported capabilities are labeled;
- at least one validation report is publishable.

### End of Month 4: Product Wedge Gate

Proceed only if:

- service MVP can run the vertical demo;
- at least two design partners have engaged deeply;
- one vertical workflow produces a clear business report;
- Pro/Enterprise scope is understood.

### End of Month 6: Commercial Readiness Gate

Proceed only if:

- release candidate passes required gates;
- validation and benchmark evidence is public or customer-shareable;
- at least one paid pilot, LOI, or procurement process exists;
- support and deployment playbooks are usable by someone other than the core author.

---

## 10. Immediate Next 10 Actions

1. Create the family capability registry and mark all families as validated, experimental, or unsupported per feature.
2. Implement artifact schema v2 and roundtrip tests.
3. Remove full training data from default JSON model artifacts.
4. Add malicious formula parser tests.
5. Build validation reports for `NO`, `GA`, and `PO` first.
6. Draft the insurance risk demo data generator and notebook.
7. Define the Pro/Enterprise one-page product offer.
8. Build a target account list of 20 insurance or risk analytics teams.
9. Publish the bilingual roadmap announcement.
10. Schedule the first monthly release checkpoint with explicit go/no-go criteria.
