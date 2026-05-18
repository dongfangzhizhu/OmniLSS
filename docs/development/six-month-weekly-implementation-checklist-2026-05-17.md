# OmniLSS Six-Month Weekly Implementation Checklist (2026-05-17)

[中文版本](six-month-weekly-implementation-checklist-2026-05-17_cn.md)

This checklist operationalizes the [six-month execution plan](six-month-execution-plan-2026-05-17.md) into Week 1 through Week 26 sequencing. Work should proceed in order: complete or explicitly defer the current week before starting later-week production work.

| Week | Dates | Primary workstream | Implementation gate | Status |
|---|---|---|---|---|
| 1 | 2026-05-17 to 2026-05-23 | D1/D2 trustworthy artifacts and parser safety | Schema v2 present, default training-data redaction, structured prediction errors, robust smooth/tensor argument parsing, artifact validator | Complete for core production path; monitor inline review fixes |
| 2 | 2026-05-24 to 2026-05-30 | D1/D2 prediction hardening | Legacy prediction entry points audited; public artifact-schema examples added | In progress; artifact-schema docs, validator CLI, legacy/indirect prediction entry-point schema audit, plot/report wrapper propagation tests, public error-envelope example, and gRPC prediction error envelope added; HTTP metadata error envelope, POST payload-limit gate, prototype structured event hook, and gRPC route-capability preflight added |
| 3 | 2026-05-31 to 2026-06-06 | D3 capability gates | Runtime capability matrix fully aligned with docs and strict routing | Started; matrix artifact, HTTP/gRPC responses, `gamlss()` strict routing, and HTTP and gRPC service route-admission reports now share the method-routing contract; importable `method_route_feature()` / `require_method_route()` helpers, generated JSON `method_routes` alias, schema version, and matrix validator are synchronized |
| 4 | 2026-06-07 to 2026-06-16 | Month 1 release gate | Core trust checkpoint with reproducible test evidence | Pending |
| 5 | 2026-06-17 to 2026-06-23 | D4 validation matrix | Prioritized family validation schema and first JSON outputs | Pending |
| 6 | 2026-06-24 to 2026-06-30 | D4 validation matrix | Density/CDF/quantile checks for first core family batch | Pending |
| 7 | 2026-07-01 to 2026-07-07 | D5 R/Python consistency | Optional R bridge report harness stabilized | Pending |
| 8 | 2026-07-08 to 2026-07-16 | D6 academic metrics | PIT/CRPS/coverage benchmark outputs available | Pending |
| 9 | 2026-07-17 to 2026-07-23 | D7 secure API | Authn/authz boundary and error envelope design | Pending |
| 10 | 2026-07-24 to 2026-07-30 | D7 job runtime | Async fit job lifecycle and status API | Pending |
| 11 | 2026-07-31 to 2026-08-06 | D8 model registry | Persistent model registry and artifact retention policy | Pending |
| 12 | 2026-08-07 to 2026-08-16 | D9 resource limits | Service MVP checkpoint with quotas and observability | Pending |
| 13 | 2026-08-17 to 2026-08-23 | Vertical workflow | Insurance workflow data contract and template | Pending |
| 14 | 2026-08-24 to 2026-08-30 | Vertical workflow | Frequency/severity/zero-inflation model templates | Pending |
| 15 | 2026-08-31 to 2026-09-06 | Calibration reports | Quantile/risk interval reporting template | Pending |
| 16 | 2026-09-07 to 2026-09-16 | Pilot readiness | Vertical demo and pilot-readiness checkpoint | Pending |
| 17 | 2026-09-17 to 2026-09-23 | Performance backend | Profiling baseline and optimization targets | Pending |
| 18 | 2026-09-24 to 2026-09-30 | Observability | Service metrics/logging/tracing hardening | Pending |
| 19 | 2026-10-01 to 2026-10-07 | Deployment automation | Repeatable local/container deployment path | Pending |
| 20 | 2026-10-08 to 2026-10-16 | Paid beta readiness | Performance report and beta readiness gate | Pending |
| 21 | 2026-10-17 to 2026-10-23 | Release candidate | RC blocker list and documentation freeze scope | Pending |
| 22 | 2026-10-24 to 2026-10-30 | Support playbook | Support boundaries, escalation, and SLA draft | Pending |
| 23 | 2026-10-31 to 2026-11-06 | Launch materials | Public tutorials, benchmark summary, webinar assets | Pending |
| 24 | 2026-11-07 to 2026-11-13 | Commercial close | Pilot/LOI package and procurement materials | Pending |
| 25 | 2026-11-14 to 2026-11-16 | RC final validation | Release candidate evidence package complete | Pending |
| 26 | 2026-11-17 | Six-month checkpoint | Go/no-go decision, launch notes, and next plan | Pending |

## Current Week 1 Evidence

- JSON artifacts now expose a versioned validator report with typed issue severity that checks archive structure, schema versions, parameter schema coverage, coefficient/schema consistency, smooth metadata availability, categorical levels, numeric-transform AST metadata, and training-data inclusion warnings, plus schema migration policy; they also expose a capability-snapshot compatibility report for comparing saved evidence tiers against the current runtime registry.
- Structured prediction errors expose stable machine-readable fields for client routing.
- Formula parser hardening covers nested bracket arguments and quoted comma strings.
- Week 2 has started only after the core Week 1 gate: public artifact-schema examples and a validator CLI are now available; legacy, indirect prediction, scoring, and validation-wrapper entry points now reuse or propagate schema-safe prediction errors, and gRPC prediction failures preserve the structured error envelope; see the [legacy prediction entry-point audit progress note](month1-legacy-prediction-entrypoint-audit-2026-05-18.md) and the [service prediction error envelope progress note](month1-service-prediction-error-envelope-2026-05-18.md).
