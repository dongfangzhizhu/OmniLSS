# 30-Day Refactor Status and Documentation Maintenance Tasks

Current status is scoped to the architecture-freeze work merged into `main` as of
May 15, 2026. The project remains in **Architecture > Features** mode.

## Task Status

| Priority | Task | Status | Notes |
| --- | --- | --- | --- |
| P0 | Task 1: Feature Freeze | Complete / ongoing | Freeze policy and forbidden feature areas are documented. Enforce for at least 30 days. |
| P0 | Task 2: Directory Structure | Partial | Transitional `core`, `api`, and `smooth` namespaces exist; large legacy modules still need incremental migration. |
| P0 | Task 3: Distribution Protocol | Partial | `DistributionProtocol` and legacy adapter exist; individual families still need migration and conformance checks. |
| P0 | Task 4: Optimizer Interface | Partial | `OptimizerProtocol` and Optax adapter exist; RS/CG/Mixed/L-BFGS still need protocol wrappers. |
| P0 | Task 5: Parameter System | Partial | Canonical `Parameter`, links, and constraints exist; distribution-specific parameter logic still needs removal. |
| P1 | Task 6: Formula System Freeze | Complete / ongoing | Formula freeze is documented; no new DSL syntax should be accepted without a new ADR. |
| P1 | Task 7: Test Matrix | Partial | Test matrix and benchmark gate docs exist; CI now skips R-only tests when R is unavailable. More snapshot/gradient coverage needed. |
| P1 | Task 8: Benchmark System | Partial | Validation gate enforces consistency-before-performance and reports cold/warm/memory. GPU and batch-scaling reporting remain future work. |
| P1 | Task 9: ADRs | Complete | ADR-001 through ADR-005 have been added. |
| P2 | Task 10: JAX-native computational graph | Not started | Requires a separate design phase after P0/P1 stabilization. |
| P2 | Task 11: Modular package split | Not started | Current work only creates boundaries; package split is future work. |
| P2 | Task 12: gRPC protocol boundary | Not started | Placeholder boundary exists; protobuf/API design is not implemented. |

## Documentation Maintenance Task List

- Keep root `README.md`, `omnilss/README.md`, `benchmarks/README.md`, and
  `omnilss/tests/README.md` aligned with the architecture-freeze status.
- Remove fixed speedup claims unless they cite a generated benchmark report with
  hardware, backend, data sizes, formulas, repetitions, and R availability.
- Update this status table when a task moves from Partial to Complete.
- Link every new protocol or boundary decision to an ADR.
- Document every benchmark run by command, timestamp, generated artifact path,
  and whether native R `gamlss` was actually available.
- Do not document new feature APIs during the freeze unless an ADR explicitly
  lifts the freeze for that area.

## CI Maintenance Notes

- General CI must not require R. Tests that need native R or `gamlss.dist` should
  skip at collection or test setup when R is unavailable.
- R-backed consistency should run in a dedicated environment with R installed,
  not as an unconditional requirement for the Python matrix.
- Keep Linux, Windows, and macOS Python 3.10-3.12 smoke coverage focused on
  package importability, non-R tests, and numerical stability checks.
