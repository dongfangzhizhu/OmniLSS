# Technical Note 3 — Designing a Differentiable Distribution Runtime

## Phase 0 protocol baseline
- Distribution protocol with stateless interfaces
- Optimizer protocol with initialize/step/converged contract
- Tensor shape protocol enforcing `(batch,)` and `(batch, features)`

## Why this matters
A stable protocol layer decouples families, optimizers, and serving boundaries.

## Compatibility target
- R-consistency for MVP families
- reproducible benchmark harness metrics
- CI gate for consistency + stability checks
