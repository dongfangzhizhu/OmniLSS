# Milestone M0 manifest (v0.4.0)

- generated: 2026-11-14T23:12:05Z
- source commit: 2b85b5929c52fae4f0df1ca481e36da0e8063a44 (branch main)
- environment pinned: True
- python 3.10.19 / jax 0.6.2 (x64=True) / numpy 2.2.6 / scipy 1.15.3
- R: R version 4.6.0 (2026-04-24 ucrt) / gamlss 5.5.0 / gamlss.dist 6.1.1
- coverage: missing=173 matched=907 (diff ok=True)
- seeds: be_y_x_seed_20260811=20260811, bi_y_x_seed_20260809=20260809, ga_y_x_seed_20260525=20260525, gu_y_x_seed_20260813=20260813, ig_y_x_seed_20260522=20260522, nbi_y_x_seed_20260810=20260810, no_y_x_seed_20260524=20260524, po_y_x_seed_20260523=20260523, zaga_y_x_seed_20260812=20260812
- tolerances: {'default': 1e-06, 'fitting': 1e-06, 'dpqr': 1e-10}

| gate | result | mode |
|---|---|---|
| p0_clear | PASS |  |
| coverage_regenerable | PASS | --check executed |
| tiered_collection | PASS |  |
| devcontainer | PASS | static + fresh-clone simulation (no docker in CI env) |
| golden_manifest_gate | PASS | --require-r |

M0 gate bundle per DEVELOPMENT_PLAN 22.4. Reproducibility is anchored by coverage provenance content hashes (HEAD-independent); source.commit is informational. Regenerate after any symbol/test change.
