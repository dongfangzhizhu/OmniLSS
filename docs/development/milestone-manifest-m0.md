# Milestone M0 manifest (v0.4.0)

- generated: 2026-11-21T03:02:44Z
- source commit: 4e458a31bdb5103964fea71bd2b1170784ed252d (branch main)
- environment pinned: True
- python 3.10.16 / jax 0.6.2 (x64=True) / numpy 2.2.6 / scipy 1.15.3
- R: R version 4.6.0 (2026-04-24 ucrt) / gamlss 5.5.0 / gamlss.dist 6.1.1
- coverage: missing=41 matched=1039 (diff ok=True)
- seeds: be_y_x_seed_20260811=20260811, beinf_y_x_seed_20261122=20261122, bi_y_x_seed_20260809=20260809, exp_y_x_seed_20261117=20261117, ga_y_x_seed_20260525=20260525, geom_y_x_seed_20261118=20261118, gu_y_x_seed_20260813=20260813, ig_y_x_seed_20260522=20260522, igamma_y_x_seed_20261119=20261119, lo_y_x_seed_20261121=20261121, logno2_y_x_seed_20261128=20261128, logno_y_x_seed_20261116=20261116, nbi_y_x_seed_20260810=20260810, nbii_y_x_seed_20261121=20261121, net_y_x_seed_20261127=20261127, no2_y_x_seed_20261120=20261120, no_y_x_seed_20260524=20260524, pe_y_x_seed_20261123=20261123, po_y_x_seed_20260523=20260523, rg_y_x_seed_20261122=20261122, wei_y_x_seed_20261115=20261115, zaga_y_x_seed_20260812=20260812
- tolerances: {'default': 1e-06, 'fitting': 1e-06, 'dpqr': 1e-10}

| gate | result | mode |
|---|---|---|
| p0_clear | PASS |  |
| coverage_regenerable | PASS | --check executed |
| tiered_collection | PASS |  |
| devcontainer | PASS | static + fresh-clone simulation (no docker in CI env) |
| golden_manifest_gate | PASS | --require-r |

M0 gate bundle per DEVELOPMENT_PLAN 22.4. Reproducibility is anchored by coverage provenance content hashes (HEAD-independent); source.commit is informational. Regenerate after any symbol/test change.
