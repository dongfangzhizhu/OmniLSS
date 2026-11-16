# M2 assessment (target 0.6.0)

- live replay: True
- promoted families: 22
- static fit+newdata evidence: 22/22
- live newdata predictions ok: 22/22
- live JSON round-trip ok: 22/22
- live converged: 22/22
- live errors: (none)

| criterion | met | detail |
|---|---|---|
| p2_defects_classified | yes | 90 non-pass entries across three gates, unclassified=0 |
| layered_parity_our_side | yes | 66 cells; counts={'flat_ridge': 11, 'pass': 54, 'r_error': 1}; mismatch=0, py_error=0 |
| fit_newdata_evidence_static | yes | 22/22 promoted fixtures carry deviance + coefficient blocks + convergence + newdata + predicted |
| fit_newdata_evidence_live | yes | 22/22 refits reproduce R's newdata predictions within atol=0.0002 |
| serialization_roundtrip | yes | 22/22 JSON round-trips drift <= 1e-10 |
| failures_explainable | yes | every non-pass carries a category: P2 classification (90 entries) plus layered r_error=['NBII'], r_not_converged=0, flat_ridge=7 |
| convergence_complete | yes | 22/22 refits report converged=True |

**m2_gate_met: True**

M2 is met when the P2 classification is exhaustive, our three fitting routes have no mismatch, every promoted fixture carries fit + newdata + convergence evidence, our refits reproduce R's newdata predictions and survive a JSON round-trip, and every non-pass has a category. flat_ridge and R-side errors are classified, not counted as our failures.

## Per-family evidence

| family | case | dev | coef blocks | newdata | predicted | convergence | pred gap | roundtrip |
|---|---|---|---|---|---|---|---|---|
| BE | be_y_x_seed_20260811 | True | 2 | True | True | True | 2.13e-07 | 0.0e+00 |
| BEINF | beinf_y_x_seed_20261122 | True | 4 | True | True | True | 1.35e-05 | 0.0e+00 |
| BI | bi_y_x_seed_20260809 | True | 1 | True | True | True | 2.25e-13 | 0.0e+00 |
| EXP | exp_y_x_seed_20261117 | True | 1 | True | True | True | 6.34e-07 | 0.0e+00 |
| GA | ga_y_x_seed_20260525 | True | 2 | True | True | True | 5.03e-09 | 0.0e+00 |
| GEOM | geom_y_x_seed_20261118 | True | 1 | True | True | True | 5.60e-05 | 0.0e+00 |
| GU | gu_y_x_seed_20260813 | True | 2 | True | True | True | 8.37e-06 | 0.0e+00 |
| IG | ig_y_x_seed_20260522 | True | 2 | True | True | True | 4.85e-04 | 0.0e+00 |
| IGAMMA | igamma_y_x_seed_20261119 | True | 2 | True | True | True | 1.18e-07 | 0.0e+00 |
| LO | lo_y_x_seed_20261121 | True | 2 | True | True | True | 1.37e-04 | 0.0e+00 |
| LOGNO | logno_y_x_seed_20261116 | True | 2 | True | True | True | 2.46e-13 | 0.0e+00 |
| LOGNO2 | logno2_y_x_seed_20261128 | True | 2 | True | True | True | 1.17e-15 | 0.0e+00 |
| NBI | nbi_y_x_seed_20260810 | True | 2 | True | True | True | 1.95e-04 | 0.0e+00 |
| NBII | nbii_y_x_seed_20261121 | True | 2 | True | True | True | 3.33e-04 | 0.0e+00 |
| NET | net_y_x_seed_20261127 | True | 2 | True | True | True | 1.67e-05 | 0.0e+00 |
| NO | no_y_x_seed_20260524 | True | 2 | True | True | True | 1.13e-12 | 0.0e+00 |
| NO2 | no2_y_x_seed_20261120 | True | 2 | True | True | True | 1.54e-05 | 0.0e+00 |
| PE | pe_y_x_seed_20261123 | True | 3 | True | True | True | 3.26e-05 | 0.0e+00 |
| PO | po_y_x_seed_20260523 | True | 1 | True | True | True | 1.14e-08 | 0.0e+00 |
| RG | rg_y_x_seed_20261122 | True | 2 | True | True | True | 1.14e-04 | 0.0e+00 |
| WEI | wei_y_x_seed_20261115 | True | 2 | True | True | True | 1.09e-04 | 0.0e+00 |
| ZAGA | zaga_y_x_seed_20260812 | True | 3 | True | True | True | 4.35e-07 | 0.0e+00 |
