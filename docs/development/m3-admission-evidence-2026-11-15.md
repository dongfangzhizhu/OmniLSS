# M3 production_safe admission evidence inventory

- scope: production_safe admission evidence for the frozen promoted families
- promoted families: 22
- admitted: 22 (BE, BEINF, BI, EXP, GA, GEOM, GU, IG, IGAMMA, LO, LOGNO, LOGNO2, NBI, NBII, NET, NO, NO2, PE, PO, RG, WEI, ZAGA)
- eligible but not admitted: 0
- blocked: 0
- hard gates: deriv_gate_pass, dpqr_gate_pass, fit_gate_pass, r_consistency_validated, golden_fit_fixture, live_newdata_predictions, live_json_roundtrip, live_converged, no_route_mismatch
- soft flags: flat_ridge_cells, r_side_route_errors

| family | status | missing gates | flat_ridge | r-side error |
|---|---|---|---|---|
| BE | admitted | - | - | - |
| BEINF | admitted | - | - | - |
| BI | admitted | - | - | - |
| EXP | admitted | - | - | - |
| GA | admitted | - | - | - |
| GEOM | admitted | - | - | - |
| GU | admitted | - | MIXED | - |
| IG | admitted | - | - | - |
| IGAMMA | admitted | - | MIXED | - |
| LO | admitted | - | - | - |
| LOGNO | admitted | - | - | - |
| LOGNO2 | admitted | - | - | - |
| NBI | admitted | - | CG, MIXED, RS | - |
| NBII | admitted | - | MIXED | CG |
| NET | admitted | - | - | - |
| NO | admitted | - | - | - |
| NO2 | admitted | - | - | - |
| PE | admitted | - | CG, MIXED | - |
| PO | admitted | - | - | - |
| RG | admitted | - | CG, MIXED | - |
| WEI | admitted | - | MIXED | - |
| ZAGA | admitted | - | - | - |

## Registry claim audit

| set | family | backed | requirement |
|---|---|---|---|
| _PRODUCTION_SAFE | BE | True | production_safe |
| _PRODUCTION_SAFE | BEINF | True | production_safe |
| _PRODUCTION_SAFE | BI | True | production_safe |
| _PRODUCTION_SAFE | EXP | True | production_safe |
| _PRODUCTION_SAFE | GA | True | production_safe |
| _PRODUCTION_SAFE | GEOM | True | production_safe |
| _PRODUCTION_SAFE | GU | True | production_safe |
| _PRODUCTION_SAFE | IG | True | production_safe |
| _PRODUCTION_SAFE | IGAMMA | True | production_safe |
| _PRODUCTION_SAFE | LO | True | production_safe |
| _PRODUCTION_SAFE | LOGNO | True | production_safe |
| _PRODUCTION_SAFE | LOGNO2 | True | production_safe |
| _PRODUCTION_SAFE | NBI | True | production_safe |
| _PRODUCTION_SAFE | NBII | True | production_safe |
| _PRODUCTION_SAFE | NET | True | production_safe |
| _PRODUCTION_SAFE | NO | True | production_safe |
| _PRODUCTION_SAFE | NO2 | True | production_safe |
| _PRODUCTION_SAFE | PE | True | production_safe |
| _PRODUCTION_SAFE | PO | True | production_safe |
| _PRODUCTION_SAFE | RG | True | production_safe |
| _PRODUCTION_SAFE | WEI | True | production_safe |
| _PRODUCTION_SAFE | ZAGA | True | production_safe |
| _RS_FIT_EVIDENCE | BE | True | rs_fit |
| _RS_FIT_EVIDENCE | BEINF | True | rs_fit |
| _RS_FIT_EVIDENCE | BI | True | rs_fit |
| _RS_FIT_EVIDENCE | EXP | True | rs_fit |
| _RS_FIT_EVIDENCE | GA | True | rs_fit |
| _RS_FIT_EVIDENCE | GEOM | True | rs_fit |
| _RS_FIT_EVIDENCE | GU | True | rs_fit |
| _RS_FIT_EVIDENCE | IG | True | rs_fit |
| _RS_FIT_EVIDENCE | IGAMMA | True | rs_fit |
| _RS_FIT_EVIDENCE | LO | True | rs_fit |
| _RS_FIT_EVIDENCE | LOGNO | True | rs_fit |
| _RS_FIT_EVIDENCE | LOGNO2 | True | rs_fit |
| _RS_FIT_EVIDENCE | NBI | True | rs_fit |
| _RS_FIT_EVIDENCE | NBII | True | rs_fit |
| _RS_FIT_EVIDENCE | NET | True | rs_fit |
| _RS_FIT_EVIDENCE | NO | True | rs_fit |
| _RS_FIT_EVIDENCE | NO2 | True | rs_fit |
| _RS_FIT_EVIDENCE | PE | True | rs_fit |
| _RS_FIT_EVIDENCE | PO | True | rs_fit |
| _RS_FIT_EVIDENCE | RG | True | rs_fit |
| _RS_FIT_EVIDENCE | WEI | True | rs_fit |
| _RS_FIT_EVIDENCE | ZAGA | True | rs_fit |
| _RS_FIT_EVIDENCE | ZIP | True | rs_fit |
| _PREDICTION_EVIDENCE | BE | True | prediction |
| _PREDICTION_EVIDENCE | BEINF | True | prediction |
| _PREDICTION_EVIDENCE | BI | True | prediction |
| _PREDICTION_EVIDENCE | EXP | True | prediction |
| _PREDICTION_EVIDENCE | GA | True | prediction |
| _PREDICTION_EVIDENCE | GEOM | True | prediction |
| _PREDICTION_EVIDENCE | GU | True | prediction |
| _PREDICTION_EVIDENCE | IG | True | prediction |
| _PREDICTION_EVIDENCE | IGAMMA | True | prediction |
| _PREDICTION_EVIDENCE | LO | True | prediction |
| _PREDICTION_EVIDENCE | LOGNO | True | prediction |
| _PREDICTION_EVIDENCE | LOGNO2 | True | prediction |
| _PREDICTION_EVIDENCE | NBI | True | prediction |
| _PREDICTION_EVIDENCE | NBII | True | prediction |
| _PREDICTION_EVIDENCE | NET | True | prediction |
| _PREDICTION_EVIDENCE | NO | True | prediction |
| _PREDICTION_EVIDENCE | NO2 | True | prediction |
| _PREDICTION_EVIDENCE | PE | True | prediction |
| _PREDICTION_EVIDENCE | PO | True | prediction |
| _PREDICTION_EVIDENCE | RG | True | prediction |
| _PREDICTION_EVIDENCE | WEI | True | prediction |
| _PREDICTION_EVIDENCE | ZAGA | True | prediction |
| _PREDICTION_EVIDENCE | ZIP | True | prediction |

- registry claim violations: 0

This is an inventory, not an admission decision. Of 22 promoted families: 22 admitted, 0 eligible but unregistered, 0 blocked. Soft flags (flat_ridge / r_side_route_error) are carried in the family rows and must travel with any admission record. M3 also requires P3-1/P3-2/P3-3, which this artifact does not cover.
