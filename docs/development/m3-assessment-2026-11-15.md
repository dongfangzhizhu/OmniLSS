# M3 assessment (target 0.9.0)

- decision: D2.245
- criteria met: 6/6
- admitted families: 22/22

| criterion | met | detail |
|---|---|---|
| p3_contracts_frozen | yes | family / link / formula / prediction contracts are frozen and checked against the live package (D2.242) |
| gen_family_fits_like_r | yes | logNO RS fit reproduces R deviance 140.3831 (D2.243 derivative wrapper); converged=True |
| fix_on_every_route | yes | sigma.fix=TRUE with sigma.start=0.5 reaches R deviance 67.1324138595735 on RS/CG/MIXED with no sigma coefficient (D2.244) |
| categorical_prediction_rebuilt | yes | CG/MIXED factor fits rebuild predictions from the contrasts golden for both the deviance and pred_mu (D2.244 intercept fix) |
| admission_closed | yes | 22/22 promoted families are production_safe (D2.241) |
| prior_milestones_intact | yes | M1 and M2 gate verdicts are still true |

## Live probes

- gen.Family: ok=True, converged=True

| route | fix deviance | sigma not estimated | sigma frozen |
|---|---|---|---|
| RS | True | True | True |
| CG | True | True | True |
| MIXED | True | True | True |

| route | categorical deviance | categorical prediction |
|---|---|---|
| CG | True | True |
| MIXED | True | True |

**m3_gate_met: True**

M3 is met when P3-1..P3-4 are closed and the two behavioural claims (gen.Family fitting, fix across routes) hold on a live probe, not only in a test name. M4 (surrounding packages, services, docs, release artifacts) is untouched by this verdict.

## Deliverables

- P3-1: docs/development/p3-contracts-2026-11-15.json
- P3-2: omnilss/src/omnilss/family_generator.py + test_family_generator.py
- P3-3: omnilss/tests/test_p3_fix_methods.py
- P3-4: docs/development/m3-admission-evidence-2026-11-15.json
