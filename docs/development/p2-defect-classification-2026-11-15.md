# P2 defect classification

- scan: docs/development/rs-parity-scan-2026-11-15.json (sha256 e973b0e806d3..., 146 families)
- non-pass total: 90
- all classified: True

| gate | status counts |
|---|---|
| deriv | pass=145, skip=1 |
| dpqr | pass=121, skip=25 |
| rsdev | pass=82, skip=28, warn=36 |

## Categories

### deriv

| code | count | families |
|---|---|---|
| skip_no_support_grid | 1 | LOGSHASHO |

### dpqr

| code | count | families |
|---|---|---|
| no_r_reference | 25 | BCCGUNTR, BCPEUNTR, BCTUNTR, BINOM_1_31, BINOM_2_33, BINOM_3_33, CONTR01_2_13, CONTR01_4_33, CONTRPLUS_1_1, CONTRPLUS_2_11, CONTRPLUS_3_13, CONTRPLUS_4_33, CONTR_2_12, CONTR_3_11, CONTR_4_13, COUNT_1_22, COUNT_1_31, COUNT_2_32, COUNT_2_32R, COUNT_2_33, COUNT_3_32, COUNT_3_33, LOGSHASHO, MULTIN, ZILG |

### rsdev

| code | count | families |
|---|---|---|
| r_fit_error | 28 | BINOM_1_31, BINOM_2_33, BINOM_3_33, CONTR01_2_13, CONTR01_4_33, CONTRPLUS_1_1, CONTRPLUS_2_11, CONTRPLUS_3_13, CONTRPLUS_4_33, CONTR_2_12, CONTR_3_11, CONTR_4_13, COUNT_1_22, COUNT_1_31, COUNT_2_32, COUNT_2_32R, COUNT_2_33, COUNT_3_32, COUNT_3_33, LOGSHASH, LOGSHASHO, MULTIN, RGE, SEP, SICHEL, ZASICHEL, ZILG, ZISICHEL |
| r_not_converged | 19 | BCCG, BCCGO, BCT, BCTO, BNB, DBURR12, EGB2, GB1, GB2, GG, JSU, JSUO, SEP1, SHASHO, SI, SN2, ST3, ST3C, ST5 |
| registered_path_divergent | 1 | SHASH |
| registered_r_density_recycled | 2 | ZANBI, ZAP |
| registered_r_diverged | 4 | GT, SEP2, SEP3, SEP4 |
| registered_r_early_stop | 8 | GAF, GP, PARETO2, PARETO2O, PE2, WARING, ZAZIPF, ZIPIG |
| registered_r_reference_invalid | 2 | ZABNB, ZIBNB |

Every non-pass carries a category. 'no_r_reference' means R's own package cannot produce the reference; 'registered_*' are the scanner's adjudicated R-side outcomes (early stop / divergence / path / density recycling); flat-ridge families are those whose deviance matched but the optimum did not.
