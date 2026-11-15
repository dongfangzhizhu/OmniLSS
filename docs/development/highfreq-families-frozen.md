# P1-1 frozen high-frequency family list

- generated: 2026-11-15T21:57:45Z
- source commit: 94c7d95f2c8da21700f5717401ab99e39c68c1f1
- validated families: 40
- promoted: 22 / pending: 2 / excluded: 16

## Promoted (golden evidence present)

| family | golden cases |
|---|---|
| BE | be_y_x_seed_20260811 |
| BEINF | beinf_y_x_seed_20261122 |
| BI | bi_y_x_seed_20260809 |
| EXP | exp_y_x_seed_20261117 |
| GA | ga_y_x_seed_20260525 |
| GEOM | geom_y_x_seed_20261118 |
| GU | gu_y_x_seed_20260813 |
| IG | ig_y_x_seed_20260522 |
| IGAMMA | igamma_y_x_seed_20261119 |
| LO | lo_y_x_seed_20261121 |
| LOGNO | logno_y_x_seed_20261116 |
| LOGNO2 | logno2_y_x_seed_20261128 |
| NBI | nbi_y_x_seed_20260810 |
| NBII | nbii_y_x_seed_20261121 |
| NET | net_y_x_seed_20261127 |
| NO | no_y_x_seed_20260524 |
| NO2 | no2_y_x_seed_20261120 |
| PE | pe_y_x_seed_20261123 |
| PO | po_y_x_seed_20260523 |
| RG | rg_y_x_seed_20261122 |
| WEI | wei_y_x_seed_20261115 |
| ZAGA | zaga_y_x_seed_20260812 |

## Pending (not probed)

| family | note |
|---|---|
| BCPE | four-parameter Box-Cox; expect a flat nu/tau ridge, probe at n.cyc>=500 first. |
| BCT | four-parameter Box-Cox; same caution as BCPE. |

## Excluded (probed, rejected)

| family | code | note |
|---|---|---|
| BCCG | unstable_seed | 2e-3..3e-1 coefficient gap across seeds; one seed jumped to another optimum. |
| DEL | coefficient_gap | sigma gap ~5e-2 and nu ~1.3e1 at n.cyc=100. |
| DELAPORT | no_r_reference | installed gamlss.dist 6.1.1 has no DELAPORT family / r function. |
| GT | diverged | R and our fit diverged at n.cyc=100 (mu gap ~8e7). |
| IGA | no_r_reference | installed gamlss.dist 6.1.1 has no IGA family (gamlss.family('IGA') errors). |
| JSUo | flat_nu_tau_ridge | coefficient gap ~5e-3 at n.cyc=100; 4-parameter heavy-tail ridge. |
| PARETO2 | flat_mu_sigma_ridge | deviance matches to 4 digits but mu/sigma differ ~1e-2 on every seed (heavy-tail shape ridge); heavy-tail samples (ymax>2000) also fail initial-value validation. |
| SHASH | flat_nu_tau_ridge | 4-parameter flat ridge; coefficient gap ~1e-1 while deviance matched. |
| SIMPLEX | coefficient_gap | coefficient gap ~1e-3 at n.cyc=100. |
| SN1 | flat_nu_ridge | nu ridge; coefficient gap ~8e-2 at n.cyc=100. |
| TF | flat_nu_tau_ridge | n.cyc=100 leaves a large coefficient gap from a different local optimum; deviance matched. |
| ZABI | r_density_vectorization_defect | R's vectorised dZABI collapses the hurdle correction to row 1 (scalar dBI(0, ..) truncates via ifelse), so R's own fit optimises a defective objective and its coefficients are not a parity target. Ours is the elementwise density; pinned by test_binomial_zero_variants_bd.py::TestTheRReferenceIsDefective. |
| ZALG | unstable_mu | 1/3 seeds within 2e-4; mu gap 1.2e-3..4.1e-3 while deviance matched (flat mu ridge). |
| ZIBI | r_density_vectorization_defect | R's vectorised dZIBI pastes row 1's base zero mass onto every y==0 row (same scalar-dBI collapse); pinned by test_binomial_zero_variants_bd.py::TestTheRReferenceIsDefective. |
| ZIP | coefficient_gap | sigma/nu gap ~2e-3 at n.cyc=100 on the probe sample. |
| ZIPF | unstable_seed | 2/3 seeds pass 2e-4; third mu gap 7.7e-4 (no sigma in rZIPF). |
