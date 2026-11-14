# P1-1 frozen high-frequency family list

- generated: 2026-11-14T23:39:55Z
- source commit: fc36e53a6bc6ed4d8e3ea774a4ce03143cf473e5
- validated families: 40
- promoted: 20 / pending: 9 / excluded: 11

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
| NBI | nbi_y_x_seed_20260810 |
| NBII | nbii_y_x_seed_20261121 |
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
| IGA | not_yet_probed |
| LOGNO2 | R simulation needs a positive-mu parameterisation before probing. |
| NET | not_yet_probed |
| PARETO2 | probe sample produced extreme mu rejected by our validation; re-scale x. |
| ZABI | needs the binomial two-column (bd) payload. |
| ZALG | not probed yet. |
| ZIBI | needs the binomial two-column (bd) payload. |

## Excluded (probed, rejected)

| family | code | note |
|---|---|---|
| BCCG | unstable_seed | 2e-3..3e-1 coefficient gap across seeds; one seed jumped to another optimum. |
| DEL | coefficient_gap | sigma gap ~5e-2 and nu ~1.3e1 at n.cyc=100. |
| DELAPORT | no_r_reference | installed gamlss.dist 6.1.1 has no DELAPORT family / r function. |
| GT | diverged | R and our fit diverged at n.cyc=100 (mu gap ~8e7). |
| JSUo | flat_nu_tau_ridge | coefficient gap ~5e-3 at n.cyc=100; 4-parameter heavy-tail ridge. |
| SHASH | flat_nu_tau_ridge | 4-parameter flat ridge; coefficient gap ~1e-1 while deviance matched. |
| SIMPLEX | coefficient_gap | coefficient gap ~1e-3 at n.cyc=100. |
| SN1 | flat_nu_ridge | nu ridge; coefficient gap ~8e-2 at n.cyc=100. |
| TF | flat_nu_tau_ridge | n.cyc=100 leaves a large coefficient gap from a different local optimum; deviance matched. |
| ZIP | coefficient_gap | sigma/nu gap ~2e-3 at n.cyc=100 on the probe sample. |
| ZIPF | unstable_seed | 2/3 seeds pass 2e-4; third mu gap 7.7e-4 (no sigma in rZIPF). |
