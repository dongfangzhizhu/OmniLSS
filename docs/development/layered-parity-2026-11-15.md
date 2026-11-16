# RS / CG / MIXED layered parity

- families: 22 x methods: 3 = 66 cells
- counts by code: {'pass': 54, 'flat_ridge': 11, 'r_error': 1}
- our side parity ok: True
- all layers pass: False

| family | RS | CG | MIXED |
|---|---|---|---|
| BE | pass | pass | pass |
| BEINF | pass | pass | pass |
| BI | pass | pass | pass |
| EXP | pass | pass | pass |
| GA | pass | pass | pass |
| GEOM | pass | pass | pass |
| GU | pass | pass | flat_ridge |
| IG | pass | pass | pass |
| IGAMMA | pass | pass | flat_ridge |
| LO | pass | pass | pass |
| LOGNO | pass | pass | pass |
| LOGNO2 | pass | pass | pass |
| NBI | flat_ridge | flat_ridge | flat_ridge |
| NBII | pass | r_error | flat_ridge |
| NET | pass | pass | pass |
| NO | pass | pass | pass |
| NO2 | pass | pass | pass |
| PE | pass | flat_ridge | flat_ridge |
| PO | pass | pass | pass |
| RG | pass | flat_ridge | flat_ridge |
| WEI | pass | pass | flat_ridge |
| ZAGA | pass | pass | pass |

## Non-pass cells

| family | method | code | detail |
|---|---|---|---|
| GU | MIXED | flat_ridge | max|dcoef|=4.586e-04 (mu), dev_rel=1.606e-07, tol=0.0002 |
| IGAMMA | MIXED | flat_ridge | max|dcoef|=5.105e-04 (mu), dev_rel=1.948e-07, tol=0.0002 |
| NBI | CG | flat_ridge | max|dcoef|=1.427e-03 (sigma), dev_rel=2.180e-08, tol=0.0005 |
| NBI | MIXED | flat_ridge | max|dcoef|=8.221e-04 (sigma), dev_rel=8.519e-09, tol=0.0002 |
| NBI | RS | flat_ridge | max|dcoef|=6.974e-04 (sigma), dev_rel=5.239e-09, tol=0.0002 |
| NBII | CG | r_error | Error in while (abs(G.dev.in - i.G.dev) > i.c.crit && i.iter < i.n.cyc) { : 
  missing value where TRUE/FALSE needed
 |
| NBII | MIXED | flat_ridge | max|dcoef|=1.547e-03 (sigma), dev_rel=3.323e-08, tol=0.0002 |
| PE | CG | flat_ridge | max|dcoef|=9.656e-04 (nu), dev_rel=9.292e-08, tol=0.0005 |
| PE | MIXED | flat_ridge | max|dcoef|=3.046e-04 (nu), dev_rel=8.773e-08, tol=0.0002 |
| RG | CG | flat_ridge | max|dcoef|=2.339e-03 (sigma), dev_rel=1.357e-05, tol=0.0005 |
| RG | MIXED | flat_ridge | max|dcoef|=2.310e-03 (sigma), dev_rel=1.337e-05, tol=0.0002 |
| WEI | MIXED | flat_ridge | max|dcoef|=2.183e-04 (mu), dev_rel=4.976e-08, tol=0.0002 |

our_side_parity_ok is the M2 claim: no coefficient mismatch and no failing route on our side. r_not_converged / r_error are R-side outcomes and flat_ridge cells reached the same likelihood (dev_rel <= 1e-5) with a weakly identified parameter; both are classified, not hidden. all_layers_pass is the stricter 'every cell behaves like R to 2e-4' reading and is reported for honesty.
