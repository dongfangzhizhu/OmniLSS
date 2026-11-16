# P3-1 frozen contracts (family / link / formula / prediction)

- decision: D2.242
- promoted families: 22
- supported links: 18
- formula cases: 7
- contract frozen: True

| criterion | met | detail |
|---|---|---|
| family_contract_holds | yes | 22 promoted families: nopar == len(parameters), estimable == parameters minus fixed (covers BI's bd outside the tuple and NET's fixed nu/tau inside it), d/p/q/r callable |
| link_contract_holds | yes | 18 supported links each have a probe; round-trip within 1e-9 (including the documented 1e-5 offset for logshiftto1/2) and a finite non-zero mu_eta |
| formula_contract_holds | yes | ParsedFormula fields frozen; 7 canonical formulas parse to a recorded structure |
| prediction_contract_holds | yes | predict_params keys == estimable parameters, shapes follow newdata, missing column raises PredictionSchemaError, extra column tolerated |

## Families

| family | parameters | nopar | estimable | fixed | links |
|---|---|---|---|---|---|
| BE | mu, sigma | 2 | mu, sigma | - | mu, sigma |
| BEINF | mu, sigma, nu, tau | 4 | mu, sigma, nu, tau | - | mu, nu, sigma, tau |
| BI | mu | 1 | mu | bd | mu |
| EXP | mu | 1 | mu | - | mu |
| GA | mu, sigma | 2 | mu, sigma | - | mu, sigma |
| GEOM | mu | 1 | mu | - | mu |
| GU | mu, sigma | 2 | mu, sigma | - | mu, sigma |
| IG | mu, sigma | 2 | mu, sigma | - | mu, sigma |
| IGAMMA | mu, sigma | 2 | mu, sigma | - | mu, sigma |
| LO | mu, sigma | 2 | mu, sigma | - | mu, sigma |
| LOGNO | mu, sigma | 2 | mu, sigma | - | mu, sigma |
| LOGNO2 | mu, sigma | 2 | mu, sigma | - | mu, sigma |
| NBI | mu, sigma | 2 | mu, sigma | - | mu, sigma |
| NBII | mu, sigma | 2 | mu, sigma | - | mu, sigma |
| NET | mu, sigma, nu, tau | 4 | mu, sigma | nu, tau | mu, nu, sigma, tau |
| NO | mu, sigma | 2 | mu, sigma | - | mu, sigma |
| NO2 | mu, sigma | 2 | mu, sigma | - | mu, sigma |
| PE | mu, sigma, nu | 3 | mu, sigma, nu | - | mu, nu, sigma |
| PO | mu | 1 | mu | - | mu |
| RG | mu, sigma | 2 | mu, sigma | - | mu, sigma |
| WEI | mu, sigma | 2 | mu, sigma | - | mu, sigma |
| ZAGA | mu, sigma, nu | 3 | mu, sigma, nu | - | mu, nu, sigma |

## Links

| link | offset | round-trip | mu_eta |
|---|---|---|---|
| identity | 0 | True | True |
| log | 0 | True | True |
| logit | 0 | True | True |
| probit | 0 | True | True |
| cauchit | 0 | True | True |
| cloglog | 0 | True | True |
| inverse | 0 | True | True |
| 1/mu^2 | 0 | True | True |
| mu^2 | 0 | True | True |
| sqrt | 0 | True | True |
| slog | 0 | True | True |
| logshiftto0 | 0 | True | True |
| logshiftto1 | 1e-05 | True | True |
| logshiftto2 | 1e-05 | True | True |
| [-1,1] | 0 | True | True |
| (0,2] | 0 | True | True |
| (0,5] | 0 | True | True |
| (1,3] | 0 | True | True |

## Formulas

| formula | linear | smooth | tensor | intercept |
|---|---|---|---|---|
| y ~ x | x | - | - | True |
| y ~ x + z | x, z | - | - | True |
| y ~ x - 1 | x | - | - | False |
| y ~ 1 | - | - | - | True |
| y ~ pb(x) | - | pb(x) | - | True |
| y ~ s(x, k=3) | - | s(x) | - | True |
| y ~ te(x, z) | - | - | te(x,z) | True |

## Prediction

- schema_error_is_value_error: True
- keys_are_estimable_parameters: True
- values_follow_newdata_length: True
- missing_column_raises_schema_error: True
- extra_column_tolerated: True
- predict_response_length_follows_newdata: True

Frozen at the public shape M3 builds on. The link offsets are part of the contract, not a defect: logshiftto1/2 mirror R's asymmetric branch. A new link without a probe, a changed parameter tuple or a changed formula parse fails --check.
