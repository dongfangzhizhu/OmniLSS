# Model Selection & Diagnostics

[中文版本](tutorial-model-selection_cn.md)

Distributional regression requires selecting both the distribution family and
each parameter's predictor terms. OmniLSS provides R-aligned information
criteria, hypothesis tests, stepwise selection, and quantile-residual
diagnostics.

## Deviance, GAIC, and Hypothesis Tests

```python
from omnilss.operations import deviance, gaic, gaic_scaled, gaic_table
from omnilss.LR_test_12_06_2013 import LR_test
from omnilss.Rsq import Rsq
from omnilss.VuongClarkTest import VC_test

dev = deviance(model)                     # -2 * logLik
aic = gaic(model, k=2.0)                  # AIC (k=log(n) for BIC/SBC)
table = gaic_table(m0, m1)                # multi-penalty comparison table
scaled = gaic_scaled(m0, m1)              # relative AIC weights
lr = LR_test(m0, m1)                      # nested likelihood-ratio test
r2 = Rsq(m1, type="both")                 # Cox-Snell and Cragg-Uhler R^2
vc = VC_test(m_ga, m_logno)               # Vuong & Clarke non-nested tests
```

## Stepwise & Distribution Selection

- `fitDist` / `chooseDist`: compare admissible candidate families over support
  groups (`_realline`, `_realplus`, `_real0to1`, `_counts`, `_binom`).
- `stepGAIC`, `stepGAICAll_A`, `stepGAICAll_B`, `stepGAIC_CH`: stepwise term
  selection across `mu`, `sigma`, `nu`, `tau`.
- `stepTGD`, `stepTGDAll_A`, `gamlssVGD`, `gamlssCV`: validation global
  deviance and cross-validation selection.

## Quantile Residuals & Diagnostics

```python
from omnilss import bp, calibration, fittedPlot, get_rqres, rqres_plot
from omnilss.acfResid import acfResid
from omnilss.operations import residuals

z = residuals(model)                      # normalized quantile residuals
acf = acfResid(model, max_lag=10)         # ACF of z, z^2, z^3, z^4
rq_samples = get_rqres(model, howmany=4)  # randomized quantile residual draws
rq_plot = rqres_plot(model, howmany=4)    # QQ / worm-plot diagnostic payloads
bucket = bp(model)                        # bucket plot (moment skewness/kurtosis)
```

## Bootstrap Uncertainty

```python
from omnilss.bootstrap import compute_confidence_intervals, nonparametric_bootstrap

result = nonparametric_bootstrap(model, data, n_bootstrap=200)
cis = compute_confidence_intervals(result)
```

## Next Steps

- [R Consistency & Golden Artifacts](tutorial-r-consistency.md)
