# 模型选择与诊断

[English version](tutorial-model-selection.md)

分布回归需要同时确定分布族与各分布参数的预测项。OmniLSS 提供了与 R `gamlss` 对齐的
信息准则、假设检验、逐步回归与分位残差诊断体系。

## 偏差、GAIC 与假设检验

```python
from omnilss.operations import deviance, gaic, gaic_scaled, gaic_table
from omnilss.LR_test_12_06_2013 import LR_test
from omnilss.Rsq import Rsq
from omnilss.VuongClarkTest import VC_test

dev = deviance(model)                     # -2 * logLik
aic = gaic(model, k=2.0)                  # AIC（k=log(n) 即 BIC/SBC）
table = gaic_table(m0, m1)                # 多惩罚系数对比表
scaled = gaic_scaled(m0, m1)              # 相对 AIC 权重
lr = LR_test(m0, m1)                      # 嵌套模型似然比检验
r2 = Rsq(m1, type="both")                 # Cox-Snell 与 Cragg-Uhler 伪 R^2
vc = VC_test(m_ga, m_logno)               # Vuong & Clarke 非嵌套检验
```

## 分布族筛选与逐步回归

- `fitDist` / `chooseDist`：按支撑集分组（`_realline`、`_realplus`、`_real0to1`、
  `_counts`、`_binom`）自动比较候选分布族。
- `stepGAIC`、`stepGAICAll_A`、`stepGAICAll_B`、`stepGAIC_CH`：跨 `mu`、`sigma`、
  `nu`、`tau` 的逐步变量选择。
- `stepTGD`、`stepTGDAll_A`、`gamlssVGD`、`gamlssCV`：验证集全局偏差与交叉验证筛选。

## 分位残差与诊断载荷

```python
from omnilss import bp, calibration, fittedPlot, get_rqres, rqres_plot
from omnilss.acfResid import acfResid
from omnilss.operations import residuals

z = residuals(model)                      # 标准化分位残差
acf = acfResid(model, max_lag=10)         # z, z^2, z^3, z^4 自相关诊断
rq_samples = get_rqres(model, howmany=4)  # 随机分位残差抽样
rq_plot = rqres_plot(model, howmany=4)    # QQ / 蠕虫图诊断载荷
bucket = bp(model)                        # 桶图（偏度/峰度诊断）
```

## Bootstrap 不确定性估计

```python
from omnilss.bootstrap import compute_confidence_intervals, nonparametric_bootstrap

result = nonparametric_bootstrap(model, data, n_bootstrap=200)
cis = compute_confidence_intervals(result)
```

## 下一步

- [R 一致性与 Golden 工件体系](tutorial-r-consistency_cn.md)
