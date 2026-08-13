# 模型选择与诊断

分布回归的核心问题：**哪个分布族 + 哪些项**。OmniLSS 提供选择与诊断工具。

## 偏差（Deviance）

```python
from omnilss.operations import deviance
deviance(model)   # -2 * logLik（R `deviance.gamlss` 语义）
```

- 嵌套模型比较用偏差差（卡方近似）；不同族之间**不可**直接比偏差。

## 信息准则

```python
from omnilss.operations import ic, gaic
ic(model)                    # AIC（k=2，R `GAIC()` 语义）
gaic(model, k=2)             # AIC；k=log(n) → BIC/SBC（R GAIC 同款）
```

- 族选择（如 `NO` vs `NBI` vs `ZAGA`）用 AIC/SBC 比较。

## 拟合值 / 残差

```python
fitted_mu = model.fitted_values["mu"]
from omnilss.operations import residuals
resid = residuals(model)             # 分位残差（R residuals.gamlss 语义）
```

- 分位残差对分布回归更稳健，可做 QQ 图检查族选择。

## 能力矩阵（族 × 方法）

```python
from omnilss.family_capabilities import capability_matrix
capability_matrix()   # 查看哪些族 × 方法可用/实验性/占位
```

- 自动选择族时先用能力矩阵过滤，避免选到占位族。

## 安全网：bootstrap

```python
from omnilss.bootstrap import nonparametric_bootstrap, compute_confidence_intervals
result = nonparametric_bootstrap(model, data, n_bootstrap=200)
cis = compute_confidence_intervals(result)   # 参数 bootstrap 置信区间
```

- 对参数不确定性给出 bootstrap 置信区间（商业层 AutoML 使用同源机制）。

## 下一步

- [R 一致性验证与 golden artifact](tutorial-r-consistency.md)
