# 快速上手（Quickstart）

`OmniLSS` 是 R `gamlss` 生态的 Python/JAX 实现：分布回归（对均值、尺度、
形状等参数同时建模）。

## 安装

```bash
pip install omnilss
# 可选：JAX GPU 后端（Phase 7 性能专项后支持）
```

## 最小示例

```python
import numpy as np
from omnilss import gamlss

rng = np.random.default_rng(0)
n = 120
x = rng.normal(size=n)
# 模拟：mu = exp(1.0 + 0.5x)，sigma = 0.3
y = rng.gamma(shape=1 / 0.09, scale=np.exp(1.0 + 0.5 * x) * 0.09)

data = {"y": y, "x": x}
model = gamlss("y ~ x", family="GA", data=data)
print(model.summary())
print("mu 系数:", model.coefficients["mu"])   # 期望接近 [1.0, 0.5]
print("sigma:", np.exp(model.coefficients["sigma"]))  # 期望接近 0.3
```

## 预测

```python
newdata = {"y": np.zeros(5), "x": np.linspace(-2, 2, 5)}
pred = model.predict(newdata=newdata)   # 返回 mu
pred_all = model.predict(newdata=newdata, type="response")
```

## 三件事

1. **分布族**：`NO`/`GA`/`PO`/`BI`/`NBI`/`BE`/`ZAGA`/`GU`（八核心，
   R 一致性已验证）及其余约 20 个族。
2. **算法**：`method="RS"`（默认）/ `"CG"` / `"MIXED"`。
3. **平滑**：`ps(x)` / `cs(x)` / `pb(x)` 加入公式即用。

## 下一步

- [分布族与参数](tutorial-families.md)
- [拟合算法 RS/CG/MIXED](tutorial-fitting-algorithms.md)
- [平滑器 pb/ps/cs](tutorial-smoothers.md)
