# 预测与序列化

[English version](tutorial-prediction.md)

模型拟合完成后，OmniLSS 会对任意 `newdata` 重建各参数设计矩阵（含分类对比编码与
平滑基），无需传入响应列，也与训练样本行数无关。

## 核心预测接口

```python
import numpy as np
from omnilss import GA, gamlss

model = gamlss("y ~ x", family=GA(), data=train)
newdata = {"x": np.linspace(-2.0, 2.0, 5)}

mu_hat = model.predict(newdata=newdata)                            # 响应尺度 mu
eta_mu = model.predict(newdata=newdata, type="link")               # 线性预测子
params = model.predict_params(newdata=newdata)                     # {"mu": ..., "sigma": ...}
quantiles = model.predict_quantiles(newdata, quantiles=[0.05, 0.5, 0.95])
```

| 方法 / 函数 | 返回内容 |
|---|---|
| `model.predict(newdata)` | 条件位置参数 `mu` |
| `model.predict(newdata, type="link")` | 线性预测子 `eta_mu` |
| `model.predict_params(newdata)` | 全部分布参数（`mu`、`sigma`、`nu`、`tau`） |
| `model.predict_quantiles(newdata, quantiles=...)` | 基于分布 `q()` 的条件分位数 |
| `predictAll(model, newdata=...)` | R `predictAll()` 风格多参数结构化预测 |
| `centiles_pred(model, xname=..., xvalues=...)` | R `centiles.pred()` 分位曲线 / z-scores |
| `lpred(model, what="mu", type="terms", se_fit=True)` | 中心化分项贡献与标准误 |

## 平滑项与分类对比编码

含 `pb()`、`ps()`、`cs()` 或分类因子对比编码（`contr.treatment`、`contr.sum`、
`contr.helmert`、`contr.poly`）的模型会持久化样条基与编码状态，确保在 `RS`、`CG`、
`MIXED` 路由下 `newdata` 预测语义完全一致：

```python
model = gamlss("y ~ pb(x)", family="NO", data=train)
pred = model.predict(newdata={"x": x_new})
```

## JSON 序列化与往返一致性

全部 22 个生产准入族均在 `docs/development/m2-assessment-2026-11-15.json` 中通过了
JSON 保存/加载往返预测漂移 `<= 1e-10` 校验：

```python
from omnilss import load_model_json, save_model_json

save_model_json(model, "model.omnilss.json")
restored = load_model_json("model.omnilss.json")
assert np.allclose(
    model.predict_params(newdata)["mu"],
    restored.predict_params(newdata)["mu"],
    atol=1e-10,
)
```

## 下一步

- [模型选择与诊断](tutorial-model-selection_cn.md)
