# 预测与 newdata

拟合完成后可对任意 newdata 预测。预测时**重建设计矩阵**（含平滑基），
因此与训练数据行数无关。

## 基本预测

```python
import numpy as np
from omnilss import gamlss

model = gamlss("y ~ x", family="GA", data=train)
pred = model.predict(newdata=new)   # 返回 mu（条件均值）
```

## 预测内容

| 方法 | 返回 |
|---|---|
| `model.predict(newdata)` | 条件均值 mu |
| `model.predict(newdata, type="link")` | 线性预测子 η_mu |
| `model.predict_params(newdata)` | 所有分布参数（mu/sigma/...） |
| `model.predict_quantiles(newdata, quantiles=[0.05, 0.5, 0.95])` | 条件分位数 |

```python
pars = model.predict_params(newdata)                      # {"mu": ..., "sigma": ...}
q90 = model.predict_quantiles(newdata, quantiles=[0.9])["0.9"]
```

## 平滑项预测

含 `pb`/`ps`/`cs` 的模型预测同样支持——内部保存平滑状态并重建基：

```python
model = gamlss("y ~ pb(x)", family="NO", data=train)
pred = model.predict(newdata={"y": y_new, "x": x_new})  # x_new 任意长度
```

## 序列化与再预测

```python
from omnilss.serialization import save_model_json, load_model_json

save_model_json(model, "model.json")
m2 = load_model_json("model.json")
pred2 = m2.predict(newdata=new)   # 等价于原模型
```

## 注意

- `newdata` 必须包含公式引用的所有变量；缺失 → `PredictionSchemaError`。
- 分布参数外推（如 sigma）遵循链接函数约束，极端外推可能触发 eta 裁剪。

## 下一步

- [模型选择与诊断](tutorial-model-selection.md)
