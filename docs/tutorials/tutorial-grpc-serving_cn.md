# gRPC 与 HTTP 托管服务

[English version](tutorial-grpc-serving.md)

OmniLSS 同时提供 gRPC 托管服务层（`omnilss.api.grpc` + `omnilss-pro` 客户端）与
FastAPI HTTP 服务（`omnilss-server/main.py`），两者共享同一套 fail-closed 能力门禁。

## gRPC 服务与接口

| Service | RPC | 用途 |
|---|---|---|
| `FitService` | `Fit`、`BatchFit` | 单模型与批量拟合 |
| `PredictService` | `Predict` | 样本外参数与分位数预测 |
| `SampleService` | `Sample` | 预测分布抽样 |
| `CapabilityService` | `CapabilityMatrix`、`RouteCapability`、`Version`、`Health` | 能力矩阵、路由检查、版本与健康探针 |

启动 gRPC 服务：

```bash
python -m omnilss.api.grpc.server --help
```

Python 客户端（`omnilss_pro.client`）：

```python
from omnilss_pro.client import OmniLSSClient

client = OmniLSSClient("localhost:50051")
fit = client.fit(data, "y ~ x", family="GA")
pred = client.predict(fit.model_id, newdata)
client.health()
client.version()
```

## FastAPI HTTP 服务（`omnilss-server`）

`omnilss-server/main.py` 暴露 `/health`、`/fit`、`/predict`、`/distributions`、
`/distributions/select`、`/models` 与 `/models/{id}` 端点，并内置：

- 可选 Bearer Token 鉴权（`OMNILSS_API_TOKEN`）与按 IP 滑动窗口限流（含客户端表上限回收）。
- `/fit` 与 `/distributions/select` 强制执行 `require_production_family_route`
  fail-closed 门禁（未准入的 `experimental`/`compat_alias` 族返回 HTTP `400`）。
- 输入 DoS 上限保护（`MAX_ROWS`、`MAX_COLS`、`max_iter` 限制在 `1..1000`）。

## 安全基线与契约冻结

- 生产环境暴露 gRPC 服务前必须启用 TLS，可通过 `tools/tls_readiness.py` 检查证书有效性。
- `omnilss/src/omnilss/api/grpc/proto/*.proto` 为冻结契约，桩代码由
  `tools/generate_grpc_stubs.py` 生成。

## 下一步

- [OmniLSS 文档首页](../index_cn.md)
