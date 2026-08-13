# gRPC 托管服务（L3）

商业层（omnilss-pro）通过 gRPC 暴露模型生命周期；开源核心提供
`omnilss.api.grpc` 的契约与实现。

## 服务与端点

| Service | RPC | 用途 |
|---|---|---|
| `FitService` | `Fit` / `BatchFit` | 拟合/批量拟合 |
| `PredictService` | `Predict` | 预测 |
| `SampleService` | `Sample` | 采样 |
| `CapabilityService` | `CapabilityMatrix` / `RouteCapability` / `Version` / `Health` | 能力矩阵 / 路由能力 / 版本 / 健康 |

## 启动服务

```bash
python -m omnilss.api.grpc.server --help
```

服务加载已保存模型（JSON 工件 + 能力快照回放），支持运行时注册新模型。

## 客户端（omnilss_pro.client）

```python
from omnilss_pro.client import OmniLSSClient

client = OmniLSSClient("localhost:50051")
m = client.fit(data, "y ~ x", family="GA")     # 拟合（远端）
pred = client.predict(m.model_id, newdata)      # 预测
client.health()                                  # 健康探针
client.version()                                 # 版本（__version__ + x64）
```

## 安全基线（托管层）

- 认证 / 请求 DoS 上限 / 输入脱敏（服务层）
- 序列化安全：加载工件时重放能力快照，防止不一致模型进入服务
- TLS 配置入口就绪（生产部署必须启用）

## 契约冻结

`omnilss/src/omnilss/api/grpc/proto/*.proto` 为冻结契约；`*_pb2.py`
由 `tools/generate_grpc_stubs.py` 生成（CI 执行，需 `grpcio-tools`）。

## 下一步

- [README 概览](../../README.md)
