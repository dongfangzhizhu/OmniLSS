# CG 平滑服务层准入一致性

CG 的 full-Hessian/IRLS-cross 后端没有惩罚矩阵输入，因此公开 API 必须在
服务边界统一 fail-closed，不能等到底层返回泛化异常。

- gRPC Fit/BatchFit 在调用 gamlss 前已有能力检查；本次将 ValueError 转为
  结构化 JSON `method_capability_error`，code 为
  `smooth_terms_unsupported_for_method`。
- HTTP FastAPI `/fit` 和候选分布选择入口共用平滑检查，返回 HTTP 400 及同样
  的 type/code/method/message 结构。
- 直接 Python、MIXED 和 CG 入口的错误语义保持一致；服务不会保存错误模型。

验证（项目 .venv）：

```text
python -m pytest omnilss/tests/test_service_smooth_admission.py omnilss/tests/test_cg_smooth_admission.py omnilss/tests/test_mixed_explicit_smooth.py -q
38 passed, 1 skipped
```

gRPC 单次/批量结构化错误及现有 MIXED/CG admission 回归通过。HTTP 测试因
当前 `.venv` 缺少 FastAPI 而跳过；安装 FastAPI 后会执行同一 helper 测试。
本轮未实现 CG 平滑惩罚，也未执行R/GPU验证。

下一项建议：把同一 `method_capability_report` 扩展为包含平滑能力的运行时
报告，接入 HTTP/gRPC route-capability，使客户端在提交 Fit 前可预检 CG
平滑不支持，而不必等请求失败。
