# CG 平滑能力预检报告

扩展 `method_route_capability_report` 接受可选 `formulas` mapping。对于
method=CG，预检共享 `has_smooth_terms` 检测；发现平滑时在调度前返回：

- `ok=false`
- `code=smooth_terms_unsupported_for_method`
- `feature=smooth_terms`
- `smooth_parameters`：受影响参数列表
- 统一 CG/RS/auto 建议消息

线性公式继续按原 family/method capability 规则报告。旧调用不传 formulas
时保持完全兼容；gRPC RouteCapability 旧协议尚未携带公式字段，因此继续
提供 family/method 级报告，Fit 请求使用上一轮结构化错误。

Python fitting 入口把 mu/sigma/nu/tau 的有效公式传入预检，避免 capability
报告与实际 CG 拒绝规则漂移。服务层当前协议没有公式字段，HTTP Fit 的
请求模型也暂未暴露 parameter_formulas 字典，后续可扩展协议。

项目 `.venv`：
`python -m pytest omnilss/tests/test_capability_smooth_preflight.py omnilss/tests/test_cg_smooth_admission.py omnilss/tests/test_service_smooth_admission.py omnilss/tests/test_family_capabilities.py -q`
结果 **40 passed, 1 skipped**；HTTP 测试因本地缺 FastAPI 跳过。

下一项建议：扩展 capability proto/HTTP route-capability 请求以携带公式，
让远端客户端也能在 Fit 前精确预检具体参数中的平滑项，而不是只做家族/方法
级预检。
