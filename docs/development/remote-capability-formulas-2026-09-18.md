# 远程 capability 公式预检协议

扩展 `RouteCapabilityRequest`，新增 `formula`、`sigma_formula`、
`nu_formula`、`tau_formula` 四个可选字符串字段；core/pro 两套 proto 与
生成的 capability stubs 已同步。旧客户端省略字段时行为不变。

gRPC RouteCapability 将字段转为 formulas mapping，调用共享
method_route_capability_report；HTTP `/route-capability` 和
`/method-route-capability` 接受同名 query 参数。CG 携带如
`sigma_formula=~ pb(x)` 时可在 Fit 前得到
`smooth_terms_unsupported_for_method` 和 `smooth_parameters=["sigma"]`。

验证（项目 .venv）：

```text
python -m pytest omnilss/tests/test_capability_smooth_preflight.py omnilss/tests/test_generate_grpc_stubs_tool.py omnilss/tests/test_grpc_server.py::test_grpc_route_capability_requires_family_and_method omnilss/tests/test_http_capability_formula_query.py -q
7 passed, 1 skipped
```

HTTP 集成测试因本地 `.venv` 未安装 FastAPI 跳过；共享报告和 gRPC 新字段
实际通过。生成器也已运行，非 capability 的无关生成文件已恢复，避免协议
变更污染其它 stub。未执行R/GPU验证。

下一项建议：为客户端 SDK 增加公式字段参数和 `route_capability` 方法封装，
让调用方无需手工构造 protobuf/query；随后在CI安装FastAPI执行HTTP集成测试。
