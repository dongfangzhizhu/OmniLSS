# GB2 概率契约补齐

## 交付

- 负 sigma 通过交换 Beta 形状参数并使用绝对 sigma，实现正确的 CDF/分位数方向。
- CDF 和生存函数分别计算小参数 Beta 尾部，不再直接使用 `1-CDF`。
- 分位数分别计算两个互补 Beta 坐标，移除 epsilon 裁剪；日志概率接近零时用 expm1 保留尾部。
- 无效参数返回 NaN，采样复用相同校验并显式使用 float64 随机数。
- GB2 进入代码派生 experimental 状态并加入全族扫描；重生成能力快照。
- placeholder 拒绝测试和证据生成器改用仍未完成的 GIG，兼容别名测试使用 DBI。

## 验证

项目 `.venv/Scripts/python.exe -m pytest` 执行下列范围：
`test_gb2_probability_contract.py`、`test_gb2_density_contract.py`、
`test_gb2_dpqr.py`、`test_family_capabilities.py`、
`test_generate_capability_matrix_tool.py`、`test_release1_family_capability_gate.py`、
`test_grpc_server.py::test_grpc_fit_blocks_placeholder_family_before_backend`、
`test_all_families_dpqr_scan.py`：**50 passed**。

实际 R d/p/q 对照覆盖小、较大及负 sigma；解析 Lomax 特例验证 1e-100
上尾和接近零的日志概率。采样使用概率积分变换检查分布。

扩大执行完整 test_grpc_server.py 时，3 项错误消息断言失败：预测 schema
信封增加字段、两项仍期待未经脱敏的异常消息。本次相关拒绝测试通过，
不声称全库回归通过。

## 范围与后续

p/q/r 使用 CPU SciPy 特殊函数，不支持 JAX tracing；密度仍支持 JIT/AD。
日志概率仍受浮点及特殊函数下溢限制。支持域端点按数学契约处理；
不复制 R 旧版 qGB2 对负 sigma 端点覆写的倒置行为。
默认参数保留现有 Python API。此次不代表 RS/CG 拟合或 production-safe 准入。

下一项优先补齐 ZALG 固定 10000 项求和及分位数搜索上限；此前边界修复
并未消除大 mu 下截断尾部的问题。
