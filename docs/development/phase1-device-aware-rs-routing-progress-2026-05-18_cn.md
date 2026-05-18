# 第一阶段进展：设备感知 RS 路由

English version: [phase1-device-aware-rs-routing-progress-2026-05-18.md](phase1-device-aware-rs-routing-progress-2026-05-18.md)

## 本步骤已完成

- 将默认 `method="RS"` 入口接入现有的设备感知后端选择器。
- 保持 CPU 行为以及无限阈值加速器配置下的 NumPy RS 行为，同时允许已配置 GPU/TPU 阈值时为支持的 family 选择 `RS_JAX`。
- 保留 `method="auto"` 作为相同行为的兼容别名。
- 对显式使用 `method="RS_JAX"` 增加 `DeprecationWarning`；用户应优先使用 `method="RS"` 并配置 crossover 阈值。
- 将方法路由诊断输出改为英文，使运行时消息遵循项目默认语言。
- 增加集成测试，验证默认 `method="RS"` 在配置加速器阈值后可以路由到 JAX。

## 说明

本步骤完成了第一阶段的路由任务，同时不会在当前默认 GPU/TPU 阈值下强制启用 JAX。项目仍需要基准测试证据，才能将占位的 `math.inf` 加速器 crossover 阈值替换为有限默认值。
