# 第 5 阶段进展：JOSS 基准测试与发表准备 — 2026-05-18

English version: [phase5-joss-benchmark-readiness-progress-2026-05-18.md](phase5-joss-benchmark-readiness-progress-2026-05-18.md)

## 已完成工作

- 将 JAX RS 基准测试入口拆分为明确的 `no-r`、`optional-r` 与 `optional-gpu` 套件，使 CI、发布验证和加速器探索不再隐含相同依赖集合。
- 在 CI 中加入不需要 R/GPU 的 benchmark smoke 作业。该作业只验证基准测试框架可执行，不声明 R 一致性或 GPU 性能结论。
- 扩展 JAX RS 基准测试产物，加入重复 warm 计时样本与 95% 置信区间，同时将 cold 编译时间单独保留。
- 移除旧的基准测试措辞，避免把 JAX RS 路径描述为使用 NumPy RS warm-start；面向发表的文本现在说明 JAX RS 使用数据感知 cold-start 初始化。
- 扩展 JOSS 论文草稿，补充需求说明、算法描述、验证与基准测试方法、保守局限性，以及带交叉链接的中文文档。

## 已确认的报告规则

- Cold 与 warm 运行时间必须分开报告。
- 仅 Python 的 benchmark smoke 运行只是健康检查，不是等价性证据。
- 发布 R 比较结论前必须生成 optional-R 产物。
- Optional-GPU 产物必须说明是否有可用 JAX GPU 设备。
