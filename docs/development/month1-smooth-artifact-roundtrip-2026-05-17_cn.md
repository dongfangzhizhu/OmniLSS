# 第 1 月平滑项 Artifact Roundtrip 进展（2026-05-17）

[English version](month1-smooth-artifact-roundtrip-2026-05-17.md)

## 背景

本文推进[六个月执行计划](six-month-execution-plan-2026-05-17_cn.md)中第 1 月可信 Core 工作流，重点覆盖平滑模型在 fit → predict → serialize → load → predict 路径上必须安全 roundtrip，或以结构化 schema 错误失败的要求。

## 实现进展

- 平滑项拟合 metadata 现在携带预测所需的关键基函数信息：节点序列、样条 degree、惩罚 order，以及当 `s()` 等公式别名解析到底层实现时所使用的具体 basis smoother。
- JSON 模型 artifact 会在训练数组之外持久化紧凑的平滑项 metadata；默认仍不保存训练数据，但保留足够 schema 信息来重建已支持的平滑预测设计矩阵块。
- 设计矩阵 schema 现在为需要平滑 metadata 的参数嵌入 `smooth_basis_metadata`，便于 schema 消费方审计 serialized artifact 是否具备 schema-safe smooth prediction 能力。
- 预测路径同时支持运行时的 `SmoothDesignInfo` 对象和 JSON load 后的平滑 metadata；缺少必需 metadata 或变量时会抛出 `PredictionSchemaError`。
- 新增回归测试验证 `pb(x)` JSON save/load 后的预测 roundtrip，满足第 1 月验收标准中的 `rtol <= 1e-7` 容差。

## 当前边界

- 目前已验证 `pb`/`ps` B-spline basis 重建的 schema-safe smooth prediction。
- 其他 smoother 类别在对应 basis metadata 和预测重建路径完成验证前，仍通过明确的结构化错误保护。
- JSON artifact 默认仍不保存完整训练数据；预测依赖 schema metadata，而不是序列化训练数组。

## 下一步

1. 将 schema-safe smooth prediction 测试扩展到 `s(x, smoother="ps")` 别名和多参数公式。
2. 增加缺少 knots、缺少平滑变量、以及不支持 smoother 的负向测试。
3. 在所有不支持 smoother 边界枚举完成后，将紧凑平滑 metadata 形状提升到更完整的 artifact schema 文档中。
