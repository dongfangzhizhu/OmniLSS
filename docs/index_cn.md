# 中文版本：OmniLSS 文档

[English version](index.md)

OmniLSS 是基于 Python/JAX 的广义加性位置、尺度与形状模型（GAMLSS）实现，完整对齐
R `gamlss` 核心生态（11 个范围内 R 包 1,036/1,036 符号 100% 表面覆盖、0 未验证符号、
22 个生产准入分布族）。

## 从这里开始

- [快速上手教程](tutorials/quickstart_cn.md)：安装、`uv` 环境管理、模型拟合与 22 个生产准入族。
- [分布族与能力门禁](tutorials/tutorial-families_cn.md)：146 个分布族与 fail-closed 准入体系。
- [拟合算法（RS / CG / MIXED / HYBRID / RS_JAX）](tutorials/tutorial-fitting-algorithms_cn.md)
- [平滑器与随机效应](tutorials/tutorial-smoothers_cn.md)：`pb`、`ps`、`cs`、`cy`、`lo`、`re`。
- [预测与序列化](tutorials/tutorial-prediction_cn.md)：`predict_params`、`predict_quantiles`、JSON 往返。
- [模型选择与诊断](tutorials/tutorial-model-selection_cn.md)：GAIC、LR/VC 检验、分位残差与蠕虫图。
- [R 一致性与 Golden 工件体系](tutorials/tutorial-r-consistency_cn.md)
- [gRPC 与 HTTP 托管服务](tutorials/tutorial-grpc-serving_cn.md)
