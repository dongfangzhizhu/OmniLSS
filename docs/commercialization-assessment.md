# OmniLSS 商业化评估（Phase 8.4）

> 本文档从审计商业视角出发，将已验证能力映射为产品化路线。所有能力宣称
> 均以开源核心（`omnilss/`）中的 R 一致性证据、golden artifact 与能力矩阵
> 为支撑（见 `docs/architecture/core-pro-boundary.md` 的边界规则）。
>
> ⚠️ **P0-2 措辞校准**：本文的价值主张只针对「已由 validated 证据覆盖」的
> 范围，不代表全库与 R `gamlss` 数值等价，也不是完成率声明。当前逐族/逐符号
> 证据层级与范围以可重生成工件 `docs/development/r-symbol-coverage.json`
> （`stats.per_layer`：surface matched ≠ 数值等价 ≠ validated parity）与
> `docs/development/symbol-coverage-diff.json` 为准，本文不写死计数。
> **`gamlss2` 为明确 OUT OF SCOPE**（`tools/coverage_scope.py`），不计入 1.0 交付目标。

## 1. 价值主张

**分布回归（Distributional Regression）全栈**：对 Release 1 已由 R 一致性 /
validated 证据覆盖的八核心分布族（NO/GA/PO/BI/NBI/BE/ZAGA/GU）、三算法
（RS/CG/MIXED）、平滑器（pb/ps/cs）与预测，提供可规模化、可商业化的
Python/JAX 实现；数值对齐声明仅限上述已验证范围，非全库 blanket 等价，
具体覆盖以能力矩阵与 `docs/development/r-symbol-coverage.json` 为准。

## 2. 产品分层

| 层 | 产品 | 渠道 | 收入模式 |
|---|---|---|---|
| L1/L4 | 开源核心（GPL-3.0） | PyPI/GitHub | 生态与信任 |
| L3 | 托管 API（gRPC/HTTP） | 商业通道（omnilss-pro） | 按用量订阅 |
| AutoML | 候选族自动选择 + bootstrap 区间 | 商业层附加 | 高级版 |

## 3. 变现矩阵

| 能力 | 开源（L1/L4） | 商业（L3/AutoML） | 定价信号 |
|---|---|---|---|
| 八核心族拟合/预测 | ✅ | ✅（托管） | 基础 |
| 三算法 + 平滑 | ✅ | ✅（托管） | 基础 |
| gRPC/HTTP 服务 | — | ✅ | 按调用量 |
| AutoML 族选择 | — | ✅ | 高级版 |
| 托管编排/弹性/治理 | — | ✅ | 企业版 |
| 认证/DoS/脱敏/序列化安全 | — | ✅（服务层） | 合规必需 |

## 4. SLA 与运营

- **SLA 目标**：托管 API 99.9% 可用性（多实例 + 健康端点 `Health`/`Version`
  已就绪）；请求级超时（客户端默认 60s）与资源上限（AutoML 治理、DoS 上限）
  已内建。
- **GPU 集群方案**：Phase 7 性能专项完成后，RS_JAX 后端可在 GPU 上提供
  大样本（n ≥ 交叉阈值）加速；当前 CPU 路径为默认。

## 5. 风险与依赖

| 风险 | 缓解 |
|---|---|
| 非核心族拟合差异（14 族例外清单） | 公开文档化，列入拟合引擎路线图 |
| JIT 模块级重构待专项 | 性能专项（GPU/CI 环境） |
| 参考项目（ondil/gamlss-python）不修改 | 仅对齐设计，独立开发 |
| GPL 兼容 | Open-Core 边界规则：商业层不复制核心内部 |

## 6. 结论

Release 1 的**验证体系（R 一致性 + golden artifact + 门禁）是商业化的信任
基础**；托管层安全底线（认证/DoS/脱敏/序列化/TLS 配置）已就绪。商业化
路线：开源核心建立信任 → 托管 API 按用量变现 → AutoML 与编排构成高级版。
