# OmniLSS 六个月每周实施 Checklist（2026-05-17）

[English version](six-month-weekly-implementation-checklist-2026-05-17.md)

本 checklist 将[六个月执行计划](six-month-execution-plan-2026-05-17_cn.md)拆解为第 1 周到第 26 周的顺序实施节奏。开发应按顺序推进：完成或明确延期当前周任务后，再进入后续周的生产实现。

| 周次 | 日期 | 主要工作流 | 实现门禁 | 状态 |
|---|---|---|---|---|
| 第 1 周 | 2026-05-17 至 2026-05-23 | D1/D2 可信 artifact 与 parser 安全 | Schema v2、默认训练数据脱敏、结构化预测错误、稳健 smooth/tensor 参数解析、artifact validator | 核心生产路径已完成；继续跟进 inline review 修复 |
| 第 2 周 | 2026-05-24 至 2026-05-30 | D1/D2 预测加固 | 审计 legacy prediction 入口；补充公开 artifact-schema 示例 | 进行中；已添加 artifact-schema 文档、validator CLI、legacy/indirect prediction 入口 schema-safe 审计、plot/report wrapper 传播测试、公开 error-envelope 示例和 gRPC prediction error envelope；HTTP metadata error envelope、POST payload-limit gate 和 prototype structured event hook 已添加 |
| 第 3 周 | 2026-05-31 至 2026-06-06 | D3 能力门禁 | Runtime capability matrix 与文档和严格路由完全对齐 | 待开始 |
| 第 4 周 | 2026-06-07 至 2026-06-16 | 第 1 月 release gate | Core trust checkpoint 和可复现实验证据 | 待开始 |
| 第 5 周 | 2026-06-17 至 2026-06-23 | D4 验证矩阵 | 优先 family validation schema 和首批 JSON 输出 | 待开始 |
| 第 6 周 | 2026-06-24 至 2026-06-30 | D4 验证矩阵 | 首批核心 family 的 density/CDF/quantile 检查 | 待开始 |
| 第 7 周 | 2026-07-01 至 2026-07-07 | D5 R/Python 一致性 | 稳定 optional R bridge report harness | 待开始 |
| 第 8 周 | 2026-07-08 至 2026-07-16 | D6 学术指标 | PIT/CRPS/coverage benchmark 输出 | 待开始 |
| 第 9 周 | 2026-07-17 至 2026-07-23 | D7 安全 API | 鉴权边界和错误 envelope 设计 | 待开始 |
| 第 10 周 | 2026-07-24 至 2026-07-30 | D7 job runtime | 异步 fit job 生命周期和状态 API | 待开始 |
| 第 11 周 | 2026-07-31 至 2026-08-06 | D8 模型注册表 | 持久化模型注册表和 artifact 保留策略 | 待开始 |
| 第 12 周 | 2026-08-07 至 2026-08-16 | D9 资源限制 | 带 quota 和 observability 的服务 MVP checkpoint | 待开始 |
| 第 13 周 | 2026-08-17 至 2026-08-23 | 垂直工作流 | 保险工作流 data contract 和模板 | 待开始 |
| 第 14 周 | 2026-08-24 至 2026-08-30 | 垂直工作流 | 频率/严重度/零膨胀模型模板 | 待开始 |
| 第 15 周 | 2026-08-31 至 2026-09-06 | 校准报告 | 分位数/风险区间报告模板 | 待开始 |
| 第 16 周 | 2026-09-07 至 2026-09-16 | Pilot readiness | 垂直 demo 和 pilot-ready checkpoint | 待开始 |
| 第 17 周 | 2026-09-17 至 2026-09-23 | 性能 backend | Profiling baseline 和优化目标 | 待开始 |
| 第 18 周 | 2026-09-24 至 2026-09-30 | 可观测性 | 服务 metrics/logging/tracing 加固 | 待开始 |
| 第 19 周 | 2026-10-01 至 2026-10-07 | 部署自动化 | 可重复本地/container 部署路径 | 待开始 |
| 第 20 周 | 2026-10-08 至 2026-10-16 | Paid beta readiness | 性能报告和 beta readiness gate | 待开始 |
| 第 21 周 | 2026-10-17 至 2026-10-23 | Release candidate | RC blocker list 和文档冻结范围 | 待开始 |
| 第 22 周 | 2026-10-24 至 2026-10-30 | 支持 playbook | 支持边界、升级流程和 SLA 草案 | 待开始 |
| 第 23 周 | 2026-10-31 至 2026-11-06 | Launch materials | 公开教程、benchmark 摘要、webinar 素材 | 待开始 |
| 第 24 周 | 2026-11-07 至 2026-11-13 | 商业推进 | Pilot/LOI package 和采购材料 | 待开始 |
| 第 25 周 | 2026-11-14 至 2026-11-16 | RC final validation | Release candidate 证据包完成 | 待开始 |
| 第 26 周 | 2026-11-17 | 六个月 checkpoint | Go/no-go 决策、launch notes 和下一阶段计划 | 待开始 |

## 当前第 1 周证据

- JSON artifact 现在提供 validator，可检查 archive 结构、schema 版本、参数 schema 覆盖、系数/schema 一致性、smooth metadata 可用性，以及训练数据包含 warning。
- 结构化预测错误提供稳定的机器可读字段，便于客户端路由。
- 公式 parser 加固已覆盖嵌套括号参数和带逗号的引号字符串。
- 第 2 周工作仅在第 1 周核心门禁完成后启动：公开 artifact-schema 示例和 validator CLI 已可用；legacy、间接 prediction、scoring 与 validation-wrapper 入口现在会复用或传播 schema-safe 预测错误，gRPC prediction failure 也会保留结构化 error envelope；详见 [legacy prediction 入口审计进展](month1-legacy-prediction-entrypoint-audit-2026-05-18_cn.md) 与 [服务端 prediction 错误 envelope 进展](month1-service-prediction-error-envelope-2026-05-18_cn.md)。
