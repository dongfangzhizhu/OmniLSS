# Core / Pro 边界

[English version](core-pro-boundary.md)

## 目标

明确 Open-Core 边界：开源**核心**（OmniLSS）与商业 **Pro** 层（omnilss-pro）
各自的范围，确保 GPL 核心内部从不被复制进专有模块，Pro 层仅通过公开的
进程级边界与核心交互。

## 核心 — 开源（GPL-3.0，`omnilss/`）

**L1 统计引擎**（Release 1 范围）：
- 分布族（d/p/q/r）与分布注册表
- 拟合算法（RS / CG / MIXED）与数值地基
- 平滑器（pb / ps / cs）与预测（saved-state newdata 重建）
- 模型选择 / 诊断 / 序列化（JSON + pickle）

**L4 验证与交付**：
- R 一致性测试套件与固定 R golden artifact
- 能力证据矩阵与 family capability 门禁
- 容差审计、`--require-r` fail-closed 阶段门禁
- CI 工作流与发布工具

## Pro — 商业层（`omnilss-pro/`）

**L3 托管服务**：
- gRPC / HTTP 服务层与客户端 SDK（`omnilss_pro.client`）
- AutoML（候选族排序、bootstrap 区间）——资源治理
- 托管编排 / 部署工具 / 企业适配器

## 边界规则

1. **禁止复制**：Pro 模块从不导入 GPL 核心的*内部*；仅通过公开 API（如
   gRPC batch-fit 边界）或核心包的文档化 Python API 交互。
2. **进程边界**：托管编排通过 gRPC/HTTP 调用核心；Pro 包不得
   `import omnilss.<私有模块>`。
3. **独立发布通道**：核心以 GPL-3.0 发布；Pro 层通过独立商业通道与版本线。
4. **验证归属核心**：R 一致性证据、golden artifact 与能力矩阵保存在开源
   核心中，保证宣称可审计；Pro 在此基础上叠加资源治理，而非自证正确性。
