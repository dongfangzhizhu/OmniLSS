# MIXED/HYBRID 权重和参数公式传递

公开 `gamlss(method="MIXED")` 和 `gamlss(method="HYBRID")` 均会把 `weights` 与 `parameter_formulas` 传到实际后端。公式 mapping 会先复制，显式 mapping 优先于旧的 `nu_formula`/`tau_formula`，调用者对象不会被修改。

自 2026-09-24 起，两种语义明确分离：

- `MIXED` 对齐 R `mixed(n1, n2)`：先运行 RS，再把全部可估参数的 coefficient、linear predictor 和 fitted value 校验后传给 eta-scale CG。公开参数为 `mixed_n1`（默认 1）和 `mixed_n2`（默认 20）。
- `HYBRID` 保留历史自动调度：根据样本量、family 和平滑项选择单一 RS 或 L-BFGS 后端。直接 API 为 `hybrid_fit(..., algorithm="auto")`。

当前 CG 阶段尚不能保留惩罚平滑状态，因此 `MIXED` 遇到任一参数公式中的 smooth term 会 fail-closed；需要平滑时使用 `RS`，或使用会自动留在 RS 的 `HYBRID`。配置 family 若 RS 保存的 eta 无法由其 beta 和 design matrix 重建，也会在 MIXED 阶段边界被拒绝，而不是静默重新初始化。

验证范围包括：加权 NO、sigma 协变量公式、RS→CG 手工流水线等价、零 CG 迭代时完整 RS 状态保持、阶段迭代参数和元数据、HYBRID 旧选择规则，以及 smooth fail-closed。R live parity 仍由单独的 required-R suite 负责；在该门禁完成前，MIXED 保持 experimental capability。
