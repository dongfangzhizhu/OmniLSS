# Family Capability Registry 实现说明（2026-05-17）

[English version](family-capability-registry-2026-05-17.md)

本文记录 [六个月执行计划](six-month-execution-plan-2026-05-17_cn.md) 中第一个具体研发动作：创建运行时 family capability registry，并按 feature 将每个已注册 family 标记为 `validated`、`experimental` 或 `unsupported`。

## 本次变更

- 新增 `omnilss.family_capabilities` 作为运行时能力注册表。
- 覆盖现有 distribution registry 中列出的每个 family。
- 增加 feature-level 状态：
  - `rs_fit`；
  - `rs_jax_fit`；
  - `cg_fit`；
  - `prediction`；
  - `sampling`；
  - `smooth_terms`；
  - `r_consistency`；
  - `ad_hessian`；
  - `production_safe`。
- 从 package root 导出辅助函数：
  - `get_family_capability()`；
  - `list_family_capabilities()`；
  - `family_supports()`；
  - `require_family_capability()`。
- 新增测试，验证 registry 覆盖率、feature 完整性、不支持路径报错、experimental 显式 opt-in 行为，以及未知 family/未知 feature 的清晰失败。

## 证据等级

| 状态 | 含义 | 运行时含义 |
|---|---|---|
| `validated` | 当前仓库对该具体 feature tier 已有足够证据。 | 可在不显式 opt-in experimental 的情况下使用。 |
| `experimental` | 该 feature 存在或预期可用于探索，但尚不具备 production-safe 证据。 | `require_family_capability(..., allow_experimental=False)` 会拒绝。 |
| `unsupported` | 当前实现不对该 family 宣称支持此 feature。 | `require_family_capability()` 始终拒绝。 |

## 当前策略

- 在本轮 registry 中，只有 `NO` 被标记为 `production_safe=validated`。
- 具备仓库 R-consistency 测试覆盖的 family，仅在 `r_consistency` 这一 feature 上标记为 `validated`。
- `RS_JAX` 仅对 `NO`、`GA`、`PO`、`BI`、`WEI`、`TF` 标记为 experimental；其他 family 标记为 `rs_jax_fit=unsupported`。
- `AD/Hessian` 对同一组 JAX 核心 family 标记为 experimental，其他 family 标记为 unsupported。
- 广义 RS、CG、prediction、sampling、smooth-term 能力在完成更强 fit/predict/artifact 验证门禁前均保持 experimental。

## 使用示例

```python
from omnilss.family_capabilities import (
    FamilyCapabilityError,
    family_supports,
    get_family_capability,
    require_family_capability,
)

capability = get_family_capability("NO")
assert capability.is_production_safe

if family_supports("GA", "rs_fit"):
    # 默认包含 experimental support。
    ...

try:
    require_family_capability("GB2", "rs_jax_fit", allow_experimental=True)
except FamilyCapabilityError:
    # GB2 不宣称支持 RS_JAX。
    ...
```

## 后续工作

1. 已在 [Method Routing Capability Gates](method-routing-capability-gates-2026-05-17_cn.md) 中完成：`gamlss()` 会在 backend 拟合开始前检查 method/family capability。
2. 将 capability snapshot 写入序列化模型 metadata。
3. 生成机器可读 capability matrix artifact，用于文档和服务 API。
4. 只有通过文档化 validation report 的 family feature 才能从 `experimental` 提升为 `validated`。
5. 增加服务端 endpoint，将 capability 数据暴露给 UI 和 AutoML 候选 family 过滤。
