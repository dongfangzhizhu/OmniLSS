# 直接 CG 平滑能力准入

核查 `fitting_cg.fit_cg` 接口只接收设计矩阵，没有平滑惩罚状态；直接
`gamlss(method="CG")` 此前仍会构造平滑设计并进入无惩罚 CG，结果语义错误。

本次在进入参数公式解析前，对 mu、sigma 和 parameter_formulas 中的
nu/tau 统一检测平滑调用，发现后 fail-closed，消息明确建议 RS/auto。
无平滑的直接 CG 路径保持可用。MIXED 显式兼容后端已有同等检查。

项目 `.venv`：

```text
python -m pytest omnilss/tests/test_cg_smooth_admission.py omnilss/tests/test_mixed_explicit_smooth.py omnilss/tests/test_smoother_admission.py -q
46 passed
```

覆盖四个参数位置的 s/pb/ps/cs、无平滑直接 CG 可用性、MIXED 显式兼容
后端拒绝及已有 admission 回归。测试中的非平滑 CG 仅运行两轮，出现
不收敛警告是预期的预算设置，不代表 CG 数值准入。尚未实现 CG 平滑惩罚，
也未执行 R/GPU 验证。

下一项建议：若需要 CG 平滑支持，应设计并实现惩罚矩阵/EDF/lambda 状态
进入 full-Hessian 和 IRLS-CG 的完整链路；否则继续加强直接 CG 的能力矩阵
和服务层 fail-closed 报告，避免将该路线误标为可用平滑后端。
