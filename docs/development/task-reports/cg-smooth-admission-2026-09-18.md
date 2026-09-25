# 直接 CG 平滑准入契约

核查直接 `gamlss(method="CG")`：full-Hessian 与 IRLS-cross 的 fit_cg
接口都没有平滑惩罚矩阵，因此不能把平滑设计矩阵当普通线性列直接拟合。
入口已有统一拒绝逻辑，本轮补齐防回退契约测试，覆盖：

- cg_backend: full_hessian、irls_cross、CG_FULL_HESSIAN、CG_IRLS_CROSS；
- mu、sigma、nu、tau 公式中的 s/pb/ps/cs/lo；
- method="CG" 及 algorithm="CG" 兼容写法；
- parameter_formulas 覆盖旧 nu_formula 时，按有效映射检测。

不支持请求在构建设计矩阵或后端调用前抛出：
`method='CG' does not support penalized smooth terms; use method='RS' or method='auto'.`
完整惩罚 CG 尚未实现，未将其标记为支持。

项目 .venv：
`python -m pytest omnilss/tests/test_cg_smooth_admission.py omnilss/tests/test_mixed_explicit_smooth.py omnilss/tests/test_smoother_admission.py -q`
结果 **59 passed**。本轮仅能力准入与错误契约测试，无R/GPU数值变更。

下一项建议：若要实现完整 CG 平滑，需要把 SmoothFitInfo 的 penalty、lambda、
EDF 状态接入 full-Hessian/IRLS-cross 目标和线性系统；否则保持当前 fail-closed。
