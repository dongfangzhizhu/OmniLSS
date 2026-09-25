# MIXED 显式兼容后端平滑能力检查

核查发现历史 joint_lbfgs_fit 包装器实际调用 CG full-Hessian 后端，
fit_cg 接口没有平滑惩罚矩阵输入。仅传入展开后的设计矩阵不能等价于
惩罚平滑拟合。

本轮在两个入口明确拒绝：
- mixed_fit 显式 algorithm='lbfgs'/'l-bfgs'/'cg' 且有效公式包含平滑项；
- 直接 joint_lbfgs_fit 的平滑请求。

检查覆盖mu/sigma以及合并后的nu/tau/参数映射，抛ValueError并提示RS/auto。
保留非平滑兼容路径和自动选择RS行为；未实现CG平滑惩罚，不宣称该能力可用。
直接gamlss(method='CG')的平滑路径尚需单独检查。

项目 .venv：
`python -m pytest omnilss/tests/test_mixed_explicit_smooth.py omnilss/tests/test_mixed_smooth_detection.py omnilss/tests/test_mixed_algorithm.py omnilss/tests/test_mixed_inputs.py -q`
结果 **70 passed**，涵盖后端调用前拒绝、别名、参数映射优先级、
既有真实自动平滑拟合及加权/非平滑回归。本次无R/GPU数学变更或验证。

下一项建议：给直接gamlss(method='CG')建立相同能力契约，检查full_hessian
与irls_cross后端对平滑惩罚的实际支持，并统一错误或实现完整惩罚路径。
