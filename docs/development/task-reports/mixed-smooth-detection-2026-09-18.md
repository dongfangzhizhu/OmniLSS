# MIXED 平滑项检测统一

把核心拟合的平滑调用检测抽取到 formula_parser.has_smooth_terms。
fitting._has_smooth_terms 保留为兼容入口，MIXED 使用相同检测函数。
识别通用s()、pb/ps/cs及尚未准入的平滑器，允许函数名后的空格或tab。
尚未准入不等于忽略：路由进入RS后仍由标准admission gate明确拒绝。

项目 .venv：
`python -m pytest omnilss/tests/test_mixed_smooth_detection.py omnilss/tests/test_mixed_inputs.py omnilss/tests/test_smoother_admission.py -q`
结果 **64 passed**。

覆盖mu/sigma/nu/tau公式在500样本阈值处的选择，非平滑表达式排除，
实际s(x)与直接RS拟合一致，以及lo(x)标准UnsupportedSmootherError。
检测仍为共享正则，并非完整AST语义分析；不宣称所有自定义表达式均可解析。
本次是CPU路由修复，没有执行R数值或GPU基准。

下一项建议：补齐MIXED显式选择非RS后端时的平滑能力校验。
目前自动选择可避免平滑误路由，但显式algorithm='lbfgs'仍应按后端
真实能力明确处理，不能依赖下游忽略或降级。
