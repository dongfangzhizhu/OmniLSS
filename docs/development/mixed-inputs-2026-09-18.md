# MIXED 权重和参数公式传递

公开 gamlss(method="MIXED") 此前未传 weights/parameter_formulas，导致
用户请求的加权模型与多参数公式可能被静默忽略。

本次将两个参数沿 mixed_fit 的 RS 和历史 lbfgs 兼容分支完整传递。
兼容包装器最终调用 CG 时也传递它们。新选项为仅关键字参数，不改变旧
位置参数含义。公式字典复制后使用，mapping优先于旧nu_formula/tau_formula。
自动选择检查完整公式集合，nu/tau里的pb/ps/cs也触发RS路径。

项目 .venv：
`python -m pytest omnilss/tests/test_mixed_inputs.py omnilss/tests/test_mixed_algorithm.py omnilss/tests/test_max_iter_contract.py -q`
结果 **51 passed**。

真实NO加权截距模型匹配解析加权均值/标准差；sigma协变量公式实际拟合
与直接RS一致，系数维度验证为2。nu/tau透传及兼容CG入口由调用边界测试
覆盖。此次未验证nu/tau协变量模型的数值收敛或GPU，不宣称R拟合等价。

下一项建议：统一MIXED平滑项识别与核心公式解析器。当前仅识别pb/ps/cs
字符串，s(...)/其它合法表达式可能被错误分配到不支持平滑的后端。
