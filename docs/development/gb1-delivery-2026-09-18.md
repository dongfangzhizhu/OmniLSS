# GB1 数值内核交付

依据实际安装的 gamlss.dist::dGB1/pGB1/qGB1/GB1 函数核对，替换旧版
“simplified parameterization”骨架。GB1 支持域是 (0,1)，不是 (0,mu)。
mu、sigma 均在 (0,1)，nu、tau 为正；sigma 链接由 log 改为 R 的 logit。

令 a=mu*(sigma^-2-1)，b=(1-mu)*(sigma^-2-1)，t=x^tau，
则 z=t/(nu+(1-nu)*t) 服从 Beta(a,b)。密度含该变换的 Jacobian。
新 gb1.py 提供共享 JAX 密度和 CPU SciPy CDF/逆CDF；采样使用 JAX key
驱动逆变换；family 与 dpqr_functions 均可访问四个函数。

支持 NumPy 广播、上下尾及日志概率；无效参数返回 NaN。域外密度为零、
日志密度为负无穷；CDF 在 q>=1 时为1，分位数端点为0/1。这些数学边界
有意不复制当前 R pGB1 在 q>=1 时返回0以及 qGB1 拒绝概率端点的行为。

验证：test_gb1_contract.py 中三组 nu/tau 参数实际调用 R d/p/q，
容差 rtol=1e-9、atol=1e-11；另验证 Beta 特例、积分归一化、JIT/AD、
广播、尾部、无效参数和采样概率积分变换。状态为 experimental；
没有宣称全族拟合一致、生产准入或 GPU p/q/r 支持。

项目 .venv 实测命令：
`python -m pytest omnilss/tests/test_gb1_contract.py omnilss/tests/test_family_capabilities.py omnilss/tests/test_generate_capability_matrix_tool.py omnilss/tests/test_all_families_dpqr_scan.py -q`
结果 **32 passed**；能力快照由原生成器重生成并通过一致性校验。

下一项建议：补齐 BB 的密度与小 sigma Binomial 极限分支。
目前 BB 的 p/q/r 已有该分支，密度/拟合内核仍需统一，以避免同一族
似然与预测概率不一致。
