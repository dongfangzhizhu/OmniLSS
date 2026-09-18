# ZIBB 完整概率与采样交付

新增 zibb.py，复用 BB 核实现零膨胀密度、CDF、生存概率、分位数及独立
混合采样。零质量为 nu+(1-nu)*P_BB(0)，正整数质量乘以1-nu。
密度与拟合共享内核，修正旧实现将负数当零点的问题。

保留默认 bd=10；原始及兼容构造器均支持 ZIBB(bd=...)，核函数显式接收 bd，
不新增可估计参数。p/q 支持广播与 log/upper-tail，q 在有限支持上二分
搜索首个交叉点，不使用 R qZIBB 中固定1e-7概率偏移。
参数有效域沿用 BB，增加0<nu<1；bd=0 正确退化为零。
概率端点采用数学契约，非法参数返回NaN；采样沿用rBB样本数校验。

修复旧 R 测试遗漏 bd、观测超过支持域的问题，实际 d/p/q 对照通过。
测试包含归一化、CDF增量、分位数相邻交叉点、日志上尾、退化边界、
JIT密度、广播及10000样本的质量频率检验。

项目 .venv 实测命令：
`python -m pytest omnilss/tests/test_zibb_contract.py omnilss/tests/test_r_consistency_zibb.py omnilss/tests/test_bb_sampling_contract.py omnilss/tests/test_family_capabilities.py omnilss/tests/test_generate_capability_matrix_tool.py omnilss/tests/test_all_families_dpqr_scan.py -q`
结果 **35 passed**，能力快照已运行原生成器校验。

范围：p/q 使用CPU SciPy，未验证完整拟合、序列化固定bd或GPU性能。
R端点异常与固定概率偏移不作为数学契约复制。

下一项建议：实现 ZABB 正计数条件分布的完整DPQR，重点处理
1-P_BB(0)归一化、小正质量、bd=0无法定义的正计数分布及R对照。
