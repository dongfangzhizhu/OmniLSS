# LG/ZALG 密度契约补齐

- 密度不再将非法 mu/sigma 裁剪为有效值，与 p/q 一致返回 NaN。
- 有效的极小参数保持原值，如 ZALG sigma=1e-30。
- ZALG 复用 LG 密度核；零点使用独立质量，其 mu 导数为零。
- 显式 d 入口绕开通用包装器的数组压平，保留 NumPy 多维广播。
- LG x<1、ZALG 非零且 x<1，以及正无穷均返回零密度；日志密度为负无穷。
- NaN 观测传播；安全分支保证有效参数处 JIT/自动微分可用。

验证命令（项目 .venv）：

```text
python -m pytest omnilss/tests/test_logarithmic_density_contract.py omnilss/tests/test_lg_tail_contract.py omnilss/tests/test_zalg_dpqr_contract.py omnilss/tests/test_r_consistency_zalg.py omnilss/tests/test_family_capabilities.py -q
37 passed
```

新增真实 R LG/ZALG 日志密度对照在正整数支持点执行，容差 1e-10。
当前安装 R 在支持域外的日志返回行为与仓库旧源码不一致，本次不复制
这些边界特例，明确采用 log(0)=-Inf。正实数非整数 x>=1 的公式扩展
沿用现有兼容约定，不代表离散 PMF 的额外支持点。

还验证了归一化、非法参数、极小零质量、多维广播、JIT、解析 sigma
梯度及零点 mu 梯度。未执行全库拟合或 GPU 基准，不作 production-safe 晋级。

下一项建议：实现 GB1 的真实 d/p/q/r，核对 R 的参数化与支持域，
补齐 Beta 变换、边界、广播及实际 R 验证后再更新能力状态。
