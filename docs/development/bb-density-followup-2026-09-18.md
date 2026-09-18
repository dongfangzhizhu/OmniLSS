# BB 密度与 Binomial 极限统一

- 公开 dBB 与拟合 _bb_log_pdf 共享 JAX 内核，sigma<1e-4 使用 Binomial 分支。
- 参数契约与 p/q 一致：0<mu<1、有限 sigma>0、非负整数 bd。
- 非整数及支持域外观测密度为零，日志密度负无穷；无效参数及 NaN 观测返回 NaN。
- bd=0 为退化零计数；显式 d 入口保留多维广播。
- 不再裁剪参数。未选中的 Beta 分支使用安全参数，保证极小 sigma 的 AD 正常。
- 普通分支使用 Gamma 恒等式，避开当前 JAX betaln 在小形状参数处约1e-6的误差。

项目 .venv 验证：
`python -m pytest omnilss/tests/test_bb_density_contract.py omnilss/tests/test_bb_dpqr.py omnilss/tests/test_family_capabilities.py -q`
结果 **31 passed**。四组 sigma（1e-12、9e-5、1e-4、0.2）实际调用 R dBB；
另比较密度与 CDF 增量、Binomial PMF，验证 JIT/梯度、广播及支持域。

当前 R dBB 在 sigma=1e-4 处有标量/向量分支不一致；Python 统一采用
严格小于阈值，和现有 p/q 一致。R 对 mu 端点及非整数观测的某些特殊
行为不复制，采用上述明确契约。未声称全库、拟合或 GPU 验证通过。

下一项：rBB 仍裁剪非法参数并将 bd=0 改成1，应统一采样校验与退化边界，
补充向量 bd 及零膨胀 BB 衍生族回归。
