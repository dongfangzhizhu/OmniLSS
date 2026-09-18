# BB 采样校验与退化边界

- rBB 与 d/p/q 使用相同参数域：0<mu<1、有限 sigma>0、非负整数 bd。
- 无效元素返回 NaN，不再裁剪成有效参数；有效元素独立产生样本。
- bd=0 正确返回0；参数需广播到 (n,)，不兼容形状明确报错。
- n 为非负静态整数，n=0 返回空数组，负数/布尔/非整数拒绝。
- 小 sigma 的 Binomial 分支及退化分支使用安全的未选中 Beta 参数，
  不再计算巨大的 Beta 形状。随机数显式采用 float64。

项目 .venv 实测：

```text
python -m pytest omnilss/tests/test_bb_sampling_contract.py omnilss/tests/test_bb_density_contract.py omnilss/tests/test_bb_dpqr.py omnilss/tests/test_family_capabilities.py -q
36 passed
```

验证包括混合有效/无效逐样本参数、零 bd、JIT、同 key 可复现、
sigma=1e-300 的 Binomial 直接抽样对照；常规与小 sigma 各20000个样本
按实时 R dBB 的概率质量检验频率（6倍标准误差加有限计数项）。
当前 JAX 对部分次正规数清零，sigma=1e-310 不在本次有效数值保证内。

扩大检查 ZABB/ZIBB 的已有 R 测试时，因未传 bd 而被 R 默认 bd=1
拒绝（测试观测包含0..19），两模块共4个子用例失败。源码进一步确认
这些 family 仍缺 p/q/r 接线，不调用 rBB；不宣称其回归通过。

下一项：实现 ZABB/ZIBB 的完整 DPQR 和显式 bd 契约，修正 R 对照输入，
覆盖零质量、截断归一化和退化边界后单独交付。
