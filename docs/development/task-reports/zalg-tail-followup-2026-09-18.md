# ZALG 无限尾部与动态分位数

## 修复

此前 CDF 固定只累加前 10000 项，分位数未找到交叉点时静默返回
10000，导致 mu 接近 1 时概率和采样错误。本次使用无限尾部积分：

令 m=floor(q)+1，
sum(mu^k/k, k=m..inf) = mu^m/m * integral(exp(-u)/(1-mu*exp(-u/m)), 0..inf)。

积分采用 SciPy quad（绝对/相对容差 1e-11），通过 expm1 稳定计算分母；
在对数尺度保留上尾，避免小概率下溢影响分位数比较。CDF 使用 -expm1(logSF)。
分位数先倍增区间，再二分定位首个交叉点。

## API

- `max_value=None` 为默认，最多搜索到 float64 精确整数上限 2**53-1。
- 显式非负整数上限继续可用；搜索失败返回 NaN，不再伪造上限分位数。
- p/q 校验 0<mu<1、0<sigma<1，非法值返回 NaN；支持 NumPy 广播。
- 采样使用修复后的逆变换，无固定 10000 上限。
- p/q/r 是 CPU 路径，不支持 JAX tracing；拟合密度未在本次修改。

## 验证

项目 `.venv/Scripts/python.exe`：

```text
python -m pytest omnilss/tests/test_zalg_dpqr_contract.py omnilss/tests/test_r_consistency_zalg.py omnilss/tests/test_family_capabilities.py -q
27 passed
```

既有实际 R d/p/q 对照通过。极端参数使用独立前 50000 项求和校验 CDF、
相邻整数交叉点校验分位数、log 上尾 -1000 校验和固定种子采样频率检验。
R 常规对照不等于已验证 R 的极端搜索上限行为。

## 下一项

补齐 LG 基础族：目前其 CDF 对输入取 round 而非 floor，分位数仍存在
固定上限和概率裁剪。建议与 ZALG 共享经过验证的尾部核，统一边界，
增加 R 对照。更大规模采样的 CPU 性能优化及极端积分误差诊断仍待后续。
