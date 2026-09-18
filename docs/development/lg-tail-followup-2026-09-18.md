# LG 概率与分位数补齐

## 实现

- LG 与 ZALG 共用无限尾部积分、概率变换和动态分位数搜索内核。
- LG 的 CDF 按 floor 取整，修复原先 round 对非整数输入提前包含质量的问题。
- 删除固定 10000 搜索上限和概率 epsilon 裁剪；显式 max_value 不足返回 NaN。
- 支持标量及 NumPy 多维广播，处理无穷、NaN、非法参数和概率端点。
- qLG(0)=0 与 R 约定一致；有限但极小的日志下尾概率映射为首个支持点 1。
- ZALG 对 sigma 的严格 (0,1) 校验保留；共享内核以 sigma=0 表示 LG。
- 采样复用动态反演，能够生成超过 10000 的值。

## 验证

```text
.venv/Scripts/python.exe -m pytest omnilss/tests/test_lg_tail_contract.py omnilss/tests/test_zalg_dpqr_contract.py omnilss/tests/test_r_consistency_zalg.py omnilss/tests/test_family_capabilities.py -q
31 passed
```

LG 新增测试实际调用 R pLG/qLG，覆盖上下尾；ZALG 原有 R d/p/q 对照保持通过。
极端 mu=0.99999 使用独立 50000 项求和校验，并测试相邻整数分位数交叉点、
日志上尾 -1000、概率端点、显式上限和固定种子采样频率。

## 范围

这是 CPU 概率与采样路径，不支持 JAX tracing；未修改拟合密度，也未执行
GPU 基准或全库回归。数值积分仍受浮点精度及积分器误差限制。
标量概率返回零维数组；不再使用旧版一维包装或 R 式循环回收数组参数。

下一项建议：统一 LG/ZALG 密度参数校验（当前密度仍裁剪参数，而 p/q
返回 NaN），增加支持域与拟合内核一致性测试；之后再推进 GB1 placeholder。
