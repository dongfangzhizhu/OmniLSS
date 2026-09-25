# ZABB 正计数归一化与完整 DPQR

## 实现

- ZABB(bd=...) 及兼容构造器接入真实 d/p/q/r，默认 bd=10 不变。
- 正计数分布为 BB 条件于 X>0；用正计数日志质量的 logsumexp 归一化，
  不做 1-P_BB(0) 的直接相减。mu=1e-20 的归一化回归通过。
- 零质量为 nu，密度和拟合共享内核，负值不再被当成零点。
- 概率通过有限支持日志表计算上下尾；分位数定位首个 CDF 交叉点，
  不复制 R 的固定1e-7偏移；采样使用JAX key驱动逆变换。
- bd 必须为正静态整数：bd=0 无正计数条件分布，明确抛 ValueError。
- 无效分布参数返回 NaN，域外密度为0，概率端点按数学契约处理。
- 固定 bd 接口不新增估计参数；支持参数广播，密度支持 JIT/AD。

## 验证

项目 .venv：

```text
python -m pytest omnilss/tests/test_zabb_contract.py omnilss/tests/test_r_consistency_zabb.py omnilss/tests/test_zibb_contract.py omnilss/tests/test_family_capabilities.py omnilss/tests/test_generate_capability_matrix_tool.py omnilss/tests/test_all_families_dpqr_scan.py -q
33 passed
```

修正原 R 测试未传 bd、观测越界问题，实际 R d/p/q 比较通过。测试还覆盖
归一化、CDF增量、零点质量、bd=1、极小正质量、非法参数、广播、日志尾部、
分位数交叉点、JIT及梯度和5000样本质量频率。能力快照已重生成。

## 限制与下一项

每组参数枚举1..bd，复杂度随分母和广播批量增长，未完成大bd性能验收；
分位数/采样为CPU路径，不声明GPU支持。未完成全族拟合或序列化固定bd验收。
有限精度CDF累计在极端上尾仍可能提前舍入到1，后续可改用生存概率搜索。

下一项建议：验证并补齐 ZABB/ZIBB 模型保存加载对自定义 bd 的恢复。
目前固定bd由构造器闭包保存，若加载只按家族名重建，会恢复默认10，
需要通过实际roundtrip测试确认并按模型状态契约修复。
