# 交叉导数验证报告（Week 1 Day 5）

> 日期：2026-05-19  
> English version: [CROSS_DERIVATIVES_VERIFICATION_2026_05_19.md](./CROSS_DERIVATIVES_VERIFICATION_2026_05_19.md)

## 范围

本文档验证新引入的跨参数导数基础设施：

- `omnilss/src/omnilss/derivatives/cross_derivatives.py`
- 通用 AD mixed-Hessian 工具 `cross_hessian(...)`
- OmniLSS 分布族 eta 尺度 mixed 导数 `cross_hessian_from_family(...)`

## 验证项

1. **分布族结构验证（NO / GA / WEI）**
   - `(param_i, param_j)` 结果 shape 正确
   - 有限值检查
   - Hessian 交叉项对称性检查

2. **通用 AD 正确性验证**
   - 解析二次函数 (`ll_i = x_i^2 + 3 x_i y_i + 2 y_i^2`) 的 Hessian 精确值对照
   - 通过 `jax.test_util.check_grads` 进行二阶有限差分梯度校验

## 结果

- Week 1 导数单元测试在本地全部通过。
- 解析二次函数案例与预期 Hessian 精确值一致。
- 二阶梯度有限差分校验（`order=2`）通过。

## 与 Week 1 验收映射

- 交叉导数基础设施：**已完成**。
- NO 分布及配套分布的初步数值验证：**已完成（本地 AD 校验）**。
- 与 R 内部导数（`d2ldmds`）直接对照：待后续接入可直接调用导数的 R bridge 后完成。

## 执行命令

```bash
python -m pytest omnilss/tests/test_cross_derivatives.py -q
```
