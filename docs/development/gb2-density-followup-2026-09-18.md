# GB2 拟合密度稳定性补齐

## 本轮交付

- `distributions_b8._gb2_log_pdf` 复用公开 `dGB2(log=True)`，消除拟合与预测的重复公式。
- 对数密度采用 `-nu*softplus(-z)-tau*softplus(z)`，其中
  `z=sigma*(log(x)-log(mu))`，避免比值溢出、指数溢出和大项相减。
- 不裁剪有效参数；无效参数返回 NaN，非正观测与正无穷密度为零
  （对数密度为负无穷），NaN 观测保持 NaN。
- 新增极端观测、JIT、自动微分、边界及实时 R 密度对照测试。

## 验证

使用项目 `.venv/Scripts/python.exe`：

```text
python -m pytest omnilss/tests/test_gb2_density_contract.py omnilss/tests/test_gb2_dpqr.py omnilss/tests/test_family_capabilities.py -q
30 passed
```

其中 3 项参数化用例实际调用 R `dGB2(log=TRUE)`，涵盖
`sigma=0.05,tau=150`、`sigma=20` 和负 `sigma=-2`。
密度与 R 的容差为 `rtol=atol=1e-10`。本次没有执行全家族拟合或 GPU 基准。

## 尚未完成

上一轮的“GB2 全部完成”说明范围过宽。当前 `pGB2/qGB2` 仍须处理负
`sigma` 的概率方向、极端尾部和参数校验；分位数对概率及 Beta 变量的
epsilon 裁剪也需要移除。能力扫描目前仍可能将 `qGB2` 的条件 NaN
误判为 placeholder，不能用迁移队列中的手工标记代替运行时能力状态。

下一项建议：完整补齐 GB2 CDF/分位数概率契约与实时 R 对照后，再修正
代码派生能力分类及迁移报告。密度验证不代表拟合准入或 production-safe。
