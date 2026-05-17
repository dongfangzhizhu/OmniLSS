# 第 1 月公式安全进展（2026-05-18）

> English version: [month1-formula-safety-progress-2026-05-18.md](month1-formula-safety-progress-2026-05-18.md)
>
> 父计划：[six-month-execution-plan-2026-05-17_cn.md](six-month-execution-plan-2026-05-17_cn.md)

## 范围

本文记录第 1 月 / 工作流 D2：**公式安全与解析器加固** 的进展。

## 已实现进展

- 数值公式表达式通过严格的 AST 白名单求值。
- 属性访问会被拒绝，包括 `np.<function>` 调用；应使用直接白名单函数，例如用 `sqrt(x)` 替代 `np.sqrt(x)`。
- 下标访问、lambda、推导式、布尔/比较表达式、集合字面量以及过深表达式会在求值前被拒绝。
- 恶意或不支持的表达式会以确定性的 `ValueError` 失败，不执行任意 Python 代码。
- 回归测试覆盖了直接白名单函数，以及属性调用、下标访问、lambda、推导式和非白名单 import 的拒绝路径。

## D2 剩余工作

- 将 prediction 以及 smooth/tensor 参数处理中剩余的临时字符串拆分替换为共享的受限解析器工具。
- 增加结构化公式错误类型，包含 parameter、term 和 reason 字段。
- 将安全覆盖扩展到所有公开拟合别名和服务输入使用的公式路径。
