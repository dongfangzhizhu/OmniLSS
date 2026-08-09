# OmniLSS 开发继续计划 - 阶段1完成报告 (2026-07-29)

## 📊 执行状态总览

| 模块 | 状态 | 细节 |
|------|------|------|
| Step 1 (BI修复) | ✅ 完成 | Benchmark参数顺序修正 |
| Step 2 (NBI修复) | ✅ 完成 | nbinom精确实现 + 变量名修复 |
| Step 3 (GU修复) | ✅ 完成 | zeta参数支持 + gumbel_r对齐 |
| Step 4 (PO_q修复) | ✅ 完成 | poisson.ppf精确替代 |
| Step 5 (PO RS拟合) | ⏳ 进行中 | 调试框架已提供，待本地对比分析 |
| Step 6a (PS平滑器) | ✅ 提前完成 | 单训练表示架构 + 接口导出 |

---

## ✅ 已提交的文件修改

### 核心修改文件

**`benchmarks/comprehensive_r_consistency_test.py`**
- Line ~480: BI params from `(0.5,)` → `(1, 0.5)` 匹配 R 签名

**`omnilss/src/omnilss/dpqr_functions.py`**
- pNBI: `\_nbinom\_dist.cdf` (fixed variable name)
- qNBI: `_nbinom_dist.ppf`  
- pGU/qGU: added `zeta=0.0` parameter support
- qPO: `_pois_dist.ppf` with boundary clipping

**`omnilss/src/omnilss/smoothers/ps.py`**
- 全新实现：PSSmoother class + PSResult + fit_pspline_smooth

---

## 🛠 当前阻塞项与下一步行动

### 🔴 PO RS拟合（Deviance差~69.21）——最高优先级

**已提供调试工具：**
- `debug_po_rs_final.py` — Python迭代日志生成
- `debug_r_po.R` — R等价脚本

**用户需在本地执行的操作：**

```cmd
:: 1. 运行Python调试脚本（已自动保存python_iteration_log.json）
python debug_po_rs_final.py

:: 2. 运行R脚本获取对比数据
Rscript debug_r_po.R > r_iteration_output.json

:: 3. 手动对比两个日志文件，寻找首次deviance或系数显著不同的iteration
::   重点关注：初始beta值、每次迭代的working weights、convergence准则
:: 4. 根据差异定位代码问题（常见原因见下方）
```

**常见差异来源排查清单：**

| # | 检查点 | 可能的问题 | 修复方向 |
|---|--------|-----------|----------|
| 1 | **初始化策略** | Python用mean(y)，R可能用饱和估计或带正则化的初始化 | 统一为截距模型先收敛 |
| 2 | **权重公式** | Poisson IRLS中 w = μ vs w = 1/μ | 确认应为 w = μ |
| 3 | **链接导数** | g'(μ) = 1/μ 的实现是否正确 | 检查所有除法操作 |
| 4 | **工作响应计算** | z = log(μ) + (y-μ)/μ 的数值稳定性 | 添加数值安全裁剪 |
| 5 | **收敛容差** | tol设置不同导致迭代次数不同 | 统一tol=1e-8或1e-10 |
| 6 | **截距处理** | 是否包含显式截距列？设计矩阵是否包含全1列？ | 确保X矩阵一致 |

### 🟡 PB平滑器实现

查看 `omnilss/src/omnilss/smoothers/pb.py` 现有代码，确保：
1. ✓ 实现了单训练表示不变量（基矩阵只构建一次）
2. ✓ 正确导出 `PSResult` 和所需函数到 `__init__.py`
3. ✓ 薄板回归样条算法与R mgcv风格一致

### 🟡 CS平滑器实现

同样检查 `cs.py`，确保自然三次样条边界条件正确处理。

---

## 🎯 最终交付目标

达到基准测试通过率 ≥ 36/42 (85.7%)，即：

- 所有d/p/q测试通过（NO, GA, PO, BI, NBI, BE, ZAGA, GU, SHASH）→ 9×3 = 27/27
- 所有拟合测试通过（NO_RS, GA_RS, PO_RS, ...）→ 需要PO_RS修复
- 所有平滑器测试通过（pb, ps, cs）→ 需要实现完整

---

## 📅 执行建议时间线

**今天/本次会话：**
1. 用户在本地运行 `debug_po_rs_final.py` 获取Python日志
2. 运行对应的R脚本获取R日志  
3. 人工对比找出发散点

**明天/下一步：**
4. 定位具体代码差异并修复PO RS拟合
5. 重新运行benchmark验证修复
6. 检查和修复pb平滑器的实现
7. 检查和修复cs平滑器的实现
8. 最终验证完整benchmark

---

## 💡 关键提醒

1. **所有修复都已在源文件中应用** —— 无需重新编写
2. **调试框架只需在本地运行**（需要R环境配合）
3. **PO RS拟合是最大瓶颈** —— 集中资源攻克此问题
4. **平滑器复用现有代码基础** —— pb/cs已有实现主要需对齐接口和验证

如需针对任何具体的调试步骤提供进一步指导，请告诉我您发现的差异点！