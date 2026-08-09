# OmniLSS 开发继续计划 - 执行状态摘要 (2026-07-28)

## ✅ 已完成的任务

### Step 1: BI族修复
- [x] Benchmark中BI参数顺序从 `(mu,)` → `(bd, mu)` = `(1, 0.5)` 
- [x] BI d/p/q已验证正确（与理论值匹配）

### Step 2: NBI族修复  
- [x] pNBI使用 `_nbinom_dist.cdf` (已修正变量名)
- [x] qNBI使用 `_nbinom_dist.ppf`
- [x] 数值验证：与scipy.nbinom完全一致（diff=0）

### Step 3: GU族修复
- [x] pGU添加zeta参数支持
- [x] qGU添加zeta参数支持
- [x] 数值验证：与gumbel_r完全一致（diff=0）

### Step 4: PO_q修复
- [x] qPO使用 `_pois_dist.ppf`
- [x] 边界裁剪增强（1e-12）
- [x] 数值验证：与scipy.poisson.ppf完全一致（diff=0）

### Step 6a: ps平滑器实现（提前完成）
- [x] 单训练表示架构基础类 `SmootherBase`
- [x] 具体实现 `PSSmoother` (截断幂基)
- [x] 导出接口 `PSResult` + `fit_pspline_smooth`
- [x] 测试通过，与接口兼容

---

## 🎯 下一步：Step 5 & 6b

目前待办按优先级：
1. **Step 5**: 调试PO RS拟合（deviance差69.21）— 需要迭代日志对比
2. **Step 6b**: pb平滑器（薄板样条）— 最复杂的平滑器
3. **Step 6c**: cs平滑器（三次样条）— 自然边界条件

建议先运行完整benchmark查看当前精确的通过率，再决定优先处理哪个。由于无法可靠运行完整benchmark，我需要提供可执行的调试命令给用户本地执行。
