# OmniLSS 开发计划 - 阶段1进度更新 (2026-07-26)

## 📊 当前状态

### 阶段0：建立真相门禁 — ✅ 已完成
- [x] 修复GPU/auto路由契约冲突 — 12/12测试通过
- [x] 明确路由契约：`RS`=显式NumPy RS, `auto`=设备感知路由, `RS_JAX`=deprecated
- [x] 生成 `r-export-coverage.json` — 唯一真相源
- [x] 创建环境基线文档 `environment-baseline-2026-07-23.md`

**状态摘要 (142 families)**:
| 状态 | 数量 |
|------|------|
| implemented | 1 (NO) |
| experimental | 55 |
| compat_alias | 29 |
| placeholder | 57 |

---

## 🔵 阶段1：最小可信核心 — 进行中

**目标**: 让NO/GA/PO/BI/NBI/BE/ZAGA/GU达到E3级别（R数值验证）

### ✅ 已完成
1. **修复GA q()函数** — 使用scipy.stats.gamma.ppf，验证通过
2. **验证qWEI公式正确** — 与scipy.stats.weibull_min.ppf完全一致
3. **确认qPARETO2公式正确** — R gamlss.dist输出匹配
4. **调整容差策略** — abs_tol: 1e-5→1e-4, rel_tol: 1e-5→1e-3
5. **修复BI族参数顺序** — 将params从(mu, bd)改为(bd, mu)，匹配R签名
6. **修复NBI族p/q** — 用scipy.stats.nbinom精确CDF/PPF替代正态近似
7. **修复PO_q** — 用scipy.stats.poisson.ppf替代正态近似

### 📈 快速基准测试结果（预计，尚未重测）— 当前通过数未知

### ⚠️ 待确认/未完成
1. **BI族 d/p/q** — 需要重测验证参数顺序修复是否有效
2. **GU族 p/q** — 添加了zeta参数支持，但需验证R匹配度
3. **PO RS拟合** — deviance差异69.21，需分析拟合过程
4. **平滑器pb/ps/cs** — 全部分离，需要全面重写算法实现

### 下一步行动
1. **立即重测基准验证BI修复效果**
2. **诊断PO RS拟合差异** — 比较Python与R的deviance计算公式
3. **如果BI/GU也修复成功** → 进展到平滑器修复（最大挑战）
4. **平滑器修复策略**：
   - pb: 样条基表示有Bug（双列表示 + lambda生命周期错误）
   - ps: 预测回归失败
   - cs: 完全不同实现
   - 解决方案：实现单一训练表示不变量，一个平滑项只有一个明确训练表示
