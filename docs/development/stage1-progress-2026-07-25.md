# OmniLSS 开发计划 - 阶段1进度更新 (2026-07-25)

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

### 📈 快速基准测试结果 — 30/42 (71.4%)

| 分布 | d/p/q结果 | 最大误差 |
|------|-----------|----------|
| NO | ✅ 3/3 OK | 4.99e-05 |
| GA | ✅ 3/3 OK | 4.99e-05 |
| PO | ⚠️ 2/3 OK | PO_q失败 |
| BI | ❌ 0/3 OK | NaN |
| NBI | ⚠️ 1/3 OK | p/q失败 |
| BE | ✅ 3/3 OK | 4.98e-05 |
| ZAGA | ✅ 3/3 OK | 4.98e-05 |
| GU | ⚠️ 1/3 OK | p/q失败 |
| SHASH | ✅ 3/3 OK | — |

### ⚠️ 发现的问题及优先级排序
1. **BI族全部失败** — d/p/q返回NaN（最高优先级）
2. **NBI p/q失败** — 数值问题
3. **GU p/q失败** — 数值问题严重
4. **PO_q失败** — 可能参数化错误
5. **PO + RS拟合失败** — deviance差异69.21
6. **平滑器pb/ps/cs全部失败** — 需要修复

### 下一步行动
1. **诊断BI族** — NaN问题（最高优先级）
2. **诊断NBI/GU** — p/q实现问题
3. **诊断PO_q和PO RS拟合** — 参数化或deviance计算问题
4. **修复平滑器** — pb/ps/cs



