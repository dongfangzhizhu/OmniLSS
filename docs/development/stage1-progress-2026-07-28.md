# OmniLSS 开发计划 - 进度更新 (2026-07-28)

## 📊 阶段0：建立真相门禁 — ✅ 已完成

### 已验证通过的修复
- [x] GPU/auto路由契约冲突 — 12/12测试通过（所有测试均确认`RS`和`auto`路由正确）
- [x] 路由契约明确化 — `RS`=显式NumPy RS, `auto`=设备感知路由, `RS_JAX`=deprecated
- [x] `r-export-coverage.json` — 唯一真相源（142个distribution家族的状态分类）
- [x] 环境基线文档 `environment-baseline-2026-07-23.md` — 固定JAX x64、Python/R版本要求

**分布族状态摘要 (142 families)**:
| 状态 | 数量 | 说明 |
|------|------|------|
| implemented | 1 (NO) | 仅NO被标记为implemented |
| experimental | 55 | 需要进一步验证 |
| compat_alias | 29 | R名称的兼容别名 |
| placeholder | 57 | 未实现的占位符 |

---

## 🔵 阶段1：最小可信核心 — 进行中

**目标**: 让NO/GA/PO/BI/NBI/BE/ZAGA/GU达到E3级别（R数值验证，d/p/q拟合平滑器全部通过）

### 第一步：运行完整基准测试获取当前状态 ⏳

由于之前应用的批量修复脚本中部分模式匹配可能不完全成功，我们需要先重新运行基准测试来了解当前的实际通过率。这将是制定后续修复策略的依据。

```powershell
set JAX_ENABLE_X64=True
set PYTHONPATH=%cd%\omnilss\src
python benchmarks/comprehensive_r_consistency_test.py --quick --require-r
```

预期输出会显示具体的失败测试列表，帮助我们确定哪些修复生效了，哪些还需要处理。

### 第二步：从NO族开始逐族修复 d/p/q 数值问题

在获得基准测试结果后，我们将按照以下优先级顺序修复各分布族的d/p/q问题：

**修复优先级（由影响范围和难度决定）：**
1. **NO族** — 最简单的基础分布，作为验证环境的"健康检查"
2. **GA族** — 已经完成修复，需重新验证
3. **PO族** — q函数已用scipy.poisson.ppf替换
4. **BI族** — 参数顺序已调整，需验证
5. **NBI族** — 正态近似已替换为精确nbinom，需验证
6. **GU族** — 已添加zeta支持，需验证与R的一致性
7. **BE/ZAGA等** — 按列表逐一检查

**每次修复的标准流程：**
1. 查看当前实现 vs R/gamlss的理论公式
2. 编写与scipy/R的对比测试
3. 修正代码中的参数化或近似错误
4. 通过基准测试验证
5. 将修正添加到回归测试集

---

### 关键注意事项

**关于参数化一致性的提醒：**
- gamlss中的分布参数可能与scipy不同（如sigma的定义在不同包中可能代表标准差还是方差）
- R的`lower_tail`参数处理（P(X<=q) vs P(X>q))需要与Python精确对齐
- 边界情况（p接近0或1，或mu/sigma接近0）需要特殊处理以避免NaN

**关于数值精度的提醒：**
- 使用float64（JAX Enable X64）进行所有计算
- 相对误差容忍度建议设置为1e-3，绝对误差1e-4（已在基准测试中调整）
- 某些分布在极端参数下可能有数值不稳定，需要clip或正则化
