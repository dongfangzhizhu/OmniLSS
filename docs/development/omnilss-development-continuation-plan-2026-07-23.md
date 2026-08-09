# OmniLSS 开发继续计划 - 基于2026-07-22全面审计

**制定日期**: 2026-07-23  
**基线**: `omnilss-r-gamlss-migration-comprehensive-audit-2026-07-22.md`  
**目标**: 建立可信最小核心，逐步完成 R `gamlss` + `gamlss.dist` 迁移

---

## 一、现状评估与核心问题

### 1.1 关键事实

| 维度 | 状态 | 风险等级 |
|------|------|----------|
| 已注册分布 | 142个 | 🟡 名称可解析，不等价于统计实现 |
| 真正implemented | 仅1个(NO) | 🔴 严重低估 |
| placeholder/alias | 86个(57+29) | 🔴 阻碍发布声明 |
| d/p/q 数值验证 | 0/27通过 | 🔴 无法宣称分布功能完成 |
| 拟合验证 | 11/12通过 | 🟡 PO/RS偏差需修复 |
| 平滑器验证 | 0/3通过 | 🔴 pb/ps/cs均需重做 |
| GPU/auto路由 | 3测试失败 | 🟡 契约不一致 |

### 1.2 根因分析

1. **精度问题**: JAX x64默认关闭，影响数值验证
2. **实现真相未对齐**: 大量compat_alias/placeholder被误计为实现
3. **平滑器设计缺陷**: ps/cs双列表示、lambda生命周期错误、预测回归失败
4. **缺乏固定R基线**: vendored与installed R版本不同步
5. **测试门禁不足**: Python-only成功不能证明R等价

---

## 二、阶段0：建立真相门禁（第1周）

**目标**: 阻断新功能开发，确保所有后续工作基于真实状态

### 2.1 生成r-export-coverage.json

```python
# 扫描R/目录下所有13个项目，建立符号→状态映射
# 状态: missing, placeholder, compat_alias, experimental, implemented, unsupported_with_rationale
```

**验收标准**:
- [ ] 覆盖 `R/gamlss`, `R/gamlss.dist`, `R/gamlss.add`, `R/gamlss.cens`, `R/gamlss.tr`, `R/gamlss.mx`, `R/gamlss.spatial`, `R/gamlss.foreach`, `R/gamlss.prepdata`, `R/gamlss.ggplots`, `R/gamlss2`, `R/gamlss.data`, `R/WeatherGermany`
- [ ] 每个Python family可追溯到底层数学核、R符号和证据状态
- [ ] 删除手写"完成率"文档，改为自动生成

### 2.2 固定开发环境基线

```powershell
# CI必须使用固定版本
$env:JAX_ENABLE_X64 = 'true'
$env:PYTHONPATH = "$PWD\omnilss\src"

# 记录到 constraints/lock文件
# - JAX 0.6.2
# - NumPy 2.2.6
# - SciPy 1.15.3
# - R 4.6.0 (从R/gamlss源码安装)
# - gamlss 5.5-3 (vendored)
# - gamlss.dist 6.1-8 (vendored)
```

**验收标准**:
- [ ] `.venv` 默认启用x64或明确拒绝x32运行
- [ ] R从vendored源码安装，禁止使用外部library
- [ ] 每个测试结果写入版本/commit/dirty状态

### 2.3 默认拒绝不可信入口

```python
# fitting.gamlss() 必须默认拒绝:
# - compat_alias families
# - placeholder families
# - staged smoothers
# 除非显式传入 allow_experimental=True / allow_compat_alias=True
```

**验收标准**:
- [ ] 84→87测试通过（修复3个GPU/auto路由测试）
- [ ] 模型对象标注是否可用于生产

---

## 三、阶段1：最小可信核心（第2-5周）

**目标**: 让8个优先族达到E3级别（R数值验证）

### 3.1 修复d/p/q数值问题

**优先族列表**: NO, GA, PO, BI, NBI, BE, ZAGA, GU

#### 任务1.1: 诊断并修复NO族

当前问题: d/p/q最大误差约`4.8e-5`

```python
# 检查项:
# - dtype是否为float64
# - 桥接层是否引入截断
# - 容差设计是否合理
# - 支持域边界处理
```

**验收标准**:
- [ ] NO d/p/q在`abs<1e-5, rel<1e-5`下通过live-R对照
- [ ] 记录R种子、数据、公式、参数、容差

#### 任务1.2: 修复GA族q函数

当前问题: q误差`3.615`，可能是P0数学缺陷

```python
# 检查项:
# - Gamma分布参数化是否与R一致
# - 量化函数实现方法
# - 极端概率尾部处理
```

**验收标准**:
- [ ] GA q函数误差降至`<1e-3`
- [ ] 补充中心/尾部/超尾部三组参数剖面

#### 任务1.3: 修复PO族离散分位数

当前问题: q误差`1`，离散分布定义与R最小整数规则差异

```python
# 检查项:
# - Poisson CDF实现
# - 分位数定义: min{x: F(x)>=p}
# - 边界值处理
# - 与R的离散分布对比
```

**验收标准**:
- [ ] PO q函数在离散点处与R一致
- [ ] 零值、大均值、极端概率场景通过

#### 任务1.4: 修复BI/NBI/BE/ZAGA/GU

**NBI**: p误差`0.2602`，q误差`2` — 参数化/CDF/量化逻辑为P0缺陷  
**BE**: 接近零密度处相对误差异常 — 混合绝对/相对规则  
**ZAGA**: q误差`1.392` — 零质量/连续部分缩放/逆CDF  
**GU**: p误差`0.2643`，q误差`3.073` — CDF/量化实现  

**通用修复流程**:
1. 对比R源实现
2. 检查参数化是否一致
3. 验证支持域和边界条件
4. 测试单调性、有限值、shape
5. 建立R golden artifacts

**验收标准**:
- [ ] 8个族全部达到E3（固定R基线下d/p/q/r通过）
- [ ] 任一NaN、shape错误、超阈值使门禁失败

### 3.2 修复PO RS拟合失配

当前问题: deviance diff=`69.215`, max_abs=`69.21`

```python
# 检查项:
# - score函数是否正确
# - Hessian是否正定
# - link函数是否匹配R
# - 初始值选择
# - 收敛判定标准
```

**验收标准**:
- [ ] PO RS拟合deviance差<`1e-3`
- [ ] 补充截距/线性mu/sigma/nu/tau/权重/缺失数据测试矩阵

### 3.3 验证RS/CG/Mixed三种路由

**测试矩阵**:
- 每个族至少含: 截距模型、线性mu、sigma公式、权重、固定参数、缺失数据
- RS、CG、Mixed分别报告: 收敛、deviance、系数、fitted values
- 计数族覆盖: 零、多零、离散响应边界、极端均值

**验收标准**:
- [ ] NO/GA/PO/BI在三种路由下均有小数据R artifact
- [ ] 任何Hessian floor/weight clipping/eta clip必须写入诊断

---

## 四、阶段2：恢复可信平滑与预测（第6-9周）

**目标**: 让pb/ps/cs在固定R基线上通过端到端验证

### 4.1 先锁定pb smoother

**当前状态**: deviance差`5.9587`，形状可能接近但不满足1e-3门槛

```python
# 需要同时验证:
# - basis函数
# - knots位置
# - penalty矩阵
# - df/lambda选择
# - EDF计算
# - 预测重建
```

**验收标准**:
- [ ] pb在300样本NO数据上deviance差<`1e-3`
- [ ] 记录R seed=1,2,3三个对照

### 4.2 重设计ps/c

当前问题严重:
- **双列平滑**: ps/cs先加ones占位，再hstack初次拟合值
- **Lambda生命周期错误**: 首lambda固定，不回写EDF/knots
- **控制参数漂移**: ps_intervals不存在SmoothFitInfo，回退到20
- **预测回归**: 训练3列 vs 预测23列B-spline → PredictionSchemaError

**修复方案A**: 保留可重建完整basis+penalty+coefficient
**修复方案B**: 定义并序列化真正的单列smoother对象与其predict(newdata)

**不要混用两种方案**。

**验收标准**:
- [ ] 一个平滑项只拥有一个明确训练表示
- [ ] 状态持久化后predict(newdata)通过
- [ ] lambda/df/EDF/knots正确更新

### 4.3 每一轮RS记录smoother更新

```python
# 日志必须包含:
# - old/new lambda
# - EDF
# - working response summary
# - weight summary
# - solver状态
```

**验收标准**:
- [ ] pb/ps/cs在固定R基线上通过parsing/fit/EDF/multiple smoother/mixed parameter/newdata predict

---

## 五、阶段3：逐波迁移剩余分布与经典核心（持续）

### 5.1 波次1: 常用连续/计数/混合族

**目标**: BCCG, BCPE, BCT, BEINF, BI, EXP, GA, GEOM, GT, GU, IG, IGAMMA, LO, LOGNO, LOGNO2, NBI, NBII, NO, NO2, PARETO2, PE, PO, RG, SHASH, SIMPLEX, SN1, TF, WEI, ZIP, ZAGA

**验收标准**:
- [ ] 全部达到E3（R数值验证）
- [ ] 无compat_alias/placeholder混入

### 5.2 波次2: Box-Cox与重尾/偏态族

**目标**: BCCGo, BCTo, BCPEo, BCTuntr, BCPEuntr, BCCGuntr, EGB2, SEP系列, SST, ST系列

**验收标准**:
- [ ] 专用数值实现替换compat_alias
- [ ] R parity证据

### 5.3 波次3: 复杂离散、零修正和多参数族

**目标**: DBI, DBURR12, ZI系列, MULTIN, gen.Family skeletons

**验收标准**:
- [ ] 不再使用PO/NO骨架伪装
- [ ] 独立likelihood/score/Hessian

### 5.4 波次4: 经典核心函数

**优先**: Formulae2data, gen.likelihood, gamlssML, vcov/rvcov/get.K, dtop, par.plot, prof.dev, find.hyper, lms, quantSheets, stepGAIC

**验收标准**:
- [ ] data-only入口建立R shape/value契约
- [ ] 图形API至少返回正确绘图数据

---

## 六、阶段4：生态扩展与范围决策（Phase 5+）

对11个附带R项目作三选一决策:

| 项目 | 建议 |
|------|------|
| `gamlss.tr` | 迁移: 通用截断核 |
| `gamlss.cens` | 迁移: 区间删失框架 |
| `gamlss.add` | 互操作: R bridge/文件格式 |
| `gamlss.mx` | 明确不支持 |
| `gamlss.spatial` | 明确不支持 |
| `gamlss.foreach` | 互操作: Bootstrap/rolling |
| `gamlss.prepdata` | 独立Python utilities |
| `gamlss.ggplots` | 可视化适配层 |
| `gamlss2` | 独立大项目 |
| `gamlss.data` | 数据分发/加载 |
| `WeatherGermany` | 数据分发/加载 |

---

## 七、开发工作流与质量门禁

### 7.1 每个PR最小要求

- [ ] 标明涉及的R包、R符号、Python API、数学参数化、capability变化
- [ ] 新功能先添加失败的R对照测试，再实现
- [ ] 运行带固定x64与R基线的目标测试
- [ ] 更新生成的capability/coverage artifact
- [ ] alias/placeholder/staged改动说明为何不会被生产路径误用
- [ ] 性能改动附带相同case的correctness artifact

### 7.2 日常命令

```powershell
$env:JAX_ENABLE_X64 = 'true'
$env:PYTHONPATH = "$PWD\omnilss\src"

# 静态编译检查
.\.venv\Scripts\python.exe -m compileall -q omnilss\src\omnilss

# 核心问题测试
.\.venv\Scripts\python.exe -m pytest `
  omnilss/tests/test_family_capabilities.py `
  omnilss/tests/test_method_routing_integration.py `
  omnilss/tests/test_r_consistency_smooth.py

# 正式gate（需要R）
.\.venv\Scripts\python.exe benchmarks\comprehensive_r_consistency_test.py `
  --quick --require-r
```

### 7.3 Release候选门禁

```powershell
# 必须通过以下所有检查
.\.venv\Scripts\python.exe benchmarks\run_local_validation.py --quick --require-r

# 门禁语义:
# - R依赖缺失/跳过/空结果/全NaN → 失败
# - compat_alias/placeholder默认拒绝
# - 每一个E3分布有d/p/q+r+拟合例
# - 每一个E3 smoother有parsing/fit/EDF/predict/R对照
# - 产出JSON/Markdown/版本清单
```

---

## 八、风险与缓解

| 风险 | 缓解措施 |
|------|----------|
| 兼容性别名冒充实现 | 强制implementation_status元数据，默认拒绝 |
| Placeholder数值函数 | 静态识别+测试门禁 |
| 静默数值漂移 | 结构化诊断+parity工件 |
| 基准假阳性 | `--require-r` fail-hard语义 |
| 服务过度暴露 | capability gate+请求限制 |
| R基线漂移 | vendored固定版本+CI锁定 |
| JAX精度问题 | 强制x64或明确float32策略 |

---

## 九、时间线与里程碑

```
Week 1:  [x] 阶段0 - 真相门禁建立
         [ ] r-export-coverage.json生成
         [ ] 环境基线固定
         [ ] 默认拒绝机制实施

Week 2-5: [ ] 阶段1 - 最小可信核心
          [ ] NO/GA/PO/BI/NBI/BE/ZAGA/GU达E3
          [ ] PO RS拟合修复
          [ ] RS/CG/Mixed验证

Week 6-9: [ ] 阶段2 - 平滑器恢复
          [ ] pb锁定R定义
          [ ] ps/cs重设计
          [ ] 预测重建通过

Week 10+: [ ] 阶段3 - 逐波迁移
          [ ] 波次1-4完成
          [ ] 经典核心函数补齐

持续:    [ ] 阶段4 - 生态决策
          [ ] 11个项目签署迁移/互操作/不支持
```

---

## 十、下一步立即行动

1. **本周内**: 运行`comprehensive_r_consistency_test.py --quick --require-r`获取当前精确状态
2. **本周内**: 修复3个GPU/auto路由测试失败
3. **下周开始**: 聚焦NO族d/p/q修复，建立第一个E3案例
4. **同步**: 创建`r-export-coverage.json`生成脚本
