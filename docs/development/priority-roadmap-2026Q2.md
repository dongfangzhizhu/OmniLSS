# OmniLSS 任务优先级清单（2026 Q2）

> 基于代码深度分析，按「重要 × 紧急」四象限排列。  
> 原则：个人开发者精力有限，先做能直接提升项目可信度和用户数的事。

---

## 🔴 第一象限：重要且紧急（立即执行）

### T-01 修复 Formula Parser 的交互项支持
**当前状态（2026-05-15）**：🟡 Partial

**为什么紧急**：当前不支持 `x1*x2`、`x1:x2` 等标准 R 公式语法，会直接导致有经验的 R/statsmodels 用户碰壁后放弃，是影响第一印象的硬伤。

- [ ] 在 `formula_parser.py` 中增加 `:` （交互项）解析
- [ ] 增加 `*` （主效应 + 交互项）的展开逻辑
- [ ] 增加因子/类别变量（`factor(x)`）的哑变量展开
- [ ] 补充对应的测试用例，与 R 的 `model.matrix()` 输出做数值比对
- [ ] 更新文档说明支持的公式语法范围

### T-02 消除 RS 算法中的 np/jnp 频繁互转
**当前状态（2026-05-15）**：🟡 Partial

**为什么紧急**：这是“JAX 加速”声明的可信度问题。当前 `rs_algorithm.py` 中有大量 `np.asarray()/jnp.asarray()` 互转，IRLS 热路径跑在 NumPy 上，GPU 加速实际上不生效。如果用户测试后发现 benchmark 不实，会直接损害项目声誉。

- [ ] 梳理 `rs_algorithm.py` 中所有 np/jnp 互转点，标注哪些在热路径上
- [ ] 将 `compute_working_weights_and_response` 改写为纯 JAX 计算
- [ ] 将 IRLS 内层循环的 WLS 求解改为 `jnp.linalg.lstsq`
- [ ] 用 `jax.lax.while_loop` 替换 Python `for` 循环（可分步进行）
- [ ] 对 NO/GA/NBI 分布重新运行 benchmark，确认 JIT 编译后速度提升真实
- [ ] 更新 README 中的性能数据表格，确保数字诚实

---

## 🟠 第二象限：重要不紧急（计划执行，1-3 个月内）

### T-03 补充 Property-Based Testing
**当前状态（2026-05-15）**：🟡 Partial (Early stage)

**为什么重要**：当前测试以固定种子的数值比对为主，边界条件（极小参数、零值、负值输入）覆盖不足。商业用户生产环境数据多样，分布函数的数值稳定性必须经过系统验证。

- [ ] 引入 `hypothesis` 库（加入 dev 依赖）
- [ ] 为每个分布族的 `logpdf` 编写 property test：输入合法范围内随机参数，验证输出有限且非 NaN
- [ ] 为 `p/q` 函数编写单调性验证：`p` 应严格单调递增，`q(p(x))` 应近似还原 `x`
- [ ] 为 RS 算法编写收敛性 property test：合成数据拟合后，deviance 不应大于 intercept-only 模型
- [ ] 将 property tests 加入 CI（允许随机种子但设置固定 `deriving_from` 数据库）

### T-04 完成 CG 算法的真实实现
**当前状态（2026-05-15）**：🟡 Partial

**为什么重要**：当前 `cg_fit()` 完全委托给 `gamlss(method="CG")`，自身写的 `_compute_cross_derivatives`（用 `jax.vmap` 计算交叉导数）未被使用。这是技术债，也是论文投稿时的弱点。

- [ ] 将 `_compute_cross_derivatives` 接入 `cg_fit()` 的主路径
- [ ] 实现完整的 CG 外层循环（使用交叉导数修正）
- [ ] 对比 RS 和 CG 在相关参数分布（如 BCT、SHASH）上的收敛速度
- [ ] 更新 README，将 CG 状态从 “Simplified” 改为 “Complete”
- [ ] 补充 CG 算法的测试用例

### T-05 Deep GAMLSS 端到端示例与文档
**为什么重要**：这是项目超越 R gamlss 的差异化功能，但当前缺少完整可运行示例。这是吸引 ML 背景用户的关键抓手，也是 JOSS 论文的核心创新点之一。

- [ ] 编写完整 Jupyter Notebook：数据生成 → 拟合 Deep GAMLSS → 与传统 GAMLSS 比较 → 可视化分布参数非线性关系
- [ ] 选择真实数据集（如 LIDAR、儿童身高体重）作为演示
- [ ] 验证 `flax` + `optax` 训练循环在 CPU/GPU 均可运行
- [ ] 补充 `deep_gamlss.py` 的预测接口文档字符串
- [ ] 在文档网站（MkDocs）增加 Deep GAMLSS 专页

### T-06 建立 gRPC 接口设计（proto 定义）
**为什么重要**：双层架构的基础。即使 `omnilss-pro` 还不存在，先把 proto 接口设计好，可以锁定边界，避免未来核心库接口设计走偏。

- [ ] 设计 `FitRequest`：包含公式、数据（Arrow/Parquet）、家族名、控制参数
- [ ] 设计 `FitResponse`：包含系数、deviance、AIC/BIC、收敛状态
- [ ] 设计 `PredictRequest` / `PredictResponse`：支持参数预测和分位数预测
- [ ] 设计 `SampleRequest`：从拟合分布中抽样
- [ ] 编写 `.proto` 文件并生成 Python stub
- [ ] 在 `api/grpc/__init__.py` 中实现服务端处理逻辑（核心层部分）

---

## 🟡 第三象限：紧急不重要（快速处理，降低维护成本）

### T-07 清理 `distributions_b5_optimized.py` 冗余文件
- [ ] 确认 `distributions_b5_optimized.py` 与 `distributions_b5.py` 差异
- [ ] 合并有效优化内容到 `distributions_b5.py`
- [ ] 删除 `distributions_b5_optimized.py`，更新 import 引用
- [ ] 运行测试确保无回归

### T-08 统一 `gamlssML` 接口的暴露方式
当前 `gamlssML.py` 仅作为 `fitting.gamlss_ml` 的别名，但 `__init__.py` 导出混用了两个名字。

- [ ] 在 `__init__.py` 中统一只暴露 `gamlss_ml`（Python 风格）
- [ ] 保留 `gamlssML` 作为废弃别名并加 `DeprecationWarning`
- [ ] 更新 docs 的 API Reference 页面

### T-09 修复 CI/CD 发布流程
README 提到 `release.yml` workflow，但项目尚未验证 PyPI 实际发布链路。

- [ ] 手动触发 `release.yml` 验证流程完整性
- [ ] 确认 PyPI token 已配置到 GitHub Secrets
- [ ] 确认版本号 bump 脚本（`bump_version.ps1`）在 Linux 上替代方案
- [ ] 补充 `CHANGELOG.md` 的 v0.3.0 条目

---

## ⚪ 第四象限：不重要不紧急（有余力再做）

### T-10 Sklearn 兼容层完善
当前 `GAMLSSRegressor` 已有基础框架，但缺少 `get_params()` / `set_params()` 完整实现，无法用于 `GridSearchCV`。

- [ ] 实现完整 sklearn estimator 协议
- [ ] 添加 `score()`（基于 deviance）
- [ ] 编写 `Pipeline` 集成示例

### T-11 AutoML 分布选择模块
当前 `automl/__init__.py` 为空，`chooseDistParallel.py` 有实现但未通过 AutoML 接口暴露。

- [ ] 将 `chooseDist` 封装为高层 `auto_select_family()` 接口
- [ ] 支持并行评估（复用 `ProcessPoolExecutor`）
- [ ] 增加 AIC/BIC/GAIC 多准则排名报告
- [ ] 评估是否纳入 `omnilss-pro` 候选功能

### T-12 Memory Optimization 模块集成
`memory_optimization.py` 已实现，但尚未接入主流程。

- [ ] 评估 n > 100K 场景下实际收益
- [ ] 将内存优化选项作为 `gamlss_control()` 可选参数暴露
- [ ] 编写大数据 benchmark 示例

---

## 执行原则

> **核心原则**：T-02（JAX 真实加速）是最影响项目声誉的技术债，应在 Sprint 2 内完成；在此之前不应对外强调性能优势。
