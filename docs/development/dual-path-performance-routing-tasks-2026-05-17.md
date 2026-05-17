# OmniLSS 双路径性能路由系统 — 任务拆分

> **背景说明**：`config.py` 已有完整的框架骨架（`auto_select_method()`、`GPU_CROSSOVER_N`、`TPU_CROSSOVER_N`、环境变量覆盖）。所有阈值当前为 `math.inf`（永不切换到 JAX），JAX 支持的分布族仅 6 个（NO/GA/PO/BI/WEI/TF）。以下任务均从这一现状出发，**不重复造轮子**。

---

## T1 · 基准测试套件构建

> **目标**：为 GPU/TPU 切换阈值提供实测数据，用真实测量值替换所有 `math.inf`。

### T1-A：GPU 基准测试矩阵设计

**优先级**：P0 · **依赖**：无

**测试变量矩阵**（全量交叉）：

| 变量 | 取值 |
|------|------|
| `n`（观测数） | 100 · 500 · 1,000 · 5,000 · 10,000 · 50,000 · 100,000 · 500,000 |
| `p`（设计矩阵列数） | 2 · 5 · 10 · 20 · 50 |
| 分布族 | NO · GA · PO · BI · WEI · TF（当前 6 个 JAX 族） |
| 方法 | `RS`（NumPy）vs `RS_JAX`（JAX JIT 预热后） |

**关键注意事项**：
- JAX 计时必须包含至少 3 次预热调用（`jax.block_until_ready()`），取预热后**中位数**，而非最快值
- NumPy 路径计时同样取中位数（避免 OS 调度抖动）
- 每个 `(n, p, family)` 组合独立测 5 次，报告 p50 和 p95
- 记录 GPU 显存占用（`nvidia-smi`）防止大 n 下 OOM 扭曲结果

**交付物**：
- `benchmarks/gpu_crossover_sweep.py` — 自动化测试脚本
- `docs/benchmarks/gpu_crossover_<timestamp>.md` — 结果报告
- `benchmarks/results/gpu_crossover_<timestamp>.json` — 机器可读原始数据

---

### T1-B：从基准数据提取切换阈值

**优先级**：P0 · **依赖**：T1-A 完成

**分析步骤**：

1. 对每个 `(family, p)` 组合，找到满足 `time(RS_JAX) < time(RS)` 的最小 `n`，记为 `crossover_n(family, p)`
2. 判断切换阈值是否与 `p` 强相关：
   - 若 `p` 影响显著（跨 2x 以上）→ 阈值字典需加入 `p` 维度（见 T3-A）
   - 若 `p` 影响可忽略 → 仅保留 `n` 维度，取各 `p` 下阈值的保守上界（`max`）
3. 若 GPU 上全范围 `RS_JAX` 仍慢于 `RS`（与 RTX 3060 初测结果一致）→ 阈值保持 `math.inf`，但在注释中记录具体倍数供未来参考

**结果写入** `config.py` 中对应字典，并注明测试硬件/JAX 版本/日期。

---

### T1-C：TPU 基准测试占位符维护

**优先级**：P2 · **依赖**：无（可独立完成）

**当前状态**：`TPU_CROSSOVER_N` 已存在，所有值为 `math.inf`。

**本任务工作**：
- 在 `config.py` 中为 TPU 补全所有当前 JAX 支持族的占位符条目（与 `GPU_CROSSOVER_N` 结构对齐）
- 在 `docs/benchmarks/tpu_crossover_placeholder.md` 中记录：预期测试环境（TPU v2/v3/v4）、待测变量矩阵、手动更新方法
- 确保环境变量 `OMNILSS_TPU_CROSSOVER_N` 已在 `config.py` 中正确解析（✅ 已存在，但需验证多族覆盖）

**手动更新入口**（测试完成后由维护者操作）：

```python
# config.py 中维护者直接编辑
TPU_CROSSOVER_N: dict[str, float] = {
    "NO":  10_000,   # TODO: 填入实测值
    "GA":  math.inf, # TODO: 待测
    "default": math.inf,
}
```

---

## T2 · 用户可配置 API 强化

> **目标**：在现有 `cfg.GPU_CROSSOVER_N["NO"] = 50_000` 基础上，提供更完整、更防错的用户接口。

### T2-A：`set_crossover()` 便捷函数

**优先级**：P1 · **依赖**：T1-B

在 `config.py` 中新增 `set_crossover(device, n, family="default")`，用于设置指定设备和分布族的切换阈值。

**输入验证**：
- `device` 不在 `{"gpu", "tpu"}` → 抛 `ValueError`（CPU 永不切换，无需设置）
- `n < 0` → 抛 `ValueError`
- `family` 不在已知族且不为 `"default"` → 发出 `UserWarning`（允许用户为未来族预设值）

---

### T2-B：`crossover_summary()` 诊断函数

**优先级**：P1 · **依赖**：T2-A

在 `config.py` 中新增 `crossover_summary(verbose: bool = False) -> None`，打印当前设备、自动切换状态、强制 JAX 状态，以及 GPU/TPU 阈值表。

---

### T2-C：`crossover_config` 上下文管理器（临时覆盖）

**优先级**：P2 · **依赖**：T2-A

新增 `crossover_config(gpu: dict | None = None, tpu: dict | None = None)`，临时覆盖切换阈值，退出 `with` 块后自动恢复原值。

---

### T2-D：配置文件支持（YAML / JSON）

**优先级**：P2 · **依赖**：T2-A

支持从配置文件持久化用户的阈值设置：

**文件查找顺序**（`config.py` 初始化时自动加载）：
1. 环境变量 `OMNILSS_CONFIG_FILE` 指定的路径
2. 当前工作目录 `./omnilss_config.yaml`
3. 用户主目录 `~/.omnilss/config.yaml`

**配置文件格式**：

```yaml
auto_method_enabled: true
force_jax: false

gpu_crossover_n:
  NO: 50000
  GA: 80000
  default: .inf

tpu_crossover_n:
  NO: 10000
  default: .inf
```

**实现要求**：
- 文件不存在时静默跳过（不报错）
- 文件格式错误时发出 `UserWarning` 并使用默认值
- 环境变量优先级高于配置文件（环境变量可覆盖文件设置）

---

## T3 · 切换逻辑扩展

> **目标**：处理 T1-B 中可能发现的“阈值与设计矩阵列数 `p` 强相关”情况，以及扩展 JAX 支持族范围。

### T3-A：（条件）二维切换阈值 `(n, p)`

**优先级**：P1 · **依赖**：T1-B（根据分析结果决定是否实施）

**触发条件**：T1-B 分析发现 `p` 对切换阈值影响超过 2 倍。若触发，扩展 `auto_select_method()` 签名并将阈值结构升级为 `(n_threshold, p_threshold)` 元组；若不触发，保持当前纯 `n` 维度并在注释中记录分析结论。

---

### T3-B：扩展 JAX 支持族

**优先级**：P1 · **依赖**：T1-A（测试新族性能），T1-B（确认有切换价值）

**当前状态**：`JAX_SUPPORTED_FAMILIES = {"NO", "GA", "PO", "BI", "WEI", "TF"}`（6 个）

**扩展候选族**（按实现难度排序）：
1. `LOGNO`（对数正态，与 NO 结构相近）
2. `EXP`（指数，单参数）
3. `IG`（逆高斯）
4. `GU` / `RG`（Gumbel，已有 AD 实现）
5. `NBI` / `NBII`（负二项，离散）

每个新族需要实现 `FamilyJAXSpec`、注册支持族、添加 GPU/TPU 占位符条目并运行对应 R 一致性测试。

---

## T4 · `gamlss()` 调用方签名与文档

> **目标**：让用户在调用层能直接控制路由行为，`method` 参数已支持，补充缺失的 `n_features` 传递链路（若 T3-A 实施）。

### T4-A：`method` 参数文档完善

**优先级**：P1 · **依赖**：T2-A

在 `fitting.py` 的 `gamlss()` docstring 中补充 `RS`、`RS_JAX`、`auto` 的使用场景、限制和手动强制路由示例。

---

### T4-B：verbose 模式输出路由信息

**优先级**：P2 · **依赖**：T4-A

当 `verbose=True` 且 `method="auto"` 时，`gamlss()` 应打印路由决策，包括设备、`n`、分布族、阈值和提示信息。

---

## T5 · 集成测试

> **目标**：保证路由逻辑在各种配置组合下行为正确，防止回归。

### T5-A：路由逻辑单元测试

**优先级**：P0 · **依赖**：无（可在任何阶段添加）

在 `tests/test_config_auto_method.py` 中补充 `set_crossover()`、非法设备、上下文恢复、`FORCE_JAX`、`AUTO_METHOD_ENABLED`、YAML 配置、环境变量覆盖、未知族 warning 等用例。

---

### T5-B：端到端路由集成测试

**优先级**：P1 · **依赖**：T2-A

新增或补充端到端路由测试，覆盖 CPU 自动路由、GPU 环境下手动 RS、`RS_JAX` 不支持族的清晰报错，以及 GPU 超阈值进入 JAX 路径。

---

## T6 · 文档

> **目标**：消除用户对双路径的困惑，提供清晰的“何时用哪条路径”指南。

### T6-A：`docs/benchmarks/device-method-selection.md` 更新

**优先级**：P1 · **依赖**：T1-B 完成后更新实测数据

按“快速结论 / 为什么 CPU 上 JAX 更慢 / 基准测试结果 / 如何自定义阈值 / TPU 配置指南”的结构重写或补充现有指南。

---

### T6-B：API 文档更新（`docs/api/` 目录）

**优先级**：P2 · **依赖**：T2-A, T4-A

- `docs/api/algorithms.md`：新增“方法路由”章节，引用 `config.py` 公开函数
- `docs/api/config.md`：新增（描述 `set_crossover()`、`crossover_summary()`、`crossover_config()`、配置文件格式）

---

## 任务依赖关系

```text
T1-A（基准测试脚本）
  └── T1-B（提取阈值）
        └── T3-A（是否需要 p 维度）（条件）
        └── T3-B（扩展 JAX 族）
        └── T6-A（文档更新实测数据）

T1-C（TPU 占位符）— 独立

T2-A（set_crossover）
  └── T2-B（诊断输出）
  └── T2-C（上下文管理器）
  └── T2-D（YAML 配置）
  └── T4-A（docstring）
  └── T5-A（单元测试）
  └── T5-B（集成测试）

T5-A — 可与 T2-A 并行开始
```

---

## 优先级总览

| 任务 | 优先级 | 预估工作量 | 可并行 |
|------|--------|-----------|--------|
| T1-A 基准测试脚本 | P0 | 1–2 天 | 是 |
| T1-B 阈值提取 | P0 | 0.5 天 | 否（等 T1-A） |
| T5-A 路由单元测试 | P0 | 0.5 天 | 是 |
| T2-A `set_crossover()` | P1 | 0.5 天 | 是 |
| T2-B `crossover_summary()` | P1 | 0.5 天 | 是 |
| T4-A docstring | P1 | 0.5 天 | 是 |
| T5-B 集成测试 | P1 | 1 天 | 否（等 T2-A） |
| T3-A 二维阈值（条件） | P1 | 1 天 | 否（等 T1-B） |
| T3-B JAX 族扩展 | P1 | 2–4 天/族 | 是 |
| T6-A 文档重写 | P1 | 1 天 | 否（等 T1-B） |
| T1-C TPU 占位符 | P2 | 0.5 天 | 是 |
| T2-C 上下文管理器 | P2 | 0.5 天 | 是 |
| T2-D YAML 配置 | P2 | 1 天 | 是 |
| T4-B verbose 路由信息 | P2 | 0.5 天 | 是 |
| T6-B API 文档 | P2 | 0.5 天 | 否（等 T2-A） |

---

## 开发推进状态（2026-05-17）

| 任务 | 状态 | 说明 |
|------|------|------|
| T1-A GPU 基准测试脚本 | ✅ 已实现 | `benchmarks/gpu_crossover_sweep.py` 覆盖默认 `(n, p, family)` 矩阵、JAX 预热、p50/p95、`nvidia-smi` 快照、JSON 与 Markdown 输出。 |
| T1-B 阈值提取 | ⚠️ 等待硬件实测 | 脚本已内置保守阈值提取逻辑；真实 `config.py` 默认值需在 GPU/TPU 环境完整跑完后人工审查写入。 |
| T1-C TPU 占位符 | ✅ 已实现 | `TPU_CROSSOVER_N` 已与 GPU 表覆盖同一组 JAX 支持族，并新增 TPU placeholder 文档。 |
| T2-A `set_crossover()` | ✅ 已实现 | 支持 GPU/TPU、非负阈值校验、未知族 warning。 |
| T2-B `crossover_summary()` | ✅ 已实现 | 输出设备、开关和 GPU/TPU 阈值表。 |
| T2-C `crossover_config()` | ✅ 已实现 | 支持临时覆盖并在退出后恢复。 |
| T2-D 配置文件支持 | ✅ 已实现 | 支持 YAML/JSON、三段查找顺序、环境变量覆盖文件配置。 |
| T4-A `method` 参数文档 | ✅ 已实现 | `gamlss()` docstring 已补充 RS/RS_JAX/auto 说明和示例。 |
| T4-B verbose 路由信息 | ✅ 已实现 | `method="auto"` 且 `verbose=True` 时打印路由决策。 |
| T5-A 路由单元测试 | ✅ 已实现 | 已覆盖配置 API、YAML/env、warning 与 TPU 占位符。 |
| T5-B 端到端路由集成测试 | ✅ 已实现 | 已覆盖 CPU 自动 RS、GPU 环境手动 RS、超阈值 JAX、unsupported family 报错。 |
| T6-A 方法路由指南 | ✅ 已更新 | `docs/benchmarks/device-method-selection.md` 已按快速结论、原因、基准、配置、TPU 指南重写。 |
| T6-B API 文档 | ✅ 已实现 | 新增 `docs/api/config.md` 与 `docs/api/algorithms.md` 方法路由章节。 |
| T3-A 二维阈值 | ⏸️ 暂缓 | 依赖完整 T1-B 实测判断 `p` 是否造成超过 2x 的阈值差异。 |
| T3-B 扩展 JAX 支持族 | ⏸️ 暂缓 | 依赖 T1-A/T1-B 证明新族有自动切换价值，并需要逐族 R 一致性测试。 |
