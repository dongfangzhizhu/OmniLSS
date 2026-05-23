# OmniLSS Phase-1 (`gamlss` / `gamlss.dist`) 全量对齐 Checklist（2026-05-23）

基线文档：`docs/development/r-gamlss-coverage-and-plan-2026-05-22.md`。  
本清单用于把“第一期目标（覆盖 R `gamlss` + `gamlss.dist` 功能）”拆解成可打勾项，按 **先开发后验证** 的顺序推进。

## 0. 统计口径

- ✅ 已完成：仓库中存在可调用实现，且 API/行为路径已经接入主流程。
- 🟡 部分完成：存在实现但缺 R parity 证据、或功能仅别名/兼容层。
- ⬜ 未完成：缺实现或仅有占位。

---

## A. 基础设施与可追溯性（Phase A）

- ✅ A1 CI 快速 R parity gate（devcontainer 内 `run_local_validation.py --quick`）。
- ✅ A2 R coverage matrix 生成脚本与 CI 漂移检查。
- ✅ A3 JSU 重复工厂收敛（`distributions_b11` legacy alias 已移除）。
- ✅ A4 关键迁移模块补充 `R reference:` 注释（含 smoothers）。

---

## B. 分布覆盖（Phase B）

### B1. 已注册 family 的 R-backed d/p/q 证据补齐

- 🟡 已推进：新增 `GIG` 的 R consistency（d/p/q）测试文件。
- ⬜ 待完成：其余大量 `experimental` family 的 d/p/q parity（约 70+）。

### B2. 计划中“缺失分布/接口”

> 进度说明（2026-05-23）：以上分布缺口已全部具备“兼容层可调用入口 + registry 可解析”，
> 当前剩余重点转向“专用数值内核替换 + R 一致性证据从脚手架升级为稳定覆盖”。

- ✅ `MULTIN` 构造接口（`MULTIN(3|4|5) -> MN3|MN4|MN5`）。
- 🟡 `TF2` 接口：已提供构造器与注册；并新增基于 R `TF` 的 d/p/q 一致性测试，后续仍需针对 R `TF2` 专用参数化逐项比对。
- 🟡 ST1（已提供兼容构造器与 registry，当前共用 ST3C 数值内核；并新增基于 ST3C 参考路径的 d/p/q R 一致性测试脚手架）
- 🟡 ST2（已提供兼容构造器与 registry，当前共用 ST3C 数值内核；并新增基于 ST3C 参考路径的 d/p/q R 一致性测试脚手架）
- 🟡 ST3（已提供兼容构造器与 registry，当前共用 ST3C 数值内核；并新增基于 ST3C 参考路径的 d/p/q R 一致性测试脚手架）
- 🟡 ST4（已提供兼容构造器与 registry，当前共用 ST3C 数值内核；并新增基于 ST3C 参考路径的 d/p/q R 一致性测试脚手架）
- 🟡 DBI（已提供兼容构造器与 registry，当前共用 BI 数值内核；并新增基于 BI 参考路径的 d/p/q R 一致性测试脚手架）
- 🟡 DBURR12（已提供兼容构造器与 registry，当前共用 SICHEL 数值内核；并新增基于 SICHEL 参考路径的 d/p/q R 一致性测试脚手架）
- 🟡 BCCGuntr（已提供兼容构造器与 registry，当前共用 BCCG 数值内核；并新增基于 BCCG 参考路径的 d/p/q R 一致性测试脚手架）
- 🟡 BCPEuntr（已提供兼容构造器与 registry，当前共用 BCPE 数值内核；已新增基于 BCPE 参考路径的 d/p/q R 一致性测试脚手架）
- 🟡 BCTuntr（已提供兼容构造器与 registry，当前共用 BCT 数值内核；已新增基于 BCT 参考路径的 d/p/q R 一致性测试脚手架）
- 🟡 `centileSK`：已有实现、基础单测与 R 一致性测试脚手架；仍需扩大输入覆盖并纳入 CI R 证据统计。
- ✅ `flexDist`, `momentSK`, `hazardFun`, `gen.hazard`：已具备实现与基础单测覆盖。

### B3. `gen.Family` skeleton builder

- 🟡 已新增统一兼容 builder 入口（`build_gen_family_skeleton`）并将 19 个常见 skeleton 名称注册到 registry；已为代表性 count/continuous skeleton 新增 d/p/q R 一致性测试脚手架；仍待逐个替换为专用数值实现。

### B4. `distributions3` 风格接口

- 🟡 已完成第一版 helper：`pdf/log_pdf/cdf/quantile/random/is_continuous/is_discrete/support`。
- 🟡 已补：`mean/variance/skewness/kurtosis` 支持统一参数与 sample 回退策略；并新增 LOGNO/NO2/GA/EXP/PO/BI 的解析矩实现与单测（含 GA 偏度/峰度解析式），仍待扩展更多 family 的解析解与精度基准。

---

## C. 拟合与诊断缺口（Phase C）

- 🟡 已有模块雏形/数据接口：`Formulae2data`, `getSmo/edfAll`, `dtop`, `par_plot`, `prof_dev`, `find_hyper`, `ECDF`, `loglogplot`, `gen_likelihood`, `lms`, `quantSheets` 等。
- ⬜ 待完成：上述函数的 R 对齐验收（输入语义、列名、边界行为、数值容差）。
- ⬜ 待完成：`gamlssML` 相关 `predict/summary/vcov` 的 R parity 完整验证。

---

## D. 平滑器覆盖（Phase D）

- ✅ 已实现主干：`pb`, `ps`, `cs` + `te/ti`, `tps` 扩展。
- ⬜ 待实现：`pbm`, `pbz`, `pbo`, `pbc`, `cy`, `pvc`, `fp`, `bfp`, `lo`, `vis.lo`, `re`, `random`, `ri`, `pcat`, `plotDF`, `plotLambda`。
- 🟡 parser 层已识别部分 smooth term；仍需“完整参数级解析 + R parity”补齐。

---

## E. 服务层与持久化（Phase E）

- ⬜ REST / gRPC 对新分布与诊断入口的系统化暴露仍待补齐。
- ⬜ SQLite 模型仓 capability snapshot schema 升级与回放验证待完成。

---

## F. 发布门禁与文档（Phase F）

- 🟡 CI 已具备部分门禁（quick parity + coverage matrix drift）。
- ⬜ “所有暴露 family 均有 R-backed d/p/q 或明确 experimental 标注”尚未达成。
- ⬜ README / 中英文文档 / capability 页面全量对齐待完成。

---

## 下一批开发优先级（从高到低）

1. **兼容层 family 的 R consistency 证据深化**：已将 ST1-4、DBI、DBURR12、BCCGuntr、BCPEuntr、BCTuntr 的脚手架扩展到多参数区间与边界场景（central/tail、edge-heavy μ、更宽 p 区间）；其余兼容层 family 持续补齐中。  
2. **将兼容层逐步替换为专用数值实现**：优先 ST1/ST2/ST3/ST4、DBI、DBURR12、Box-Cox untr 系列。  
3. `distributions3` 统计矩接口从“部分 family 支持”升级为“可配置回退 + 全家族可用性声明”（已新增 `moment_availability` 声明接口与单测，后续继续扩展解析矩覆盖）。  
4. 为当前已注册但无证据的 family 批量补 d/p/q R consistency（按家族分组推进）。

---

## 人工检查点（本轮）

- 请优先确认该清单状态标注是否符合你对“一期目标”的验收口径。  
- 若认可该口径，下一轮将直接按“下一批开发优先级”第 1 项开始实现。
