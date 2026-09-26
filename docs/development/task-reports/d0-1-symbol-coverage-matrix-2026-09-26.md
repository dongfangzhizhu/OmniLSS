# D0.1 真实 R↔Python 逐符号 coverage 矩阵（2026-09-26）

## 目标

`docs/development/deliverable-plan-2026-09-26.md` 的 D0.1：把 registry inventory 式的
coverage 工具替换为**真实**逐符号矩阵，解析 12 个 NAMESPACE、R 函数定义、Python
registry/exports、测试与 golden manifest，JSON/Markdown 可从源码重生成，每条含
r/python 符号、四层状态、证据路径与 commit。

## 实现

- 新工具 `tools/generate_symbol_coverage.py`：
  - R 侧：`parse_namespace()` 解析 `export*()`/`S3method`（丢弃 `exportPattern` 等模式）；
    `index_r_definitions()` 扫描 `name <- function` 定义并记录 `文件:行号`；
    `r_package_version()` 读取 vendored 包 DESCRIPTION 版本。
  - Python 侧：`index_python_surface()` 用 AST 索引各模块 `__all__`、非下划线定义与
    `raise NotImplementedError` stub；`index_test_evidence()` 反向索引 418 个 `test_*.py`
    的标识符；`load_family_facts()` 读取 capability registry 与 golden manifest。
  - 匹配：R 名先精确匹配 Python 名，再按 `.` → `_` 归一（如 `centiles.com` → `centiles_com`）；
    `d/p/q/r<FAMILY>` 映射到注册族（family-linked）。
  - 四层：`surface`（public_api / module_export / module_def / registry / missing）、
    `kernel`（implemented / stub / placeholder / compat_alias / experimental）、
    `parity`（`r_consistency_test` / `golden:<tier>` / `capability:<tier>` / tested / unverified）、
    `admission`（production_safe / blocked / exported_api / internal / not_available）。
  - 生成物：`docs/development/r-symbol-coverage.json`、`docs/development/r-symbol-coverage.md`；
    已接入 `tools/generate_coverage_artifacts.py`，并提供 `--check` 供 CI 复核新鲜度。
- 溯源策略：`R/` 是 vendored 且被 `.gitignore` 忽略的源码，因此逐行证据为
  「定义文件内容哈希 + 上游 DESCRIPTION 版本 + Python 主模块最后提交」，
  顶层记录 NAMESPACE / Python surface / 测试套件 / golden manifest 的内容哈希。
  全流程无时间戳、无 HEAD 引用，故生成物可重复且可 diff。

## 证据与结果

- 12 个 R 包（gamlss 5.5-3、gamlss.dist 6.1-8、…；gamlss2 标记为 out of scope）；
  1080 个 R 导出符号（in scope 1036）：surface matched 678 / missing 402；
  family-linked 531（其中 dpqr 命名 424）。
- 分层计数：surface `missing 402 / module_def 151 / module_export 96 / public_api 191 / registry 240`；
  kernel `implemented 152 / experimental 440 / compat_alias 11 / placeholder 75 / missing 402`；
  parity `r_consistency_test 126 / golden:validated 36 / capability:validated 99 /
  capability:unsupported 285 / tested 95 / unverified 439`；
  admission `production_safe 5 / blocked 526 / exported_api 85 / internal 62 / not_available 402`。
- 典型行：`gamlss.dist::dNO` → family `NO`(dpqr)、surface `module_export`、kernel `implemented`、
  parity `golden:validated`（golden case `no_y_x_seed_20260524`）、admission `production_safe`；
  `gamlss::gamlss` → `public_api`/`implemented`/`r_consistency_test`/`exported_api`（118 个测试引用）。
- 真实缺口（首次以逐符号方式可见）：`gamlss.ggplots` 109 个导出仅 1 个有 Python surface；
  `gamlss.prepdata` 59 个导出 0 匹配；`gamlss.data` 数据集未打包。这些是 D1/D2 的输入，
  而非历史文档中的推算值。

## 验证

- `python tools/generate_symbol_coverage.py`：约 11s 生成两份生成物；
  连续两次运行文件哈希一致；`python tools/generate_symbol_coverage.py --check` 返回 0。
- `cd omnilss; python -m pytest tests/test_symbol_coverage_tool.py -q`：14 项通过
  （含 1 项 `slow` 的 CLI `--check`；覆盖 NAMESPACE 解析、定义行号、行尾无关哈希、
  JSON 往返与逐行结构、四层自洽、生成物新鲜度、CLI `--help`）。

## 已知边界

- S3 方法（`plot.gamlss`、`predict.gamlss`）在 R NAMESPACE 中未必导出，Python 侧同名方法
  通过 generic 提供，矩阵只在 R 侧确实导出时才产生行；不要用行数代替行为等价性判断。
- 家族行不调用 `resolve_family()`（单族初始化可达 10s，会使再生不可接受）；
  kernel 深度以 capability registry 的 `implementation_status` 为准，不重复验证 dpqr 数值等价。
- 生成物哈希覆盖测试套件，因此新增/修改测试后必须重新生成，否则 `--check` 会（正确地）报失效。

## 后续

- D0.2：golden manifest 模板与 `--require-r` fail-closed 门禁；
- D0.3：omnilss / ondil / gamlss-python 分层 collection 固化到 CI；
- D0.4：在本矩阵之上输出 missing/extra/alias/placeholder 差异报告并阻断回归。
