# D0.2 golden manifest 模板与 `--require-r` 门禁（2026-09-26）

## 目标

`docs/development/deliverable-plan-2026-09-26.md` D0.2：固定 R/包/Python/JAX 版本、源码 hash、
x64、seed、容差，使任一 parity 测试可复现；无 R 时只能标 `smoke-only`；
`--require-r` 门禁 fail-closed（依赖缺失/版本漂移/未开启 x64 即失败，而非 skip）。

## 实现

- 模板 `omnilss/tests/fixtures/r_golden/manifest_template.json`（schema_version 1）：
  - `pinned`：`R_version_prefix`、`gamlss_version`、`gamlss_dist_version`、
    `python_version_prefix`、`jax_version_prefix`、`jax_enable_x64: true`；
  - `required_case_fields`：`family/formula/seed/n/R_version/gamlss_version/
    gamlss_dist_version/content_hash`（是 `tools/stage0_gate.py` 的
    `PROVENANCE_FIELDS` 的严格超集，二者一致不冲突）；
  - `tolerances`：`default` / `fitting` / `dpqr`（容差唯一数据源），case 可用
    `tolerance` 字段覆盖；`cases`：9 个期望 case 及其 kind。
- 门禁 `tools/golden_manifest_gate.py`：
  - `--check`：只校验模板 + manifest + fixture 内容哈希（快路径，不探环境）；
  - 默认：校验 + 环境探针，输出 `pinned` / `smoke-only` 标签（发布外运行仅标签，退出码只由校验决定）；
  - `--require-r`：校验 + 探针，环境问题也计入失败（fail-closed）；
  - `--json` 输出机器可读报告；`--template/--manifest/--fixtures-dir` 可注入路径供测试。
  - 探针：单次 `Rscript -e` 取 R 版本 + `gamlss`/`gamlss.dist` 版本（缺失输出 `MISSING`），
    Python/JAX 版本与 `jax_enable_x64` 通过「先 `import omnilss`（与 parity 测试一致，任务 0.7 的
    x64 开启点）再读 `jax.config`」获得；探针器与 `runner` 均可注入，测试无需 R/JAX。
  - 可复现性核心：按 `generate_golden.R` 的 djb2（对 JSON 文本、去掉行尾换行）复算
    9 个 fixture 的 `content_hash`，篡改或未再生成都会被 `--check` 拦下。

## 证据与结果

- `python tools/golden_manifest_gate.py --check` → `validation: ok (9 cases, tolerance 1e-06)`，exit 0。
- `python tools/golden_manifest_gate.py --require-r` →
  `python 3.10.19, jax 0.6.2 (x64=True), r=available [R version 4.6.0; gamlss=5.5.0, gamlss.dist=6.1.1]`，
  `label: pinned`，exit 0（本机与模板完全一致）。
- 9/9 fixture 内容哈希在 Python 侧重算一致（`djb2_hex(text.rstrip("\n"))`）。
- 顺带修复过期断言：`tests/test_r_golden_manifest.py` 原先用父目录 glob 比对
  `regenerated/manifest.json` 的 key，被后来新增的 `sichel_dpqr_high_counts_20260912`（历史锚点，
  由 `generate_sichel_golden.R` 生成）打破 → 改为比对 `REGEN_DIR`，并补充说明父目录还存放
  历史锚点与 `manifest_template.json` 等自维护输入。

## 验证

- `cd omnilss; python -m pytest tests/test_golden_manifest_gate.py tests/test_r_golden_manifest.py tests/test_r_golden_fixture_schema.py -q` → exit 0。
- 新增 18 项（17 快 + 1 `slow` 真实 CLI `--require-r --json`）：模板校验、9/9 哈希复算、
  篡改检测、case 集合漂移、字段缺失、版本漂移、容差解析（默认/kind/case 覆盖）、
  `classify` 四种环境（pin、无 R、版本漂移+缺包、x64 关闭）、探针解析与 `FileNotFoundError`、
  `--require-r` fail-closed vs 仅标签语义、CLI `--check` 快路径。

## 已知边界

- 模板 pin 的是 `pinned.*_version(_prefix)` 期望值；升级 R/JAX 时必须同步改模板（这是刻意的
  fail-closed 行为：不改模板就无法宣称 pinned）。
- `generate_golden.R` 未改（避免在不重新生成 fixture 的情况下引入不可验证的 R 变更）；因此
  manifest 仍由 R 侧写入现有字段，容差的唯一来源是模板，`effective_tolerance()` 负责解析。
- x64 结论依赖「parity 测试会 `import omnilss`」这一前提；探针显式模拟了该行为并在
  探针失败时 fail-closed（`jax_error` → smoke-only）。

## 后续

- D0.3：omnilss / ondil / gamlss-python 分层 collection 固化到 CI（本门禁接入其中）；
- D0.4：在 D0.1 矩阵之上做 registry/R 源码差异报告。
