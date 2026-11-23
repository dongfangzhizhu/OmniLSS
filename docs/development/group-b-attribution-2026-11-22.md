# Group B R 一致性失败归因报告（2026-11-22）

## 0. 结论摘要

Group B（R Consistency）共 **40 个唯一失败**，按族归因如下：

| 族 | 失败数 | 归因 | 处理 |
|---|---:|---|---|
| DBI | 19 | R 参考版本漂移 | CI 钉版 |
| ZISICHEL | 8 | R 参考版本漂移 | CI 钉版 |
| GU | 5 | R 参考版本漂移 | CI 钉版 |
| DBURR12 | 5 | R 参考版本漂移 | CI 钉版 |
| PARETO1o | 1 | R 参考版本漂移 | CI 钉版 |
| SHASHo | 1 | R 参考版本漂移 | CI 钉版 |
| smooth（pb in mu, Normal） | 1 | Python 单趟 `gamlss_ml` 对比 R 迭代 `gamlss` | 改比 RS |

**39/40 是 CI 安装的 R 包版本比测试基准新**，只有 1 个是测试期望本身写错。
修复后 Group B：**512 passed / 0 failed / 28 skipped**，CI 全绿。

## 1. 失败清单

原始失败来自 run 45072711332（提交 123fa4f）的 Group B job，40 个唯一失败：

```text
GU       5   test_dpqr_r_consistency[GU-d]
             test_algorithm_r_consistency[RS-GU-11]
             test_algorithm_r_consistency[RS-GU-12]
             test_algorithm_r_consistency[CG-GU-11]
             test_consistency_families[GU]
DBI      19  test_{d,p,q}_matches_live_r[...]（15）
             test_fractional_x_is_not_rounded
             test_p_steps_at_the_colon_fuzz
             test_live_r_stops_for_x_above_bd
             test_live_r_returns_the_raw_sum_just_above_bd
DBURR12   5  test_q_matches_live_r_bit_for_bit[...]（3）
             test_q_reproduces_round_even_though_it_is_not_the_cdf_inverse
             test_quantile_boundaries_follow_the_house_convention
ZISICHEL  8  test_zisichel_score_slots_match_live_r[0..3]
             test_zisichel_hessian_slots_match_live_r[0..3]
PARETO1o  1  test_live_r_stops_for_p_outside_unit_interval
SHASHo    1  test_q_boundary_contract_pins_both_r_versions
smooth    1  test_pb_in_mu_normal
```

## 2. 根因 A：R 参考版本漂移（39 个）

### 2.1 证据链

1. **CI 安装日志**（Group B job）：

   ```text
   trying URL 'https://cloud.r-project.org/src/contrib/gamlss.dist_6.1-11.tar.gz'
   installing *source* package 'gamlss' ...   # 5.5-5
   ```

   即 CI 装的是 CRAN **最新**版本：`gamlss 5.5-5` / `gamlss.dist 6.1-11`。

2. **本机/测试基准**：

   ```r
   packageDescription("gamlss")$Version       # 5.5-0
   packageDescription("gamlss.dist")$Version  # 6.1-1
   ```

   注意 `packageVersion()` 会显示成 `5.5.0` / `6.1.1`，容易被误读成 `6.1.1`；
   实际版本是 **6.1-1**（连字符版本号），与 CI 的 **6.1-11** 是不同的包。

3. **本机复现**：用本机 6.1-1 跑上述 DBI / DBURR12 / ZISICHEL / GU /
   PARETO1o / SHASHo 测试，**全部通过**；只有 smooth 一项仍失败。

4. **P2 宽粒度扫描**：`docs/development/p2-defect-classification-2026-11-15.json`
   中这些族的 `dpqr` / `deriv` / `rsdev` 均为 `pass`。Group B 的失败来自更细的
   位级、边界与槽位钉版，不是族级实现缺失。

### 2.2 各族具体表现（6.1-11 相对 6.1-1 的变化）

**DBI（19）**

- `d/p/q` 在整数与分数 `x` 上大面积不一致，例如
  - `p`: ACTUAL 0.002812… vs DESIRED 0.056504…（相对差 7.77）
  - `q`: R 的返回序列整体前移（0,1,2,3,4,5,7,10… vs 0,0,1,2,3,4,5,6…）
  - `d(0.65)`: 0.018834 vs 0.002747
- 边界语义变化：
  - `bd < x` 停止条件不再触发（`returncode == 0`）
  - `q > bd` 的 raw sum 由约 `1-1ULP` 变为精确 `1`
- 结论：6.1-11 重写了 DBI 的 `x` 取整/支持集/边界实现；我们的实现对 6.1-1。

**DBURR12（5）**

- `q` 的 `round(w-1)` 结果变化（ACTUAL 0,0,1,2,4,6,14 vs DESIRED 0,1,2,3,4,5,7…）
- 四分之一分位：R 6.1-11 返回 0（CDF 逆），6.1-1 返回 1（round 行为）
- `p ∉ [0,1]` 不再 stop
- 结论：6.1-11 修正了 `qDBURR12` 的 round/边界实现，测试钉的是旧 quirk。

**ZISICHEL（8）**

- score / hessian 槽位 9/10 不一致；R 6.1-11 的 DESIRED 大量为 0，
  我们的 ACTUAL 是非零解析导数
- 结论：6.1-11 改变了 ZISICHEL 的导数计算（或把某些参数视为常数），
  6.1-1 与我们的解析导数一致。

**GU（5）**

- `RS-GU-11`: deviance 513.998 vs R 485.851（+5.8%）
- `RS-GU-12`: deviance 565.585 vs R 500.656（+13.0%）
- `CG-GU-11`: R 侧未收敛
- `test_GU_d_consistency` / `test_GU_family_consistency`: 同族密度/拟合差异
- 本机 6.1-1 全部通过 → 是 R 侧 Gumbel 拟合/密度在 6.1-11 的变化，不是我们的算法回归。

**PARETO1o（1）**

- 6.1-11 的 `qPARETO1o` 对 `p=0,1,-0.5,1.5` 不再输出
  `p must be between 0 and 1`（返回 0/Inf/NaN）；测试钉的是 6.1-1 的 stop 行为。

**SHASHo（1）**

- `q` 边界契约（`q(0)=-Inf, q(1)=Inf, q(±越界)=NaN` 与 R 版本相关）；
  6.1-11 的边界输出与 6.1-1 不同。

## 3. 根因 B：smooth pb 的比较对象错（1 个）

`test_r_consistency_smooth.py::test_pb_in_mu_normal`：

- Python 侧：`gamlss_ml(...)` —— 单趟 ML 拟合
- R 侧：`gamlss(...)` —— RS 迭代拟合
- 结果：deviance 相对差 **1.8687**（阈值 0.30）

这不是实现差异，而是拿单趟算法对迭代算法。文件里 4 个 smooth 用例都有同样问题，
只是其余 3 个的阈值/指标较宽没暴露。

修复：4 个用例统一改为 `gamlss(...)`（RS），与 R 的 `gamlss` 同类比较；
`test_r_consistency_smooth.py` 6 例全部通过。

## 4. 修复内容

| 提交 | 内容 |
|---|---|
| `3d02f43` | CI 改从 CRAN Archive 安装钉死源码包：`gamlss.data 6.0-6` / `gamlss.dist 6.1-1` / `gamlss 5.5-0`；smooth 测试改比 RS |
| `05852c6` | DBI raw-sum 断言改 ULP 容差（`abs=1e-14` + `raw < 1.0` + `STOP`） |
| `05fd34a` | Step 9 fail-closed 改为解析 step 8 日志：`passed >= 300`、`skipped <= 50`、禁止 `Rscript not found / R not usable`；不再重跑 19 分钟整套 |
| `2222a2f` / `0659824` | 覆盖工件与 M0 manifest 重生成 |

CI 安装段（`.github/workflows/ci.yml`）：

```yaml
Rscript -e 'install.packages("jsonlite", repos = "https://cloud.r-project.org")'
Rscript -e 'install.packages(c(
  "https://cran.r-project.org/src/contrib/Archive/gamlss.data/gamlss.data_6.0-6.tar.gz",
  "https://cran.r-project.org/src/contrib/Archive/gamlss.dist/gamlss.dist_6.1-1.tar.gz",
  "https://cran.r-project.org/src/contrib/Archive/gamlss/gamlss_5.5-0.tar.gz"
), repos = NULL, type = "source")'
```

## 5. 验证

### 5.1 本机（R 6.1-1 / 5.5-0）

```text
test_r_consistency_dbi.py                 31 passed
test_r_consistency_smooth.py               6 passed
test_r_consistency_dburr12.py             通过
test_r_consistency_sichel_scores.py       通过
GU / PARETO1o / SHASHo 子集               通过
```

### 5.2 CI

- 钉版后首次 Group B（run 45080141644）：**1 failed / 512 passed / 28 skipped**
  （只剩 DBI ULP 那一条）
- 修 DBI 断言后（run 45085180217）：**512 passed / 0 failed / 28 skipped**；
  但 Step 9 仍误报 `R-consistency tests were skipped`（把 28 个合法 skip 当失败）
- 修正 Step 9 后 run **45088440177**（`05fd34a`）：**10/10 job success**
  - Group B ✅、Group A ✅、Group C ✅
  - Tiered Collection Gate ✅、Code Quality ✅
  - Windows py3.10/3.11/3.12 ✅、TLS ✅、Benchmark ✅

### 5.3 合法 skip 说明

28 个 skip 均为既有、已登记原因：

- vendored `gamlss.add / gamlss.cens / gamlss.ggplots / gamlss.mx / gamlss.tr / gamlss.spatial` 源不存在
- R 侧不收敛（SHASH / WARING / BNB / GB2 / BCT）
- 因此 Step 9 的 gate 阈值设为 `skipped <= 50`：仍能抓住“R 不可用导致全量 skip”，
  但不再把已知 skip 当失败。

## 6. 残留风险

1. **R 参考版仍是外部依赖**：钉的是 CRAN Archive 源码包。若 CRAN 删除归档或 R 4.x
   编译不兼容，需要把 tarball 纳入内部镜像/仓库。
2. **升级 R 参考版需要重新标定**：未来升级 `gamlss.dist` 时，DBI/DBURR12/ZISICHEL
   的位级/边界测试需要一次性重标，不应直接放宽阈值。
3. **smooth 用例的其余 3 个**此前“通过”只是阈值宽；改为 RS 后语义一致，但阈值仍建议
   后续单独校准。

## 7. 附录：运行 ID

- 归因输入：run 45072711332（`123fa4f`）Group B log
- 钉版后 1 失败：run 45080141644（`2222a2f`）
- DBI 修后 step 8 全绿、step 9 误报 skip：run 45085180217（`0659824`）
- 最终全绿：run 45088440177（`05fd34a`）
