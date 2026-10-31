# OmniLSS 交接文档（2026-10-31）

> 用途：新会话入口。先读本文件，再读 `.workbuddy/memory/2026-10-23.md`（本轮全部实测细节）与
> `DEVELOPMENT_PLAN.md`（阶段状态）。本文只写结论、命令、待办、坑。

## 0. 环境事实

- 仓库：`d:\AI\my\githubprojs\OmniLSS`，分支 `main`，交接时 HEAD = `6567edf`。
- R 可用：R 4.6.0 + `gamlss` / `gamlss.dist` / `jsonlite`。vendored R 源码在 `R/gamlss.dist/R/*.R`（124 个文件，
  **包含未导出的族**，如 LOGSHASH / LOGSHASHo / DELAPORT）。
- Python：`d:\miniconda3\python.exe`；测试用 `python -m pytest`。
- **前台命令 2 分钟会被 SIGTERM 杀掉**（表现为"假失败"）；长任务一律日志重定向 + 后台运行。

## 1. 当前状态（可交付判定）

**结论：可以交付**（依据连续两次一次性跑完的全族复扫）。

```
python tools/rs_family_parity_scan.py --json tmp/scan.json      # 146 族，约 45~60 分钟

deriv : 144/146 pass, skip=2
dpqr  : 120/146 pass, skip=26
rsdev : 82/146 pass, skip=29, warn=35        <- 没有 fail 行
```

- 该结果在 D2.117 与 D2.122 各跑出一次，数字完全一致。
- 功能性覆盖：rqres 覆盖 146/146、`has_dqr` 146/146、placeholder 族 0。
- 已知限制（建议写进 release note）：
  - `deriv` 2 skip：GT（探针 >180 s 被杀，Bessel 路径慢）、LOGSHASHO（无支撑网格能给出有限密度——R 的
    dLOGSHASHO 本身不是密度）。
  - `dpqr` 26 skip：R 侧无对应函数（18 个 BINOM_/CONTR_/COUNT_ 系列、BCCGUNTR/BCTUNTR、MULTIN、ZILG 等）；
    其中 22 个是真无参考，非缺陷。
  - 35 个 `rsdev` warn：R 未收敛 / R 发散 / R 的 sigma 地板 / R 密度不逐元素 / R 早停——每一个都有可核对的原因，
    没有一个是"我方密度不一致"。
  - Sichel 三族拟合 172 s/周期（见待办 3.2）。

## 2. 已完成（本轮会话，D2.92–D2.126）

### 2.1 把 13 个 rsdev fail 全部定性（D2.94–D2.103）

从"14 个无法解释的 fail"收口到"0 fail"，每个非 pass 都有归属：

| 判定 | 族 | 依据 |
|---|---|---|
| R 密度缺陷（不逐元素 / 不是密度） | ZANBI、ZAP | R 的 dNBI / dPO 不回收标量 x |
| R 发散（扫 n_cyc 见其振荡） | SEP2、SEP3、SEP4、GT | R 的 deviance 随 n_cyc 来回跳 |
| R 的 sigma 地板（数值不可信） | ZABNB、WARING、ZIBNB | R 的 sigma = 2.220446e-16 |
| R 早停 / 未收敛 | ZAZIPF、GAF、GP、PARETO2、PE2、ZIPIG、PARETO2O | R 的轨迹一平到底或仍在缓降 |

数值修好：SEP3（inf 到 123.46，补 R 初始值 + rs_fit 历史最优回退）、SEP4（1.3e27 到 119，补 R 初始值）、
DELAPORT（223.87 到与 R 一致，缺 DEL 的初始值）。

### 2.2 工具与工程

- **D2.105**：扫描器建 job 250 s 到 183 s；同进程重复调用到 0 s（进程内 payload 缓存 + 非上报族提前跳过）。
- **D2.107**：vendored 源码参考通道（tools/r_parity_batch.R 的 vend_env，包优先）→ LOGSHASH 的 dpqr 从 skip
  变 pass；新增 ("LOGSHASH","q"): 1e-4 容差及说明（qLOGSHASH 字面就是 exp(qSHASH(...))）。
- **D2.108/109**：DELAPORT 即 R 的 DEL（d/p/q 逐位一致 ~1e-15），接回别名并补上它缺的 DEL 初始值 → 三闸全绿；
  新增 test_delaport_is_r_del.py、test_delaport_fits_like_del.py。
- **D2.111**：IGA 补 IG 的初始值 → IG/IGA 三闸全绿；新增 test_iga_fits_like_ig.py。
- **D2.120**：BEo 与 GEOMo 补 live R 的初始值 → 三闸全绿；新增 test_beo_geomo_initial_values.py。
- **D2.121**：LOGSHASHo 共享 LOGSHASH 的 4 条初始值（R 源码逐字相同）；PARETO2o 试过 R 值但我们更差
  （176.072 到 178.926），已回退并在代码里注释防复发。
- **D2.117/122**：两次全量复扫确认 0 fail。
- **D2.125**：修掉 _RSDEV_R_STOPPED_EARLY 里 GAF/GP/PARETO2/PE2 的重复条目（D2.101 追加时引入）。
- **D2.126**：确认 2 个测试失败先于本轮存在（在旧 commit 851ccf6 上也失败）。

### 2.3 防呆测试（都不需要 R，跑得快）

- omnilss/tests/test_rsdev_registry_keys.py：4 张 rsdev 表不得有重复键；源码条目数等于运行时表长；
  一族最多落一张表。（D2.125 就是它抓到的）
- omnilss/tests/test_family_name_branches_are_live.py：_fitting_init.py 不得出现"是注册表键但不是族对象名"
  的字面量（等于永远不触发的分支）；并钉住 10 组键到名的映射。
- test_delaport_fits_like_del.py / test_iga_fits_like_ig.py / test_beo_geomo_initial_values.py。
## 3. 待完成（按优先级）

### 3.1 修 2 个既有测试失败（最高优先级，非本轮引入）

- test_igamma_local_fitting.py::TestIGAMMALocalFitting::test_intercept_model_converges_locally：
  实际 0.649055 vs 期望 0.648814，rel 3.7e-4 大于 1e-4 容差。先按"扫 n_cyc 再定性"的流程判断是收敛差还是期望错。
- test_r_consistency_batch3_4_remaining.py::TestBatch34RemainingConsistency::test_beoi_intercept_only：
  需 live R；先在 live R 上复现再判。
- 验收：两个测试变绿，且不引入 rsdev fail。

### 3.2 log_bessel_kv 的 JAX 原生实现（唯一挂账的功能项）

- 现状：omnilss/src/omnilss/bessel.py 用 jax.pure_callback 调 scipy（每次求值一次主机往返，并切断 jit 融合）。
- 已测量的量级（D2.124）：jit 下 n=48、nu=-0.5：primal 1.05 ms、grad 2.18 ms（grad/primal = 2.1）。
  所以 172 s/周期来自调用次数（每周期十几万次），不是"回调次数多"；减少回调次数收益有限，真正收益来自
  消除主机往返 + 让循环能在设备侧融合。
- 验收标准（两条都要）：1. 与 scipy 在 nu 取 -5 到 20、x 取 1e-8 到 1e3 的对数网格上逐点 1e-12 相对误差；
  2. Sichel 拟合的周期时间下降（不只看 deviance 一致）。
- 建议单独一轮、预算充足再做：特殊函数最怕"看着对、精度不对"。

### 3.3 按族独立播种 + 一次性重登记证据（工程债）

- 现状：为让子集与全量同流，单族请求也要走完 146 族构建（183 s，D2.105 后）；缓存是进程内的，单独 CLI 调用
  仍各付一次。
- 做法：给每族独立播种（不再依赖共享 RNG 顺序），然后一次跑完把所有登记表里的 dev_ours / dev_r 重测一遍
  （现在表里不少数字是旧样本的，逐族手工重测很慢）。
- 若改播种，扫描器样本会整体变化，所有登记数字必须同时重测，否则闸会大面积 fail。

### 3.4 跑一次完整测试套件

- python -m pytest omnilss/tests -q（全量超过 1.5 h，分块 + 后台跑）。
- 本轮用定向子集漏掉了失败，全量跑一次才能知道全库真实状态。

### 3.5 Phase 8 收尾（DEVELOPMENT_PLAN.md 原有项）

- 8.3 用户文档与教程（API 参考 + 迁移对照表 + 8 篇教程校准）。
- 8.5 gamlss() method 名与行为一致性核对 + 全库 lint/type（CI 目前只有测试，没有 lint/type 步骤）。

### 3.6 可选：继续减少 dpqr 的 26 个 skip

- 真无参考的 22 个不要动；可争取的是有 vendored 源码的族（通道已建好，见 D2.107）。

## 4. 必须遵守的坑（本轮全部踩过）

1. **字典字面量里重复键是静默错误**：后出现的覆盖先出现的。往 _RSDEV_* 表加条目前必须确认该族是否已在表里；
   若已存在就改它的数字，不要追加（D2.96、D2.125）。test_rsdev_registry_keys.py 会抓这个。
2. **按族名分支必须用 resolve(key).name，不能用注册表键**。所有"原参数化"别名二者不一致：JSUO 到 JSUo、
   SHASHO 到 SHASHo、BCCGO 到 BCCGo、GEOMO 到 GEOMo、BEO 到 BEo、PARETO2O 到 PARETO2o、
   LOGSHASHO 到 LOGSHASHo（DELAPORT、IGA 是一致的）。写注册表键等于死代码（D2.118）；也曾因此把原本
   生效的分支改死，让闸从 0 fail 变成 1.3e13（D2.113）。
3. **判断 R 是"早停"还是"发散"必须扫 n_cyc**：早停是单调上升后平掉，发散是来回跳。判错会把真相记歪（D2.94）。
4. **R 报得越低越可疑**：sigma 撞 2.220446e-16 地板时，R 可能给出偏小的数（ZABNB 202.8 对平台 212.7、
   WARING 208 对 221.7），也可能给出荒谬大数（SEP2/3/4 的 1e4 到 1e27）。每次都要实测安全 sigma 平台（1e-4 到 1e-10）。
5. **"我们更差"先检查 R 那个数是在哪里打的分**（D2.98 推翻过一次误判）；也别急着把 R 的初始值搬过来，
   PARETO2O 搬了反而更差（D2.121）。
6. **判断"这个失败是不是我引入的"**：git worktree add --detach 临时目录 旧commit，在旧代码上跑，跑完
   git worktree remove --force（比 git stash 安全，不动当前工作树）（D2.126）。
7. **测试通过不等于验证有效**：断言"通用默认值恰好满足"的测试是空测试（D2.118 删过一个）。
8. **改初始值类改动必须跑真实闸验证**，不能靠局部 monkeypatch 探针外推：SHASHO 逐个隔离时每条分支都无害，
   但源码层面改就炸（D2.113/115）。
9. **梯度 / 初值类改动用"同数据上两族拟合必须一致"的测试钉住**（模板：test_delaport_fits_like_del.py）。

## 5. 关键文件索引

- tools/rs_family_parity_scan.py：三闸扫描器 + 全部登记表（_RSDEV_R_DIVERGED、_RSDEV_R_STOPPED_EARLY、
  _RSDEV_R_REFERENCE_INVALID、_RSDEV_R_DENSITY_RECYCLED、_RSDEV_PATH_DIVERGENT、_DPQR_TOLERANCE 及
  _DPQR_TOLERANCE_NOTE、_R_FAMILY_ALIASES、_R_ARGNAME_MAP）。
- tools/r_parity_batch.R：单个 R 进程跑全部族；vendored 源码回退（vend_env）；族名解析（精确 + 大小写不敏感）。
- omnilss/src/omnilss/_fitting_init.py：初始值表（_initial_parameter_value / _initial_mu_target /
  _initial_mu_beta / _initial_sigma），分支用 family.name。
- omnilss/src/omnilss/bessel.py：log_bessel_iv / log_bessel_kv（scipy + pure_callback，待原生化）。
- .workbuddy/memory/2026-10-23.md：本轮全部实测数据（D2.82 到 D2.126）。
- DEVELOPMENT_PLAN.md：阶段状态表（已含 R 一致性里程碑行）。

## 6. 如何取 R 侧事实（最快的方式）

```r
# 1) 某族是否已导出
suppressPackageStartupMessages(library(gamlss.dist))
"LOGSHASHo" %in% getNamespaceExports(asNamespace("gamlss.dist"))

# 2) 某族的初始值（权威，比读源码可靠）
fs <- get("BCCGo")(); fs$mu.initial; fs$sigma.initial; fs$nu.initial

# 3) 逐周期轨迹（判断早停 vs 发散）
for (k in c(2,5,10,14,20,40,60)) cat(k, deviance(gamlss(y ~ x, sigma.formula=~1,
    data=d, family=ZABNB, n.cyc=k, trace=FALSE)), "\n")

# 4) 安全 sigma 平台（判断 R 的数是否打在地板上）
for (sg in c(1e-4,1e-6,1e-8,1e-10)) cat(sg, -2*sum(dZABNB(y, mu=mu,
    sigma=rep(sg,length(y)), nu=rep(nu,length(y)), tau=rep(tau,length(y)), log=TRUE)), "\n")
```

扫描器样本可以缓存下来反复用（省掉 183 s 的构建）：

```python
import sys, json; sys.path.insert(0, "tools"); sys.path.insert(0, "omnilss/src")
import rs_family_parity_scan as S
jobs, index = S._build_jobs(["SHASHO", "SHASHO2"], 8)
json.dump({f: {"fit": jobs[index[f]["fit"]]["data"]} for f in ("SHASHO", "SHASHO2")},
          open("tmp/shasho_sample.json", "w"))
```

## 7. 一句话总结

「对齐 R」这条线已全部完成并有闸验证支撑（146 族三闸 0 fail，连续两次一次跑完）。剩下的工作里，只有 3.1 的
两个既有测试失败可能挡发布；3.2 是唯一挂账的功能项（性能，需单独一轮）；3.3 到 3.5 是工程债与交付收尾。