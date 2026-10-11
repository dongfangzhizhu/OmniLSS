# OmniLSS 项目长期记忆

> 纪律条目按"踩过的坑"组织。新条目先找归属，合并而非追加。

## 跑测试 / 覆盖工件

- tests/ 425 文件全量 >1.5h → **分块跑**；慢批 `run_in_background`（前台 2min 超时=SIGTERM 假失败）。
- **tests/ 或源码变更 → 同 commit 再生覆盖工件**：`generate_coverage_artifacts.py --date` + `coverage_diff_report.py` → `result: OK`。既有失败判定：已提交 `git checkout <c>~1`，未提交 `git stash -u`。

## 四条通用纪律

1. 既有用例失败，先分代码错 vs 期望错（D2.35：7 个失败 5 个是期望错）。
2. "R 没说" ≠ 常识可补；假拒绝比漏检更糟。
3. 写断言前先看实际数值、也看对通道（`-inf` 被把 exp 成 0.0；密度非有限走 log=True）。
4. **活 R 对照是唯一可信 parity 证据**。bridge：`tests._r_bridge_helper`；family 传字符串名；R 的 stop() → RuntimeError。

## parity 五判据（D2.18）

1. 复现 gate 对齐三件事：族列表（共享 rng(20261006) 流顺序敏感）、网格 `--n`（默认 8）、拟合配置（sigma_formula="~1", max_iter=20）。正确入口 `_build_jobs`。
2. rel 方向定性质：**我们更高=缺陷；我们更低=先疑 R 提前停**。两步判定（①R 轨迹 ②R 的 d\* 评我们系数，先 self-check）缺一不可；登记 `_RSDEV_R_STOPPED_EARLY`，数字漂移退回 fail。已登记 ST1/BCPE/ZINBI/ZIPIG/JSU。
3. 系数是 link 尺度；R fit 顶层按参数名平铺取 coefficients；`y~1` 让 mu 标量。
4. kernel 开关分支单独核；R 自己不一致：**域内复现 R**（DBI 窗口），**域外不复现**（PIG 非整数）。
5. parity 断言问活 R，不硬编码 16 位常数。

## 族 parity 扫描器（排查族缺陷默认入口）

`tools/rs_family_parity_scan.py`，146 族**必须后台** + `--timeout`（默认 180s 必需；慢族 300s 补跑）。坑：deriv 对响应约定错位盲（dpqr 才抓）；skip≠fail 且可能是探针元数错伪装；探测参数必须落在**参数自己的合法域**（D2.28）。

## 收敛与地板

- converged=True 只是步长塌缩 ≠ 极小。**量级分离不是判据，"沿梯度走一步能否降 deviance"才是**。
- 分轴切片前先确认基准点是驻点；地板正当与否唯一判据：**活 R 在边界另一侧是否有定义**。
- 加预算反而更差 = 发散，入口常是初值；漏分支让它 raise，静默兜底就是缺陷。
- `g_dev` 已是完整 deviance，别再乘 2。同名参数按角色不按前缀。monkey-patch 确认落到被用的绑定。
- plateau 只是症状：地板比 R 落点高 → 根因在别的参数。

## rel 荒谬 = 约定层问题

1e90/inf 不是数值分歧，是参数/约定送错的签名。共享 rng 流级联判据："**R 侧数字也变**"。

## 密度/导数分裂

- "密度对、导数错"根因常在密度；修好一处可能更差（TF2），**不要单步变好就停**。
- 同一密度两实现（dpqr_functions vs distributions 的 `_log_density`）**必须同步改**。
- 转发壳先问：被转发的是同一个函数吗（TF2 签名偷换）。

## R 探针读取要点

$nu.initial 是 expression 要 eval；jsonlite 丢命名用 as.list；R 的 NULL 双判空；gamlss.family 无 d/p/q 槽；`G.dev.incr` 收**逐观测拟合值**（eta 重构+逆 link）。

## 零变体 p/q/r（D2.27/D2.28）

build_ad_family 不传 p/q/r 装 **nan 占位**（能力缺失静默）。hurdle `p=tau@0`，否则 `tau+(1-tau)(F-F0)/(1-F0)`；mixture `tau+(1-tau)F`；组合 `_zero_variant_tails`；epsilon 封闭词汇：ZAP=0、16 族 1e-10、ZIBNB/ZINBI=1e-7。

## 数值库精度（D2.29）

jax.scipy.special.betaln 自带 ~3.5e-8 误差 → 手写 `_lbeta`。**跨库同名数值函数先做精度对照再信。**

## 登记绑定样本流；cross-check 先验 R 自洁（D2.30）

登记数字只在**全族流**有效；子列表跑 → 样本移位假警报（D2.36 重踩确认：`--families` 子集使 R 侧数字整体漂移，targeted 扫描 pass/fail 全部不可信）。cross-check 前先验证 **R 自身自洁**（R 的 dZAP recycle artifact；实现失误不复现，约定差异才复现）。helper 缓存排序前缀到目标族为止（精确等价快 4 倍）。

## 算法对齐粒度（D2.36）

- "对齐 R 的 glim.fit" 的对齐单位是**内层循环结构**：`while(|olddv-dv|>cc && itn<cyc)`，cc=0.001、**cyc=50**（活 R；GLIMControl 旧值 20 从未对过）；减半 break `(olddv-dv)>cc` 非 `dv<=olddv`。docstring 声称的 "matching R" 必须对照源码逐条核。
- **初值采纳与其阻塞因素必须同 commit**：内层对齐后 Box-Cox sigma=0.1+BCCG/BCT nu=0.5 直 pass（BCPE 122.816 vs R 122.826，登记撤除）。改内层语义前先 monkeypatch 影响面实验（登记族漂移 ≤2.4e-5 rel 则登记数字仍有效）。
- monkeypatch 实验与实现**语义必须完全一致**（d236b 只 patch 了 max_iter/tol，实现还改了减半判据 → ZIPIG 数字又漂）。
- **JSU 拟合轨迹缺陷待修**（D2.37 候选）：与 R 同初值下多数 20 点样本我们停在 ~640 劣质驻点（R<6），scan 样本侥幸走对。
- fixture 病态铁证三例：BCT 6 点（D2.35）、JSU 6 点（D2.36，R 振荡 -14→10355）；修法=20 点双侧收敛样本 + 共享 helper。

## 同名不同义与占位（D2.34/D2.35）

- JSU/JSUo、SHASHo/BCPEo 的 "o"=original 参数化（公式完全不同）；**BCCG/BCT/BCPE 的 mu/nu link=identity，o 变体才是 log**。三关同挂+量级 0.3~1=送错分布签名，先活 R dump 验 link 身份。
- 提前停登记新情形：默认 20 步我们**高于** R（nu 长平台）→ 登记 `n_cyc=200`，gate 按建立 case 的预算跑。**现存登记：ZIPIG/ZINBI**（BCPE/ST1/JSU 已因 D2.36 内层对齐 gate 直 pass 而撤除）。
- R-backed 测试首次真跑=测试的验收不是回归（skip 假绿）；测试族名用 R 真名大小写。
- 符号证据按测试文件**标识符引用**判定：fam.p(...) 间接调用不算，按名 import 才算。
- b14 组 SHASHo/SHASHo2/BCPEo/BCTo 仍是 nan 占位（不在扫描器 146 族名单），待补。

## 环境（用户 2026-10-10 明确）

- `.venv/` = 项目 Python 环境（仓库根，Windows 3.10.16 + jax 0.6.2），**优先 `.venv/Scripts/python.exe`**。
- `debianjaxgpu/` = WSL Debian 的 JAX GPU 环境，非本项目源码，**勿清理勿修改**。
