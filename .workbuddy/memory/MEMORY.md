# OmniLSS 项目长期记忆

> 纪律条目按"踩过的坑"组织。新条目先找归属，合并而非追加。

## 跑测试 / 覆盖工件

- 491 个测试文件全量 >1.5h → **按文件分 4 片并行后台跑**（前台 2min 超时=SIGTERM 假失败）；单进程 `-q` 只出点。
- **tests/ 或源码变更 → 同 commit 再生覆盖工件**：`generate_coverage_artifacts.py --date` + `coverage_diff_report.py` → `result: OK`。**改完测试要最后再生成一次**（先改测试再生成仍会报过期）。
- **归属判定**：`git worktree add --detach` 到改动前 HEAD 跑同一批 id；但 **worktree 里没有 gitignored 的 `R/`** → `build_coverage` 出全 0 矩阵，**覆盖/能力矩阵类测试在 worktree 的结论不可信**，只信“主树失败、worktree 通过”那几条。
- **pytest 9 的 FAILED 行带 ` - <reason>` 后缀**：当 id 传回去会报 `file or directory not found: -`，要 `split(" - ")[0]`；长列表用 `pytest @argfile`。
- 全库现状（2026-11-01 实测）：75 failed，其中 71 个先于 D2.129 存在（簇见 DEVELOPMENT_PLAN 第十六节：CG/MIXED 路由 ~10、ZABB/ZIBB `bd` 6、`SmoothFitInfo` 签名 6、`RTestBus.eval_expr` 4、旧登记数字 5…）。

- 交接文档：仓库根 `HANDOFF.md`（2026-10-31）——新会话入口，含当前状态、已完成/待完成、必踩的坑。

## 响应域 (family$y.valid) —— 只能由活 R dump 生成（D2.128/D2.129）

- R 在 `gamlss-5.R` 拟合前 `if (!family$y.valid(y)) stop("response variable out of range")`；我们在 `fitting.py:1048`（**最前面，先于任何路由**）同样硬拒。`omnilss/src/omnilss/response_domain.py` 的表 **82 条**，逐条由 `tools/generate_response_domain_golden.py`（跑 `tools/r_y_valid_dump.R`）对活 R 生成并钉进 `omnilss/tests/fixtures/r_golden/response_domains_y_valid.json`（`--check` 判过期）。**手工加条目 = 回到 D2.128 之前的漏检面**。
- 活 R 的 120 个谓词只有 9 种形状：`all(y>=0)` 40 / `TRUE` 40 / `all(y>0)` 24 / `0<y<1` 5 / `0<=y<1` 2 / `0<y<=1` 2 / `0<=y<=1` 1 / `y>1` 1 / `y>=1` 1 / `1<=y<=4` 1 / `1<=y<=5` 1 / MN3+MULTIN 因子分支 2。**`TRUE` 的族（NO/SHASH/ST*/JSU*/PARETO2*/IGAMMA/GB2…）不要自己加限制**：假拒绝比漏检更糟。
- 22 个无 R 参照的族保持宽松，并且**清单化**在测试里：19 个 `BINOM_/CONTR_/COUNT_` 骨架族 + ZILG + LOGSHASH/LOGSHASHo（后者 vendored 源码写的是 `y.valid=TRUE`）。
- 名字坑：`BCCGUNTR/BCPEUNTR/BCTUNTR` 的 R 名是 `BCCGuntr/BCPEuntr/BCTuntr`；四个内核别名族（IGA→IG、DPO1→DPO、GENERALISEDPOISSON→GPO、DELAPORT→DEL）按我们 dispatch 的名入表。查表做大小写不敏感兜底，但**表键仍是 `family.name`**。
- **R 会纠正我们的“宽松”断言**：PO 曾被当成“R 无 y.valid”的族，实测 `PO()$y.valid=all(y>=0)`。凡“R 没说”的结论都要先 dump 一次再信。
- 域闸先于路由/优化器：**用不合法数据测“更靠后的报错”会先撞 `ResponseDomainError`**（D2.129 踩过 RS_JAX 路由用例与 LOGNO 用例）；修法=换合法数据。

## 四条通用纪律

1. 既有用例失败，先分代码错 vs 期望错（D2.35：7 个失败 5 个是期望错）。
2. "R 没说" ≠ 常识可补；假拒绝比漏检更糟。
3. 写断言前先看实际数值、也看对通道（`-inf` 被把 exp 成 0.0；密度非有限走 log=True）。
4. **活 R 对照是唯一可信 parity 证据**。bridge：`tests._r_bridge_helper`；family 传字符串名；R 的 stop() → RuntimeError。**R 桥拼名必须对 namespace exports 精确优先 + 大小写兜底**（D2.42：`paste0(func, family)` 对 11 个混合大小写族必挂——R 导出名小写尾 `o`：PARETO1o/BCCGo/BCPEo/BCTo/JSUo/SHASHo/SHASHo2/BEo/EXGAUS/GEOMo/PARETO2o，我们别名用大写 `O`；探针报错被 skip 吞掉 = 门禁静默失效）。

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

$nu.initial 是 expression 要 eval；jsonlite 丢命名用 as.list；R 的 NULL 双判空；gamlss.family 无 d/p/q 槽；`G.dev.incr` 收**逐观测拟合值**（eta 重构+逆 link）。**gamlss 的 `numeric.deriv` 是前向差分**（`get("numeric.deriv", envir=asNamespace("gamlss"))`，delta=0.001；实测 = (f(x+δ)-f(x))/δ，非中心差），且按名重求值调用帧表达式——**变量必须先绑定**（`ND(exp(0.0),"nu")` 报 `object 'nu' not found`）。**仓库 vendored 的 R 源码不可全信（D2.50：NBII.R pNBII 分支条件与活 R 相反）**——initial/分支语义以活 R `deparse(body(...))` 为准。

## 零变体 p/q/r（D2.27/D2.28）

build_ad_family 不传 p/q/r 装 **nan 占位**（能力缺失静默）。hurdle `p=tau@0`，否则 `tau+(1-tau)(F-F0)/(1-F0)`；mixture `tau+(1-tau)F`；组合 `_zero_variant_tails`；epsilon 封闭词汇：ZAP=0、16 族 1e-10、ZIBNB/ZINBI=1e-7。

## 数值库精度（D2.29）

jax.scipy.special.betaln 自带 ~3.5e-8 误差 → 手写 `_lbeta`。**跨库同名数值函数先做精度对照再信。** JAX `norm.ppf`（ndtri）在 p 距 1 的 1e-5 内相对精度退化 ~1e-6，被 sinh（导数 cosh(r)~50）放大成 ~3e-4 往返误差（D2.43 实测；R qnorm 同区更准）——q 类函数的尾部 roundtrip 断言要掩码避开该区。

## 登记绑定样本流；cross-check 先验 R 自洁（D2.30）

登记数字只在**全族流**有效；子列表跑 → 样本移位假警报（D2.36 重踩确认：`--families` 子集使 R 侧数字整体漂移，targeted 扫描 pass/fail 全部不可信）。cross-check 前先验证 **R 自身自洁**（R 的 dZAP recycle artifact；实现失误不复现，约定差异才复现）。helper 缓存排序前缀到目标族为止（精确等价快 4 倍）。

## 算法对齐粒度（D2.36）

- "对齐 R 的 glim.fit" 的对齐单位是**内层循环结构**：`while(|olddv-dv|>cc && itn<cyc)`，cc=0.001、**cyc=50**（活 R；GLIMControl 旧值 20 从未对过）；减半 break `(olddv-dv)>cc` 非 `dv<=olddv`。docstring 声称的 "matching R" 必须对照源码逐条核。
- **初值采纳与其阻塞因素必须同 commit**：内层对齐后 Box-Cox sigma=0.1+BCCG/BCT nu=0.5 直 pass（BCPE 122.816 vs R 122.826，登记撤除）。改内层语义前先 monkeypatch 影响面实验（登记族漂移 ≤2.4e-5 rel 则登记数字仍有效）。
- monkeypatch 实验与实现**语义必须完全一致**（d236b 只 patch 了 max_iter/tol，实现还改了减半判据 → ZIPIG 数字又漂）。
- **JSU 拟合轨迹缺陷待修**（D2.37 候选）：与 R 同初值下多数 20 点样本我们停在 ~640 劣质驻点（R<6），scan 样本侥幸走对。
- fixture 病态铁证三例：BCT 6 点（D2.35）、JSU 6 点（D2.36，R 振荡 -14→10355）；修法=20 点双侧收敛样本 + 共享 helper。

## IRLS 的 Hessian 形式是 parity 的第二道初值（D2.60/D2.62 确立）

- **R 的 d2ldX2 常常不是真二阶导，而是观测信息近似 -dldX²**，且逐族地板不同（JSUO/SEP1/SHASH/ST5 全是 -1e-15；ST5 的 nu/tau 是 -1e-4；ZABNB 的 d2ldt2 是 -1/(tau(1-tau))）。我们的 build_ad_family 默认喂**真二阶导**——初值对齐后轨迹仍在第一个内层步分道（JSUO：第一周期 mu 斜率反号 3.41/-0.51 vs R 1.84/+0.36；只补初值 147.365 vs R 146.778 仍 fail，补 hessian 后 146.843 pass）。
- **诊断顺序**：①活 R dump 四个 dld* 与 d2ld*2（`deparse(f$d2ldm2)`）；②若 hessian 是 -score² 形式 → 写 `hessian_overrides`；③先用 monkeypatch 验证（直接改 `fam.hessian_functions[par]` 再 fit，family 对象每次新建所以安全），确认 deviance 贴近 R 再落源码。
- **score 也要转写时优先闭式**：用 `jax.vmap(jax.grad(...))` 造 hessian 会在**标量输入**下炸（vmap 要求 rank≥1）——闭式 elementwise 无此问题（D2.62 踩过）。
- **不是所有族都在分道**：修好初值后多数族直接 pass；hessian 形式只在"初值对 + 仍差 1e-3~1e-2"时才是首选嫌疑。判断依据是**逐周期系数对比**（max_iter=k, k=1..6 双方 dump），不是单点 deviance。

## d2ldt2 特殊形式清单（活 R，遇族先查）

- ZABNB：d2ldt2 = **-1/(tau*(1-tau))**（不是 -score²）；其余三参数是 hurdle 修正后的 -dld²（`dld0 = BNB$dld(y)+dBNB(0)*BNB$dld(0)/(1-dBNB(0))`，y==0 时 dld≡0）。
- ZIBNB：dldt 分段（y==0 时 `(1-f0)/(tau+(1-tau)f0)`，否则 `-1/(1-tau)`），d2ld*2 全 -dld²。
- ST5：mu/sigma 地板 -1e-15、nu/tau 地板 **-1e-4**。

## 同名不同义与占位（D2.34/D2.35）

- JSU/JSUo、SHASHo/BCPEo 的 "o"=original 参数化（公式完全不同）；**BCCG/BCT/BCPE 的 mu/nu link=identity，o 变体才是 log**。三关同挂+量级 0.3~1=送错分布签名，先活 R dump 验 link 身份。
- 提前停登记新情形：默认 20 步我们**高于** R（nu 长平台）→ 登记 `n_cyc=200`，gate 按建立 case 的预算跑。**现存登记：ZIPIG/ZINBI/PE2**（D2.38：PE2 是"R 停在劣质驻点"变体——R 长预算 200 周期也不跟进我们更低且 R 密度逐位验证的点）。BCPE/ST1/JSU 已因 D2.36 内层对齐撤除。

## 逐观测初值 roll-out（D2.38）

- **初始状态是算法对齐的第一粒度**：内层对齐后轨迹仍分道，唯一自由度是初值。R 的 X.initial 是逐观测 fitted values，压缩成系数再重构会改变第一个内层步的工作权重；rs_fit 现把逐观测初值直接交 rs_step（`_initial_mu_target`/`_initial_sigma_fitted`）。
- 71 族共享 mu.initial=(y+mean(y))/2（live R dump 全量 146 族存 omnilss/tmp/d238_all_initials.json）。**R 的 var()/sd() 全是 n-1 分母**（np.var 默认 ddof=0，移植逐个核对）。
- roll-out 的回归面 = 所有默认分支的族（两头不靠：旧 warm=mean(y) 常数、roll-out 默认=target=y 都不是 R 表达式）→ 16+2 族回归，port 后全转 pass。**绿族不碰**：ZINBI/PO/GEOM 等 gate 绿族保持旧起点。targeted 判定照旧不可信，登记数字绑定全族流。
- **重尾族常数起点不能被默认分支覆盖**：PARETO mu.initial=rep(2) 常数，target=y 逐观测 exp(y_i) 在 Pareto(2) 样本爆到 -inf。
- **`_initial_mu_target` 漏族名 = 指数爆炸（D2.50 NBII）**：通用 fallback `target=y` 是 **response 尺度**，rs_fit 按 **link 尺度**消费——log-link 族漏分支 → mu=exp(y)（e^7..e^15）→ 初值 deviance 4.8e6。**通用 fallback 只对 identity-link 族安全**；sigma 初值同理必须用 R 的原始矩表达式（var(y)，n-1），不能用对初始 mu 的残差方差。
- **族对象 `.name` 大小写 ≠ registry key（D2.51 exGAUS）**：registry key 大写 `EXGAUS`，但 `distributions_b2` 的 `build_ad_family(name="exGAUS")` 让族对象 `.name` 是**小写**。`_initial_*` 全部门 `family.name`——分支写大写静默不匹配（无报错，直接继续走 fallback，症状=缺陷依旧）。加族条目前先 `resolve_family(X).name` 打印核实；漏族名与写错大小写是同一个静默 fallback 坑。
- **通用 sigma 默认值 = 未登记族的默认陷阱（D2.51 exGAUS，第三例）**：族没有 sigma 条目时，通用默认对**初始 mu** 求残差尺度；若 mu 也落 `target=y` 默认 → RMSE(y,y)=0 → eps 地板 1e-6 → log-link IRLS 权重塌缩、deviance 差 66 倍（exGAUS 9856.41 vs R 146.905；SHASH 族 D2.40 同款，NBII D2.50 是 mu 侧同源）。**mu/sigma/nu 三个初值要一起补**——只补 mu 不补 sigma 仍在地板上。
- **RS 外循环收敛判据 = 纯绝对（D2.50，R gamlss-5.R:305）**：`G.dev.old<-G.dev+1` 播种 + `abs(change)>c.crit`；R 2003 年主动移除过相对变体。带相对项会在**伪造首查**上误触发（初值 deviance>1/rtol → rel<rtol → "0 迭代收敛" converged=True）——发散起点被报成收敛拟合。`_has_converged` 已改绝对-only。
- **门禁样本小 y 掩盖初值缺陷（D2.50）**：146 族扫描样本 y≤7 时 exp(y) 可恢复 → 全绿；只有 n=100/y≤15 的 consistency 测试暴露。**gate 绿 ≠ 族健康**，初值类缺陷要专门构造重尾/大值样本验。
- **NET 缺陷组（D2.39 修复中，语义已修正）**：R 的 NET.R 在 family 对象里标记 `parameters=list(mu=T,sigma=T,nu=F,tau=F)`——**族级** non-estimable 标记（JSU 等族未给 formula 仍估计，实测 nu.coefficients 非 NULL），不是全局顶层语义；nu/tau 固定于 nu.initial/tau.initial=1.5/2，给了 formula 也忽略（nu.coefficients=NULL）。修法=族级 fixed_parameters+fixed_parameter_defaults（新机制）+link identity；density 里 tau=max(tau,nu) 静默 clamp vs R dNET 的 stop 是另一处 dpqr 对齐待办。
- **LNO intercept-only 既有失败**（stash 验证 HEAD 同挂）：候选后续项。
- **R 的 score 槽可能与 G.deviance 自相矛盾（D2.42 PARETO1o 幽灵项）**：dldd 无 support 守卫而密度边界为 0，mu=min(y) 使最小观测永在边界 → IRLS 固定点是 `sigma=n/Σlog(y/mu)`（含边界观测的幽灵 1/σ 项，其 deviance 贡献为 0）。R 落点与闭式 10 位一致 + c.crit=1e-10 不动 = 固定点非提前停；我们低于/高于都要先算这个闭式再定性。修法=score_overrides/hessian_overrides 装 R 闭式（D2.40 SHASHo2 模式）。**固定 mu 的数据依赖默认**（rep(min(y))）走 `_resolve_fixed_parameter_values(family, data, n_obs, y)` 族名特例，不走常数 defaults；residual 路径要显式取 fixed_parameter_values 的 mu。
- R-backed 测试首次真跑=测试的验收不是回归（skip 假绿）；测试族名用 R 真名大小写。
- 符号证据按测试文件**标识符引用**判定：fam.p(...) 间接调用不算，按名 import 才算。
- b14 组 BCPEo/BCTo 仍是 nan 占位（SHASHo/SHASHo2 已于 D2.40/D2.42/D2.43 交付）；混合大小写族 BCCGO/BCPEO/BCTO/BEO/GEOMO/PARETO2O 的 dpqr 接线状态待查（R 桥大小写缺陷 D2.42 已修，门禁可真实探测）。**EXGAUS dpqr 已验（D2.51：d/p pass，q 的 1.71e-06 残留是 R uniroot 默认容差误差，已入 _Q_UNIROOT_MEASURED）**。
- **SHASH（非 o）密度积分缺陷待查**（D2.43 新发现）：同 profile 下 SHASH 密度积分 [-40,40]=0.9601、[-400,400]=0.9998（SHASHo=1.0 精确）——SHASH 族 dpqr 路径既存问题。
- **LOGSHASH 已交付（D2.52）**：安装版 gamlss.dist 6.1.1 **没有** LOGSHASH/LOGSHASHo（vendored 是更新开发版；exports 实测无）——无活 R 直接对照时，若 R 把该族定义为另一族的**变换**（LOGSHASH = SHASH 的 log 变换，Jacobian 1/x），就对**活 R 的基族在变换后参数上**验全部契约；拟合 parity 用 G.dev_新族(y) = G.dev_基族(变换后) + 常数（对数雅可比参数无关）。旧实现的密度是另一套自洽参数化（过得了 AD deriv 闸但不是 R 的分布）——**deriv 闸只验自洽，不验是 R 的分布**。qSHASH 求逆已从 z 空间 Newton 加固为 s=asinh(z) 空间括号二分（旧法尾部导数塌缩+clip 冻结：rel 0.81 vs R、p 距 1 的 1e-16 内 NaN）。
- **R 自己的 q 函数会在极端 profile 挂死**（D2.52：qSHASH 在 nu=0.05/tau=2.0 下 uniroot 区间扩张 while 不返回）——测试网格要避开两侧都算不出的 profile 并记录理由；"R 没说"在这里的形态是"R 算不出"。
- **特殊函数包装的 eps 钳制陷阱（D2.53，影响全部 Bessel 族）**：`_log_bessel_kv_scipy` 曾用 `maximum(kv, eps)` 防下溢再取 log——但 K_nu(x) 在 x >~ 35 就小于 eps（K_2(50)=3.4e-23），此后所有 log(K) 静默变成 log(eps)=-36.04，Sichel 归一化差 log K(α)-log K(1/σ) 归零 → SI 密度最大偏 1.04e+04 倍、质量和不归一。**对特殊函数取 log 必须用指数缩放变体**（scipy `kve(v,x)=K_v(x)e^x` → `log K = log(kve) - x`；iv 用 `ive`）。修后对活 R 1e-12、对 scipy 全域 1.5e-15。dpqr 闸没暴露只因校准网格都在小参数区——**闸绿 ≠ 内核全域正确**。
- **跨库签名坑**：R 的 `besselK` 是 **(x, nu)**，scipy 的 `kv` 是 **(nu, x)**——写反静默对照转置族（D2.53 R 探针踩过）；R 的 `besselK(x, 负nu)` 可用、x>~740 下溢到 0（log 得 -Inf）。
- **SI rsdev 已修（D2.53）**：初值三缺陷（同 EXGAUS/NBII 签名第三族）+ 上述 Bessel 钳制。修后 rsdev 闸**全绿**（146 族无 rsdev fail；剩余既有：BCTO deriv、一批 dpqr fail/skip、三个 R 提前停登记 ZIPIG/ZINBI/PE2）。SI 似然面有极长平坦 nu 斜脊：我们精确 AD 分数沿脊爬向更优点（R 的 dldv 前向差跟不上，PE2 处置），1000 周期 nu=8.5 仍在爬、converged 长期 False——该族拟合的预期行为。

## 缺陷登记（D2.44）

- **标量内核禁用 `_prepare_jax_args`**（D2.45 坑）：它把参数 flatten 成 1-D，标量内核因此返回 `(1,)` 而非 `()`，vmap 后成 `(n,1)`，`g_dev_inc` 与 `weights` 广播 → deviance 变成 n 倍（实测 3350.48 vs 111.68）。密度/CDF 内核只用 `jnp.broadcast_arrays`；score 槽才用 `_prepare_jax_args`（它需要标量输入的 0-d 解包）。
- **DBURR12 不是 SICHEL**（D2.45 已修，D2.46 解锁准入）：离散 Burr XII，`Sur=(1+(y/mu)^sigma)^(-nu)`，nu>0、三个 link 全 log、初值 sigma=nu=2；现居 `omnilss/distributions_dburr12.py`，状态 EXPERIMENTAL。**别名继承 SICHEL 的 nu.link=identity 会让 dpqr 门禁因负 nu 网格静默 skip** —— 门禁失效可以是 metadata 造成的。安装版 `qDBURR12` = `round(w-1)` + 负值截 0 + p∉(0,1) stop，与 vendored `trunc` 不同，且 round 不是 CDF 逆。**别名判定有两条来源**：`_KNOWN_COMPAT_ALIAS_FAMILIES`（族名）+ `_COMPAT_FACTORY_MODULES`（模块名），少改一条状态不变。
- **Bessel JVP 已修（D2.47）**：旧规则用渐近切线 `-1 - nu/(x+|nu|+1)` 且丢 nu 切线 → 密度全对、AD 评分全错。现为 `d/dx log K_ν = -K_{ν+1}/K_ν + ν/x`（K 递推 `K_{ν-1}-K_{ν+1}=-(2ν/x)K_ν`；**I 用 +、K 用 −，符号相反，写反会被 ν=±0.5 解析解抓住**）+ `d/dν` 中心差（scipy 无阶数导数，h=1e-5）。SICHEL AD 对 R 闭式 dldm ≤7.8e-14；dldv 与 R 的 1e-2 差距是 R 前向差的误差。**Bessel 族拟合轨迹已改变 → ZIPIG/ZINBI 的 `_RSDEV_R_STOPPED_EARLY` 登记数字待全族流重立。**
- **SICHEL 族拟合慢（实测）**：`_sichel_log_pdf` 与 `tofySICHEL` 都是 `lax.scan` 固定 100000 步（掩码但全跑）→ 单槽 0.3~0.9s、200 周期 10~20 分钟。修法范式：`dpqr_functions.py` 7454/7547 的 `lax.while_loop` 动态上界。批量测试只构造族不拟合，故从未暴露。
- **DBURR12 别名错误**：离散 Burr XII（`Sur=(1+(y/mu)^sigma)^(-nu)`，nu>0、link 全 log、初值 sigma=nu=2）≠ SICHEL；registry 现指向 SICHEL 内核，且 dpqr gate 因 R 拒绝负 nu 校准网格而 skip → 错误密度从未被对照。
- **R 侧自身失败**：gamlss 的 ZISICHEL 在零多样本拟合报 `NA/NaN/Inf in foreign function call (arg 5)`（SICHEL/ZASICHEL 同样本正常）→ rsdev gate 对 ZISICHEL 无 R 参照。

## 环境（用户 2026-10-10 明确）

- `.venv/` = 项目 Python 环境（仓库根，Windows 3.10.16 + jax 0.6.2），**优先 `.venv/Scripts/python.exe`**。
- `debianjaxgpu/` = WSL Debian 的 JAX GPU 环境，非本项目源码，**勿清理勿修改**。
