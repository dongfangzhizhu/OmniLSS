# OmniLSS 项目长期记忆

> 纪律条目按"踩过的坑"组织。新条目先找归属，合并而非追加。

## 跑测试 / 覆盖工件

- tests/ 425 文件，全量 >1.5h → **分块跑**；慢批 `run_in_background`（前台 2min 超时打成 SIGTERM，看着像失败其实是超时）；后台 pytest 管道用 `| tail -N`；沙箱吞 `--collect-only`/重定向；tmp_path 清扫触发批量删除守卫，分小批。
- **tests/ 变更 → 同 commit 再生覆盖工件**：`tools/generate_coverage_artifacts.py --date` + `tools/coverage_diff_report.py` → `result: OK`（generate_capability_matrix.py 在 omnilss/tools/）。既有失败判定：已提交 `git checkout <c>~1`，未提交 `git stash -u`。

## 四条通用纪律

1. 既有用例失败，先分代码错 vs 期望错。
2. "R 没说" ≠ 常识可补；假拒绝比漏检更糟。
3. 写断言前先看实际数值、也要看对通道（`-inf` 被把 exp 成 0.0；密度非有限走 log=True）。
4. **活 R 对照是唯一可信 parity 证据**。bridge：`tests._r_bridge_helper`；family 传字符串名；R 的 stop() → RuntimeError。

## parity 五判据（D2.18）

1. 复现 gate 对齐三件事：族列表（共享 rng(20261006) 流顺序敏感）、网格 `--n`（默认 8）、拟合配置（sigma_formula="~1", max_iter=20）。正确入口 `_build_jobs`。
2. rel 方向定性质：**我们更高=缺陷；我们更低=先疑 R 提前停**。两步判定（①R 轨迹 ②R 的 d\* 评我们系数，须先 self-check）缺一不可；登记 `_RSDEV_R_STOPPED_EARLY` 只对我们更低开口，数字漂移退回 fail。已确认 BCPE/ZINBI/ZIPIG。
3. 系数是 link 尺度；R fit 顶层按参数名平铺取 coefficients；`y~1` 让 mu 标量。
4. kernel 开关分支单独核（两分支须可观测）；R 自己不一致：**域内复现 R**（DBI 窗口），**域外不复现**（PIG 非整数）。
5. parity 断言问活 R，不硬编码 16 位常数。

## 族 parity 扫描器（排查族缺陷默认入口）

`tools/rs_family_parity_scan.py`（+r_parity_batch.R），146 族**必须后台** + `--timeout`（默认 180s 必需；慢族 300s 补跑排除超时假象）。坑：deriv 对响应约定错位盲（dpqr 才抓）；skip≠fail；skip 可能是探针元数错/坏形状伪装；no_r_counterpart 记 pass+note；**探测参数必须落在参数自己的合法域**（D2.28：非法 nu=1.0 的 nan 被误读为"实现缺失"）。族发现走 resolve_family 报错（146 个），注册名大写。

## 收敛与地板

- converged=True 只是步长塌缩 ≠ 极小（BI 8566 全程 converged）。eta_gradient_norms 已装仪器，刻意不 gate converged。
- **量级分离不是判据，"沿梯度走一步能否降 deviance"才是**（PO/WEI2 重叠 7 倍）。
- 分轴切片前先确认基准点是驻点（BCPE"两盆地"错因：切片取在未收敛点）；地板正当与否唯一判据：**活 R 在边界另一侧是否有定义**（SST 正当 / ST3C 不正当）。
- 加预算反而更差 = 发散，入口常是初值（ZIPIG 漏分支 → 兜底 1.0 → logit=+inf）；对漏分支让它 raise，静默兜底就是缺陷。
- `g_dev` 已是完整 deviance，别再乘 2。
- 同名参数按角色不按前缀（nu 有零点质量/形状/底座三种角色）；monkey-patch 要确认落到被用的绑定（no-op vs no-effect，输出逐位相同=不可区分）。
- plateau 只是症状：地板比 R 落点高 → 根因在别的参数。断言双向带容差，前提基准点已收敛。

## rel 荒谬 = 约定层问题

1e90/inf 不是数值分歧，是参数/约定送错的签名（dpqr 必须按名传参；离散网格必须整数；多缺陷可能叠加才显症状；`type_`/元数据先骗探针）。共享 rng 流级联：修 type 会移位其后每族样本，判据"**R 侧数字也变**"。

## 密度/导数分裂

- "密度对、导数错"根因常在密度；修好一处可能更差（TF2 3399→6835），**不要单步变好就停**。
- 同一密度两实现（dpqr_functions vs distributions 的 `_log_density`）**必须同步改**。
- 转发壳先问：被转发的是同一个函数吗（TF2 签名偷换；hessian 与被偷换 score 自洽到 1e-11）。

## Box-Cox 簇（已修完）

三组九族：截断 identity / `o` 后缀**只改 mu.link=log** / untr 无 d,p,q。截断只进三处：密度 -logF、dldd/dldv/dldt hazard（**dldm 不含**）、p/q 重标。修法顺序密度→导数→link。教训：连粗糙一起抄 R；density 里禁 scipy.stats（jit 内炸）；BCT()$dld\* 向量化有 ifelse 标量坍缩，R 对照逐元素；"我们的族等于 R 的某族"断言会锁死缺陷。

## 二项族（D2.22/D2.23）

- y=成功计数、mu=比例，**入口统一约定**；R 的 BI/DBI/BB mu.initial=(y+0.5)/(bd+1)。
- 固定参数（bd）必须一路传到似然（rs_fit other_params/dev_kwargs 会连带排除 → Bernoulli 分支静默出错）。
- DBI 真族：C 是 getBI_C2.c 有限求和；GetBI_C 先 sigma=1/sigma；dDBI 在 |sigma-1|<0.001 切 dbinom；capability EXPERIMENTAL。

## R 探针读取要点（每条都踩过）

$nu.initial 是 expression 要 eval；jsonlite 丢命名 names 用 as.list；R 的 NULL 渲染成 {}/null 双判空；gamlss.family 无 d/p/q 槽；gamlss 类无 @ slot；Rscript 路径复用 `_rscript_path()`；gamlss() 的 start= 只接受 gamlss 对象；`o$parameters` 打印是 TRUE，名字在 `names()`；`G.dev.incr` 收**逐观测拟合值**不是系数（eta 重构+逆 link）。

## 零变体 p/q/r（D2.27/D2.28）

build_ad_family 不传 p/q/r 装 **nan 占位**——能力齐备但没有入口同样静默。活 R 公式：hurdle `p=tau@0`，否则 `tau+(1-tau)(F-F0)/(1-F0)`；mixture `tau+(1-tau)F`。组合实现 `_zero_variant_tails(base, altered=, epsilon=)`（b10）。tail flag：q<0 归零在 log 之后。epsilon 词汇（封闭）：ZAP=0、16 族 1e-10、ZIBNB/ZINBI=1e-7。R 侧零质量命名：ZAZIPF/ZABI/ZIBI/ZAP 叫 sigma，ZABNB/ZIBNB 叫 tau。

## 数值库精度（D2.29）

jax.scipy.special.betaln 自带 ~3.5e-8 误差；jax gammaln 准到 1.4e-14 → 手写 `_lbeta` 组合反准五个数量级（dpqr_functions）。**跨库同名数值函数先做一次精度对照再信。**

## 登记绑定样本流；cross-check 先验 R 自洁（D2.30）

登记数字只在**全族流**有效；子列表跑 → 样本移位 → "numbers moved" 是预期假警报（全族流重建见 tools/d230_refit_full_stream.py）。cross-check 前先验证 **R 自身自洁**（R 的 dZAP 在 mu 非常数向量时 recycle 长度 1 结果——dpqr pass ≠ 拟合路径正确；实现失误不复现，约定差异才复现）。

## helper 缓存范围匹配实际作用域

缓存排序前缀到目标族为止（精确等价快 4 倍）；R-backed 测试首跑 skip 长得像通过。

## 环境（用户 2026-10-10 明确）

- `.venv/` = 项目 Python 环境（Windows, 3.10.16 + jax 0.6.2），**优先 `.venv/Scripts/python.exe`**。
- `debianjaxgpu/` = WSL Debian 的 JAX GPU 环境，非本项目源码，**勿清理勿修改**。
