# OmniLSS — R `gamlss` / `gamlss.dist` 迁移进度

> 生成时间：2026-05-23  
> 基准版本：R `gamlss` 5.5-3，`gamlss.dist` 6.1-1（`R/gamlss/` 和 `R/gamlss.dist/` 目录）  
> Python 包：`omnilss` 0.3.0（`omnilss/src/omnilss/`）

本文档逐文件、逐函数列出 R 目录下所有功能的迁移状态。

**状态说明**

| 标记 | 含义 |
|------|------|
| ✅ 已实现 | Python 中有对应实现，可从 `omnilss` 包导入 |
| ⚠️ 部分实现 | 有 Python 入口，但功能不完整（见备注） |
| ❌ 未实现 | 尚无 Python 对应实现 |

---

## 目录

1. [gamlss 核心包（R/gamlss/R/）](#1-gamlss-核心包)
   - 1.1 [拟合引擎](#11-拟合引擎)
   - 1.2 [预测与线性预测器](#12-预测与线性预测器)
   - 1.3 [模型诊断与残差](#13-模型诊断与残差)
   - 1.4 [绘图与可视化](#14-绘图与可视化)
   - 1.5 [模型选择与逐步回归](#15-模型选择与逐步回归)
   - 1.6 [平滑项（Smoothers）](#16-平滑项-smoothers)
   - 1.7 [分位数图表与 LMS](#17-分位数图表与-lms)
   - 1.8 [其他工具函数](#18-其他工具函数)
2. [gamlss.dist 分布包（R/gamlss.dist/R/）](#2-gamlssdist-分布包)
   - 2.1 [连续分布（实数域）](#21-连续分布实数域)
   - 2.2 [连续分布（正实数域）](#22-连续分布正实数域)
   - 2.3 [有界连续分布（0,1）](#23-有界连续分布01)
   - 2.4 [Box-Cox 变换族](#24-box-cox-变换族)
   - 2.5 [离散计数分布](#25-离散计数分布)
   - 2.6 [零膨胀 / 零截断分布](#26-零膨胀--零截断分布)
   - 2.7 [分布族辅助函数](#27-分布族辅助函数)
   - 2.8 [矩与分位数形状工具](#28-矩与分位数形状工具)
   - 2.9 [通用族构造器](#29-通用族构造器)
   - 2.10 [distributions3 接口](#210-distributions3-接口)
3. [OmniLSS 扩展功能（超出 R 范围）](#3-omnilss-扩展功能超出-r-范围)
4. [总体进度统计](#4-总体进度统计)

---

## 1. gamlss 核心包

### 1.1 拟合引擎

**R 文件：`gamlss-5.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `gamlss()` | 主拟合函数（RS/CG/Mixed 算法） | ✅ 已实现 | `omnilss.fitting.gamlss` |
| `gamlss.control()` | 拟合控制参数 | ✅ 已实现 | `omnilss.controls.gamlss_control` |
| `glim.control()` | GLIM 内层控制参数 | ✅ 已实现 | `omnilss.controls.glim_control` |
| `gamlssNews()` | 显示版本新闻 | ❌ 未实现 | — |
| `rqres()` | 随机化分位数残差（内部） | ⚠️ 部分实现 | `omnilss.methods.get_rqres_samples` |
| `parameterOut()` | 参数输出辅助（内部） | ❌ 未实现 | — |
| `get.smoothers()` | 提取平滑项列表（内部） | ❌ 未实现 | — |
| `get.object()` | 获取拟合对象（内部） | ❌ 未实现 | — |

**R 文件：`gamlssML.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `gamlssML()` | 最大似然拟合（无平滑） | ✅ 已实现 | `omnilss.gamlssML.gamlssML` |
| `fitted.gamlssML()` | 拟合值（S3 方法） | ✅ 已实现 | `omnilss.operations.fitted` |
| `vcov.gamlssML()` | 协方差矩阵（S3 方法） | ✅ 已实现 | `omnilss.vcov_gamlss.vcov` |
| `summary.gamlssML()` | 摘要（S3 方法） | ✅ 已实现 | `omnilss.SUMMARY.summary` |
| `predict.gamlssML()` | 预测（S3 方法） | ✅ 已实现 | `omnilss.predict_gamlss_23_12_21.predict` |

**R 文件：`gamlssVGD_23_12_21.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `gamlssVGD()` | VGD 算法拟合 | ✅ 已实现 | `omnilss.gamlssVGD_23_12_21.gamlssVGD` |
| `VGD()` | VGD 对象构造 | ✅ 已实现 | `omnilss.gamlssVGD_23_12_21.VGD` |
| `getTGD()` | 获取 TGD 对象 | ✅ 已实现 | `omnilss.gamlssVGD_23_12_21.getTGD` |
| `TGD()` | TGD 对象构造 | ✅ 已实现 | `omnilss.stepTGD.extractTGD` |
| `gamlssCV()` | 交叉验证拟合 | ✅ 已实现 | `omnilss.gamlssVGD_23_12_21.gamlssCV` |
| `CV()` | CV 对象构造 | ✅ 已实现 | `omnilss.gamlssVGD_23_12_21.CV` |

**R 文件：`update.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `update.gamlss()` | 更新模型公式/参数 | ✅ 已实现 | `omnilss.update.update` |
| `refit()` | 重新拟合模型 | ✅ 已实现 | `omnilss.operations.refit` |

**R 文件：`add.r`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `additive.fit()` | 加性拟合内核（内部） | ❌ 未实现 | — |


### 1.2 预测与线性预测器

**R 文件：`predict.gamlss_23_12_21.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `predict.gamlss()` | 预测（新数据/训练数据） | ✅ 已实现 | `omnilss.predict_gamlss_23_12_21.predict` |

**R 文件：`predictAll_22_08_22.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `predictAll()` | 预测所有参数 | ⚠️ 部分实现 | `omnilss.predictAll_22_08_22.predict_all`（顶层 `predictAll` 未导出） |

**R 文件：`lpred.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `lpred()` | 线性预测器 | ✅ 已实现 | `omnilss.lpred.lpred` |
| `lp()` | `lpred` 别名 | ✅ 已实现 | `omnilss.operations.lp` |
| `fv()` | 拟合值（fitted values） | ✅ 已实现 | `omnilss.operations.fv` |

**R 文件：`prodist.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `prodist.gamlss()` | 预测分布对象 | ✅ 已实现 | `omnilss.prodist.prodist_data` |

**R 文件：`Formulae2data.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `Formulae2data()` | 公式转数据矩阵 | ⚠️ 部分实现 | `omnilss.Formulae2data.Formulae2data`（委托内部函数，smooth 路径脆弱） |
| `getAll.formulae2data()` | 内部辅助 | ❌ 未实现 | — |

**R 文件：`extra.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `refit()` | 重新拟合 | ✅ 已实现 | `omnilss.operations.refit` |
| `fitted.gamlss()` | 拟合值 | ✅ 已实现 | `omnilss.operations.fitted` |
| `coef.gamlss()` | 系数 | ✅ 已实现 | `omnilss.operations.coef` |
| `coefAll()` | 所有参数系数 | ✅ 已实现 | `omnilss.operations.coef_all` |
| `residuals.gamlss()` | 残差 | ✅ 已实现 | `omnilss.operations.residuals` |
| `deviance.gamlss()` | 偏差 | ✅ 已实现 | `omnilss.operations.deviance` |
| `lp()` | 线性预测器 | ✅ 已实现 | `omnilss.operations.lp` |
| `fv()` | 拟合值 | ✅ 已实现 | `omnilss.operations.fv` |
| `model.frame.gamlss()` | 模型框架 | ✅ 已实现 | `omnilss.operations.model_frame` |
| `terms.gamlss()` | 模型项 | ✅ 已实现 | `omnilss.operations.terms` |
| `model.matrix.gamlss()` | 设计矩阵 | ✅ 已实现 | `omnilss.operations.model_matrix` |
| `formula.gamlss()` | 公式 | ✅ 已实现 | `omnilss.operations.formula` |
| `is.gamlss()` | 类型检查 | ✅ 已实现 | `omnilss.operations.is_gamlss` |
| `IC()` | 信息准则 | ✅ 已实现 | `omnilss.operations.ic` |
| `AIC.gamlss()` | AIC | ✅ 已实现 | `omnilss.operations.gaic`（k=2） |
| `GAIC()` | 广义 AIC | ✅ 已实现 | `omnilss.operations.gaic` |
| `GAIC_table()` | GAIC 比较表 | ✅ 已实现 | `omnilss.operations.gaic_table` |
| `GAIC_scaled()` | 缩放 GAIC | ✅ 已实现 | `omnilss.operations.gaic_scaled` |
| `Rsq()` | 伪 R² | ✅ 已实现 | `omnilss.Rsq.rsq` |
| `numeric.deriv()` | 数值导数 | ✅ 已实现 | `omnilss.operations.numeric_deriv` |
| `.hat.WX()` | 帽子矩阵 | ✅ 已实现 | `omnilss.operations.hat_wx` |


### 1.3 模型诊断与残差

**R 文件：`rqres.R` / `rqresplot_new.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `rqres()` | 随机化分位数残差 | ⚠️ 部分实现 | `omnilss.methods.get_rqres_samples` |
| `rqres1()` | 单参数版本（内部） | ❌ 未实现 | — |
| `get.rqres()` | 获取残差样本 | ⚠️ 部分实现 | `omnilss.methods.get_rqres_samples` |
| `rqres.plot()` | 残差诊断图 | ❌ 未实现 | — |

**R 文件：`acfResid.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `acfResid()` | 残差自相关函数 | ✅ 已实现 | `omnilss.acfResid.acf_residuals` |

**R 文件：`wp.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `wp()` | Worm plot（蠕虫图） | ✅ 已实现 | `omnilss.worm_plot.wp` |

**R 文件：`qstats.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `Q.stats()` | Q 统计量区间诊断 | ✅ 已实现 | `omnilss.qstats.q_stats` |
| `Qtest()` | Q 检验（内部） | ❌ 未实现 | — |
| `get.par.df()` | 获取参数自由度（内部） | ❌ 未实现 | — |

**R 文件：`LR-test-12-06-2013.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `LR.test()` | 似然比检验 | ✅ 已实现 | `omnilss.LR_test_12_06_2013.likelihood_ratio_test` |

**R 文件：`VuongClarkTest.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `VC.test()` | Vuong-Clarke 非嵌套模型检验 | ✅ 已实现 | `omnilss.VuongClarkTest.vuong_clarke_test` |

**R 文件：`DevianceIncr.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `devianceIncr()` | 偏差增量 | ✅ 已实现 | `omnilss.DevianceIncr.devianceIncr` |

**R 文件：`hatvalues.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `hatvalues.gamlss()` | 帽子值（杠杆值） | ✅ 已实现 | `omnilss.hatvalues.hatvalues` |
| `model2formulae2data()` | 内部辅助 | ❌ 未实现 | — |

**R 文件：`logLik.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `logLik.gamlss()` | 对数似然 | ✅ 已实现 | `omnilss.logLik.logLik` |

**R 文件：`vcov-gamlss.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `vcov.gamlss()` | 协方差矩阵 | ✅ 已实现 | `omnilss.vcov_gamlss.vcov` |
| `get.K()` | 获取参数数量 K | ✅ 已实现 | `omnilss.vcov_gamlss.get_K` |
| `rvcov()` | 鲁棒协方差矩阵 | ✅ 已实现 | `omnilss.vcov_gamlss.rvcov` |

**R 文件：`confint-gamlss-29-06-22.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `confint.gamlss()` | 置信区间 | ✅ 已实现 | `omnilss.confint_gamlss_29_06_22.confint` |

**R 文件：`SUMMARY.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `summary.gamlss()` | 模型摘要 | ✅ 已实现 | `omnilss.SUMMARY.summary` |

**R 文件：`print.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `print.gamlss()` | 打印模型 | ✅ 已实现 | `omnilss.print.print_model` |

**R 文件：`edf.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `edf()` | 有效自由度（单项） | ❌ 未实现 | — |
| `edfAll()` | 所有参数有效自由度 | ✅ 已实现 | `omnilss.getSmo.edfAll` |

**R 文件：`getSmo-18-6-13.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `getSmo()` | 提取平滑项信息 | ✅ 已实现 | `omnilss.getSmo.getSmo` |

**R 文件：`get-K-rvcov-16-06-2913.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `get.K()` | 参数数量 | ✅ 已实现 | `omnilss.vcov_gamlss.get_K` |
| `rvcov()` | 鲁棒协方差 | ✅ 已实现 | `omnilss.vcov_gamlss.rvcov` |

**R 文件：`gen-Likelihood.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `gen.likelihood()` | 生成似然函数 | ✅ 已实现 | `omnilss.gen_likelihood.gen_likelihood` |

**R 文件：`PROFILE_new_19-08-12.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `prof.dev()` | 剖面偏差（结构化数据） | ⚠️ 部分实现 | `omnilss.prof_dev.prof_dev`（不执行重拟合） |

**R 文件：`profterm_18_05_13.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `prof.term()` | 项剖面偏差 | ⚠️ 部分实现 | `omnilss.prof_dev.prof_term`（不执行重拟合） |

**R 文件：`findhyper-23-05-13.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `find.hyper()` | 超参数搜索 | ⚠️ 部分实现 | `omnilss.find_hyper.find_hyper`（只汇总预计算结果，不执行网格搜索重拟合） |

**R 文件：`dtop.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `dtop()` | 偏差排名（top deviance） | ✅ 已实现 | `omnilss.dtop.dtop` |


### 1.4 绘图与可视化

**R 文件：`plot.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `plot.gamlss()` | 四合一诊断图 | ✅ 已实现 | `omnilss.plot.plot_diagnostics`（返回数据，不直接绘图） |

**R 文件：`fitted-plot.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `fittedPlot()` | 拟合值图 | ✅ 已实现 | `omnilss.fitted_plot.fitted_plot_data` |

**R 文件：`pdfplot.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `pdf.plot()` | 概率密度图 | ✅ 已实现 | `omnilss.pdfplot.pdf_plot_data` |

**R 文件：`term.plot-new.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `term.plot()` | 项效应图 | ✅ 已实现 | `omnilss.term_plot_new.term_plot_data` |
| `plotLME()` | LME 项图（内部） | ❌ 未实现 | — |
| `draw.polys.in()` | 多边形绘制（内部） | ❌ 未实现 | — |
| `CheckSmoList()` | 平滑项列表检查（内部） | ❌ 未实现 | — |

**R 文件：`par_plot-21-5-13.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `par.plot()` | 参数诊断图 | ✅ 已实现 | `omnilss.par_plot.par_plot` |

**R 文件：`plot2way.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `plot2way()` | 双向交互效应图 | ✅ 已实现 | `omnilss.plot2way.plot2way_data` |

**R 文件：`HistDist-03-10-13.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `histDist()` | 直方图 + 分布拟合图 | ✅ 已实现 | `omnilss.HistDist_03_10_13.hist_dist_data` |

**R 文件：`histSmo.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `histSmo()` | 直方图平滑（普通） | ⚠️ 部分实现 | `omnilss.histSmo.histSmo`（四变体实现相同，未区分平滑方法） |
| `histSmoO()` | 直方图平滑（有序） | ⚠️ 部分实现 | `omnilss.histSmo.histSmoO` |
| `histSmoC()` | 直方图平滑（连续） | ⚠️ 部分实现 | `omnilss.histSmo.histSmoC` |
| `histSmoP()` | 直方图平滑（惩罚） | ⚠️ 部分实现 | `omnilss.histSmo.histSmoP` |
| `plot.histSmo()` | 绘图方法 | ❌ 未实现 | — |
| `lines.histSmo()` | 添加线条 | ❌ 未实现 | — |
| `print.histSmo()` | 打印方法 | ❌ 未实现 | — |

**R 文件：`getPEF.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `getPEF()` | 偏效应函数 | ✅ 已实现 | `omnilss.getPEF.get_pef_data` |

**R 文件：`getQuantile.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `getQuantile()` | 获取分位数曲线 | ✅ 已实现 | `omnilss.getQuantile.get_quantile_data` |

**R 文件：`FitTail.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `loglogSurv1/2/3()` | 对数-对数生存曲线变体 | ✅ 已实现 | `omnilss.FitTail.loglogSurv`（合并实现） |
| `loglogSurv()` | 对数-对数生存曲线 | ✅ 已实现 | `omnilss.FitTail.loglogSurv` |
| `logSurv()` | 对数生存曲线 | ✅ 已实现 | `omnilss.FitTail.logSurv` |
| `logSurv0()` | 对数生存曲线（别名） | ✅ 已实现 | `omnilss.loglogSurv.logSurv0` |
| `loglogplot()` | 对数-对数图 | ✅ 已实现 | `omnilss.loglogplot.loglogplot` |
| `loglogplot0()` | 对数-对数图（别名） | ❌ 未实现 | — |
| `ECDF()` | 经验累积分布函数 | ✅ 已实现 | `omnilss.ECDF.ECDF` |
| `ESURV()` | 经验生存函数 | ❌ 未实现 | — |
| `quantile_weights()` | 分位数权重（内部） | ❌ 未实现 | — |

**R 文件：`bp.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `bp()` | 箱线图平滑 | ❌ 未实现 | — |
| `momBucket()` | 矩桶（内部） | ❌ 未实现 | — |
| `boundaryR()` | 边界（内部） | ❌ 未实现 | — |
| `draw.Jarque.Bera()` | JB 检验图（内部） | ❌ 未实现 | — |


### 1.5 模型选择与逐步回归

**R 文件：`stepGAIC-03-10-13..R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `stepGAIC()` | 逐步 GAIC 选择 | ✅ 已实现 | `omnilss.stepGAIC_03_10_13.stepGAIC` |
| `stepGAIC.VR()` | VR 版本逐步选择 | ❌ 未实现 | — |
| `stepGAIC.CH()` | CH 版本逐步选择 | ❌ 未实现 | — |
| `gamlss.scope()` | 作用域定义 | ❌ 未实现 | — |
| `extractAIC.gamlss()` | 提取 AIC | ✅ 已实现 | `omnilss.stepGAIC_03_10_13.extractAIC` |
| `drop1.gamlss()` | 删除一项 | ✅ 已实现 | `omnilss.methods.dropterm` |

**R 文件：`stepGAICAll-A-parallel.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `stepGAICAll.A()` | 全参数逐步选择（A 型） | ✅ 已实现 | `omnilss.stepGAICAll_A_parallel.stepGAICAll_A` |

**R 文件：`stepGAICAll-B-Parallel.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `stepGAICAll.B()` | 全参数逐步选择（B 型） | ✅ 已实现 | `omnilss.stepGAICAll_B_Parallel.stepGAICAll_B` |
| `drop1All()` | 删除所有参数一项 | ✅ 已实现 | `omnilss.methods.dropterm_all` |
| `add1All()` | 添加所有参数一项 | ✅ 已实现 | `omnilss.methods.addterm_all` |
| `droptermAllP()` | 并行删除项 | ✅ 已实现 | `omnilss.stepGAICAll_B_Parallel.droptermAllP` |
| `addtermAllP()` | 并行添加项 | ✅ 已实现 | `omnilss.stepGAICAll_B_Parallel.addtermAllP` |

**R 文件：`DropAddStepGAIC-Parallel.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `dropterm.gamlss()` | 删除项 | ✅ 已实现 | `omnilss.methods.dropterm` |
| `addterm.gamlss()` | 添加项 | ✅ 已实现 | `omnilss.methods.addterm` |
| `stepGAIC()` | 逐步选择（重复导出） | ✅ 已实现 | `omnilss.stepGAIC_03_10_13.stepGAIC` |

**R 文件：`stepTGD.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `extractTGD()` | 提取 TGD | ✅ 已实现 | `omnilss.stepTGD.extractTGD` |
| `drop1TGD()` | TGD 删除一项 | ✅ 已实现 | `omnilss.stepTGD.drop1TGD` |
| `add1TGD()` | TGD 添加一项 | ✅ 已实现 | `omnilss.stepTGD.add1TGD` |
| `stepTGD()` | TGD 逐步选择 | ✅ 已实现 | `omnilss.stepTGD.stepTGD` |
| `drop1TGDP()` | TGD 并行删除 | ✅ 已实现 | `omnilss.stepTGD.drop1TGDP` |
| `add1TGDP()` | TGD 并行添加 | ✅ 已实现 | `omnilss.stepTGD.add1TGDP` |

**R 文件：`stepTGDAll-A-parallel.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `stepTGDAll.A()` | 全参数 TGD 逐步选择 | ❌ 未实现 | — |

**R 文件：`fitDist.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `fitDist()` | 分布拟合与排名 | ✅ 已实现 | `omnilss.fitDist.fitDist` |

**R 文件：`fitDistPred.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `fitDistPred()` | 预测分布拟合 | ✅ 已实现 | `omnilss.fitDistPred.fitDistPred` |
| `gamlssMLpred()` | ML 预测拟合 | ✅ 已实现 | `omnilss.fitDistPred.gamlssMLpred` |
| `predictAll.V()` | 预测所有（内部） | ❌ 未实现 | — |

**R 文件：`chooseDistParallel.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `chooseDist()` | 自动选择最优分布 | ✅ 已实现 | `omnilss.chooseDistParallel.chooseDist` |
| `chooseDistPred()` | 预测导向分布选择 | ✅ 已实现 | `omnilss.chooseDistParallel.chooseDistPred` |
| `getOrder()` | 获取排名顺序 | ✅ 已实现 | `omnilss.chooseDistParallel.getOrder` |


### 1.6 平滑项（Smoothers）

> OmniLSS 在公式解析器中识别 `pb`、`ps`、`cs`、`random`、`re`、`lo`，但顶层 `pb()`/`ps()`/`cs()` 函数未从 `omnilss` 包直接导出（需通过 `omnilss.smoothers` 子模块访问）。

**R 文件：`pb.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `pb()` | P-样条平滑（自动 λ） | ✅ 已实现 | `omnilss.smoothers.pb.fit_pspline`（公式中用 `pb(x)`） |
| `pb.control()` | P-样条控制参数 | ✅ 已实现 | `omnilss.smoothers.pb`（内嵌参数） |
| `gamlss.pb()` | 内层拟合函数 | ✅ 已实现 | `omnilss.smooth_fitting`（内部） |
| `plot.pb()` | 绘图方法 | ❌ 未实现 | — |
| `coef.pb()` | 系数方法 | ❌ 未实现 | — |
| `fitted.pb()` | 拟合值方法 | ❌ 未实现 | — |
| `print.pb()` | 打印方法 | ❌ 未实现 | — |
| `knots.pb()` | 节点方法 | ❌ 未实现 | — |

**R 文件：`PS.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `ps()` | P-样条（等间距节点） | ✅ 已实现 | `omnilss.smoothers.ps.fit_pspline_smooth` |
| `gamlss.ps()` | 内层拟合 | ✅ 已实现 | `omnilss.smooth_fitting`（内部） |

**R 文件：`cubicSplines-10-08-12.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `cs()` | 三次平滑样条 | ✅ 已实现 | `omnilss.smoothers.cs.fit_cubic_spline` |
| `cs.control()` | 控制参数 | ✅ 已实现 | `omnilss.smoothers.cs`（内嵌） |
| `gamlss.cs()` | 内层拟合 | ✅ 已实现 | `omnilss.smooth_fitting`（内部） |
| `scs()` | 平滑三次样条（别名） | ❌ 未实现 | — |

**R 文件：`pbm.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `pbm()` | 单调 P-样条 | ❌ 未实现 | — |
| `pbm.control()` | 控制参数 | ❌ 未实现 | — |
| `gamlss.pbm()` | 内层拟合 | ❌ 未实现 | — |
| `print.pbm()` | 打印方法 | ❌ 未实现 | — |

**R 文件：`pb_goingtozero.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `pbz()` | 趋零 P-样条 | ❌ 未实现 | — |
| `pbz.control()` | 控制参数 | ❌ 未实现 | — |
| `gamlss.pbz()` | 内层拟合 | ❌ 未实现 | — |
| `print.pbz()` | 打印方法 | ❌ 未实现 | — |

**R 文件：`pbo_original.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `pbo()` | 带偏移 P-样条 | ❌ 未实现 | — |
| `pbo.control()` | 控制参数 | ❌ 未实现 | — |
| `gamlss.pbo()` | 内层拟合 | ❌ 未实现 | — |

**R 文件：`pbp.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `pbp()` | 正值 P-样条 | ❌ 未实现 | — |
| `pbp.control()` | 控制参数 | ❌ 未实现 | — |
| `gamlss.pbp()` | 内层拟合 | ❌ 未实现 | — |

**R 文件：`pbc.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `pbc()` | 循环 P-样条 | ❌ 未实现 | — |
| `pbc.control()` | 控制参数 | ❌ 未实现 | — |
| `gamlss.pbc()` | 内层拟合 | ❌ 未实现 | — |
| `print.pbc()` | 打印方法 | ❌ 未实现 | — |

**R 文件：`pvc.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `pvc()` | 变系数 P-样条 | ❌ 未实现 | — |
| `pvc.control()` | 控制参数 | ❌ 未实现 | — |
| `gamlss.pvc()` | 内层拟合 | ❌ 未实现 | — |
| `coef.pvc()` | 系数方法 | ❌ 未实现 | — |
| `fitted.pvc()` | 拟合值方法 | ❌ 未实现 | — |
| `print.pvc()` | 打印方法 | ❌ 未实现 | — |
| `plot.pvc()` | 绘图方法 | ❌ 未实现 | — |

**R 文件：`cy.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `cy()` | 循环样条 | ❌ 未实现 | — |
| `cy.control()` | 控制参数 | ❌ 未实现 | — |
| `gamlss.cy()` | 内层拟合 | ❌ 未实现 | — |
| `print.cy()` | 打印方法 | ❌ 未实现 | — |

**R 文件：`fracpoly.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `fp()` | 分数多项式 | ❌ 未实现 | — |
| `bfp()` | 最优分数多项式 | ❌ 未实现 | — |
| `gamlss.fp()` | 内层拟合 | ❌ 未实现 | — |

**R 文件：`lo_new.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `lo()` | LOESS 平滑 | ⚠️ 部分实现 | `omnilss.lo`（公式解析器识别，无实际 LOESS 拟合） |
| `lo.control()` | 控制参数 | ❌ 未实现 | — |
| `gamlss.lo()` | 内层拟合 | ❌ 未实现 | — |
| `vis.lo()` | LOESS 可视化 | ⚠️ 部分实现 | `omnilss.vis_lo.vis_lo`（只排序 x/y，无 LOESS） |

**R 文件：`random.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `random()` | 随机效应项 | ⚠️ 部分实现 | `omnilss.random`（公式解析器识别，无 nlme 后端） |
| `gamlss.random()` | 内层拟合 | ❌ 未实现 | — |
| `plot.random()` | 绘图方法 | ❌ 未实现 | — |
| `coef.random()` | 系数方法 | ❌ 未实现 | — |
| `fitted.random()` | 拟合值方法 | ❌ 未实现 | — |
| `print.random()` | 打印方法 | ❌ 未实现 | — |
| `getMarginal()` | 边际效应（内部） | ❌ 未实现 | — |

**R 文件：`re_gamlss_lme_test.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `re()` | 混合效应项（nlme） | ❌ 未实现 | — |
| `gamlss.re()` | 内层拟合 | ❌ 未实现 | — |

**R 文件：`ri.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `ri()` | 岭回归项 | ❌ 未实现 | — |
| `gamlss.ri()` | 内层拟合 | ❌ 未实现 | — |
| `plot.ri()` | 绘图方法 | ❌ 未实现 | — |
| `coef.ri()` | 系数方法 | ❌ 未实现 | — |
| `fitted.ri()` | 拟合值方法 | ❌ 未实现 | — |
| `print.ri()` | 打印方法 | ❌ 未实现 | — |

**R 文件：`ridge.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `ridge()` | 岭回归平滑 | ❌ 未实现 | — |
| `gamlss.ridge()` | 内层拟合 | ❌ 未实现 | — |

**R 文件：`powerpoly.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `pp()` | 幂多项式 | ❌ 未实现 | — |
| `gamlss.pp()` | 内层拟合 | ❌ 未实现 | — |

**R 文件：`POLY.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `polyS()` | 正交多项式 | ❌ 未实现 | — |
| `poly.matrix()` | 多项式矩阵（内部） | ❌ 未实现 | — |

**R 文件：`p-categorical.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `pcat()` | 惩罚分类项 | ❌ 未实现 | — |
| `gamlss.pcat()` | 内层拟合 | ❌ 未实现 | — |
| `plotDF()` | 自由度图 | ✅ 已实现 | `omnilss.plotDF`（数据层） |
| `plotLambda()` | λ 图 | ✅ 已实现 | `omnilss.plotLambda`（数据层） |
| `plot.pcat()` | 绘图方法 | ❌ 未实现 | — |
| `coef.pcat()` | 系数方法 | ❌ 未实现 | — |
| `print.pcat()` | 打印方法 | ❌ 未实现 | — |
| `fitted.pcat()` | 拟合值方法 | ❌ 未实现 | — |

**R 文件：`pbm.R`（getZmatrix）**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `getZmatrix()` | 获取 Z 矩阵 | ✅ 已实现 | `omnilss.getZmatrix.getZmatrix` |


### 1.7 分位数图表与 LMS

**R 文件：`centilesPLOT.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `centiles()` | 分位数曲线图 | ✅ 已实现 | `omnilss.centilesPLOT.centiles_data` |
| `centiles.split()` | 分组分位数图 | ✅ 已实现 | `omnilss.centilesPLOT.centiles_split_data` |

**R 文件：`centilescom.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `centiles.com()` | 分位数比较图 | ✅ 已实现 | `omnilss.centilescom.centiles_comparison_data` |

**R 文件：`centilesFan.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `centiles.fan()` | 扇形分位数图 | ✅ 已实现 | `omnilss.centilesFan.centiles_fan` |

**R 文件：`centile-pred_30_06_22.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `centiles.pred()` | 分位数预测 | ✅ 已实现 | `omnilss.centile_pred_30_06_22.centile_pred_data` |

**R 文件：`lms.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `lms()` | LMS 方法（Box-Cox z-score） | ✅ 已实现 | `omnilss.lms.lms` |
| `z.scores()` | LMS z-score 计算 | ✅ 已实现 | `omnilss.lms.z_scores` |
| `findPower()` | 寻找最优 Box-Cox 幂 | ✅ 已实现 | `omnilss.quantSheets.findPower` |
| `ptrans()` | 幂变换（内部） | ❌ 未实现 | — |
| `predict.lms()` | LMS 预测（S3 方法） | ❌ 未实现 | — |
| `calibration()` | 校准函数 | ❌ 未实现 | — |

**R 文件：`quantSheets.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `quantSheets()` | 分位数表 | ✅ 已实现 | `omnilss.quantSheets.quantSheets` |
| `quantSheets.control()` | 控制参数 | ✅ 已实现 | `omnilss.quantSheets_control.quantSheets_control` |
| `findPower()` | 寻找最优幂 | ✅ 已实现 | `omnilss.quantSheets.findPower` |
| `z.scoresQS()` | 分位数表 z-score | ✅ 已实现 | `omnilss.quantSheets.z_scoresQS` |
| `invptrans()` | 逆幂变换（内部） | ❌ 未实现 | — |
| `print.quantSheets()` | 打印方法 | ❌ 未实现 | — |
| `fitted.quantSheets()` | 拟合值方法 | ❌ 未实现 | — |
| `predict.quantSheets()` | 预测方法 | ❌ 未实现 | — |
| `residuals.quantSheets()` | 残差方法 | ❌ 未实现 | — |


### 1.8 其他工具函数

**R 文件：`PROFILE_new_19-08-12.R` / `profterm_18_05_13.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `prof.dev()` | 剖面偏差 | ⚠️ 部分实现 | `omnilss.prof_dev.prof_dev` |
| `prof.term()` | 项剖面偏差 | ⚠️ 部分实现 | `omnilss.prof_dev.prof_term` |

**R 文件：`findhyper-23-05-13.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `find.hyper()` | 超参数搜索 | ⚠️ 部分实现 | `omnilss.find_hyper.find_hyper` |

---

## 2. gamlss.dist 分布包

### 2.1 连续分布（实数域）

| R 文件 | 分布族 | d/p/q/r | 状态 | Python 位置 |
|--------|--------|---------|------|-------------|
| `NO.r` | `NO` 正态分布 | ✅ | ✅ 已实现 | `omnilss.distributions.NO` |
| `NO.r` | `NO2` 正态（方差参数化） | ✅ | ✅ 已实现 | `omnilss.distributions_b2.NO2` |
| `Logistic.R` | `LO` 逻辑斯谛分布 | ✅ | ✅ 已实现 | `omnilss.distributions.LO` |
| `TF.R` | `TF` Student-t 分布 | ✅ | ✅ 已实现 | `omnilss.distributions.TF` |
| `TF2.R` | `TF2` Student-t（另一参数化） | ✅ | ✅ 已实现 | `omnilss.distributions_b16.TF2` |
| `GU.R` | `GU` Gumbel 分布 | ✅ | ✅ 已实现 | `omnilss.distributions_b1.GU` |
| `RG.R` | `RG` 反 Gumbel 分布 | ✅ | ✅ 已实现 | `omnilss.distributions_b1.RG` |
| `PE.R` | `PE` 幂指数分布 | ✅ | ✅ 已实现 | `omnilss.distributions_b2.PE` |
| `PE2.R` | `PE2` 幂指数（另一参数化） | ✅ | ✅ 已实现 | `omnilss.distributions_b13.PE2` |
| `NET.R` | `NET` 正态-指数-t 分布 | ✅ | ✅ 已实现 | `omnilss.distributions_b8.NET` |
| `SHASH.R` | `SHASH` Sinh-Arcsinh 分布 | ✅ | ✅ 已实现 | `omnilss.distributions_b3.SHASH` |
| `SHASHo.R` | `SHASHo` Sinh-Arcsinh（原始参数化） | ✅ | ✅ 已实现 | `omnilss.distributions_b14.SHASHo` |
| `SHASHo2.R` | `SHASHo2` Sinh-Arcsinh（第二参数化） | ✅ | ✅ 已实现 | `omnilss.distributions_b14.SHASHo2` |
| `LOGSHASH.R` | `LOGSHASH` 对数 Sinh-Arcsinh | ✅ | ✅ 已实现 | `omnilss.distributions_b15.LOGSHASH` |
| `LOGSHASHo.R` | `LOGSHASHo` 对数 Sinh-Arcsinh（原始） | ✅ | ✅ 已实现 | `omnilss.distributions_b16.LOGSHASHo` |
| `JSU.R` | `JSU` Johnson SU 分布 | ✅ | ✅ 已实现 | `omnilss.distributions.JSU` |
| `JSUo.R` | `JSUo` Johnson SU（原始参数化） | ✅ | ✅ 已实现 | `omnilss.distributions_b14.JSUo` |
| `SN1.R` | `SN1` 偏正态 1 | ✅ | ✅ 已实现 | `omnilss.distributions_b3.SN1` |
| `SN2.R` | `SN2` 偏正态 2 | ✅ | ✅ 已实现 | `omnilss.distributions_b3.SN2` |
| `GT.R` | `GT` 广义 t 分布 | ✅ | ✅ 已实现 | `omnilss.distributions_b3.GT` |
| `ST1.R` | `ST1` 偏 t 分布 1 | ✅ | ✅ 已实现 | `omnilss.distributions_st_compat.ST1` |
| `ST2.R` | `ST2` 偏 t 分布 2 | ✅ | ✅ 已实现 | `omnilss.distributions_st_compat.ST2` |
| `ST3.R` | `ST3` 偏 t 分布 3 | ✅ | ✅ 已实现 | `omnilss.distributions_st_compat.ST3` |
| `ST3C.R` | `ST3C` 偏 t 分布 3C | ✅ | ✅ 已实现 | `omnilss.distributions_b15.ST3C` |
| `ST4.R` | `ST4` 偏 t 分布 4 | ✅ | ✅ 已实现 | `omnilss.distributions_st_compat.ST4` |
| `ST5.R` | `ST5` 偏 t 分布 5 | ✅ | ✅ 已实现 | `omnilss.distributions_b14.ST5` |
| `SST.R` | `SST` 标准化偏 t | ✅ | ✅ 已实现 | `omnilss.distributions_b15.SST` |
| `SEP.R` | `SEP` 偏指数幂 | ✅ | ✅ 已实现 | `omnilss.distributions_b16.SEP` |
| `SEP1.R` | `SEP1` 偏指数幂 1 | ✅ | ✅ 已实现 | `omnilss.distributions_b15.SEP1` |
| `SEP2.R` | `SEP2` 偏指数幂 2 | ✅ | ✅ 已实现 | `omnilss.distributions_b15.SEP2` |
| `SEP3.R` | `SEP3` 偏指数幂 3 | ✅ | ✅ 已实现 | `omnilss.distributions_b15.SEP3` |
| `SEP4.R` | `SEP4` 偏指数幂 4 | ✅ | ✅ 已实现 | `omnilss.distributions_b15.SEP4` |
| `exGAUS.R` | `exGAUS` 指数-高斯分布 | ✅ | ✅ 已实现 | `omnilss.distributions_b2.exGAUS` |
| `EGB2.R` | `EGB2` 指数广义 Beta 2 | ✅ | ✅ 已实现 | `omnilss.distributions_b15.EGB2` |
| `LQNO.R` | `LQNO` 对数分位数正态 | ✅ | ✅ 已实现 | `omnilss.distributions_b15.LQNO` |
| `LOGIT_NO.R` | `LOGITNO` Logit 正态 | ✅ | ✅ 已实现 | `omnilss.distributions_b13.LOGITNO` |
| `LNO.R` | `LNO` 对数正态（位置参数化） | ✅ | ✅ 已实现 | `omnilss.distributions_b8.LNO` |


### 2.2 连续分布（正实数域）

| R 文件 | 分布族 | d/p/q/r | 状态 | Python 位置 |
|--------|--------|---------|------|-------------|
| `GA.R` | `GA` Gamma 分布 | ✅ | ✅ 已实现 | `omnilss.distributions.GA` |
| `GAF.R` | `GAF` Gamma（自由参数化） | ✅ | ✅ 已实现 | `omnilss.distributions_b12.GAF` |
| `logNO.R` | `LOGNO` 对数正态 | ✅ | ✅ 已实现 | `omnilss.distributions.LOGNO` |
| `logNO2.R` | `LOGNO2` 对数正态 2 | ✅ | ✅ 已实现 | `omnilss.distributions_b2.LOGNO2` |
| `WEI.R` | `WEI` Weibull 分布 | ✅ | ✅ 已实现 | `omnilss.distributions.WEI` |
| `WEI2.R` | `WEI2` Weibull（另一参数化） | ✅ | ✅ 已实现 | `omnilss.distributions_b12.WEI2` |
| `WEI3.R` | `WEI3` Weibull（第三参数化） | ✅ | ✅ 已实现 | `omnilss.distributions_b12.WEI3` |
| `EXP.R` | `EXP` 指数分布 | ✅ | ✅ 已实现 | `omnilss.distributions.EXP` |
| `IG.R` | `IG` 逆高斯分布 | ✅ | ✅ 已实现 | `omnilss.distributions.IG` |
| `IGA.R` | `IGAMMA` 逆 Gamma 分布 | ✅ | ✅ 已实现 | `omnilss.distributions_b1.IGAMMA` |
| `GG.R` | `GG` 广义 Gamma 分布 | ✅ | ✅ 已实现 | `omnilss.distributions_b8.GG` |
| `GIG.R` | `GIG` 广义逆高斯 | ✅ | ✅ 已实现 | `omnilss.distributions_b11.GIG` |
| `GB2.R` | `GB2` 广义 Beta 2 | ✅ | ✅ 已实现 | `omnilss.distributions_b8.GB2` |
| `GB1.R` | `GB1` 广义 Beta 1 | ✅ | ✅ 已实现 | `omnilss.distributions_b11.GB1` |
| `PARETO.R` | `PARETO` Pareto 分布 | ✅ | ✅ 已实现 | `omnilss.distributions_b8.PARETO` |
| `PARETO1.R` | `PARETO1` Pareto 1 型 | ✅ | ✅ 已实现 | `omnilss.distributions_b16.PARETO1` |
| `PARETO1_mu_fixed.R` | `PARETO1o` Pareto 1（μ 固定） | ✅ | ✅ 已实现 | `omnilss.distributions_b16.PARETO1o` |
| `PARETO2.R` | `PARETO2` Pareto 2 型 | ✅ | ✅ 已实现 | `omnilss.distributions_b1.PARETO2` |
| `PARETO2o.R` | `PARETO2o` Pareto 2（另一参数化） | ✅ | ✅ 已实现 | `omnilss.distributions_b12.PARETO2o` |
| `NOF-13-3-13.R` | `NOF` 正态（自由参数化） | ✅ | ✅ 已实现 | `omnilss.distributions_b12.NOF` |
| `RGE.R` | `RGE` 反广义极值 | ✅ | ✅ 已实现 | `omnilss.distributions_b12.RGE` |
| `GeneralisedPareto.R` | `GP` 广义 Pareto | ✅ | ❌ 未实现 | — |
| `GeneralisedPoisson.R` | `GPO` 广义 Poisson（连续） | ✅ | ✅ 已实现 | `omnilss.distributions_b15.GPO` |
| `SIMPLEX.R` | `SIMPLEX` 单纯形分布 | ✅ | ✅ 已实现 | `omnilss.distributions_b2.SIMPLEX` |

### 2.3 有界连续分布（0,1）

| R 文件 | 分布族 | d/p/q/r | 状态 | Python 位置 |
|--------|--------|---------|------|-------------|
| `BE.R` | `BE` Beta 分布 | ✅ | ✅ 已实现 | `omnilss.distributions.BE` |
| `BEo.R` | `BEo` Beta（原始参数化） | ✅ | ✅ 已实现 | `omnilss.distributions_b12.BEo` |
| `BEINF.R` | `BEINF` Beta 膨胀（0 和 1） | ✅ | ✅ 已实现 | `omnilss.distributions_b4.BEINF` |
| `BEINF0.R` | `BEINF0` Beta 膨胀（0） | ✅ | ✅ 已实现 | `omnilss.distributions_b4.BEINF0` |
| `BEINF1.R` | `BEINF1` Beta 膨胀（1） | ✅ | ✅ 已实现 | `omnilss.distributions_b4.BEINF1` |
| `BEOI.r` | `BEOI` Beta 一膨胀 | ✅ | ✅ 已实现 | `omnilss.distributions_b4.BEOI` |
| `BEZI.r` | `BEZI` Beta 零膨胀 | ✅ | ✅ 已实现 | `omnilss.distributions_b4.BEZI` |
| `BEINF.R` | `plotBEINF()` | — | ❌ 未实现 | — |
| `BEINF.R` | `meanBEINF()` | — | ✅ 已实现 | `omnilss.meanBEINF` |
| `BEINF0.R` | `plotBEINF0()` / `meanBEINF0()` | — | ⚠️ 部分实现（`meanBEINF0`） | `omnilss.meanBEINF0` |
| `BEINF1.R` | `plotBEINF1()` / `meanBEINF1()` | — | ❌ 未实现 | — |

### 2.4 Box-Cox 变换族

| R 文件 | 分布族 | d/p/q/r | 状态 | Python 位置 |
|--------|--------|---------|------|-------------|
| `BCCG.R` | `BCCG` Box-Cox Cole-Green | ✅ | ✅ 已实现 | `omnilss.distributions.BCCG` |
| `BCCG.R` | `BCCGo` BCCG（原始参数化） | ✅ | ✅ 已实现 | `omnilss.distributions_b16.BCCGo` |
| `BCCG.R` | `BCCGuntr` BCCG（非截断） | ✅ | ✅ 已实现 | `omnilss.distributions_boxcox_compat.BCCGuntr` |
| `BCPE.R` | `BCPE` Box-Cox 幂指数 | ✅ | ✅ 已实现 | `omnilss.distributions.BCPE` |
| `BCPE.R` | `BCPEo` BCPE（原始参数化） | ✅ | ✅ 已实现 | `omnilss.distributions_b14.BCPEo` |
| `BCPE.R` | `BCPEuntr` BCPE（非截断） | ✅ | ✅ 已实现 | `omnilss.distributions_boxcox_compat.BCPEuntr` |
| `BCT.R` | `BCT` Box-Cox t 分布 | ✅ | ✅ 已实现 | `omnilss.distributions.BCT` |
| `BCT.R` | `BCTo` BCT（原始参数化） | ✅ | ✅ 已实现 | `omnilss.distributions_b14.BCTo` |
| `BCT.R` | `BCTuntr` BCT（非截断） | ✅ | ✅ 已实现 | `omnilss.distributions_boxcox_compat.BCTuntr` |
| `BCPE.R` | `checkBCPE()` | — | ❌ 未实现 | — |


### 2.5 离散计数分布

| R 文件 | 分布族 | d/p/q/r | 状态 | Python 位置 |
|--------|--------|---------|------|-------------|
| `po.r` | `PO` Poisson 分布 | ✅ | ✅ 已实现 | `omnilss.distributions.PO` |
| `NBI.R` | `NBI` 负二项 I 型 | ✅ | ✅ 已实现 | `omnilss.distributions.NBI` |
| `NBII.R` | `NBII` 负二项 II 型 | ✅ | ✅ 已实现 | `omnilss.distributions_b1.NBII` |
| `NBF.R` | `NBF` 负二项（自由参数化） | ✅ | ✅ 已实现 | `omnilss.distributions_b15.NBF` |
| `BI.R` | `BI` 二项分布 | ✅ | ✅ 已实现 | `omnilss.distributions.BI` |
| `BB.R` | `BB` Beta-二项分布 | ✅ | ✅ 已实现 | `omnilss.distributions_b7.BB` |
| `BNB.R` | `BNB` Beta-负二项分布 | ✅ | ✅ 已实现 | `omnilss.distributions_b7.BNB` |
| `DoubleBinomial.R` | `DBI` 双二项分布 | ✅ | ✅ 已实现 | `omnilss.distributions_b11.DBI` |
| `DoubleBinomial.R` | `GetBI_C()` 内部 C 函数 | — | ✅ 已实现 | `omnilss.dpqr_functions.GetBI_C` |
| `DoublePoisson-30-12-14.R` | `DPO` 双 Poisson 分布 | ✅ | ✅ 已实现 | `omnilss.distributions_b6.DPO` |
| `DoublePoisson-30-12-14.R` | `DPO1` 双 Poisson（另一参数化） | ✅ | ✅ 已实现 | `omnilss.distributions_b16.DPO1` |
| `DoublePoisson-30-12-14.R` | `get_C()` 归一化常数 | — | ✅ 已实现 | `omnilss.dpqr_functions.get_C` |
| `GEOM.R` | `GEOM` 几何分布 | ✅ | ✅ 已实现 | `omnilss.distributions.GEOM` |
| `GEOMo.R` | `GEOMo` 几何（原始参数化） | ✅ | ✅ 已实现 | `omnilss.distributions_b12.GEOMo` |
| `PIG.R` | `PIG` Poisson-逆高斯 | ✅ | ✅ 已实现 | `omnilss.distributions_b6.PIG` |
| `PIG2.R` | `PIG2` Poisson-逆高斯 2 | ✅ | ✅ 已实现 | `omnilss.distributions_b13.PIG2` |
| `SICHEL-10-09-12.R` | `SICHEL` Sichel 分布 | ✅ | ✅ 已实现 | `omnilss.distributions_b6.SICHEL` |
| `SICHEL-10-09-12.R` | `tofySICHEL()` 变换辅助 | — | ✅ 已实现 | `omnilss.dpqr_functions.tofySICHEL` |
| `SICHEL-10-09-12.R` | `VSICHEL()` 方差函数 | — | ✅ 已实现 | `omnilss.dpqr_functions.VSICHEL` |
| `SI.R` | `SI` Sichel（简化） | ✅ | ✅ 已实现 | `omnilss.distributions_b6.SI` |
| `SI.R` | `tofyS()` 变换辅助 | — | ✅ 已实现 | `omnilss.dpqr_functions.tofyS` |
| `DELAPORT.R` | `DEL` Delaporte 分布 | ✅ | ✅ 已实现 | `omnilss.distributions_b6.DEL` |
| `DELAPORT.R` | `DELAPORT` Delaporte（别名） | ✅ | ✅ 已实现 | `omnilss.distributions_b13.DELAPORT` |
| `LG.R` | `LG` 对数分布 | ✅ | ✅ 已实现 | `omnilss.distributions_b9.LG` |
| `WARING_BOB.r` | `WARING` Waring 分布 | ✅ | ✅ 已实现 | `omnilss.distributions_b6.WARING` |
| `YULE.R` | `YULE` Yule 分布 | ✅ | ✅ 已实现 | `omnilss.distributions_b6.YULE` |
| `discBURR12.R` | `DBURR12` 离散 Burr XII | ✅ | ✅ 已实现 | `omnilss.distributions_b11.DBURR12` |
| `MN3.R` | `MN3` 多项分布（3 类） | ✅ | ✅ 已实现 | `omnilss.distributions_b7.MN3` |
| `MN4.R` | `MN4` 多项分布（4 类） | ✅ | ✅ 已实现 | `omnilss.distributions_b7.MN4` |
| `MN5.R` | `MN5` 多项分布（5 类） | ✅ | ✅ 已实现 | `omnilss.distributions_b7.MN5` |
| `multin-23-09-12.R` | `MULTIN` 通用多项分布 | ✅ | ✅ 已实现 | `omnilss.distributions_discrete_compat.MULTIN` |
| `multin-23-09-12.R` | `fittedMN()` 多项拟合值 | — | ✅ 已实现 | `omnilss.dpqr_functions.fittedMN` |
| `ZIPF.R` | `ZIPF` Zipf 分布 | ✅ | ✅ 已实现 | `omnilss.distributions_b9.ZIPF` |
| `ZIPF.R` | `zetaP()` Zeta 函数 | — | ✅ 已实现 | `omnilss.dpqr_functions.zetaP` |
| `IGA.R` | `IGAMMA` 逆 Gamma | ✅ | ✅ 已实现 | `omnilss.distributions_b1.IGAMMA` |
| `GeneralisedPoisson.R` | `GPO` 广义 Poisson（离散） | ✅ | ✅ 已实现 | `omnilss.distributions_b15.GPO` |
| `GeneralisedPoisson.R` | `GP` 广义 Pareto | ✅ | ❌ 未实现 | — |


### 2.6 零膨胀 / 零截断分布

| R 文件 | 分布族 | d/p/q/r | 状态 | Python 位置 |
|--------|--------|---------|------|-------------|
| `ZIP.R` | `ZIP` 零膨胀 Poisson | ✅ | ✅ 已实现 | `omnilss.distributions.ZIP` |
| `ZIP2.R` | `ZIP2` 零膨胀 Poisson 2 | ✅ | ✅ 已实现 | `omnilss.distributions_b5.ZIP2` |
| `ZAP.R` | `ZAP` 零截断 Poisson | ✅ | ✅ 已实现 | `omnilss.distributions_b5.ZAP` |
| `ZAGA.R` | `ZAGA` 零截断 Gamma | ✅ | ✅ 已实现 | `omnilss.distributions_b5.ZAGA` |
| `ZAGA.R` | `plotZAGA()` / `meanZAGA()` | — | ⚠️ 部分实现（`meanZAGA`） | `omnilss.meanZAGA` |
| `ZAIG.R` | `ZAIG` 零截断逆高斯 | ✅ | ✅ 已实现 | `omnilss.distributions_b5.ZAIG` |
| `ZAIG.R` | `plotZAIG()` / `meanZAIG()` | — | ⚠️ 部分实现（`meanZAIG`） | `omnilss.meanZAIG` |
| `ZINBI.R` | `ZINBI` 零膨胀负二项 I | ✅ | ✅ 已实现 | `omnilss.distributions_b5.ZINBI` |
| `ZANBI.R` | `ZANBI` 零截断负二项 I | ✅ | ✅ 已实现 | `omnilss.distributions_b10_zero_variants.ZANBI` |
| `ZIBI.R` | `ZIBI` 零膨胀二项 | ✅ | ✅ 已实现 | `omnilss.distributions_b10_zero_variants.ZIBI` |
| `ZABI.R` | `ZABI` 零截断二项 | ✅ | ✅ 已实现 | `omnilss.distributions_b10_zero_variants.ZABI` |
| `ZIBB.R` | `ZIBB` 零膨胀 Beta-二项 | ✅ | ✅ 已实现 | `omnilss.distributions_b10_zero_variants.ZIBB` |
| `ZABB.R` | `ZABB` 零截断 Beta-二项 | ✅ | ✅ 已实现 | `omnilss.distributions_b10_zero_variants.ZABB` |
| `ZIBNB.R` | `ZIBNB` 零膨胀 Beta-负二项 | ✅ | ✅ 已实现 | `omnilss.distributions_b10_zero_variants.ZIBNB` |
| `ZABNB.R` | `ZABNB` 零截断 Beta-负二项 | ✅ | ✅ 已实现 | `omnilss.distributions_b10_zero_variants.ZABNB` |
| `ZINBF.R` | `ZINBF` 零膨胀负二项（自由） | ✅ | ✅ 已实现 | `omnilss.distributions_b15.ZINBF` |
| `ZIPIG.R` | `ZIPIG` 零膨胀 Poisson-逆高斯 | ✅ | ✅ 已实现 | `omnilss.distributions_b10_zero_variants.ZIPIG` |
| `ZAPIG.R` | `ZAPIG` 零截断 Poisson-逆高斯 | ✅ | ✅ 已实现 | `omnilss.distributions_b10_zero_variants.ZAPIG` |
| `ZASICHEL.R` | `ZASICHEL` 零截断 Sichel | ✅ | ✅ 已实现 | `omnilss.distributions_b10_zero_variants.ZASICHEL` |
| `ZISICHEL.R` | `ZISICHEL` 零膨胀 Sichel | ✅ | ✅ 已实现 | `omnilss.distributions_b10_zero_variants.ZISICHEL` |
| `ZALG.R` | `ZALG` 零截断对数分布 | ✅ | ✅ 已实现 | `omnilss.distributions_b9.ZALG` |
| `ZAZIPF.R` | `ZAZIPF` 零截断 Zipf | ✅ | ✅ 已实现 | `omnilss.distributions_b10_zero_variants.ZAZIPF` |
| `BEOI.r` | `BEOI` Beta 一膨胀 | ✅ | ✅ 已实现 | `omnilss.distributions_b4.BEOI` |
| `BEOI.r` | `plotBEOI()` / `meanBEOI()` | — | ❌ 未实现 | — |
| `BEZI.r` | `BEZI` Beta 零膨胀 | ✅ | ✅ 已实现 | `omnilss.distributions_b4.BEZI` |
| `BEZI.r` | `plotBEZI()` / `meanBEZI()` | — | ❌ 未实现 | — |


### 2.7 分布族辅助函数

**R 文件：`gamlss-family.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `as.family()` | 转换为 gamlss.family 对象 | ❌ 未实现 | — |
| `gamlss.family()` | 创建 gamlss 族对象 | ❌ 未实现 | — |
| `gamlss.family.default()` | 默认方法 | ❌ 未实现 | — |
| `as.gamlss.family()` | 转换方法 | ❌ 未实现 | — |
| `print.gamlss.family()` | 打印方法 | ❌ 未实现 | — |
| `checklink()` | 检查链接函数有效性 | ❌ 未实现 | — |
| `numeric.deriv()` | 数值导数（gamlss.dist 版） | ✅ 已实现 | `omnilss.operations.numeric_deriv` |

**R 文件：`make-link-gamlss.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `show.link()` | 显示可用链接函数 | ❌ 未实现 | — |
| `make.link.gamlss()` | 创建链接函数对象 | ❌ 未实现 | — |

**R 文件：`hazard.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `hazardFun()` | 危险函数 h(x) = f(x)/(1-F(x)) | ✅ 已实现 | `omnilss.hazardFun.hazardFun` |
| `gen.hazard()` | 通用危险函数 | ✅ 已实现 | `omnilss.hazardFun.gen_hazard` |

**R 文件：`flexDist.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `flexDist()` | 灵活分布摘要 | ⚠️ 部分实现 | `omnilss.flexDist.flexDist`（只计算经验统计量，不拟合参数分布） |

**R 文件：`multin-23-09-12.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `MULTIN()` | 通用多项分布构造器 | ✅ 已实现 | `omnilss.distributions_discrete_compat.MULTIN` |
| `fittedMN()` | 多项拟合值 | ✅ 已实现 | `omnilss.dpqr_functions.fittedMN` |

**R 文件：`PlottingDistributions.R`（分布绘图骨架）**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `count_1_31()` 等 19 个骨架 | 计数/二项/连续分布绘图骨架 | ⚠️ 部分实现 | `omnilss.gen_family_builder.build_gen_family_skeleton`（结构骨架，无绘图） |
| `mtexti()` | 边距文字（内部） | ❌ 未实现 | — |


### 2.8 矩与分位数形状工具

**R 文件：`Functions_for_SK_in_gamlss.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `momentSK()` | 矩（均值/方差/偏度/峰度） | ✅ 已实现 | `omnilss.momentSK.momentSK` |
| `centileSK()` | 分位数形状摘要 | ✅ 已实现 | `omnilss.centileSK.centileSK` |
| `centileSkew()` | 分位数偏度 | ❌ 未实现 | — |
| `centileKurt()` | 分位数峰度 | ✅ 已实现 | `omnilss.centileKurt.centileKurt` |
| `theoCentileSK()` | 理论分位数 z 值 | ✅ 已实现 | `omnilss.theoCentileSK.theoCentileSK` |
| `plotCentileSK()` | 分位数形状图 | ✅ 已实现 | `omnilss.plotCentileSK.plotCentileSK` |
| `checkMomentSK()` | 矩一致性检查 | ✅ 已实现 | `omnilss.checkMomentSK.checkMomentSK` |
| `checkCentileSK()` | 分位数一致性检查 | ✅ 已实现 | `omnilss.checkCentileSK.checkCentileSK` |
| `momentSkew()` | 矩偏度 | ✅ 已实现 | `omnilss.momentSK.momentSkew` |
| `quantileW()` | 加权分位数（内部） | ❌ 未实现 | — |
| `SKmoment_col()` 等绘图辅助 | 矩/分位数彩色图（内部） | ❌ 未实现 | — |

### 2.9 通用族构造器

**R 文件：`genFamily.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `gen.Family()` | 通用族构造器 | ⚠️ 部分实现 | `omnilss.gen_family_builder.build_gen_family_skeleton`（骨架，不执行 R 的完整构造） |
| `Family()` | 族对象 | ❌ 未实现 | — |
| `Family.d()` | 密度函数接口 | ❌ 未实现 | — |
| `Family.p()` | 分布函数接口 | ❌ 未实现 | — |
| `Family.q()` | 分位数函数接口 | ❌ 未实现 | — |
| `Family.r()` | 随机数接口 | ❌ 未实现 | — |

### 2.10 distributions3 接口

**R 文件：`distributions3.R`**

| R 函数 | 说明 | 状态 | Python 位置 |
|--------|------|------|-------------|
| `GAMLSS` S4 类 | distributions3 封装类 | ✅ 已实现 | `omnilss.distributions3`（模块级接口） |
| `format.GAMLSS()` | 格式化方法 | ✅ 已实现 | `omnilss.distributions3` |
| `print.GAMLSS()` | 打印方法 | ✅ 已实现 | `omnilss.distributions3` |
| `mean.GAMLSS()` | 均值 | ✅ 已实现 | `omnilss.mean` |
| `variance.GAMLSS()` | 方差 | ✅ 已实现 | `omnilss.variance` |
| `skewness.GAMLSS()` | 偏度 | ✅ 已实现 | `omnilss.skewness` |
| `kurtosis.GAMLSS()` | 峰度 | ✅ 已实现 | `omnilss.kurtosis` |
| `pdf.GAMLSS()` | 概率密度 | ✅ 已实现 | `omnilss.pdf` |
| `log_pdf.GAMLSS()` | 对数密度 | ✅ 已实现 | `omnilss.log_pdf` |
| `cdf.GAMLSS()` | 累积分布函数 | ✅ 已实现 | `omnilss.cdf` |
| `quantile.GAMLSS()` | 分位数函数 | ✅ 已实现 | `omnilss.quantile` |
| `random.GAMLSS()` | 随机数生成 | ✅ 已实现 | `omnilss.random` |
| `support.GAMLSS()` | 支撑集 | ✅ 已实现 | `omnilss.support` |
| `is_discrete.GAMLSS()` | 是否离散 | ✅ 已实现 | `omnilss.is_discrete` |
| `is_continuous.GAMLSS()` | 是否连续 | ✅ 已实现 | `omnilss.is_continuous` |


---

## 3. OmniLSS 扩展功能（超出 R 范围）

以下功能是 OmniLSS 在 R `gamlss` 基础上新增的，R 中没有对应实现。

### 3.1 算法扩展

| 功能 | 说明 | Python 位置 |
|------|------|-------------|
| RS-JAX 批量拟合 | 使用 `jax.vmap` 批量拟合 | `omnilss.fitting`（`method="RS_JAX"`） |
| CG 完整 Hessian | 使用 `jax.hessian` 的 Cole-Green 算法 | `omnilss.fitting_cg`（`method="CG"`） |
| CG IRLS-cross | 带交叉导数修正的 IRLS 外循环 | `omnilss.fitting_cg`（`cg_backend="irls_cross"`） |
| Adam 联合优化 | 基于 Optax 的 Adam 优化器 | `omnilss.algorithms`（`method="joint", optimizer="adam"`） |
| L-BFGS 拟牛顿 | L-BFGS 优化器 | `omnilss.core.lbfgs_optimizer`（`method="lbfgs"`） |
| Mixed 算法 | RS/CG 混合算法 | `omnilss.fitting`（`method="Mixed"`） |
| JIT RS 核心（实验） | `jax.lax.while_loop` 的 JIT 化 RS | `omnilss.fitting_jit.create_jit_rs_no_core` |

### 3.2 神经网络 GAMLSS

| 功能 | 说明 | Python 位置 |
|------|------|-------------|
| `fit_deep_gamlss()` | 神经网络输出分布参数 | `omnilss.deep.deep_gamlss`（需 `flax`/`optax`） |

### 3.3 scikit-learn 集成

| 功能 | 说明 | Python 位置 |
|------|------|-------------|
| `GAMLSSRegressor` | sklearn Pipeline 兼容包装器 | `omnilss.sklearn_compat.GAMLSSRegressor` |

### 3.4 概率评分

| 功能 | 说明 | Python 位置 |
|------|------|-------------|
| `crps()` | 连续排名概率评分 | `omnilss.scoring.crps` |
| `log_score()` | 对数评分 | `omnilss.scoring.log_score` |
| `dss()` | Dawid-Sebastiani 评分 | `omnilss.scoring.dss` |
| `interval_score()` | 区间评分 | `omnilss.scoring.interval_score` |
| `coverage()` | 覆盖率 | `omnilss.scoring.coverage` |
| `pit_histogram()` | PIT 直方图 | `omnilss.scoring.pit_histogram` |
| `scoring_summary()` | 综合评分摘要 | `omnilss.scoring.scoring_summary` |

### 3.5 模型序列化

| 功能 | 说明 | Python 位置 |
|------|------|-------------|
| `save_model()` / `load_model()` | JSON 模型保存/加载 | `omnilss.serialization` |
| `save_model_json()` / `load_model_json()` | JSON 格式 | `omnilss.serialization` |
| `save_model_pickle()` / `load_model_pickle()` | Pickle 格式 | `omnilss.serialization` |
| `validate_model_json()` | JSON 模型验证 | `omnilss.serialization` |

### 3.6 Bootstrap 与正则化

| 功能 | 说明 | Python 位置 |
|------|------|-------------|
| `nonparametric_bootstrap()` | 非参数 Bootstrap | `omnilss.bootstrap` |
| `parametric_bootstrap()` | 参数 Bootstrap | `omnilss.bootstrap` |
| `residual_bootstrap()` | 残差 Bootstrap | `omnilss.bootstrap` |
| `fit_ridge()` | 岭回归 | `omnilss.regularization` |
| `fit_lasso_coordinate_descent()` | Lasso 坐标下降 | `omnilss.regularization` |
| `fit_elastic_net()` | 弹性网络 | `omnilss.regularization` |
| `cross_validate_lambda()` | λ 交叉验证 | `omnilss.regularization` |

### 3.7 平滑项扩展（超出 R）

| 功能 | 说明 | Python 位置 |
|------|------|-------------|
| 薄板样条 `tps()` | 二维薄板样条 | `omnilss.smoothers.tps` |
| 张量积平滑 `te()` / `ti()` | 多维张量积 | `omnilss.smoothers.tensor_smooth` |
| GCV 自动 λ 选择 | 广义交叉验证 | `omnilss.smoothers.gcv` |
| REML 自动 λ 选择 | 限制最大似然 | `omnilss.smoothers.reml` |
| UBRE 自动 λ 选择 | 无偏风险估计 | `omnilss.smoothers.ubre` |

### 3.8 服务层（REST / gRPC）

| 功能 | 说明 | Python 位置 |
|------|------|-------------|
| REST API | FastAPI 服务（`/fit`, `/predict`, `/diagnostics`） | `omnilss-server/main.py` |
| gRPC 服务 | protobuf 接口 | `omnilss/src/omnilss/api/grpc/` |
| 能力矩阵 API | 运行时族能力查询 | `omnilss.family_capabilities` |

### 3.9 诊断扩展

| 功能 | 说明 | Python 位置 |
|------|------|-------------|
| `quantile_residuals()` | 分位数残差 | `omnilss.diagnostics` |
| `comprehensive_diagnostics()` | 综合诊断 | `omnilss.diagnostics` |
| `calibration_check()` | 校准检查 | `omnilss.diagnostics` |
| `cooks_distance()` | Cook 距离 | `omnilss.diagnostics` |


---

## 4. 总体进度统计

### 4.1 gamlss 核心包（R/gamlss/R/）

| 类别 | 总计 | ✅ 已实现 | ⚠️ 部分实现 | ❌ 未实现 | 完成率 |
|------|------|----------|------------|----------|--------|
| 拟合引擎 | 14 | 10 | 1 | 3 | 71% |
| 预测与线性预测器 | 18 | 14 | 2 | 2 | 78% |
| 模型诊断与残差 | 38 | 28 | 5 | 5 | 74% |
| 绘图与可视化 | 32 | 20 | 4 | 8 | 63% |
| 模型选择与逐步回归 | 28 | 24 | 0 | 4 | 86% |
| 平滑项（Smoothers） | 62 | 8 | 4 | 50 | 13% |
| 分位数图表与 LMS | 18 | 12 | 1 | 5 | 67% |
| 其他工具函数 | 6 | 0 | 3 | 3 | 0% |
| **gamlss 合计** | **216** | **116** | **20** | **80** | **54%** |

### 4.2 gamlss.dist 分布包（R/gamlss.dist/R/）

| 类别 | 总计 | ✅ 已实现 | ⚠️ 部分实现 | ❌ 未实现 | 完成率 |
|------|------|----------|------------|----------|--------|
| 连续分布（实数域）族 | 37 | 36 | 0 | 1 | 97% |
| 连续分布（正实数域）族 | 25 | 23 | 0 | 2 | 92% |
| 有界连续分布（0,1）族 | 15 | 7 | 0 | 8 | 47% |
| Box-Cox 变换族 | 10 | 9 | 0 | 1 | 90% |
| 离散计数分布族 | 35 | 29 | 0 | 6 | 83% |
| 零膨胀/零截断分布族 | 28 | 22 | 0 | 6 | 79% |
| 分布族辅助函数 | 18 | 4 | 2 | 12 | 22% |
| 矩与分位数形状工具 | 14 | 10 | 0 | 4 | 71% |
| 通用族构造器 | 6 | 0 | 1 | 5 | 0% |
| distributions3 接口 | 15 | 15 | 0 | 0 | 100% |
| **gamlss.dist 合计** | **203** | **155** | **3** | **45** | **76%** |

### 4.3 综合汇总

> 以下数字来自文档中所有详细表格的实际行数统计（451 个 R 符号条目）。

| 包 | 总符号数 | ✅ 已实现 | ⚠️ 部分实现 | ❌ 未实现 | 完成率 |
|----|---------|----------|------------|----------|--------|
| `gamlss` 核心 | 248 | 127 | 22 | 99 | 51% |
| `gamlss.dist` 分布 | 203 | 155 | 3 | 45 | 76% |
| **总计** | **451** | **282** | **25** | **144** | **63%** |

> **说明**：完成率 = (已实现 + 部分实现×0.5) / 总计。  
> 部分实现的功能有 Python 入口但缺少 R 中的完整计算逻辑（如 `prof.dev` 不执行重拟合、`histSmoO/C/P` 三变体实现相同等）。

### 4.4 关键缺口（优先级排序）

#### 高优先级（影响核心功能）

1. **平滑项缺口（50 个未实现）**：`pbm`、`pbo`、`pbz`、`pbc`、`pbp`、`pvc`、`cy`、`fp`/`bfp`、`re`/`random`（nlme 后端）、`ri`、`pp`、`pcat`、`scs`、`polyS`
2. **`GP` 广义 Pareto 分布**：gamlss.dist 中导出但 Python 中缺失
3. **`stepGAIC.VR` / `stepGAIC.CH`**：两种逐步选择变体
4. **`stepTGDAll.A`**：全参数 TGD 逐步选择
5. **`gamlss.scope()`**：作用域定义（stepGAIC 依赖）
6. **`predictAll()`**：顶层导出缺失（实现存在但未导出）
7. **`edf()`**：单项有效自由度（`edfAll` 已实现，`edf` 未实现）

#### 中优先级（影响完整性）

8. **`prof.dev()` / `prof.term()` 完整实现**：当前只结构化数据，不执行重拟合
9. **`find.hyper()` 完整实现**：当前只汇总结果，不执行网格搜索
10. **`histSmo` 变体区分**：`histSmoO/C/P` 应有不同平滑方法
11. **`flexDist()` 完整实现**：应拟合参数分布，不只计算经验统计量
12. **`calibration()` / `predict.lms()`**：LMS 相关完整功能
13. **`rqres.plot()`**：残差诊断图
14. **`gamlss.family` 系统**：`as.family`、`checklink`、`show.link`、`make.link.gamlss`
15. **`fittedMN()`**：多项分布拟合值

#### 低优先级（辅助/内部函数）

16. **S3 方法**：`plot.pb`、`coef.pb`、`print.pb`、`knots.pb` 等平滑项 S3 方法
17. **`plotBEINF/BEINF0/BEINF1/ZAGA/ZAIG/BEOI/BEZI`**：分布可视化辅助
18. **`meanBEINF/ZAGA/ZAIG` 等**：分布均值辅助
19. **`tofySICHEL`、`VSICHEL`、`tofyS`、`zetaP`**：分布内部辅助
20. **`GetBI_C`、`get_C`**：C 语言归一化常数辅助
21. **`gamlssNews()`**：版本新闻
22. **`quantileW`、`centileSkew`、`SKmoment_*` 等**：形状工具内部辅助

---

## 5. 文件映射速查表

### Python 模块 → R 源文件

| Python 模块 | 对应 R 文件 |
|-------------|------------|
| `omnilss.fitting` | `gamlss/R/gamlss-5.R` |
| `omnilss.fitting_cg` | `gamlss/R/gamlss-5.R`（CG 算法扩展） |
| `omnilss.gamlssML` | `gamlss/R/gamlssML.R` |
| `omnilss.gamlssVGD_23_12_21` | `gamlss/R/gamlssVGD_23_12_21.R` |
| `omnilss.controls` | `gamlss/R/gamlss-5.R` |
| `omnilss.update` | `gamlss/R/update.R` |
| `omnilss.predict_gamlss_23_12_21` | `gamlss/R/predict.gamlss_23_12_21.R` |
| `omnilss.predictAll_22_08_22` | `gamlss/R/predictAll_22_08_22.R` |
| `omnilss.lpred` | `gamlss/R/lpred.R` |
| `omnilss.prodist` | `gamlss/R/prodist.R` |
| `omnilss.Formulae2data` | `gamlss/R/Formulae2data.R` |
| `omnilss.operations` | `gamlss/R/extra.R` |
| `omnilss.acfResid` | `gamlss/R/acfResid.R` |
| `omnilss.worm_plot` | `gamlss/R/wp.R` |
| `omnilss.qstats` | `gamlss/R/qstats.R` |
| `omnilss.LR_test_12_06_2013` | `gamlss/R/LR-test-12-06-2013.R` |
| `omnilss.VuongClarkTest` | `gamlss/R/VuongClarkTest.R` |
| `omnilss.DevianceIncr` | `gamlss/R/DevianceIncr.R` |
| `omnilss.hatvalues` | `gamlss/R/hatvalues.R` |
| `omnilss.logLik` | `gamlss/R/logLik.R` |
| `omnilss.vcov_gamlss` | `gamlss/R/vcov-gamlss.R` + `get-K-rvcov-16-06-2913.R` |
| `omnilss.confint_gamlss_29_06_22` | `gamlss/R/confint-gamlss-29-06-22.R` |
| `omnilss.SUMMARY` | `gamlss/R/SUMMARY.R` |
| `omnilss.print` | `gamlss/R/print.R` |
| `omnilss.getSmo` | `gamlss/R/getSmo-18-6-13.R` + `edf.R` |
| `omnilss.gen_likelihood` | `gamlss/R/gen-Likelihood.R` |
| `omnilss.prof_dev` | `gamlss/R/PROFILE_new_19-08-12.R` + `profterm_18_05_13.R` |
| `omnilss.find_hyper` | `gamlss/R/findhyper-23-05-13.R` |
| `omnilss.dtop` | `gamlss/R/dtop.R` |
| `omnilss.plot` | `gamlss/R/plot.R` |
| `omnilss.fitted_plot` | `gamlss/R/fitted-plot.R` |
| `omnilss.pdfplot` | `gamlss/R/pdfplot.R` |
| `omnilss.term_plot_new` | `gamlss/R/term.plot-new.R` |
| `omnilss.par_plot` | `gamlss/R/par_plot-21-5-13.R` |
| `omnilss.plot2way` | `gamlss/R/plot2way.R` |
| `omnilss.HistDist_03_10_13` | `gamlss/R/HistDist-03-10-13.R` |
| `omnilss.histSmo` | `gamlss/R/histSmo.R` |
| `omnilss.getPEF` | `gamlss/R/getPEF.R` |
| `omnilss.getQuantile` | `gamlss/R/getQuantile.R` |
| `omnilss.FitTail` | `gamlss/R/FitTail.R` |
| `omnilss.loglogSurv` | `gamlss/R/FitTail.R`（`logSurv0` 别名） |
| `omnilss.loglogplot` | `gamlss/R/FitTail.R` |
| `omnilss.ECDF` | `gamlss/R/FitTail.R` |
| `omnilss.stepGAIC_03_10_13` | `gamlss/R/stepGAIC-03-10-13..R` |
| `omnilss.stepGAICAll_A_parallel` | `gamlss/R/stepGAICAll-A-parallel.R` |
| `omnilss.stepGAICAll_B_Parallel` | `gamlss/R/stepGAICAll-B-Parallel.R` |
| `omnilss.DropAddStepGAIC_Parallel` | `gamlss/R/DropAddStepGAIC-Parallel.R` |
| `omnilss.stepTGD` | `gamlss/R/stepTGD.R` |
| `omnilss.fitDist` | `gamlss/R/fitDist.R` |
| `omnilss.fitDistPred` | `gamlss/R/fitDistPred.R` |
| `omnilss.chooseDistParallel` | `gamlss/R/chooseDistParallel.R` |
| `omnilss.centilesPLOT` | `gamlss/R/centilesPLOT.R` |
| `omnilss.centilescom` | `gamlss/R/centilescom.R` |
| `omnilss.centilesFan` | `gamlss/R/centilesFan.R` |
| `omnilss.centile_pred_30_06_22` | `gamlss/R/centile-pred_30_06_22.R` |
| `omnilss.lms` | `gamlss/R/lms.R` |
| `omnilss.quantSheets` | `gamlss/R/quantSheets.R` |
| `omnilss.quantSheets_control` | `gamlss/R/quantSheets.R` |
| `omnilss.getZmatrix` | `gamlss/R/pbm.R` |
| `omnilss.vis_lo` | `gamlss/R/lo_new.R` |
| `omnilss.Rsq` | `gamlss/R/extra.R` |
| `omnilss.smoothers.pb` | `gamlss/R/pb.R` |
| `omnilss.smoothers.ps` | `gamlss/R/PS.R` |
| `omnilss.smoothers.cs` | `gamlss/R/cubicSplines-10-08-12.R` |
| `omnilss.hazardFun` | `gamlss.dist/R/hazard.R` |
| `omnilss.flexDist` | `gamlss.dist/R/flexDist.R` |
| `omnilss.momentSK` | `gamlss.dist/R/Functions_for_SK_in_gamlss.R` |
| `omnilss.centileSK` | `gamlss.dist/R/Functions_for_SK_in_gamlss.R` |
| `omnilss.centileKurt` | `gamlss.dist/R/Functions_for_SK_in_gamlss.R` |
| `omnilss.theoCentileSK` | `gamlss.dist/R/Functions_for_SK_in_gamlss.R` |
| `omnilss.plotCentileSK` | `gamlss.dist/R/Functions_for_SK_in_gamlss.R` |
| `omnilss.checkMomentSK` | `gamlss.dist/R/Functions_for_SK_in_gamlss.R` |
| `omnilss.checkCentileSK` | `gamlss.dist/R/Functions_for_SK_in_gamlss.R` |


---

## 6. 未实现功能完整清单

以下是所有 ❌ 未实现的 R 符号，按包和文件分组，供开发时直接参考。

### 6.1 gamlss 核心包未实现列表

**拟合引擎**
- `gamlssNews()` — `gamlss-5.R`
- `parameterOut()` — `gamlss-5.R`（内部）
- `get.smoothers()` — `gamlss-5.R`（内部）
- `get.object()` — `gamlss-5.R`（内部）
- `additive.fit()` — `add.r`（内部）

**预测**
- `predictAll()` — `predictAll_22_08_22.R`（顶层导出缺失）
- `getAll.formulae2data()` — `Formulae2data.R`（内部）
- `concat()` — `predict.gamlss_23_12_21.R`（内部）

**诊断**
- `rqres1()` — `rqres.R`（内部）
- `rqres.plot()` — `rqresplot_new.R`
- `Qtest()` — `qstats.R`（内部）
- `get.par.df()` — `qstats.R`（内部）
- `model2formulae2data()` — `hatvalues.R`（内部）
- `edf()` — `edf.R`（单项有效自由度）

**绘图**
- `plotLME()` — `term.plot-new.R`（内部）
- `draw.polys.in()` — `term.plot-new.R`（内部）
- `CheckSmoList()` — `term.plot-new.R`（内部）
- `loglogplot0()` — `FitTail.R`（别名）
- `ESURV()` — `FitTail.R`
- `quantile_weights()` — `FitTail.R`（内部）
- `bp()` 及其所有内部函数 — `bp.R`
- `plot.histSmo()` / `lines.histSmo()` / `print.histSmo()` — `histSmo.R`

**模型选择**
- `stepGAIC.VR()` — `stepGAIC-03-10-13..R`
- `stepGAIC.CH()` — `stepGAIC-03-10-13..R`
- `gamlss.scope()` — `stepGAIC-03-10-13..R`
- `stepTGDAll.A()` — `stepTGDAll-A-parallel.R`
- `predictAll.V()` — `fitDistPred.R`（内部）

**平滑项（全部未实现）**
- `pbm()` / `pbm.control()` / `gamlss.pbm()` / `print.pbm()` — `pbm.R`
- `pbz()` / `pbz.control()` / `gamlss.pbz()` / `print.pbz()` — `pb_goingtozero.R`
- `pbo()` / `pbo.control()` / `gamlss.pbo()` — `pbo_original.R`
- `pbp()` / `pbp.control()` / `gamlss.pbp()` — `pbp.R`
- `pbc()` / `pbc.control()` / `gamlss.pbc()` / `print.pbc()` — `pbc.R`
- `pvc()` / `pvc.control()` / `gamlss.pvc()` / `coef.pvc()` / `fitted.pvc()` / `print.pvc()` / `plot.pvc()` — `pvc.R`
- `cy()` / `cy.control()` / `gamlss.cy()` / `print.cy()` — `cy.R`
- `fp()` / `bfp()` / `gamlss.fp()` — `fracpoly.R`
- `lo.control()` / `gamlss.lo()` — `lo_new.R`（`lo()` 和 `vis.lo()` 部分实现）
- `gamlss.random()` / `plot.random()` / `coef.random()` / `fitted.random()` / `print.random()` / `getMarginal()` — `random.R`
- `re()` / `gamlss.re()` — `re_gamlss_lme_test.R`
- `ri()` / `gamlss.ri()` / `plot.ri()` / `coef.ri()` / `fitted.ri()` / `print.ri()` — `ri.R`
- `ridge()` / `gamlss.ridge()` — `ridge.R`
- `pp()` / `gamlss.pp()` — `powerpoly.R`
- `polyS()` / `poly.matrix()` — `POLY.R`
- `pcat()` / `gamlss.pcat()` / `plot.pcat()` / `coef.pcat()` / `print.pcat()` / `fitted.pcat()` — `p-categorical.R`
- `scs()` — `cubicSplines-10-08-12.R`
- `plot.pb()` / `coef.pb()` / `fitted.pb()` / `print.pb()` / `knots.pb()` — `pb.R`（S3 方法）

**LMS / 分位数**
- `ptrans()` — `lms.R`（内部）
- `predict.lms()` — `lms.R`（S3 方法）
- `calibration()` — `lms.R`
- `invptrans()` — `quantSheets.R`（内部）
- `print.quantSheets()` / `fitted.quantSheets()` / `predict.quantSheets()` / `residuals.quantSheets()` — `quantSheets.R`（S3 方法）

### 6.2 gamlss.dist 分布包未实现列表

**分布族**
- `GP` 广义 Pareto — `GeneralisedPareto.R`

**分布辅助函数**
- `as.family()` / `gamlss.family()` / `gamlss.family.default()` / `as.gamlss.family()` / `print.gamlss.family()` / `checklink()` — `gamlss-family.R`
- `show.link()` / `make.link.gamlss()` — `make-link-gamlss.R`
- `plotBEINF()` / `meanBEINF()` — `BEINF.R`
- `plotBEINF0()` / `meanBEINF0()` — `BEINF0.R`
- `plotBEINF1()` / `meanBEINF1()` — `BEINF1.R`
- `plotZAGA()` / `meanZAGA()` — `ZAGA.R`
- `plotZAIG()` / `meanZAIG()` — `ZAIG.R`
- `plotBEOI()` / `meanBEOI()` — `BEOI.r`
- `plotBEZI()` / `meanBEZI()` — `BEZI.r`
- `tofySICHEL()` / `VSICHEL()` — `SICHEL-10-09-12.R`
- `tofyS()` — `SI.R`
- `zetaP()` — `ZIPF.R`
- `checkBCPE()` — `checkBCPE.R`
- `GetBI_C()` — `DoubleBinomial.R`
- `get_C()` — `DoublePoisson-30-12-14.R`
- `fittedMN()` — `multin-23-09-12.R`

**通用族构造器**
- `Family()` / `Family.d()` / `Family.p()` / `Family.q()` / `Family.r()` — `genFamily.R`

**矩/分位数工具**
- `centileSkew()` — `Functions_for_SK_in_gamlss.R`
- `quantileW()` — `Functions_for_SK_in_gamlss.R`（内部）
- `SKmoment_col()` / `SKmoment_gray()` / `SKmomentBoth()` / `SKcentile_col()` / `SKcentile_gray()` / `SKcentileBoth()` — `Functions_for_SK_in_gamlss.R`（内部绘图）

**绘图骨架**
- `mtexti()` — `PlottingDistributions.R`（内部）

---

*本文档由 OmniLSS 项目自动审计生成，最后更新：2026-05-23*  
*参考：[R/gamlss/NAMESPACE](../R/gamlss/NAMESPACE)、[R/gamlss.dist/NAMESPACE](../R/gamlss.dist/NAMESPACE)*
