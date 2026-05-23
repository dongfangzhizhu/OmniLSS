# OmniLSS vs R `gamlss` / `gamlss.dist` — Coverage Audit and Implementation Plan (2026-05-22)

This note inventories the R `gamlss` and `gamlss.dist` source trees that ship in
`R/gamlss/` and `R/gamlss.dist/` and compares them to the current OmniLSS Python
implementation under `omnilss/src/omnilss/`. The goal is a verifiable picture of
what is implemented, what is partial, and what is still missing — together with
a prioritized plan to close the remaining gaps and a guide for using the
devcontainer to develop and validate against R offline.

OmniLSS is **not** required to ship every R routine; this plan exists so that the
delta is explicit and so that the validation gate, capability registry, and
documentation stay honest.

Cross-references:
- Source trees: [`R/gamlss/`](../../R/gamlss/), [`R/gamlss.dist/`](../../R/gamlss.dist/)
- Registry: [`omnilss/src/omnilss/distribution_registry.py`](../../omnilss/src/omnilss/distribution_registry.py)
- Capability matrix: [`docs/development/family-capability-matrix-2026-05-18.json`](family-capability-matrix-2026-05-18.json)
- R consistency coverage: [`docs/testing/r-consistency-coverage.md`](../testing/r-consistency-coverage.md)
- Architecture freeze: [`docs/architecture/30-day-feature-freeze.md`](../architecture/30-day-feature-freeze.md)
- Offline bootstrap: [`docs/development/offline-dev-bootstrap-2026-05-17.md`](offline-dev-bootstrap-2026-05-17.md)
- R environment guide: [`docs/development/r-environment-setup.md`](r-environment-setup.md)

## 1. Method

1. List every `.R` source file under `R/gamlss/R/` and `R/gamlss.dist/R/`,
   plus the symbol set declared in each `NAMESPACE`.
2. Map each R symbol to a Python implementation by greping
   `omnilss/src/omnilss/**/*.py` for matching module names, factory functions,
   and `R reference:` doc strings.
3. Mark each R symbol as one of:
   - `implemented` — has a public Python entry point with parity intent and at
     least one R-backed test or numerical contract test;
   - `partial` — Python entry point exists but lacks R-backed validation or
     diverges in API/parameterization;
   - `missing` — no Python equivalent exists.
4. Cross-check the result against `family-capability-matrix-2026-05-18.json`
   (which records evidence tier per family/feature pair).

The categorization below is conservative: when in doubt, the entry is marked
`partial` so the validation plan does not assume coverage that is not proven.

### 1.1 Implementation-grounded correction (2026-05-23)

The current development plan must distinguish **callable API surface** from
**real statistical implementation**. A code audit on 2026-05-23 found several
registered families and helpers whose Python names are present but whose numeric
kernels are compatibility aliases or placeholders:

- `omnilss/distributions_st_compat.py`: `ST1`, `ST2`, `ST3`, and `ST4` clone
  the `ST3C` kernel while preserving distinct family names.
- `omnilss/distributions_count_compat.py`: `DBI` reuses the `BI` kernel.
- `omnilss/distributions_discrete_compat.py`: `DBURR12` reuses the `SICHEL`
  kernel.
- `omnilss/distributions_boxcox_compat.py`: `BCCGuntr`, `BCPEuntr`, and
  `BCTuntr` clone the truncated Box-Cox kernels.
- `omnilss/gen_family_builder.py`: `count_*` and `binom_*` skeletons route to
  `PO`; `contR*` skeletons route to `NO`.
- `omnilss/dpqr_functions.py`: generic fallback p/q/r helpers still return
  normal approximations or `NaN` placeholders for distributions without real
  CDF/quantile/random implementations.

For this plan, those entries are therefore **not implemented** for statistical
coverage purposes. They may remain useful as temporary import/API sentinels, but
they must not be treated as migrated R functionality, must not be promoted to
`validated`, and must not be exposed by production service routes until their
dedicated likelihood, d/p/q/r, parameterization, and R-backed tests exist.

Scope for the next development cycle:

1. Primary: migrate every required `R/gamlss` and `R/gamlss.dist` behavior into
   `omnilss/`.
2. Secondary: expose validated `omnilss` behavior through `omnilss-server/`.
3. Out of scope for this document cycle: `omnilss-pro/`.

## 2. Distribution coverage (`gamlss.dist`)

R `gamlss.dist` exports d/p/q/r quartets for ~100 named families, plus several
helpers (`flexDist`, `hazardFun`, `momentSK`, `MULTIN`, generic count/binomial
skeletons, etc.).

### 2.1 Callable in OmniLSS (registered + Python factory present)

The following families have a `FamilyDefinition` factory and are reachable
through `omnilss.distribution_registry.resolve()`:

Continuous on the real line:
NO, NO2, GU, RG, LO, TF, PE, PE2, NET, EXP, JSU, JSUo, GT, SHASH, SHASHo,
SHASHo2, LOGSHASH, LOGSHASHo, SN1, SN2, EGB2, exGAUS, SEP, SEP1, SEP2, SEP3,
SEP4, SST, ST3C, ST5, LQNO

Continuous positive (real⁺):
GA, GAF, IG, IGA, IGAMMA, LOGNO, LOGNO2, LNO, WEI, WEI2, WEI3, GG, GIG, GB2,
PARETO, PARETO1, PARETO1o, PARETO2, PARETO2o, NOF, RGE

Bounded (0,1) and inflated:
BE, BEo, BEINF, BEINF0, BEINF1, BEZI, BEOI, SIMPLEX, GB1, LOGITNO

Box-Cox / Cole-Green family:
BCCG, BCCGo, BCT, BCTo, BCPE, BCPEo

Discrete counts:
PO, NBI, NBII, NBF, GEOM, GEOMo, ZIP, ZIP2, ZAP, ZAGA, ZAIG, ZIBI, ZABI, ZINBI,
ZANBI, ZIBB, ZABB, ZIBNB, ZABNB, ZIPIG, ZAPIG, ZASICHEL, ZISICHEL, ZINBF,
ZIPF, ZAZIPF, ZALG, LG, PIG, PIG2, SICHEL, SI, DEL, DELAPORT, DPO, DPO1,
GeneralisedPoisson (R `GP`), GPO, YULE, WARING, BNB, BB

Binomial / categorical:
BI, MN3, MN4, MN5

Total: **≈105 FamilyDefinition factories** registered in
`omnilss.distribution_registry`. Almost all are tier `experimental` for fitting
features and `validated` only when there is a repository R-consistency test.
Note: factory `JSU` is exposed both from `distributions.py` and
`distributions_b11.py`; the b11 alias should be reconciled to a single source of
truth before the next release.

Implementation correction: registration is only a callable-surface fact. It is
not sufficient evidence of migrated R behavior. Families implemented as
compatibility aliases (`ST1`-`ST4`, `DBI`, `DBURR12`, untruncated Box-Cox
variants, and `gen.Family` skeleton names) must be counted separately from
families with dedicated numerical kernels. Families whose p/q/r paths use
generic `NaN` or normal-approximation fallbacks must be marked `placeholder`
until replaced.

### 2.2 Validated against native R (full d/p/q + fitting parity)

From `docs/testing/r-consistency-coverage.md` and the existing
`tests/test_r_consistency_*.py` files, repository-level R-backed test files
exist for: NO, NO2, GA, PO, BI, IG, LO, LOGNO, LOGNO2, BE, BEINF, EXP, GU, RG,
SHASH, SN1, GT, NBI, NBII, ZIP, ZAGA, GEOM, exGAUS, IGAMMA, PARETO2, PE, TF,
WEI, SIMPLEX, plus extended d/p/q tests for BB, BCCG, BCPE, BCT, BEINF, GB2,
GG, JSU, NET, PE, PIG, SHASH, ZAGA, exGAUS.

The capability matrix marks the following families with
`r_consistency = validated`:
BCCG, BCPE, BCT, BE, BEINF, BI, EXP, GA, GEOM, GT, GU, IG, IGAMMA, LO, LOGNO,
LOGNO2, NBI, NBII, NO, NO2, PARETO2, PE, PO, RG, SHASH, SIMPLEX, SN1, TF, WEI,
ZAGA, ZIP — **31 families**.

All other registered families are tier `experimental` and need extra evidence
before any equivalence claim.

### 2.3 Missing, aliased, placeholder, or untested distribution exports

R-side exports that are not yet exposed by OmniLSS, whose Python equivalent is
currently only an alias/placeholder, or whose parity evidence is incomplete:

- ST1, ST2, ST3, ST4 — factories exist, but they currently clone the `ST3C`
  numeric kernel and therefore remain `compat_alias`, not implemented R
  families.
- TF2 — alternative Student-t parameterization with degrees-of-freedom
  parameter; verify that implementation is not only a `TF` alias.
- DBI — Double Binomial; currently reuses `BI` and must be replaced.
- DBURR12 — Discrete Burr-12; currently reuses `SICHEL` and must be replaced.
- BCCGuntr, BCPEuntr, BCTuntr — untruncated Box-Cox variants; currently clone
  truncated kernels and must be replaced.
- MULTIN — multinomial fitting helper (`MN3/4/5` exist as factories but the
  generic `MULTIN(...)` constructor used in R formulas is missing).
- `flexDist`, `momentSK`, `centileSK`, `momentSkew`, `centileSkew`,
  `centileKurt`, `theoCentileSK`, `plotCentileSK`, `checkMomentSK`,
  `checkCentileSK` — distribution-shape utilities.
- `hazardFun`, `gen.hazard` — generic hazard helpers.
- Generic skeletons used by `gen.Family`: `count_1_31`, `count_1_22`,
  `count_2_32`, `count_2_32R`, `count_2_33`, `count_3_32`, `count_3_33`,
  `binom_1_31`, `binom_2_33`, `binom_3_33`, `contR_2_12`, `contR_3_11`,
  `contR_4_13`, `contRplus_1_1`, `contRplus_2_11`, `contRplus_3_13`,
  `contRplus_4_33`, `contR01_2_13`, `contR01_4_33`.
  Current compatibility builders route these names to `PO` or `NO`; they must
  become real generic builders before being counted as migrated.
- `distributions3` interface (`GAMLSS` S4 wrapper, `pdf`, `log_pdf`, `cdf`,
  `quantile`, `random`, `is_continuous`, `is_discrete`, `support`, `mean`,
  `variance`, `skewness`, `kurtosis`).

Even where a Python factory exists, every family still needs an R-backed d/p/q
parity test before the capability tier can move from `experimental` to
`validated`.

## 3. `gamlss` core function coverage (R/gamlss/R)

The following table summarizes every R source file under `R/gamlss/R/`. Each
entry lists the canonical R symbol and the closest Python module/function. An
empty Python column means the routine is missing.

### 3.1 Fitting and control

| R file | R symbol | Python equivalent | Status |
|---|---|---|---|
| `gamlss-5.R` | `gamlss` | `omnilss.fitting.gamlss`, `omnilss.gamlss_5` | implemented |
| `gamlss-5.R` | `gamlss.control`, `glim.control` | `omnilss.controls`, `omnilss.gamlss_5.gamlss_control_exact` | implemented |
| `gamlssML.R` | `gamlssML` | `omnilss.gamlssML` | partial (no R fitting parity test yet) |
| `gamlssVGD_23_12_21.R` | `gamlssVGD`, `VGD`, `getTGD`, `TGD`, `gamlssCV`, `CV` | `omnilss.gamlssVGD_23_12_21`, `omnilss.methods` | implemented |
| `update.R` | `update.gamlss`, `refit` | `omnilss.update`, `omnilss.operations.refit` | implemented |
| `predict.gamlss_23_12_21.R` | `predict.gamlss` | `omnilss.predict_gamlss_23_12_21`, `omnilss.prediction.predict_params` | implemented |
| `predictAll_22_08_22.R` | `predictAll` | `omnilss.predictAll_22_08_22.predict_all` | implemented |
| `lpred.R` | `lpred`, `lp`, `fv` | `omnilss.lpred`, `omnilss.operations.lp`, `fv` | implemented |
| `prodist.R` | `prodist.gamlss` | `omnilss.prodist.prodist_data` | implemented |
| `Formulae2data.R` | `Formulae2data` | none | missing |
| `gen-Likelihood.R` | `gen.likelihood` | none | missing |

### 3.2 Diagnostics, plots, and statistics

| R file | R symbol | Python equivalent | Status |
|---|---|---|---|
| `acfResid.R` | `acfResid` | `omnilss.acfResid.acf_residuals` | implemented |
| `confint-gamlss-29-06-22.R` | `confint.gamlss` | `omnilss.confint_gamlss_29_06_22` | implemented |
| `vcov-gamlss.R` | `vcov.gamlss`, `rvcov`, `get.K` | `omnilss.vcov_gamlss.vcov` | partial (`rvcov`, `get.K` missing) |
| `SUMMARY.R` | `summary.gamlss` | `omnilss.SUMMARY.summary` | implemented |
| `print.R` | `print.gamlss` | `omnilss.print.print_model` | implemented |
| `plot.R` | `plot.gamlss` | `omnilss.plot.plot_diagnostics` | partial (returns data instead of figures, parity not formally checked) |
| `rqres.R`, `rqresplot_new.R` | `rqres.plot`, `get.rqres` | `omnilss.diagnostics.residual_plot_data`, `omnilss.methods.get_rqres_samples` | partial (no explicit `rqres.plot`) |
| `wp.R` | `wp` | `omnilss.worm_plot.wp`, `wp_data` | implemented |
| `dtop.R` | `dtop` | none | missing |
| `qstats.R` | `Q.stats` | `omnilss.qstats.q_stats` | implemented |
| `extra.R` | `IC`, `fv`, `coefAll`, `.hat.WX` | `omnilss.operations.ic`, `fv`, `coef_all`, `hat_wx` | implemented |
| `LR-test-12-06-2013.R` | `LR.test` | `omnilss.LR_test_12_06_2013.likelihood_ratio_test` | implemented |
| `VuongClarkTest.R` | `VC.test` | `omnilss.VuongClarkTest.vuong_clarke_test` | implemented |
| `DevianceIncr.R` | `devianceIncr` | `omnilss.DevianceIncr.devianceIncr` | implemented |
| `centile-pred_30_06_22.R` | `centiles.pred` | `omnilss.centile_pred_30_06_22.centile_pred_data` | implemented |
| `centilescom.R` | `centiles.com` | `omnilss.centilescom.centiles_comparison_data` | implemented |
| `centilesFan.R` | `centiles.fan` | `omnilss.centilesFan.centiles_fan` | implemented |
| `centilesPLOT.R` | `centiles`, `centiles.split` | `omnilss.centilesPLOT.centiles_data`, `centiles_split_data` | implemented |
| `fitted-plot.R` | `fittedPlot` | `omnilss.fitted_plot.fitted_plot_data` | implemented |
| `pdfplot.R` | `pdf.plot` | `omnilss.pdfplot.pdf_plot_data` | implemented |
| `term.plot-new.R` | `term.plot` | `omnilss.term_plot_new.term_plot_data` | implemented |
| `par_plot-21-5-13.R` | `par.plot` | none | missing |
| `plot2way.R` | `plot2way` | `omnilss.plot2way.plot2way_data` | implemented |
| `getPEF.R` | `getPEF` | `omnilss.getPEF.get_pef_data` | implemented |
| `getQuantile.R` | `getQuantile` | `omnilss.getQuantile.get_quantile_data` | implemented |
| `prof.dev`, `prof.term` | `PROFILE_new_19-08-12.R`, `profterm_18_05_13.R` | none | missing |
| `findhyper-23-05-13.R` | `find.hyper` | none | missing |
| `Rsq.R` (in `extra.R`) | `Rsq.gamlss` | `omnilss.Rsq.rsq` | implemented |
| `hatvalues.R` | `hatvalues.gamlss` | `omnilss.hatvalues.hatvalues` | implemented |
| `logLik.R` | `logLik.gamlss` | `omnilss.logLik.logLik` | implemented |

### 3.3 Model selection and stepwise

| R file | R symbol | Python equivalent | Status |
|---|---|---|---|
| `stepGAIC-03-10-13..R` | `stepGAIC.VR`, `stepGAIC.CH`, `stepGAIC`, `gamlss.scope`, `extractAIC.gamlss` | `omnilss.stepGAIC_03_10_13`, `omnilss.operations.gaic`, `omnilss.methods.compare_models` | partial (both VR and CH variants are merged; `gamlss.scope` not explicitly exposed) |
| `stepGAICAll-A-parallel.R` | `stepGAICAll.A` | `omnilss.stepGAICAll_A_parallel.stepGAICAll_A` | implemented |
| `stepGAICAll-B-Parallel.R` | `stepGAICAll.B`, `addtermAllP`, `droptermAllP` | `omnilss.stepGAICAll_B_Parallel` | implemented |
| `DropAddStepGAIC-Parallel.R` | `drop1All`, `add1All`, `drop1`, `add1` | `omnilss.DropAddStepGAIC_Parallel`, `omnilss.methods.dropterm_all/addterm_all` | implemented |
| `stepTGD.R`, `stepTGDAll-A-parallel.R` | `stepTGD`, `extractTGD`, `drop1TGD`, `add1TGD`, `drop1TGDP`, `add1TGDP`, `stepTGDAll.A` | `omnilss.stepTGD` | implemented |
| `chooseDistParallel.R` | `chooseDist`, `chooseDistPred`, `getOrder` | `omnilss.chooseDistParallel`, `omnilss.fitDist`, `omnilss.fitDistPred` | implemented |
| `fitDist.R`, `fitDistPred.R` | `fitDist`, `fitDistPred`, `gamlssMLpred` | `omnilss.fitDist`, `omnilss.fitDistPred` | implemented |

### 3.4 Smoothers and special terms

OmniLSS currently recognises `pb`, `ps`, `cs`, `random`, `re`, `lo` in
`omnilss.fitting._has_smooth_terms`, but the smoother *implementation* layer
under `omnilss/smoothers/` only ports `pb`, `ps`, `cs`, plus added thin-plate
splines (`tps`) and tensor products (`te`, `ti`). The remaining R smoothers
listed in the `gamlss` NAMESPACE (`pbp`, `pbm`, `pbo`, `pbz`, `pbc`, `pvc`,
`cy`, `fp`, `bfp`, `re`, `pp`, `lo`, `random`, `ri`, `pcat`, `bp`, `scs`)
are missing or only stubbed in the formula parser.

| R smoother | OmniLSS smoother | Status |
|---|---|---|
| `pb`, `gamlss.pb`, `pb.control` | `omnilss.smoothers.pb` | implemented |
| `ps`, `gamlss.ps` | `omnilss.smoothers.ps` | implemented |
| `cs`, `gamlss.cs`, `cs.control` | `omnilss.smoothers.cs` | implemented |
| `bp` | none | missing |
| `pbp`, `gamlss.pbp`, `pbp.control` | none | missing |
| `pbm`, `gamlss.pbm`, `pbm.control` | none | missing |
| `pbo`, `gamlss.pbo`, `pbo.control` | none | missing |
| `pbz`, `gamlss.pbz`, `pbz.control` | none | missing |
| `pbc`, `gamlss.pbc`, `pbc.control` | none | missing |
| `pvc`, `gamlss.pvc`, `pvc.control` | none | missing |
| `cy`, `gamlss.cy`, `cy.control` | none | missing |
| `fp`, `gamlss.fp`, `bfp` | none | missing (fractional polynomials) |
| `re`, `gamlss.re` | none | missing (mixed-effects via nlme) |
| `pp`, `gamlss.pp` | none | missing |
| `lo`, `gamlss.lo`, `vis.lo` | parser stub only | partial (formula recognises `lo(...)` but no LOESS smoother) |
| `random`, `gamlss.random` | parser stub only | partial |
| `ri`, `gamlss.ri` | none | missing (ridge term) |
| `pcat`, `gamlss.pcat`, `plotDF`, `plotLambda` | none | missing |
| `getSmo` | none | missing |
| `edf`, `edfAll` | `omnilss.smoothers.penalties._edf_numpy` only | partial (no `edfAll`) |
| `getZmatrix` | none | missing |
| Tensor smoothers (extension) | `omnilss.smoothers.tensor_smooth` | added beyond R |
| Thin-plate splines (extension) | `omnilss.smoothers.tps` | added beyond R |
| Penalty/REML/UBRE/GCV selection | `omnilss.smoothers.{reml, ubre, gcv}` | implemented |

### 3.5 Specialized models and survival

| R file | R symbol | Status |
|---|---|---|
| `lms.R` | `lms`, `z.scores` | missing |
| `quantSheets.R` | `quantSheets`, `quantSheets.control`, `findPower`, `z.scoresQS` | missing |
| `histSmo.R` | `histSmo`, `histSmoO`, `histSmoC`, `histSmoP` | missing |
| `HistDist-03-10-13.R` | `histDist` | implemented (`omnilss.HistDist_03_10_13.hist_dist_data`) |
| `FitTail.R` | `loglogSurv`, `loglogSurv1/2/3`, `logSurv`, `logSurv0` | missing |
| `p-categorical.R` | `pcat`, `gamlss.pcat` | missing |
| `re_gamlss_lme_test.R` | `re.gamlss` test scaffold | n/a (development scaffold) |
| `bp.R` | `bp` smoother | missing |
| `cy.R`, `pbm.R`, `pb_goingtozero.R`, `pbo_original.R`, `pbp.R`, `pbc.R`, `pvc.R`, `ridge.R`, `random.R`, `ri.R`, `lo_new.R`, `fracpoly.R`, `powerpoly.R`, `POLY.R` | smoother implementations | missing |
| `ECDF.R`, `loglogplot.R` | `ECDF`, `loglogplot` | missing |
| `numeric.deriv` | `omnilss.operations.numeric_deriv` | implemented |
| `gamlssNews` | none | missing |

## 4. Algorithm and runtime coverage

OmniLSS exposes more optimization routes than R `gamlss` because the JAX
backend enables additional algorithms (Adam, L-BFGS) on top of the canonical
ones.

| Algorithm | R `gamlss` | OmniLSS | Status |
|---|---|---|---|
| RS (Rigby–Stasinopoulos backfitting) | yes | `omnilss.fitting` (default), `rs_jax`, `fitting_jit` | implemented |
| Mixed RS/CG | yes | `omnilss.fitting` | implemented |
| CG (Cole–Green) full-Hessian | yes | `omnilss.fitting_cg`, `omnilss.cg_derivatives` | implemented |
| CG IRLS-cross | n/a | experimental backend, opt-in | OmniLSS extension |
| Joint Adam | n/a | `omnilss.algorithms` (joint optimizer) | OmniLSS extension |
| L-BFGS | n/a | `omnilss.core.lbfgs_optimizer` | OmniLSS extension |
| ML (gamlssML) | yes | `omnilss.gamlssML` | implemented |
| Cross-validation routes | yes | `omnilss.gamlssVGD_23_12_21` | implemented |

Method routing strictness, capability gates, and singular-Hessian protection
are documented under `docs/development/method-routing-capability-gates-2026-05-17.md`.

## 5. Documentation, examples, and tutorials

`tutorials/`, `examples/`, `paper/`, and `docs/` already cover the implemented
public API, but they predate this audit. After the gaps in §2 and §3 are
closed, each new family/function needs:
- one runnable example in `examples/`;
- one R-backed parity script under `benchmarks/` or `omnilss/tests/`;
- a row in `docs/testing/r-consistency-coverage.md`;
- an evidence tier update in `family-capability-matrix-*.json`;
- a paragraph in the migration log of `docs/development/`.

## 6. Implementation plan

This plan is now ordered by implementation truth rather than by API surface.
The guiding rule is: **a migrated R symbol is not counted as implemented until
its Python code has a dedicated numeric implementation and R-backed evidence**.
Callable aliases, compatibility constructors, placeholder p/q/r functions, and
`NaN` fallbacks are treated as migration debt.

### Phase 0 — Truth gate and scope freeze (immediate)

P0.1. Add a machine-readable implementation status layer for every R export:

- `implemented`: dedicated Python implementation exists and has R-backed tests;
- `experimental`: dedicated Python implementation exists but evidence is
  incomplete;
- `compat_alias`: public constructor exists but reuses another family's numeric
  kernel;
- `placeholder`: code returns `NaN`, normal approximations, or unsupported
  fallbacks;
- `missing`: no Python entry point.

This status must be generated from code, not maintained only in prose. The
generator should inspect `omnilss.distribution_registry`, known compatibility
modules, `dpqr_functions.py`, and `omnilss/tests/test_r_consistency_*.py`.

P0.2. Block false coverage:

- `compat_alias` and `placeholder` families must not be counted in
  "implemented" totals.
- `gamlss()` must either reject them by default or require an explicit
  `allow_experimental=True` / `allow_compat_alias=True` style flag.
- `family_capabilities.py` must expose the distinction so tests, benchmarks,
  REST, and gRPC can enforce it.

P0.3. Keep current cycle scope strict:

- Primary work is `omnilss/`.
- Secondary work is `omnilss-server/` after `omnilss` behavior is validated.
- `omnilss-pro/` is not part of this plan.

### Phase A — `omnilss` distribution migration from `R/gamlss.dist`

A1. Replace compatibility aliases with dedicated distribution kernels, in this
priority order:

1. `ST1`, `ST2`, `ST3`, `ST4` — replace `ST3C` clones with real skew-t
   parameterizations.
2. `DBI` — replace `BI` reuse with Double Binomial likelihood, score, Hessian,
   and d/p/q/r.
3. `DBURR12` — replace `SICHEL` reuse with Discrete Burr-12 implementation.
4. `BCCGuntr`, `BCPEuntr`, `BCTuntr` — implement the untruncated Box-Cox
   variants rather than cloning truncated kernels.
5. `TF2` — ensure the constructor and d/p/q/r match R `TF2` parameterization,
   not only a `TF` alias.

A2. Replace generic p/q/r placeholders:

- Any p/q/r helper returning `NaN`, a standard normal fallback, or a generic
  random normal draw must either be replaced by a mathematically correct
  implementation or removed from the family capability surface.
- For distributions where closed-form CDF/quantile is unavailable, implement a
  bounded numerical method with explicit support, monotonicity checks, tail
  handling, and R parity tests.

A3. Port `gen.Family` skeletons as real builders:

- `count_*`, `binom_*`, `contR_*`, `contRplus_*`, and `contR01_*` must build
  `FamilyDefinition` objects from the R-style argument tuples and supplied
  density/CDF/quantile/random components.
- Do not route skeletons to `PO` or `NO` except in tests that explicitly assert
  the current debt and are marked as expected failures.

A4. For every family promoted above `experimental`, add:

- d/p/q tests against native R;
- at least one fitting test when R `gamlss` supports the family in model
  fitting;
- extreme-domain tests for support boundaries, zeros/ones, tail probabilities,
  and invalid parameter values;
- capability matrix update with evidence artifact path.

### Phase B — `omnilss` core `gamlss` migration from `R/gamlss`

B1. Make the RS implementation faithful and auditable:

- Separate "statistical update" logic from "numerical guardrail" logic in
  `fitting.py` and `algorithms/rs_algorithm.py`.
- Record when Hessian floors, weight clipping, NaN replacement, eta clipping,
  step halving, or fallback solvers are used.
- Add tests that fail when clipping silently changes convergence without a
  warning or diagnostic slot.

B2. Promote CG only behind evidence:

- `fitting_cg.py` uses full observed Hessian and line search, which is useful
  but expensive. Keep it as a validated route only for families with finite
  derivative tests, monotonic deviance tests, and R/RS comparison artifacts.
- For families with unstable Hessians, expose CG as experimental and explain
  fallback behavior in capability metadata.

B3. Finish core R helper migration before broad service work:

- fitting/control: `Formulae2data`, `gen.likelihood`, `gamlssML` parity;
- diagnostics: `dtop`, `par.plot`, `prof.dev`, `prof.term`, `find.hyper`,
  `get.K`, `rvcov`, `ECDF`, `loglogplot`;
- model selection: explicit parity for `stepGAIC*`, `drop/add`, `chooseDist`;
- survival/specialized: `loglogSurv*`, `logSurv*`, `lms`, `z.scores`,
  `quantSheets`, `histSmo*`.

B4. For diagnostics and plotting ports, keep the first implementation
data-only, but enforce R-compatible column names, vector lengths, ordering, and
edge-case behavior through tests.

### Phase C — `omnilss` smoother migration

C1. Complete first-class smoother parsing and implementation:

- Existing: `pb`, `ps`, `cs` should stay the reference ABI.
- Next: `pbm`, `pbz`, `pbo`, `pbc`, `pbp`, `pvc`, `cy`, `fp`, `bfp`, `lo`,
  `vis.lo`, `re`, `random`, `ri`, `pcat`, `plotDF`, `plotLambda`.

C2. Remove parser-level placeholders:

- `formula_parser.py` and `smooth_fitting.py` must not return intercept-like
  placeholder bases for unsupported smoothers.
- Unsupported smoothers should fail early with a structured error until their
  basis, penalty, prediction rebuild, EDF, and R parity tests exist.

C3. Add R-backed smoother tests:

- formula parsing parity;
- basis dimension and penalty shape checks;
- fitted values/deviance checks for small deterministic datasets;
- prediction matrix rebuild checks for saved JSON artifacts.

### Phase D — Benchmark gate correctness and completeness

D1. Fix `benchmarks/comprehensive_r_consistency_test.py` so it is a source of
truth, not a hand-written whitelist:

- Load the test matrix from generated capability/coverage metadata.
- Exclude `compat_alias` and `placeholder` entries unless a test explicitly
  verifies that they are blocked.
- When `--require-r` is set, any missing R dependency, skipped R comparison,
  missing R function, null reference value, or all-NaN comparison must fail the
  run.
- Python-only mode (`--no-r`) must label results as smoke-only and must never
  mark R consistency as passed.
- d/p/q tests must fail when Python returns all `NaN`, all infinities, or a
  shape-broadcasted scalar that hides vector errors.

D2. Fix fitting and smoother consistency checks:

- Compare deviance, coefficients, fitted values, convergence status, iteration
  counts, and diagnostic warnings where R exposes them.
- Make tolerances family-specific when mathematical scale requires it, and store
  those tolerances in the generated results JSON.
- Add deterministic seed and data-generation metadata to every result row.

D3. Fix `benchmarks/comprehensive_performance_test.py` so performance never
outruns correctness:

- Performance runs must require a passing consistency result for the same
  family/formula/data case, unless explicitly invoked in smoke-only mode.
- Reports must separate cold time, warm median, R in-process time, Python heap,
  and JAX device memory availability; missing device memory must be reported as
  "not measured", not zero.
- The distribution list must be generated from validated/experimental
  capability metadata, not manually maintained constants.
- If R succeeds and Python fails, the script must return non-zero.
- If Python and R both succeed but deviance exceeds tolerance, the script must
  return non-zero even if timing succeeded.

D4. Replace stale/null fixtures:

- `benchmarks/r_reference_results.json` currently contains null R values and
  must not be used as evidence.
- Generate real R fixtures inside the offline devcontainer and attach the R
  version, `gamlss`, `gamlss.dist`, data seed, family, formula, tolerance, and
  source commit.

### Phase E — `omnilss-server` alignment after `omnilss` validation

E1. Expose only validated or explicitly experimental `omnilss` behavior:

- REST and gRPC fit endpoints must reject `compat_alias` and `placeholder`
  families by default.
- `CapabilityService.RouteCapability` must include implementation status,
  validation evidence, and service availability.

E2. Harden service execution:

- fix gRPC `Fit()` error response to return the correct `FitResponse` type;
- add request size limits, maximum rows/columns, maximum candidate families,
  maximum iterations, fit timeout, and cancellation;
- require authentication by default for non-local deployments;
- avoid exposing raw internal exception strings to clients except in debug mode.

E3. Persist only safe model artifacts:

- service storage should use JSON artifacts (`save_model_json` /
  `load_model_json`) and must not load pickle/cloudpickle artifacts from
  untrusted paths;
- model registry should record family implementation status and capability
  snapshot at save time.

### Phase F — Release gating and documentation

F1. A release candidate cannot pass unless:

- every exposed `FamilyDefinition` is `implemented` or `experimental` with an
  explicit reason;
- no `compat_alias` or `placeholder` family is exposed as normal coverage;
- `python benchmarks/run_local_validation.py --quick` passes with native R;
- comprehensive consistency and performance reports are generated as artifacts.

F2. Documentation updates come after code truth:

- update README, package docs, generated capability pages, and migration logs
  from the generated coverage/capability matrix;
- do not manually advertise family counts.

## 7. Risks and acceptance criteria

- **Risk: false coverage through aliases.** A family name can resolve and fit
  while running another family's math. Mitigation: `compat_alias` status,
  default rejection in fitting/service routes, and R-backed promotion gates.
- **Risk: placeholder numerical functions.** Generic p/q/r fallbacks can return
  normal approximations or `NaN`. Mitigation: placeholder detection in tests and
  benchmark scripts; no placeholder can pass coverage.
- **Risk: silent numerical drift.** Hessian floors, clipping, and fallback
  solvers can keep a fit alive while changing the statistical update.
  Mitigation: structured diagnostics, warning tests, and parity artifacts.
- **Risk: benchmark false positives.** Python-only smoke runs, skipped R cases,
  null fixtures, or manual distribution lists can look like evidence.
  Mitigation: `--require-r` fail-hard semantics and generated matrices.
- **Risk: service overexposure.** `omnilss-server` can expose unvalidated
  families or long-running fits. Mitigation: capability gate, request limits,
  authentication, timeout, and safe JSON-only artifact loading.

Acceptance criteria for "covers R gamlss / gamlss.dist":

- every R export under `R/gamlss` and `R/gamlss.dist` appears in the generated
  coverage matrix with one of `implemented`, `experimental`, `compat_alias`,
  `placeholder`, `missing`, or `unsupported_with_rationale`;
- every `implemented` distribution has dedicated d/p/q/r code and native R
  parity evidence;
- every `implemented` fitting/diagnostic/smoother helper has R-backed tests or
  an explicitly justified data-only equivalence test;
- the benchmark validation gate passes in the offline devcontainer with
  `gamlss`, `gamlss.dist`, and `jsonlite` installed;
- `comprehensive_r_consistency_test.py` fails when R is required but unavailable
  or skipped;
- `comprehensive_performance_test.py` fails when correctness fails for a tested
  family/formula/data case;
- `omnilss-server` exposes only statuses allowed by capability metadata;
- `omnilss-pro` remains out of scope for this cycle.

## 8. Operating in the offline devcontainer

The devcontainer at `.devcontainer/Dockerfile` already builds an offline image
with R 4.4.1 plus `gamlss`, `gamlss.dist`, `jsonlite`, `renv`, and the Python
extras needed for the benchmark gate. As of this note the image build also:

- pre-downloads CRAN sources for `gamlss` and `gamlss.dist` into
  `/opt/omnilss-r-cache/` so a future offline reuse can rebuild a missing
  R package without network access;
- installs `r-base-dev`, `pandoc`, and the JSON / curl / openssl headers
  required by `gamlss`, `nlme`, and `survival` so the R toolchain remains
  functional after `apt` is offline;
- copies the local `R/gamlss/` and `R/gamlss.dist/` source trees into the
  image, so a developer can run `R CMD INSTALL` against the bundled sources
  if CRAN is unreachable.

The Dockerfile fails fast at build time if any required Python module
(`build`, `twine`, `jax`, `flax`, `optax`, `sklearn`, `scoringrules`, `grpc`,
`grpc_tools`, `google.protobuf`, `mkdocs`) or any required R package
(`renv`, `gamlss`, `gamlss.dist`, `jsonlite`, `languageserver`) is missing.

The `.codex/setup.sh` post-create script reuses the cached wheelhouse and R
library when network access is unavailable (`OMNILSS_OFFLINE=1`). It also
prints the validation-gate command so a fresh contributor can confirm that
R-backed parity actually runs.

See `docs/development/offline-dev-bootstrap-2026-05-17.md` and
`docs/development/r-environment-setup.md` for the day-to-day usage of these
files.
