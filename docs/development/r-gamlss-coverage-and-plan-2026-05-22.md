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

## 2. Distribution coverage (`gamlss.dist`)

R `gamlss.dist` exports d/p/q/r quartets for ~100 named families, plus several
helpers (`flexDist`, `hazardFun`, `momentSK`, `MULTIN`, generic count/binomial
skeletons, etc.).

### 2.1 Implemented in OmniLSS (registered + Python factory present)

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

### 2.3 Missing or untested distribution exports

R-side exports that are not yet exposed by OmniLSS or whose Python equivalent
needs to be built:

- ST1, ST2, ST3, ST4 — only ST3C, ST5, SST, SEP4 ports of the Skew-t family are
  registered. ST1/ST2/ST3/ST4 are referenced in `fitting_jit.py` warm-start
  defaults but no `FamilyDefinition` factory exists.
- TF2 — alternative Student-t parameterization with degrees-of-freedom
  parameter.
- DBI — Double Binomial.
- DBURR12 — Discrete Burr-12.
- BCCGuntr, BCPEuntr, BCTuntr — untruncated Box-Cox variants.
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

The plan is phased so that every step lands behind the existing
benchmark-validation gate (consistency before performance). Inside each phase,
work is ordered by *risk reduction* first, *coverage breadth* second.

### Phase A — Plumbing for systematic R parity (week 1–2)

A1. Lock the R-comparison runtime in CI by adding a job to
   `.github/workflows/environment-validation.yml` that runs
   `python benchmarks/run_local_validation.py --quick` in the prebuilt
   devcontainer (so consistency is exercised on every PR that affects
   distributions or fitting).

A2. Generate `docs/development/r-coverage-matrix-2026-05-22.json` from a single
   script that crawls
   - `R/gamlss.dist/NAMESPACE` and `R/gamlss/NAMESPACE` for exports,
   - `omnilss.distribution_registry.list_families()` for Python factories,
   - `omnilss/tests/test_r_consistency_*.py` for R-backed parity tests,
   and emits a row per R symbol with its Python module, validation status, and
   capability tier. This becomes the source of truth that supersedes the
   prose tables in §2 and §3.

A3. Reconcile the duplicate `JSU` factory between `distributions.py` and
   `distributions_b11.py` so the registry has one canonical entry.

A4. Add explicit `R reference:` doc strings to every Python module that ports
   a named R symbol (most exist, but `extra.py`, `lpred.py`, `pdfplot.py`, and
   the smoother modules need consistent backlinks).

### Phase B — Distribution coverage gap closure (week 2–8)

B1. Add R-backed d/p/q tests for every `experimental` family currently
   registered (≈75 families). Reuse the structure of
   `tests/test_dpqr_r_consistency.py`: scalarize R output with the existing
   `RTestBus`, compare under `1e-5` tolerances, and update the capability
   matrix.

B2. Add the missing R distribution factories listed in §2.3, in priority order:
   ST1, ST2, ST3, ST4, TF2, DBI, DBURR12, BCCGuntr, BCPEuntr, BCTuntr,
   followed by the categorical/multinomial helpers (`MULTIN`) and the shape
   utilities (`flexDist`, `momentSK`, `centileSK`, `hazardFun`, `gen.hazard`).

B3. Port the `gen.Family` skeleton families (`count_*`, `binom_*`, `contR_*`,
   `contRplus_*`, `contR01_*`) as a single generic builder rather than 17
   independent factories, exposing them through one helper that constructs a
   `FamilyDefinition` from R-style argument tuples.

B4. Add the `distributions3` interface (`pdf`, `log_pdf`, `cdf`, `quantile`,
   `random`, `is_continuous`, `is_discrete`, `support`, `mean`, `variance`,
   `skewness`, `kurtosis`) on top of `FamilyDefinition`. This mirrors the R
   `GAMLSS` S4 wrapper without a new class hierarchy.

### Phase C — Fitting / diagnostics gap closure (week 4–10)

C1. Implement the missing diagnostic helpers as data-only ports first:
   `dtop`, `par.plot`, `prof.dev`, `prof.term`, `find.hyper`, `gen.likelihood`,
   `loglogSurv`, `logSurv`, `lms`, `z.scores`, `quantSheets`, `findPower`,
   `histSmo`/`histSmoO/C/P`, `getSmo`, `get.K`, `rvcov`, `getZmatrix`,
   `vis.lo`, `Formulae2data`, `edfAll`, `ECDF`, `loglogplot`. Each port
   returns a structured dataclass with the same column names as R, then the
   plot helpers can layer on matplotlib without coupling.

C2. Promote the existing diagnostics wrappers (`predict.gamlssML`,
   `summary.gamlssML`, `vcov.gamlssML`) from partial to validated by adding
   R parity tests.

C3. Round-trip every diagnostic through the JSON model artifact so service
   clients (REST + gRPC) can request them by name, matching the existing
   capability metadata snapshot policy.

### Phase D — Smoother coverage (week 6–14)

D1. Port the next tier of R smoothers behind the same `SmoothTerm` ABI used by
   `pb`/`ps`/`cs`:
   - `pbm` (monotonic P-spline) and `pbz` (zero-going P-spline) — straightforward
     constraint extensions of `pb`;
   - `pbo` (P-spline with offset) and `pbc` (cyclic P-spline);
   - `cy` (cyclic spline), `pvc` (varying-coefficient term);
   - `fp` and `bfp` (fractional polynomials);
   - `lo` (LOESS) and `vis.lo` companion;
   - `re` and `random` (mixed-effects/random terms via a thin
     `ridge`-style backend before any nlme-equivalent landing);
   - `ri` (ridge term), `pcat` (penalized categorical), with `plotDF`,
     `plotLambda` companions.

D2. Update `omnilss.formula_parser` so each new smoother is a first-class term
   with parameter parsing instead of a regex hint, and add `getSmo`, `edfAll`
   bindings.

D3. Add R-backed smoother parity tests modelled on
   `tests/test_r_consistency_smooth.py`.

### Phase E — Service/runtime alignment (week 10–14)

E1. Expose the new diagnostics, smoother terms, and capability rows through
    REST (`omnilss-server/main.py`) and gRPC (`omnilss/api/grpc/proto/`)
    boundaries, with capability metadata advertised through the existing
    `CapabilityService.RouteCapability`.

E2. Persist new family/diagnostic artifacts through the SQLite-backed model
    store, including the capability snapshot schema `version` bump.

E3. Update the published JOSS-ready benchmark suite to cover the new
    R-validated families and smoothers.

### Phase F — Release gating and docs (week 12–16)

F1. Block release until every family with an OmniLSS factory has at least one
    R-backed d/p/q test or is explicitly marked `experimental` in the
    capability matrix.

F2. Update `README.md`, `omnilss/README.md`, the `chineseREADME` companions,
    `mkdocs.yml`, and the generated capability page to point at the new
    coverage matrix.

F3. Promote families and method routes from `experimental` to `validated`
    only after the R-equivalence run has been recorded as an artifact.

## 7. Risks and acceptance criteria

- **Risk: silent divergence.** Many `experimental` families compile and fit but
  have not been compared to R. Mitigation: every family must have a d/p/q
  parity test before its capability tier is upgraded; CI must run the
  validation gate.
- **Risk: feature creep during freeze.** The 30-day architecture freeze
  (`docs/architecture/30-day-feature-freeze.md`) is still in effect.
  Distribution and smoother additions in Phases B and D must wait until the
  freeze ends. Phase A and the test-only parts of Phase C are allowed inside
  the freeze.
- **Risk: numerical method drift.** The L-BFGS, Adam, and IRLS-cross routes are
  OmniLSS-specific; they need their own parity targets (e.g. RS reference
  fits) instead of R `gamlss` since R does not expose them.
- **Acceptance criteria for "covers gamlss":**
  - capability matrix has `r_consistency = validated` for every family that
    OmniLSS exposes;
  - every R `gamlss` core function listed in §3 is either implemented with an
    R parity test or explicitly marked unsupported with rationale;
  - the benchmark-validation gate passes in the offline devcontainer with
    `gamlss`, `gamlss.dist`, and `jsonlite` installed;
  - `omnilss/tests` reaches >90% line coverage on the touched modules and the
    `tests/consistency/` suite is green.

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
