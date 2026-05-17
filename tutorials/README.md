# OmniLSS Tutorial Series

Welcome to the OmniLSS tutorial series! These tutorials help you migrate from R gamlss
to Python OmniLSS and apply distributional regression in real-world scenarios.


## Current fitting API notes

Tutorial snippets now use the current OmniLSS model API:

- `gamlss(..., method="RS")` remains the routine CPU default.
- `gamlss(..., method="CG")` uses the auditable `CG_FULL_HESSIAN` Cole-Green backend by default and records `cg_backend` / `cg_cross_derivatives` in `model.additional_slots`.
- `gamlss(..., method="CG", cg_backend="irls_cross")` enables the experimental eta-scale IRLS cross-derivative backend.
- Model diagnostics such as AIC/BIC live in `model.additional_slots["aic"]` and `model.additional_slots["sbc"]`; coefficients are in `model.coefficients[parameter]`; fitted parameter values are in `model.fitted_values[parameter]`.

## Tutorial Structure

### Phase 1: Feature Comparison and Usage Guide

Organized by distribution family, comparing R and Python implementations.

#### 1. Continuous Distributions — Basic

**Article 1.1**: [Normal Family — NO, NO2, LOGNO, LOGNO2](phase1/01_normal_distributions.md)
- R vs Python implementation
- Basic usage comparison
- Performance benchmarks
- Migration guide

**Article 1.2**: [Gamma Family — GA, GG, IGAMMA, IG](phase1/02_gamma_distributions.md)
- Distribution characteristics
- Parameterization comparison
- Use cases and code examples

**Article 1.3**: [Exponential and Weibull — EXP, WEI, LO](phase1/03_exponential_weibull.md)
- Survival analysis applications
- Parameter estimation
- Model diagnostics

#### 2. Discrete Distributions

**Article 2.1**: [Count Distributions — PO, NBI, NBII, GEOM](phase1/04_count_distributions.md)
- Poisson and extensions
- Overdispersion handling
- Model selection

**Article 2.2**: [Binomial Family — BI, BB, BNB](phase1/05_binomial_distributions.md)
- Proportion data modeling
- Compound distributions
- Practical applications

**Article 2.3**: [Zero-Inflated Distributions — ZIP, ZINBI, ZAGA, ZAIG](phase1/06_zero_inflated.md)
- Zero-inflation problem
- Model comparison
- Diagnostic methods

#### 3. Beta and Bounded Distributions

**Article 3.1**: [Beta Family — BE, BEINF, BEZI, BEOI](phase1/07_beta_distributions.md)
- Proportion data modeling
- Boundary inflation
- Real-world case studies

**Article 3.2**: [Simplex and Other Bounded Distributions](phase1/08_bounded_distributions.md)
- SIMPLEX distribution
- Application scenarios
- Model comparison

#### 4. Skewed and Heavy-Tailed Distributions

**Article 4.1**: [Skewed Normal — SN1, SN2, SHASH, SHASHo](phase1/09_skewed_normal.md)
- Skewness modeling
- Parameter interpretation
- Practical applications

**Article 4.2**: [t-Distribution Family — TF, GT, ST series](phase1/10_t_distributions.md)
- Heavy-tailed data
- Robust modeling
- Outlier handling

**Article 4.3**: [Box-Cox Transformation — BCCG, BCT, BCPE](phase1/11_boxcox_distributions.md)
- Transformation techniques
- Flexible modeling
- Growth curves

#### 5. Special Distributions

**Article 5.1**: [Pareto and Extreme Value — PARETO, PARETO2, GU, RG](phase1/12_extreme_value.md)
- Extreme value theory
- Risk modeling
- Tail analysis

**Article 5.2**: [Complex Discrete — PIG, SICHEL, DPO, DEL](phase1/13_complex_discrete.md)
- Mixture distributions
- Special count data
- Application scenarios

**Article 5.3**: [Other Special Distributions — exGAUS, PE, NET](phase1/14_special_distributions.md)
- Reaction time data
- Power exponential distribution
- Special applications

#### 6. Smoothing and Additive Models

**Article 6.1**: [P-splines (pb) — Penalized B-splines](phase1/15_psplines.md)
- Smoothing techniques
- Parameter selection (GCV/REML)
- Practical applications

**Article 6.2**: [Cubic Splines (ps, cs)](phase1/16_cubic_splines.md)
- Spline smoothing
- Knot selection
- Model comparison

**Article 6.3**: [Mixed and Random Effects](phase1/17_random_effects.md)
- Random effects modeling
- Hierarchical models
- Case studies

#### 7. Model Selection and Diagnostics

**Article 7.1**: [Model Selection — AIC, BIC, GAIC](phase1/18_model_selection.md)
- Information criteria
- Stepwise selection
- Cross-validation

**Article 7.2**: [Model Diagnostics — Residual Analysis](phase1/19_diagnostics.md)
- Residual types
- Diagnostic plots
- Model checking

**Article 7.3**: [Prediction and Inference](phase1/20_prediction_inference.md)
- Point prediction
- Interval prediction
- Quantile regression

#### 8. Advanced Topics

**Article 8.1**: [Multi-Parameter Modeling](phase1/21_multiparameter_modeling.md)
- Location, scale, shape
- Parameter relationships
- Complex models

**Article 8.2**: [Distribution Selection and Comparison](phase1/22_distribution_selection.md)
- Distribution fitting
- Model comparison
- Automatic selection

**Article 8.3**: [Performance Optimization and GPU Acceleration](phase1/23_performance_gpu.md)
- JAX optimization tips
- GPU acceleration
- Large-scale data

---

### Phase 2: Real-World Application Scenarios

End-to-end tutorials for practical use cases.

#### Scenario 1: Insurance and Risk Management

- [Claim Amount Modeling](phase2/insurance/01_claim_amount.md) — Gamma, Pareto
- [Claim Frequency Modeling](phase2/insurance/02_claim_frequency.md) — Poisson, NBI, ZIP
- [Survival Analysis](phase2/insurance/03_survival_analysis.md) — Weibull, Exponential
- [Extreme Risk Modeling](phase2/insurance/04_extreme_risk.md) — Pareto, GEV

#### Scenario 2: Financial Analysis

- [Returns Modeling](phase2/finance/01_returns_modeling.md) — Skewed-t, SHASH
- [Credit Risk Scoring](phase2/finance/02_credit_scoring.md) — Beta, Logistic
- [Volatility Modeling](phase2/finance/03_volatility_modeling.md) — Conditional distributions
- [Derivatives Pricing](phase2/finance/04_derivatives_pricing.md) — Distribution assumptions

#### Scenario 3: E-commerce and User Behavior

- [User Activity Modeling](phase2/ecommerce/01_user_activity.md) — Poisson, ZIP
- [Conversion Rate Analysis](phase2/ecommerce/02_conversion_rate.md) — Beta regression
- [Customer Lifetime Value](phase2/ecommerce/03_customer_lifetime_value.md) — Gamma, Pareto-NBD
- [Basket Analysis](phase2/ecommerce/04_basket_analysis.md) — Multinomial distributions

#### Scenario 4: Healthcare

- [Disease Risk Prediction](phase2/healthcare/01_disease_risk.md) — Logistic, Beta
- [Length of Stay Modeling](phase2/healthcare/02_length_of_stay.md) — Gamma, Weibull
- [Dose-Response Analysis](phase2/healthcare/03_dose_response.md) — Nonlinear smoothing
- [Survival and Prognosis](phase2/healthcare/04_survival_prognosis.md) — Extended Cox models

#### Scenario 5: Environment and Energy

- [Rainfall Modeling](phase2/environment/01_rainfall_modeling.md) — Zero-inflated Gamma
- [Energy Consumption Forecasting](phase2/environment/02_energy_consumption.md) — Time series
- [Pollution Concentration](phase2/environment/03_pollution_modeling.md) — Skewed distributions

#### Scenario 6: Social Sciences

- [Income Inequality Analysis](phase2/social/01_income_inequality.md) — Pareto, Lognormal
- [Educational Achievement](phase2/social/02_educational_achievement.md) — Beta regression
- [Demographic Analysis](phase2/social/03_demographic_analysis.md) — Count data

---

## How to Use These Tutorials

### Learning Paths

**Path 1: R User Migration**
1. Start with Phase 1 in order
2. Focus on R vs Python comparisons
3. Complete exercises in each article
4. Choose relevant Phase 2 scenarios

**Path 2: New Python User**
1. Start with basic distributions (Articles 1.1–1.3)
2. Select relevant distribution types for your domain
3. Study smoothing and model selection (Articles 6.x, 7.x)
4. Move to Phase 2 application scenarios

**Path 3: Domain-Specific Application**
1. Quickly review relevant Phase 1 distributions
2. Go directly to the corresponding Phase 2 scenario
3. Return to Phase 1 for detailed reference as needed

### Tutorial Features

- ✅ **Side-by-side comparison**: R and Python code shown together
- ✅ **Runnable code**: All examples can be executed directly
- ✅ **Real data**: Uses realistic datasets
- ✅ **Complete workflow**: From data exploration to model deployment
- ✅ **Performance analysis**: Includes benchmarks and optimization tips
- ✅ **Best practices**: Practical recommendations and tips

### Code Repository

All tutorial code and data:
- Phase 1 code: `tutorials/phase1/code/`
- Phase 2 code: `tutorials/phase2/code/`
- Datasets: `tutorials/datasets/`

---

## Tutorial Goals

**Phase 1**: Help R users migrate to Python; demonstrate OmniLSS features;
provide performance comparisons; build user confidence.

**Phase 2**: Demonstrate real-world applications; provide end-to-end solutions;
share best practices; foster community exchange.

---

## Contributing

Contributions welcome:
- Bug reports and improvement suggestions
- Real datasets and case studies
- Usage experience sharing
- Translations

---

**Start here**: [Article 1.1 — Normal Distributions](phase1/01_normal_distributions.md)

**Maintainer**: OmniLSS Team | **Last updated**: 2026-05-11
