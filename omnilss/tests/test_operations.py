from dataclasses import replace
import importlib
import math
import unittest

import jax.numpy as jnp
import numpy as np

choose_dist_parallel_module = importlib.import_module("omnilss.chooseDistParallel")
drop_add_step_gaic_module = importlib.import_module("omnilss.DropAddStepGAIC_Parallel")
acf_resid_module = importlib.import_module("omnilss.acfResid")
centile_pred_module = importlib.import_module("omnilss.centile_pred_30_06_22")
centiles_fan_module = importlib.import_module("omnilss.centilesFan")
centiles_com_module = importlib.import_module("omnilss.centilescom")
centiles_plot_module = importlib.import_module("omnilss.centilesPLOT")
confint_module = importlib.import_module("omnilss.confint_gamlss_29_06_22")
deviance_incr_module = importlib.import_module("omnilss.DevianceIncr")
extra_module = importlib.import_module("omnilss.extra")
fit_dist_module = importlib.import_module("omnilss.fitDist")
fit_dist_pred_module = importlib.import_module("omnilss.fitDistPred")
fitted_plot_module = importlib.import_module("omnilss.fitted_plot")
gamlss_5_module = importlib.import_module("omnilss.gamlss_5")
gamlss_ml_module = importlib.import_module("omnilss.gamlssML")
gamlss_vgd_module = importlib.import_module("omnilss.gamlssVGD_23_12_21")
get_pef_module = importlib.import_module("omnilss.getPEF")
get_quantile_module = importlib.import_module("omnilss.getQuantile")
hist_dist_module = importlib.import_module("omnilss.HistDist_03_10_13")
hatvalues_module = importlib.import_module("omnilss.hatvalues")
loglik_module = importlib.import_module("omnilss.logLik")
lr_test_module = importlib.import_module("omnilss.LR_test_12_06_2013")
model_comparison_module = importlib.import_module("omnilss.MODEL_comparison")
predict_all_module = importlib.import_module("omnilss.predictAll_22_08_22")
predict_module = importlib.import_module("omnilss.predict_gamlss_23_12_21")
plot_module = importlib.import_module("omnilss.plot")
plot2way_module = importlib.import_module("omnilss.plot2way")
pdfplot_module = importlib.import_module("omnilss.pdfplot")
print_module = importlib.import_module("omnilss.print")
prodist_module = importlib.import_module("omnilss.prodist")
qstats_module = importlib.import_module("omnilss.qstats")
rsq_module = importlib.import_module("omnilss.Rsq")
summary_module = importlib.import_module("omnilss.SUMMARY")
lpred_module = importlib.import_module("omnilss.lpred")
step_gaic_a_module = importlib.import_module("omnilss.stepGAIC_03_10_13")
step_gaic_all_a_module = importlib.import_module("omnilss.stepGAICAll_A_parallel")
step_gaic_all_b_module = importlib.import_module("omnilss.stepGAICAll_B_Parallel")
step_tgd_module = importlib.import_module("omnilss.stepTGD")
term_plot_module = importlib.import_module("omnilss.term_plot_new")
update_module = importlib.import_module("omnilss.update")
vcov_module = importlib.import_module("omnilss.vcov_gamlss")
vuong_module = importlib.import_module("omnilss.VuongClarkTest")

from omnilss import (
    BCCG,
    BCPE,
    BCT,
    BE,
    BI,
    CV,
    BetaFamily,
    BinomialFamily,
    BoxCoxColeGreenFamily,
    BoxCoxPowerExponentialFamily,
    BoxCoxTFamily,
    EXP,
    ExponentialFamily,
    FamilyDefinition,
    GA,
    GAMLSSControl,
    GAMLSSModel,
    GEOM,
    GeometricFamily,
    GLIMControl,
    IG,
    InverseGaussianFamily,
    JSU,
    LO,
    LOGNO,
    LogNormalFamily,
    LogisticFamily,
    NBI,
    NegativeBinomialFamily,
    NO,
    PO,
    TF,
    VGD,
    WEI,
    ZIP,
    WeibullFamily,
    ZeroInflatedPoissonFamily,
    coef,
    coef_all,
    centile_pred_data,
    centiles_fan,
    centiles_comparison_data,
    centiles_comparison_coverage_data,
    centiles_coverage_data,
    centiles_data,
    centiles_fan_data,
    centiles_split_data,
    centiles_split_coverage_data,
    cv,
    choose_dist_data,
    choose_dist_pred,
    choose_dist_pred_data,
    chooseDist,
    chooseDistPred,
    gamlss_cv_data,
    compare_models,
    confint,
    deviance,
    devianceIncr,
    deviance_increment,
    drop1_tgd,
    drop1_tgd_all,
    drop1_tgdp,
    drop1TGD,
    drop1TGDP,
    dropterm,
    droptermAllP,
    dropterm_all,
    extract_aic,
    extractAIC,
    extractTGD,
    extract_tgd,
    extract_tgd_data,
    fitted,
    fitDist,
    fit_dist,
    fit_dist_data,
    fitDistPred,
    fit_dist_pred,
    fit_dist_pred_data,
    fitted_plot_data,
    formula,
    fv,
    add1_tgd,
    add1_tgd_all,
    add1_tgdp,
    add1TGD,
    add1TGDP,
    addterm,
    addtermAllP,
    addterm_all,
    gaic,
    gaic_scaled,
    gaic_table,
    get_pef_data,
    getOrder,
    get_order,
    get_quantile_data,
    getTGD,
    get_tgd,
    get_tgd_data,
    hatvalues,
    hist_dist_data,
    is_gamlss_cv,
    is_gamlss_vgd,
    gaic_weights,
    get_rqres_samples,
    gamlssCV,
    gamlssMLpred,
    gamlss_cv,
    gamlss_ml_pred,
    gamlss_ml_pred_data,
    gamlssVGD,
    gamlss_vgd,
    gamlss_vgd_data,
    gamlss,
    gamlss_control,
    gamlss_control_exact,
    gamlssML,
    gamlss_ml,
    glim_control,
    glim_control_exact,
    hat_wx,
    ic,
    is_gamlss,
    likelihood_ratio_test,
    lp,
    lpred,
    lpred_exact,
    logLik,
    log_likelihood,
    model_frame,
    model_matrix,
    numeric_deriv,
    plot_diagnostics,
    plot2way_data,
    predict,
    predict_all,
    pdf_plot_data,
    predict_gamlss_exact,
    prodist_data,
    print_gamlss_exact,
    print_model,
    q_stats,
    qq_stats,
    refit,
    resolve_family,
    residuals,
    acf_residuals,
    rsq,
    summary,
    stepGAIC,
    stepGAICAll_A,
    stepGAICAll_B,
    stepTGD,
    step_gaic,
    step_gaic_all,
    step_tgd,
    step_tgd_all,
    term_plot_data,
    vgd,
    terms,
    update,
    update_model,
    vcov,
    vuong_clarke_test,
    worm_plot_data,
)


class OperationsTest(unittest.TestCase):
    def test_r_aligned_module_files_export_expected_interfaces(self) -> None:
        self.assertTrue(callable(extra_module.refit))
        self.assertTrue(callable(extra_module.fitted))
        self.assertTrue(callable(extra_module.coef))
        self.assertTrue(callable(extra_module.coefAll))
        self.assertTrue(callable(extra_module.GAIC))
        self.assertTrue(callable(extra_module.Rsq))
        self.assertTrue(callable(summary_module.summary))
        self.assertTrue(callable(acf_resid_module.acfResid))
        self.assertTrue(callable(plot_module.plot_gamlss))
        self.assertTrue(callable(pdfplot_module.pdf_plot))
        self.assertTrue(callable(hist_dist_module.histDist))
        self.assertTrue(callable(qstats_module.Q_stats))
        self.assertTrue(callable(centiles_plot_module.centiles))
        self.assertTrue(callable(centiles_plot_module.centiles_split))
        self.assertTrue(callable(centiles_fan_module.centiles_fan))
        self.assertTrue(callable(deviance_incr_module.devianceIncr))
        self.assertTrue(callable(centiles_com_module.centiles_com))
        self.assertTrue(callable(centile_pred_module.centiles_pred))
        self.assertTrue(callable(gamlss_5_module.gamlss))
        self.assertTrue(callable(gamlss_5_module.gamlss_control))
        self.assertTrue(callable(gamlss_5_module.glim_control))
        self.assertTrue(callable(gamlss_ml_module.gamlssML))
        self.assertTrue(callable(loglik_module.logLik))
        self.assertTrue(callable(get_quantile_module.getQuantile))
        self.assertTrue(callable(hatvalues_module.hatvalues))
        self.assertTrue(callable(lpred_module.lpred))
        self.assertTrue(callable(predict_module.predict))
        self.assertTrue(callable(print_module.print))
        self.assertTrue(callable(term_plot_module.term_plot))
        self.assertTrue(callable(update_module.update))
        self.assertTrue(callable(plot2way_module.plot2way))
        self.assertTrue(callable(get_pef_module.getPEF))
        self.assertTrue(callable(lr_test_module.LR_test))
        self.assertTrue(callable(vuong_module.VC_test))
        self.assertTrue(callable(prodist_module.prodist))
        self.assertTrue(callable(rsq_module.Rsq))
        self.assertTrue(callable(model_comparison_module.compare_models))
        self.assertTrue(callable(model_comparison_module.gaic_weights))
        self.assertTrue(callable(fitted_plot_module.fitted_plot))
        self.assertTrue(callable(step_gaic_a_module.extractAIC))
        self.assertTrue(callable(step_gaic_a_module.stepGAIC))
        self.assertTrue(callable(drop_add_step_gaic_module.dropterm))
        self.assertTrue(callable(drop_add_step_gaic_module.addterm))
        self.assertTrue(callable(drop_add_step_gaic_module.stepGAIC))
        self.assertTrue(callable(step_gaic_all_a_module.stepGAICAll_A))
        self.assertTrue(callable(step_gaic_all_b_module.droptermAllP))
        self.assertTrue(callable(step_gaic_all_b_module.addtermAllP))
        self.assertTrue(callable(step_gaic_all_b_module.stepGAICAll_B))
        self.assertTrue(callable(predict_all_module.predictAll))
        self.assertTrue(callable(confint_module.confint))
        self.assertTrue(callable(vcov_module.vcov))
        self.assertTrue(callable(fit_dist_module.fit_dist))
        self.assertTrue(callable(fit_dist_module.fit_dist_data))
        self.assertTrue(callable(fit_dist_module.fitDist))
        self.assertTrue(callable(fit_dist_pred_module.fit_dist_pred))
        self.assertTrue(callable(fit_dist_pred_module.gamlss_ml_pred))
        self.assertTrue(callable(fit_dist_pred_module.fitDistPred))
        self.assertTrue(callable(fit_dist_pred_module.gamlssMLpred))
        self.assertTrue(callable(choose_dist_parallel_module.choose_dist_data))
        self.assertTrue(callable(choose_dist_parallel_module.choose_dist_pred))
        self.assertTrue(callable(choose_dist_parallel_module.chooseDist))
        self.assertTrue(callable(choose_dist_parallel_module.chooseDistPred))
        self.assertTrue(callable(choose_dist_parallel_module.getOrder))
        self.assertTrue(callable(step_gaic_a_module.extractAIC))
        self.assertTrue(callable(step_gaic_a_module.stepGAIC))
        self.assertTrue(callable(drop_add_step_gaic_module.dropterm))
        self.assertTrue(callable(drop_add_step_gaic_module.addterm))
        self.assertTrue(callable(drop_add_step_gaic_module.stepGAIC))
        self.assertTrue(callable(step_gaic_all_a_module.stepGAICAll_A))
        self.assertTrue(callable(step_gaic_all_b_module.droptermAllP))
        self.assertTrue(callable(step_gaic_all_b_module.addtermAllP))
        self.assertTrue(callable(step_gaic_all_b_module.stepGAICAll_B))
        self.assertTrue(callable(gamlss_vgd_module.gamlss_vgd))
        self.assertTrue(callable(gamlss_vgd_module.gamlss_cv))
        self.assertTrue(callable(gamlss_vgd_module.gamlssVGD))
        self.assertTrue(callable(gamlss_vgd_module.VGD))
        self.assertTrue(callable(gamlss_vgd_module.getTGD))
        self.assertTrue(callable(gamlss_vgd_module.gamlssCV))
        self.assertTrue(callable(gamlss_vgd_module.CV))
        self.assertTrue(callable(step_tgd_module.step_tgd))
        self.assertTrue(callable(step_tgd_module.step_tgd_all))
        self.assertTrue(callable(step_tgd_module.extractTGD))
        self.assertTrue(callable(step_tgd_module.drop1TGD))
        self.assertTrue(callable(step_tgd_module.add1TGD))
        self.assertTrue(callable(step_tgd_module.stepTGD))
        self.assertTrue(callable(step_tgd_module.drop1TGDP))
        self.assertTrue(callable(step_tgd_module.add1TGDP))

    def test_package_root_exports_r_style_exact_name_aliases(self) -> None:
        self.assertTrue(callable(fitDist))
        self.assertTrue(callable(fitDistPred))
        self.assertTrue(callable(gamlssMLpred))
        self.assertTrue(callable(chooseDist))
        self.assertTrue(callable(chooseDistPred))
        self.assertTrue(callable(getOrder))
        self.assertTrue(callable(gamlssVGD))
        self.assertTrue(callable(VGD))
        self.assertTrue(callable(getTGD))
        self.assertTrue(callable(gamlssCV))
        self.assertTrue(callable(CV))
        self.assertTrue(callable(extractAIC))
        self.assertTrue(callable(stepGAIC))
        self.assertTrue(callable(stepGAICAll_A))
        self.assertTrue(callable(stepGAICAll_B))
        self.assertTrue(callable(droptermAllP))
        self.assertTrue(callable(addtermAllP))
        self.assertTrue(callable(gamlssML))
        self.assertTrue(callable(gamlss_control_exact))
        self.assertTrue(callable(glim_control_exact))
        self.assertTrue(callable(devianceIncr))
        self.assertTrue(callable(centiles_fan))
        self.assertTrue(callable(logLik))
        self.assertTrue(callable(hatvalues))
        self.assertTrue(callable(lpred_exact))
        self.assertTrue(callable(predict_gamlss_exact))
        self.assertTrue(callable(print_gamlss_exact))
        self.assertTrue(callable(update))
        self.assertTrue(callable(extractTGD))
        self.assertTrue(callable(drop1TGD))
        self.assertTrue(callable(drop1TGDP))
        self.assertTrue(callable(add1TGD))
        self.assertTrue(callable(add1TGDP))
        self.assertTrue(callable(stepTGD))

    def setUp(self) -> None:
        self.refit_calls: list[dict[str, object]] = []

        def refit_callable(**kwargs: object) -> dict[str, object]:
            self.refit_calls.append(kwargs)
            return kwargs

        self.model = GAMLSSModel(
            par=("mu", "sigma"),
            family=FamilyDefinition(
                name="TEST",
                parameters=("mu", "sigma"),
                g_dev_inc=lambda y, mu, sigma: (
                    (jnp.asarray(y, dtype=jnp.float64) - mu) / sigma
                )
                ** 2,
            ),
            df_fit=3.0,
            g_dev=10.0,
            n=20,
            y=jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64),
            fitted_values={
                "mu": jnp.array([1.5, 2.5, 3.5], dtype=jnp.float64),
                "sigma": jnp.array([2.0, 2.0, 2.0], dtype=jnp.float64),
            },
            coefficients={
                "mu": jnp.array([0.0, 0.2, 0.4], dtype=jnp.float64),
                "sigma": jnp.array([1.1], dtype=jnp.float64),
            },
            linear_predictors={
                "mu": jnp.array([0.1, 0.2, 0.3], dtype=jnp.float64),
                "sigma": jnp.array([0.7], dtype=jnp.float64),
            },
            working_vectors={
                "mu": jnp.array([0.5, 0.7, 0.9], dtype=jnp.float64),
                "sigma": jnp.array([1.2], dtype=jnp.float64),
            },
            iterative_weights={
                "mu": jnp.array([4.0, 4.0, 4.0], dtype=jnp.float64),
                "sigma": jnp.array([9.0], dtype=jnp.float64),
            },
            offsets={
                "mu": jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64),
                "sigma": jnp.array([0.1], dtype=jnp.float64),
            },
            formulas={"mu": "y ~ x1 + x2", "sigma": "y ~ ."},
            terms={
                "mu": {
                    "term_labels": ["x1", "x2"],
                    "response": "y",
                    "intercept": True,
                    "formula": "y ~ x1 + x2",
                    "predictor_matrix": jnp.array(
                        [[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]], dtype=jnp.float64
                    ),
                    "se_matrix": jnp.array(
                        [[0.02, 0.03], [0.04, 0.05], [0.06, 0.07]], dtype=jnp.float64
                    ),
                    "constant": 0.15,
                    "column_names": ["x1", "x2"],
                },
                "sigma": {
                    "term_labels": ["x1", "x2"],
                    "response": "y",
                    "intercept": True,
                    "formula": "y ~ x1 + x2",
                    "predictor_matrix": jnp.array([[0.8, 0.9]], dtype=jnp.float64),
                    "se_matrix": jnp.array([[0.08, 0.09]], dtype=jnp.float64),
                    "constant": 0.25,
                    "column_names": ["x1", "x2"],
                },
            },
            design_matrices={
                "mu": jnp.array(
                    [[1.0, 1.0, 10.0], [1.0, 2.0, 20.0], [1.0, 3.0, 30.0]],
                    dtype=jnp.float64,
                )
            },
            xlevels={"mu": {"x1": [1.0, 2.0, 3.0]}},
            additional_slots={
                "P.deviance": 12.0,
                "G.deviance": 10.0,
                "noObs": 20,
                "df.residual": 17.0,
                "aic": 16.0,
                "sbc": 18.98719682,
                "method": "RS",
                "vcov": np.diag([0.04, 0.01, 0.09, 0.16]).astype(np.float64),
            },
            call={
                "callable": refit_callable,
                "kwargs": {"alpha": 5},
                "data": {
                    "y": jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64),
                    "x1": jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64),
                    "x2": jnp.array([10.0, 20.0, 30.0], dtype=jnp.float64),
                },
            },
            control={"n.cyc": 20},
            iter=7,
            weights=jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64),
            residuals=jnp.array([0.1, 0.2, 0.3], dtype=jnp.float64),
            parameters=("mu", "sigma"),
        )

    def test_is_gamlss(self) -> None:
        self.assertTrue(is_gamlss(self.model))
        self.assertFalse(is_gamlss(object()))

    def test_fitted_and_fv_follow_parameter_selection(self) -> None:
        np.testing.assert_allclose(np.asarray(fitted(self.model, "mu")), [1.5, 2.5, 3.5])
        np.testing.assert_allclose(np.asarray(fv(self.model, parameter="sigma")), [2.0, 2.0, 2.0])

    def test_coef_and_coef_all_match_r_style_slots(self) -> None:
        np.testing.assert_allclose(np.asarray(coef(self.model, "mu")), [0.0, 0.2, 0.4])
        output = coef_all(self.model, deviance=True)
        self.assertEqual(set(output.keys()), {"mu", "sigma", "deviance"})
        self.assertEqual(output["deviance"], 10.0)

    def test_deviance_lp_and_ic_helpers(self) -> None:
        self.assertEqual(deviance(self.model, "G"), 10.0)
        self.assertEqual(deviance(self.model, "P"), 12.0)
        np.testing.assert_allclose(np.asarray(lp(self.model, "mu")), [0.1, 0.2, 0.3])
        self.assertEqual(ic(self.model, k=2.0), 16.0)

    def test_refit_uses_call_component_and_doubles_cycle_count(self) -> None:
        result = refit(self.model)
        self.assertEqual(result["alpha"], 5)
        self.assertEqual(result["iter"], 7)
        self.assertEqual(result["n_cyc"], 40)
        self.assertIs(result["start_from"], self.model)
        self.assertEqual(len(self.refit_calls), 1)

    def test_model_methods_return_structured_results(self) -> None:
        loglik = log_likelihood(self.model)
        self.assertEqual(loglik.nall, 20)
        self.assertEqual(loglik.nobs, 20)
        self.assertEqual(loglik.df, 3.0)
        self.assertEqual(loglik.value, -5.0)

        covariance = vcov(self.model, type="vcov")
        self.assertEqual(covariance.shape, (4, 4))
        np.testing.assert_allclose(np.diag(covariance), [0.04, 0.01, 0.09, 0.16])

        summary_result = summary(self.model)
        self.assertEqual(summary_result.family, "TEST")
        self.assertEqual(summary_result.method, "RS")
        self.assertTrue(summary_result.converged)
        self.assertIn("mu", summary_result.coefficients)
        self.assertIn("t_value", summary_result.coefficients["mu"])
        self.assertIn("p_value", summary_result.coefficients["mu"])

        rendered = print_model(self.model)
        self.assertIn("Family: TEST", rendered)
        self.assertIn("Mu Coefficients", rendered)
        self.assertIn("Converged:", rendered)
        self.assertIn("estimate=", rendered)

        response_prediction = predict(self.model, what="mu", type="response")
        np.testing.assert_allclose(np.asarray(response_prediction), [1.5, 2.5, 3.5])
        term_prediction = predict(self.model, what="mu", type="terms")
        np.testing.assert_allclose(np.asarray(term_prediction), [[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]])

        predict_all_result = predict_all(self.model, type="response", output="list")
        self.assertEqual(predict_all_result.family, "TEST")
        self.assertIn("mu", predict_all_result.values)
        self.assertIn("sigma", predict_all_result.values)

        predict_all_matrix = predict_all(self.model, type="response", output="matrix")
        self.assertEqual(predict_all_matrix.shape, (3, 3))

        predict_all_se = predict_all(self.model, type="link", output="list", se_fit=True)
        self.assertIn("fit", predict_all_se.values["mu"])
        self.assertIn("se.fit", predict_all_se.values["mu"])

        predict_all_se_frame = predict_all(self.model, type="link", output="data.frame", se_fit=True)
        self.assertIn("mu", predict_all_se_frame)
        self.assertIn("mu_se", predict_all_se_frame)
        self.assertIn("sigma_se", predict_all_se_frame)
        self.assertEqual(np.asarray(predict_all_se_frame["sigma"]).shape, (3,))

        predict_all_se_matrix = predict_all(self.model, type="link", output="matrix", se_fit=True)
        self.assertEqual(predict_all_se_matrix.shape, (3, 5))

        predict_all_newdata_frame = predict_all(
            self.model,
            newdata={"x1": np.array([4.0, 5.0]), "x2": np.array([40.0, 50.0]), "y": np.array([0.0, 0.0])},
            type="link",
            output="data.frame",
            se_fit=True,
        )
        self.assertEqual(np.asarray(predict_all_newdata_frame["mu"]).shape, (2,))
        self.assertEqual(np.asarray(predict_all_newdata_frame["sigma"]).shape, (2,))
        self.assertEqual(np.asarray(predict_all_newdata_frame["sigma_se"]).shape, (2,))

        predict_all_terms_frame = predict_all(self.model, type="terms", output="data.frame", se_fit=True)
        self.assertIn("mu_x1", predict_all_terms_frame)
        self.assertIn("mu_x2", predict_all_terms_frame)
        self.assertIn("sigma_x1", predict_all_terms_frame)
        self.assertIn("sigma_x2", predict_all_terms_frame)
        self.assertIn("mu_x1_se", predict_all_terms_frame)
        self.assertIn("sigma_x2_se", predict_all_terms_frame)
        self.assertEqual(np.asarray(predict_all_terms_frame["mu_x1"]).shape, (3,))
        self.assertEqual(np.asarray(predict_all_terms_frame["sigma_x1"]).shape, (3,))

        predict_all_terms_matrix = predict_all(self.model, type="terms", output="matrix", se_fit=True)
        self.assertEqual(predict_all_terms_matrix.shape, (3, 9))

        updated = update_model(self.model, alpha=6)
        self.assertEqual(updated["alpha"], 6)

    def test_predict_all_use_weights_refit_supports_newdata_standard_errors(self) -> None:
        data = {
            "y": jnp.array([1.1, 1.8, 2.9, 3.7, 5.2, 6.1], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=jnp.float64),
            "x2": jnp.array([1.0, 1.3, 0.9, 1.4, 1.1, 1.5], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1 + x2",
            sigma_formula="~ x1",
            family=NO(),
            data=data,
            method="RS",
        )

        prediction = predict_all(
            fit,
            newdata={
                "x1": np.array([0.25, 1.75], dtype=np.float64),
                "x2": np.array([1.2, 1.0], dtype=np.float64),
            },
            type="link",
            se_fit=True,
            output="data.frame",
            use_weights=True,
        )
        self.assertEqual(np.asarray(prediction["mu"]).shape, (2,))
        self.assertEqual(np.asarray(prediction["mu_se"]).shape, (2,))
        self.assertEqual(np.asarray(prediction["sigma"]).shape, (2,))
        self.assertEqual(np.asarray(prediction["sigma_se"]).shape, (2,))
        self.assertTrue(np.isfinite(np.asarray(prediction["mu_se"])).all())
        self.assertTrue(np.isfinite(np.asarray(prediction["sigma_se"])).all())

        list_prediction = predict_all(
            fit,
            newdata={
                "x1": np.array([0.25, 1.75], dtype=np.float64),
                "x2": np.array([1.2, 1.0], dtype=np.float64),
            },
            type="response",
            se_fit=True,
            output="list",
            use_weights=True,
        )
        self.assertIn("mu", list_prediction.values)
        self.assertIn("sigma", list_prediction.values)
        self.assertIn("fit", list_prediction.values["mu"])
        self.assertIn("se.fit", list_prediction.values["mu"])
        self.assertEqual(np.asarray(list_prediction.values["mu"]["fit"]).shape, (2,))

        with self.assertRaises(ValueError):
            predict_all(fit, newdata={"x1": np.array([0.25]), "x2": np.array([1.2])}, type="terms", use_weights=True)

    def test_r_aligned_predict_all_module_hosts_runtime_implementation(self) -> None:
        result = predict_all_module.predict_all(self.model, type="response", output="list")
        self.assertIsInstance(result, predict_all_module.PredictAllResult)
        self.assertEqual(result.family, "TEST")
        self.assertIn("mu", result.values)

    def test_r_aligned_prodist_module_hosts_runtime_implementation(self) -> None:
        result = prodist_module.prodist_data(self.model, type="response")
        self.assertIsInstance(result, prodist_module.ProDistResult)
        self.assertEqual(result.family, "TEST")
        self.assertIn("mu", result.parameters)

    def test_residual_diagnostic_helpers_return_structured_outputs(self) -> None:
        diag_model = GAMLSSModel(
            par=("mu",),
            family=self.model.family,
            df_fit=1.0,
            g_dev=5.0,
            n=6,
            y=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=jnp.float64),
            fitted_values={"mu": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=jnp.float64)},
            coefficients={"mu": jnp.array([0.1], dtype=jnp.float64)},
            linear_predictors={"mu": jnp.array([0.0], dtype=jnp.float64)},
            residuals=jnp.array([-1.0, -0.2, 0.1, 0.4, 0.7, 1.2], dtype=jnp.float64),
            type="Continuous",
            class_name="gamlss",
        )
        acf_result = acf_residuals(diag_model, max_lag=3)
        self.assertEqual(acf_result.lags.tolist(), [1, 2, 3])
        self.assertEqual(acf_result.residual_acf.shape, (3,))
        self.assertTrue(np.isfinite(acf_result.squared_acf).all())

        qq_result = qq_stats(diag_model)
        self.assertEqual(qq_result.theoretical.shape, (6,))
        self.assertEqual(qq_result.ordered.shape, (6,))
        self.assertTrue(np.isfinite(qq_result.variance))

    def test_rqres_sample_and_worm_plot_helpers_return_structured_outputs(self) -> None:
        data = {
            "y": jnp.array([0.0, 1.0, 0.0, 2.0, 3.0], dtype=jnp.float64),
            "x1": jnp.array([0.0, 1.0, 0.5, 1.5, 2.0], dtype=jnp.float64),
        }
        fit = gamlss(formula="y ~ x1", family=PO(), data=data, method="RS")
        sample_result = get_rqres_samples(fit, howmany=4, order=True)
        self.assertEqual(sample_result.samples.shape, (5, 4))
        self.assertTrue(sample_result.ordered)
        self.assertTrue(np.isfinite(sample_result.samples).all())

        worm_result = worm_plot_data(fit, howmany=4)
        self.assertEqual(worm_result.theoretical.shape, (5,))
        self.assertEqual(worm_result.median_deviation.shape, (5,))
        self.assertEqual(worm_result.sample_deviations.shape, (5, 4))
        self.assertTrue(np.isfinite(worm_result.lower_band).all())
        self.assertTrue(np.isfinite(worm_result.upper_band).all())

    def test_plot_diagnostics_returns_four_panel_data_bundle(self) -> None:
        diag_model = GAMLSSModel(
            par=("mu",),
            family=self.model.family,
            df_fit=1.0,
            g_dev=5.0,
            n=6,
            y=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=jnp.float64),
            fitted_values={"mu": jnp.array([1.1, 2.0, 2.8, 4.2, 5.1, 5.9], dtype=jnp.float64)},
            coefficients={"mu": jnp.array([0.1], dtype=jnp.float64)},
            linear_predictors={"mu": jnp.array([0.0], dtype=jnp.float64)},
            residuals=jnp.array([-1.0, -0.2, 0.1, 0.4, 0.7, 1.2], dtype=jnp.float64),
            weights=jnp.ones(6, dtype=jnp.float64),
            type="Continuous",
            class_name="gamlss",
        )
        panels = plot_diagnostics(diag_model)
        self.assertEqual(panels.fitted_x.shape, (6,))
        self.assertEqual(panels.residual_y.shape, (6,))
        self.assertEqual(panels.index_x.shape, (6,))
        self.assertEqual(panels.qq_theoretical.shape, (6,))
        self.assertEqual(panels.qq_ordered.shape, (6,))
        self.assertGreater(panels.density_x.shape[0], 1)
        self.assertEqual(set(panels.summary_stats.keys()), {"mean", "variance", "skewness", "kurtosis", "filliben"})

    def test_term_plot_data_returns_sorted_term_curves_and_intervals(self) -> None:
        term_result = term_plot_data(self.model, what="mu", terms=["x1", "x2"], se=True, level=0.95)
        self.assertEqual(term_result.what, "mu")
        self.assertEqual(len(term_result.entries), 2)
        first = term_result.entries[0]
        self.assertEqual(first.term, "x1")
        self.assertEqual(first.x.shape, (3,))
        self.assertEqual(first.fit.shape, (3,))
        self.assertEqual(first.se_fit.shape, (3,))
        self.assertEqual(first.lower.shape, (3,))
        self.assertEqual(first.upper.shape, (3,))
        self.assertTrue(np.all(first.x[:-1] <= first.x[1:]))

    def test_fitted_plot_data_returns_sorted_parameter_curves(self) -> None:
        fitted_result = fitted_plot_data(self.model, xvar="x1", parameters=("mu", "sigma"), type="response", se_fit=True)
        self.assertEqual(fitted_result.xvar, "x1")
        self.assertEqual(fitted_result.type, "response")
        self.assertEqual(len(fitted_result.entries), 2)
        first = fitted_result.entries[0]
        self.assertEqual(first.parameter, "mu")
        self.assertEqual(first.x.shape, (3,))
        self.assertEqual(first.fit.shape, (3,))
        self.assertEqual(first.se_fit.shape, (3,))
        self.assertEqual(first.lower.shape, (3,))
        self.assertEqual(first.upper.shape, (3,))
        self.assertTrue(np.all(first.x[:-1] <= first.x[1:]))

    def test_fitted_plot_data_supports_jsu_family(self) -> None:
        data = {
            "y": jnp.array([-0.8, -0.2, 0.1, 0.7, 1.4, 2.1], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=JSU(),
            data=data,
            method="RS",
        )
        result = fitted_plot_data(fit, xvar="x1", parameters=("mu", "sigma", "nu", "tau"), type="response", se_fit=True)
        self.assertEqual(result.xvar, "x1")
        self.assertEqual(result.type, "response")
        self.assertEqual(len(result.entries), 4)
        for entry in result.entries:
            self.assertEqual(entry.x.shape, (6,))
            self.assertEqual(entry.fit.shape, (6,))
            self.assertEqual(entry.se_fit.shape, (6,))
            self.assertEqual(entry.lower.shape, (6,))
            self.assertEqual(entry.upper.shape, (6,))
            self.assertTrue(np.all(entry.x[:-1] <= entry.x[1:]))
            self.assertTrue(np.isfinite(entry.fit).all())

    def test_fitted_plot_data_supports_bct_family(self) -> None:
        data = {
            "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=BCT(),
            data=data,
            method="RS",
        )
        result = fitted_plot_data(fit, xvar="x1", parameters=("mu", "sigma", "nu", "tau"), type="response", se_fit=True)
        self.assertEqual(len(result.entries), 4)
        for entry in result.entries:
            self.assertEqual(entry.x.shape, (6,))
            self.assertEqual(entry.fit.shape, (6,))
            self.assertEqual(entry.se_fit.shape, (6,))
            self.assertTrue(np.all(np.isfinite(entry.fit)))
        self.assertTrue(np.all(result.entries[0].fit > 0.0))
        self.assertTrue(np.all(result.entries[1].fit > 0.0))
        self.assertTrue(np.all(np.isfinite(result.entries[3].fit)))

    def test_fitted_plot_data_supports_bcpe_family(self) -> None:
        data = {
            "y": jnp.array([0.9, 1.1, 1.5, 2.1, 2.9, 4.0], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=BCPE(),
            data=data,
            method="RS",
        )
        result = fitted_plot_data(fit, xvar="x1", parameters=("mu", "sigma", "nu", "tau"), type="response", se_fit=True)
        self.assertEqual(len(result.entries), 4)
        for entry in result.entries:
            self.assertEqual(entry.x.shape, (6,))
            self.assertEqual(entry.fit.shape, (6,))
            self.assertEqual(entry.se_fit.shape, (6,))
            self.assertTrue(np.all(np.isfinite(entry.fit)))
        self.assertTrue(np.all(result.entries[0].fit > 0.0))
        self.assertTrue(np.all(result.entries[1].fit > 0.0))
        self.assertTrue(np.all(result.entries[3].fit > 0.0))

    def test_fitted_plot_data_supports_logno_family(self) -> None:
        data = {
            "y": jnp.array([0.8, 1.1, 1.6, 2.4, 3.3, 4.5], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            family=LOGNO(),
            data=data,
            method="RS",
        )
        result = fitted_plot_data(fit, xvar="x1", parameters=("mu", "sigma"), type="response", se_fit=True)
        self.assertEqual(len(result.entries), 2)
        for entry in result.entries:
            self.assertEqual(entry.x.shape, (6,))
            self.assertEqual(entry.fit.shape, (6,))
            self.assertEqual(entry.se_fit.shape, (6,))
            self.assertTrue(np.all(np.isfinite(entry.fit)))
        self.assertTrue(np.all(np.isfinite(result.entries[0].fit)))
        self.assertTrue(np.all(result.entries[1].fit > 0.0))

    def test_fitted_plot_data_supports_no_family(self) -> None:
        data = {
            "y": jnp.array([1.0, 1.6, 2.2, 3.0, 3.7, 4.5], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            family=NO(),
            data=data,
            method="RS",
        )
        result = fitted_plot_data(fit, xvar="x1", parameters=("mu", "sigma"), type="response", se_fit=True)
        self.assertEqual(len(result.entries), 2)
        for entry in result.entries:
            self.assertEqual(entry.x.shape, (6,))
            self.assertEqual(entry.fit.shape, (6,))
            self.assertEqual(entry.se_fit.shape, (6,))
            self.assertTrue(np.all(np.isfinite(entry.fit)))
            self.assertTrue(np.all(entry.x[:-1] <= entry.x[1:]))
        self.assertTrue(np.all(result.entries[1].fit > 0.0))

    def test_fitted_plot_data_supports_ga_family(self) -> None:
        data = {
            "y": jnp.array([0.8, 1.1, 1.5, 2.0, 2.6, 3.4], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            family=GA(),
            data=data,
            method="RS",
        )
        result = fitted_plot_data(fit, xvar="x1", parameters=("mu", "sigma"), type="response", se_fit=True)
        self.assertEqual(len(result.entries), 2)
        for entry in result.entries:
            self.assertEqual(entry.x.shape, (6,))
            self.assertEqual(entry.fit.shape, (6,))
            self.assertEqual(entry.se_fit.shape, (6,))
            self.assertTrue(np.all(np.isfinite(entry.fit)))
            self.assertTrue(np.all(entry.fit > 0.0))

    def test_fitted_plot_data_supports_ig_family(self) -> None:
        data = {
            "y": jnp.array([0.9, 1.2, 1.7, 2.3, 3.1, 4.2], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            family=IG(),
            data=data,
            method="RS",
        )
        result = fitted_plot_data(fit, xvar="x1", parameters=("mu", "sigma"), type="response", se_fit=True)
        self.assertEqual(len(result.entries), 2)
        for entry in result.entries:
            self.assertEqual(entry.x.shape, (6,))
            self.assertEqual(entry.fit.shape, (6,))
            self.assertEqual(entry.se_fit.shape, (6,))
            self.assertTrue(np.all(np.isfinite(entry.fit)))
            self.assertTrue(np.all(entry.fit > 0.0))

    def test_fitted_plot_data_supports_wei_family(self) -> None:
        data = {
            "y": jnp.array([0.7, 1.0, 1.3, 1.9, 2.7, 3.8], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            family=WEI(),
            data=data,
            method="RS",
        )
        result = fitted_plot_data(fit, xvar="x1", parameters=("mu", "sigma"), type="response", se_fit=True)
        self.assertEqual(len(result.entries), 2)
        for entry in result.entries:
            self.assertEqual(entry.x.shape, (6,))
            self.assertEqual(entry.fit.shape, (6,))
            self.assertEqual(entry.se_fit.shape, (6,))
            self.assertTrue(np.all(np.isfinite(entry.fit)))
            self.assertTrue(np.all(entry.fit > 0.0))

    def test_pdf_plot_data_supports_model_observation_panels(self) -> None:
        result = pdf_plot_data(
            obj=self.model,
            obs=(1, 2),
            min_value=0.0,
            max_value=4.0,
            no_points=41,
        )
        self.assertEqual(result.family, "TEST")
        self.assertEqual(result.source, "model")
        self.assertEqual(len(result.entries), 2)
        first = result.entries[0]
        self.assertEqual(first.index, 1)
        self.assertEqual(first.y.shape, (41,))
        self.assertEqual(first.density.shape, (41,))
        self.assertTrue(np.all(np.isfinite(first.density)))
        self.assertTrue(np.all(first.density >= 0.0))
        self.assertIsNotNone(first.observed_value)
        self.assertIn("mu", first.parameters)
        self.assertIn("sigma", first.parameters)

    def test_pdf_plot_data_supports_direct_family_parameters(self) -> None:
        result = pdf_plot_data(
            family=NO(),
            mu=(1.0, 2.0),
            sigma=(0.5, 1.0),
            min_value=-2.0,
            max_value=5.0,
            no_points=51,
        )
        self.assertEqual(result.family, "NO")
        self.assertEqual(result.source, "parameters")
        self.assertEqual(len(result.entries), 2)
        for entry in result.entries:
            self.assertEqual(entry.y.shape, (51,))
            self.assertEqual(entry.density.shape, (51,))
            self.assertTrue(np.all(np.isfinite(entry.density)))
            self.assertTrue(np.all(entry.density >= 0.0))
            self.assertIsNone(entry.observed_value)
            self.assertIn("mu", entry.parameters)
            self.assertIn("sigma", entry.parameters)

    def test_pdf_plot_data_supports_real_family_models(self) -> None:
        data = {
            "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            family=GA(),
            data=data,
            method="RS",
        )
        result = pdf_plot_data(
            obj=fit,
            obs=(1, 3),
            min_value=0.01,
            max_value=6.0,
            no_points=61,
        )
        self.assertEqual(result.family, "GA")
        self.assertEqual(result.distribution_type, "Continuous")
        self.assertEqual(len(result.entries), 2)
        for entry in result.entries:
            self.assertTrue(np.all(np.isfinite(entry.density)))
            self.assertTrue(np.all(entry.density >= 0.0))
            self.assertTrue(np.isfinite(entry.observed_value))
            self.assertTrue(entry.parameters["mu"] > 0.0)
            self.assertTrue(entry.parameters["sigma"] > 0.0)

    def test_hist_dist_data_supports_continuous_family(self) -> None:
        result = hist_dist_data(
            y=np.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=np.float64),
            family=GA(),
            density=True,
            nbins=5,
        )
        self.assertEqual(result.family, "GA")
        self.assertEqual(result.distribution_type, "Continuous")
        self.assertEqual(result.histogram_x.shape, (5,))
        self.assertEqual(result.histogram_y.shape, (5,))
        self.assertEqual(result.fitted_x.shape, (201,))
        self.assertEqual(result.fitted_y.shape, (201,))
        self.assertTrue(np.all(np.isfinite(result.histogram_y)))
        self.assertTrue(np.all(np.isfinite(result.fitted_y)))
        self.assertTrue(np.all(result.fitted_y >= 0.0))
        self.assertIsNotNone(result.density_x)
        self.assertIsNotNone(result.density_y)
        self.assertTrue(result.fitted_parameters["mu"] > 0.0)
        self.assertTrue(result.fitted_parameters["sigma"] > 0.0)

    def test_hist_dist_data_supports_discrete_family(self) -> None:
        result = hist_dist_data(
            y=np.array([0, 1, 1, 2, 2, 3], dtype=np.float64),
            family=PO(),
        )
        self.assertEqual(result.family, "PO")
        self.assertEqual(result.distribution_type, "Discrete")
        self.assertEqual(result.histogram_x.shape, result.histogram_y.shape)
        self.assertEqual(result.fitted_x.shape, result.fitted_y.shape)
        self.assertTrue(np.all(np.isfinite(result.histogram_y)))
        self.assertTrue(np.all(np.isfinite(result.fitted_y)))
        self.assertTrue(np.all(result.histogram_y >= 0.0))
        self.assertTrue(np.all(result.fitted_y >= 0.0))
        self.assertIsNone(result.density_x)
        self.assertIsNone(result.density_y)

    def test_hist_dist_data_supports_frequency_weights(self) -> None:
        result = hist_dist_data(
            y=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            family=NO(),
            freq=np.array([1.0, 2.0, 1.0], dtype=np.float64),
            nbins=3,
        )
        self.assertEqual(result.family, "NO")
        self.assertTrue(result.used_weights)
        self.assertEqual(result.histogram_x.shape, (3,))
        self.assertEqual(result.histogram_y.shape, (3,))
        self.assertTrue(np.all(np.isfinite(result.histogram_y)))
        self.assertTrue(np.all(np.isfinite(result.fitted_y)))

    def test_prodist_data_returns_parameter_snapshot(self) -> None:
        result = prodist_data(self.model)
        self.assertEqual(result.family, "TEST")
        self.assertEqual(result.output, "list")
        self.assertEqual(result.type, "response")
        self.assertIn("mu", result.parameters)
        self.assertIn("sigma", result.parameters)
        self.assertNotIn("y", result.parameters)
        self.assertTrue(np.all(np.isfinite(np.asarray(result.parameters["mu"], dtype=np.float64))))

    def test_prodist_data_supports_newdata_and_standard_errors(self) -> None:
        result = prodist_data(
            self.model,
            newdata={"x1": np.array([1.1, 1.9], dtype=np.float64), "x2": np.array([0.8, 1.2], dtype=np.float64)},
            type="link",
            se_fit=True,
        )
        self.assertEqual(result.type, "link")
        self.assertIn("mu", result.parameters)
        mu_entry = result.parameters["mu"]
        self.assertIsInstance(mu_entry, dict)
        self.assertIn("fit", mu_entry)
        self.assertIn("se.fit", mu_entry)
        self.assertEqual(np.asarray(mu_entry["fit"], dtype=np.float64).shape, (2,))
        self.assertEqual(np.asarray(mu_entry["se.fit"], dtype=np.float64).shape, (2,))

    def test_prodist_data_supports_real_family_models(self) -> None:
        data = {
            "y": jnp.array([-0.8, -0.2, 0.1, 0.7, 1.4, 2.1], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=JSU(),
            data=data,
            method="RS",
        )
        result = prodist_data(fit)
        self.assertEqual(result.family, "JSU")
        self.assertIn("mu", result.parameters)
        self.assertIn("sigma", result.parameters)
        self.assertIn("nu", result.parameters)
        self.assertIn("tau", result.parameters)
        self.assertEqual(np.asarray(result.parameters["tau"], dtype=np.float64).shape, (6,))

    def test_fit_dist_data_ranks_realplus_families(self) -> None:
        result = fit_dist_data(
            y=np.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=np.float64),
            type="realplus",
            k=2.0,
        )
        self.assertEqual(result.type, "realplus")
        self.assertGreaterEqual(len(result.rows), 1)
        self.assertEqual(result.best_family, result.rows[0].family)
        self.assertTrue(all(result.rows[i].criterion <= result.rows[i + 1].criterion for i in range(len(result.rows) - 1)))
        self.assertIn(result.best_family, {"EXP", "GA", "IG", "LOGNO", "WEI", "BCCG", "BCT", "BCPE"})

    def test_fit_dist_data_ranks_count_families(self) -> None:
        result = fit_dist_data(
            y=np.array([0, 1, 1, 2, 3, 5], dtype=np.float64),
            type="counts",
            k=2.0,
        )
        self.assertEqual(result.type, "counts")
        self.assertGreaterEqual(len(result.rows), 1)
        self.assertEqual(result.best_family, result.rows[0].family)
        self.assertIn(result.best_family, {"PO", "GEOM", "NBI", "ZIP"})
        self.assertTrue(all(np.isfinite(row.criterion) for row in result.rows))

    def test_fit_dist_data_supports_extra_family_candidates(self) -> None:
        result = fit_dist_data(
            y=np.array([-0.8, -0.2, 0.1, 0.7, 1.4, 2.1], dtype=np.float64),
            type="realline",
            extra=("JSU",),
        )
        families = {row.family for row in result.rows}
        self.assertIn("JSU", families)

    def test_r_aligned_fit_dist_module_hosts_runtime_implementation(self) -> None:
        result = fit_dist_module.fit_dist_data(
            y=np.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=np.float64),
            type="realplus",
        )
        self.assertIsInstance(result, fit_dist_module.FitDistResult)
        self.assertGreaterEqual(len(result.rows), 1)

    def test_fit_dist_pred_data_supports_rand_split(self) -> None:
        result = fit_dist_pred_data(
            y=np.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=np.float64),
            type="realplus",
            rand=np.array([1, 1, 1, 2, 2, 2], dtype=np.int64),
        )
        self.assertEqual(result.type, "realplus")
        self.assertEqual(result.validation_size, 3)
        self.assertGreaterEqual(len(result.rows), 1)
        self.assertEqual(result.best_family, result.rows[0].family)
        self.assertTrue(
            all(
                result.rows[i].validation_global_deviance <= result.rows[i + 1].validation_global_deviance
                for i in range(len(result.rows) - 1)
            )
        )

    def test_fit_dist_pred_data_supports_newdata_path(self) -> None:
        result = fit_dist_pred_data(
            y=np.array([0.8, 1.0, 1.4, 2.0], dtype=np.float64),
            type="realplus",
            newdata={"y": np.array([2.8, 3.9], dtype=np.float64)},
        )
        self.assertEqual(result.validation_size, 2)
        self.assertGreaterEqual(len(result.rows), 1)
        self.assertTrue(all(np.isfinite(row.prediction_error) for row in result.rows))

    def test_fit_dist_pred_data_supports_count_families(self) -> None:
        result = fit_dist_pred_data(
            y=np.array([0, 1, 1, 2, 3, 5], dtype=np.float64),
            type="counts",
            rand=np.array([1, 1, 1, 2, 2, 2], dtype=np.int64),
        )
        self.assertEqual(result.type, "counts")
        self.assertEqual(result.validation_size, 3)
        self.assertIn(result.best_family, {"PO", "GEOM", "NBI", "ZIP"})

    def test_gamlss_ml_pred_data_supports_rand_split(self) -> None:
        result = gamlss_ml_pred_data(
            y=np.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=np.float64),
            family=GA(),
            rand=np.array([1, 1, 1, 2, 2, 2], dtype=np.int64),
        )
        self.assertEqual(result.family, "GA")
        self.assertEqual(result.validation_size, 3)
        self.assertEqual(result.validation_increment.shape, (3,))
        self.assertTrue(np.isfinite(result.validation_increment).all())
        self.assertTrue(np.isfinite(result.validation_global_deviance))
        self.assertTrue(np.isfinite(result.prediction_error))

    def test_gamlss_ml_pred_data_supports_newdata_path(self) -> None:
        result = gamlss_ml_pred_data(
            y=np.array([0.8, 1.0, 1.4, 2.0], dtype=np.float64),
            family=LOGNO(),
            newdata={"y": np.array([2.8, 3.9], dtype=np.float64)},
        )
        self.assertEqual(result.family, "LOGNO")
        self.assertEqual(result.validation_size, 2)
        self.assertEqual(result.validation_increment.shape, (2,))
        self.assertTrue(np.isfinite(result.validation_increment).all())
        self.assertGreater(result.validation_global_deviance, 0.0)

    def test_gamlss_ml_pred_data_returns_validation_rqres_for_counts(self) -> None:
        result = gamlss_ml_pred_data(
            y=np.array([0, 1, 1, 2, 3, 5], dtype=np.float64),
            family=PO(),
            rand=np.array([1, 1, 1, 2, 2, 2], dtype=np.int64),
        )
        self.assertEqual(result.family, "PO")
        self.assertEqual(result.validation_size, 3)
        self.assertIsNotNone(result.validation_residuals)
        self.assertEqual(result.validation_residuals.shape, (3,))
        self.assertTrue(np.isfinite(result.validation_residuals).all())

    def test_r_aligned_fit_dist_pred_module_hosts_runtime_implementation(self) -> None:
        result = fit_dist_pred_module.fit_dist_pred_data(
            y=np.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=np.float64),
            type="realplus",
            rand=np.array([1, 1, 1, 2, 2, 2], dtype=np.int64),
        )
        self.assertIsInstance(result, fit_dist_pred_module.FitDistPredResult)
        self.assertGreaterEqual(len(result.rows), 1)

        single = fit_dist_pred_module.gamlss_ml_pred_data(
            y=np.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=np.float64),
            family=GA(),
            rand=np.array([1, 1, 1, 2, 2, 2], dtype=np.int64),
        )
        self.assertIsInstance(single, fit_dist_pred_module.GAMLSSMLPredResult)
        self.assertEqual(single.validation_size, 3)

    def test_get_tgd_data_supports_continuous_models(self) -> None:
        fit = gamlss(
            formula="y ~ 1",
            family=GA(),
            data={"y": jnp.array([0.8, 1.0, 1.4, 2.0], dtype=jnp.float64)},
            method="RS",
        )
        result = get_tgd_data(
            fit,
            newdata={"y": np.array([2.8, 3.9], dtype=np.float64)},
        )
        self.assertEqual(result.family, "GA")
        self.assertEqual(result.validation_size, 2)
        self.assertEqual(result.deviance_increment.shape, (2,))
        self.assertTrue(np.isfinite(result.deviance_increment).all())
        self.assertTrue(np.isfinite(result.test_global_deviance))
        self.assertTrue(np.isfinite(result.prediction_error))
        self.assertIsNotNone(result.residuals)
        self.assertEqual(result.residuals.shape, (2,))

    def test_get_tgd_data_returns_rqres_for_discrete_models(self) -> None:
        fit = gamlss(
            formula="y ~ 1",
            family=PO(),
            data={"y": jnp.array([0, 1, 1, 2, 3, 5], dtype=jnp.float64)},
            method="RS",
        )
        result = get_tgd_data(
            fit,
            newdata={"y": np.array([2, 3, 5], dtype=np.float64)},
        )
        self.assertEqual(result.family, "PO")
        self.assertEqual(result.validation_size, 3)
        self.assertEqual(result.deviance_increment.shape, (3,))
        self.assertIsNotNone(result.residuals)
        self.assertEqual(result.residuals.shape, (3,))
        self.assertTrue(np.isfinite(result.residuals).all())

    def test_extract_tgd_data_returns_edf_and_test_deviance(self) -> None:
        fit = gamlss(
            formula="y ~ 1",
            family=GA(),
            data={"y": jnp.array([0.8, 1.0, 1.4, 2.0], dtype=jnp.float64)},
            method="RS",
        )
        result = extract_tgd_data(
            fit,
            newdata={"y": np.array([2.8, 3.9], dtype=np.float64)},
        )
        self.assertAlmostEqual(result.edf, float(fit.df_fit))
        self.assertTrue(np.isfinite(result.tgd))
        self.assertGreater(result.tgd, 0.0)

    def test_gamlss_vgd_data_returns_validation_fit_summary(self) -> None:
        result = gamlss_vgd_data(
            formula="y ~ x1",
            sigma_formula="~ x1",
            family=GA(),
            data={
                "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64),
            },
            rand=np.array([1, 1, 1, 2, 2, 2], dtype=np.int64),
            method="RS",
        )
        self.assertEqual(result.family, "GA")
        self.assertEqual(result.validation_size, 3)
        self.assertEqual(result.validation_increment.shape, (3,))
        self.assertTrue(np.isfinite(result.validation_increment).all())
        self.assertIn("VGD", result.model.additional_slots)
        self.assertTrue(np.isfinite(result.model.additional_slots["VGD"]))

    def test_validation_and_fit_aliases_delegate_to_staged_helpers(self) -> None:
        y = jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64)
        x1 = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64)
        data = {"y": y, "x1": x1}
        split = np.array([1, 1, 1, 2, 2, 2], dtype=np.int64)
        cv_split = np.array([1, 2, 3, 1, 2, 3], dtype=np.int64)

        self.assertEqual(fit_dist(y, type="realplus").best_family, fit_dist_data(y, type="realplus").best_family)
        self.assertEqual(
            fit_dist_pred(y, type="realplus", rand=split).best_family,
            fit_dist_pred_data(y, type="realplus", rand=split).best_family,
        )
        self.assertEqual(
            gamlss_ml_pred(y, family=GA(), rand=split).validation_size,
            gamlss_ml_pred_data(y, family=GA(), rand=split).validation_size,
        )

        fit = gamlss(formula="y ~ x1", family=GA(), data=data, method="RS")
        self.assertAlmostEqual(
            get_tgd(fit, newdata={"y": np.array([2.8, 3.9], dtype=np.float64), "x1": np.array([4.0, 5.0], dtype=np.float64)}).test_global_deviance,
            get_tgd_data(fit, newdata={"y": np.array([2.8, 3.9], dtype=np.float64), "x1": np.array([4.0, 5.0], dtype=np.float64)}).test_global_deviance,
        )
        self.assertAlmostEqual(
            extract_tgd(fit, newdata={"y": np.array([2.8, 3.9], dtype=np.float64), "x1": np.array([4.0, 5.0], dtype=np.float64)}).tgd,
            extract_tgd_data(fit, newdata={"y": np.array([2.8, 3.9], dtype=np.float64), "x1": np.array([4.0, 5.0], dtype=np.float64)}).tgd,
        )

        self.assertEqual(
            gamlss_vgd(formula="y ~ x1", family=GA(), data=data, rand=split, method="RS").validation_size,
            gamlss_vgd_data(formula="y ~ x1", family=GA(), data=data, rand=split, method="RS").validation_size,
        )
        self.assertEqual(
            gamlss_cv(formula="y ~ x1", family=GA(), data=data, rand=cv_split, method="RS").k_fold,
            gamlss_cv_data(formula="y ~ x1", family=GA(), data=data, rand=cv_split, method="RS").k_fold,
        )
        choose_alias = choose_dist_pred(fit, rand=split, type="realplus")
        choose_data = choose_dist_pred_data(fit, rand=split, type="realplus")
        self.assertEqual(choose_alias.best_family, choose_data.best_family)

    def test_vgd_supports_single_and_multi_model_comparisons(self) -> None:
        first = gamlss_vgd_data(
            formula="y ~ x1",
            family=GA(),
            data={
                "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64),
            },
            rand=np.array([1, 1, 1, 2, 2, 2], dtype=np.int64),
            method="RS",
        )
        second = gamlss_vgd_data(
            formula="y ~ x1",
            family=WEI(),
            data={
                "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64),
            },
            rand=np.array([1, 1, 1, 2, 2, 2], dtype=np.int64),
            method="RS",
        )
        single_value = vgd(first.model)
        comparison = vgd(first.model, second.model)
        self.assertTrue(np.isfinite(single_value))
        self.assertEqual(len(comparison.rows), 2)
        self.assertTrue(all(np.isfinite(row.pred_gd) for row in comparison.rows))
        self.assertTrue(comparison.rows[0].pred_gd <= comparison.rows[1].pred_gd)

    def test_gamlss_cv_data_returns_fold_summaries(self) -> None:
        result = gamlss_cv_data(
            formula="y ~ x1",
            family=GA(),
            data={
                "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64),
            },
            k_fold=3,
            set_seed=7,
            method="RS",
        )
        self.assertEqual(result.family, "GA")
        self.assertEqual(result.k_fold, 3)
        self.assertEqual(result.all_cv.shape, (3,))
        self.assertEqual(result.resid_cv.shape, (6,))
        self.assertEqual(result.folds.shape, (6,))
        self.assertTrue(np.isfinite(result.cv))
        self.assertTrue(np.all(np.isfinite(result.all_cv)))
        self.assertIn("CV", result.model.additional_slots)
        self.assertIn("allCV", result.model.additional_slots)
        self.assertIn("residCV", result.model.additional_slots)

    def test_cv_supports_single_and_multi_model_comparisons(self) -> None:
        first = gamlss_cv_data(
            formula="y ~ x1",
            family=GA(),
            data={
                "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64),
            },
            rand=np.array([1, 2, 3, 1, 2, 3], dtype=np.int64),
            method="RS",
        )
        second = gamlss_cv_data(
            formula="y ~ x1",
            family=WEI(),
            data={
                "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64),
            },
            rand=np.array([1, 2, 3, 1, 2, 3], dtype=np.int64),
            method="RS",
        )
        single_value = cv(first.model)
        comparison = cv(first.model, second.model)
        self.assertTrue(np.isfinite(single_value))
        self.assertEqual(len(comparison.rows), 2)
        self.assertTrue(all(np.isfinite(row.cv) for row in comparison.rows))
        self.assertTrue(comparison.rows[0].cv <= comparison.rows[1].cv)

    def test_r_aligned_gamlss_vgd_module_hosts_runtime_implementation(self) -> None:
        result = gamlss_vgd_module.gamlss_vgd_data(
            formula="y ~ x1",
            family=GA(),
            data={
                "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64),
            },
            rand=np.array([1, 1, 1, 2, 2, 2], dtype=np.int64),
            method="RS",
        )
        self.assertIsInstance(result, gamlss_vgd_module.GAMLSSVGDResult)
        self.assertTrue(gamlss_vgd_module.is_gamlss_vgd(result))

        cv_result = gamlss_vgd_module.gamlss_cv_data(
            formula="y ~ x1",
            family=GA(),
            data={
                "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64),
            },
            rand=np.array([1, 2, 3, 1, 2, 3], dtype=np.int64),
            method="RS",
        )
        self.assertIsInstance(cv_result, gamlss_vgd_module.GAMLSSCVResult)
        self.assertTrue(gamlss_vgd_module.is_gamlss_cv(cv_result))

    def test_validation_object_predicates_detect_vgd_and_cv_results(self) -> None:
        vgd_result = gamlss_vgd_data(
            formula="y ~ x1",
            family=GA(),
            data={
                "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64),
            },
            rand=np.array([1, 1, 1, 2, 2, 2], dtype=np.int64),
            method="RS",
        )
        cv_result = gamlss_cv_data(
            formula="y ~ x1",
            family=GA(),
            data={
                "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64),
            },
            rand=np.array([1, 2, 3, 1, 2, 3], dtype=np.int64),
            method="RS",
        )

        self.assertTrue(is_gamlss_vgd(vgd_result))
        self.assertTrue(is_gamlss_vgd(vgd_result.model))
        self.assertFalse(is_gamlss_vgd(self.model))
        self.assertTrue(is_gamlss_cv(cv_result))
        self.assertTrue(is_gamlss_cv(cv_result.model))
        self.assertFalse(is_gamlss_cv(self.model))

    def test_vgd_and_cv_accept_result_wrappers_directly(self) -> None:
        first_vgd = gamlss_vgd_data(
            formula="y ~ x1",
            family=GA(),
            data={
                "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64),
            },
            rand=np.array([1, 1, 1, 2, 2, 2], dtype=np.int64),
            method="RS",
        )
        second_vgd = gamlss_vgd_data(
            formula="y ~ x1",
            family=WEI(),
            data={
                "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64),
            },
            rand=np.array([1, 1, 1, 2, 2, 2], dtype=np.int64),
            method="RS",
        )
        first_cv = gamlss_cv_data(
            formula="y ~ x1",
            family=GA(),
            data={
                "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64),
            },
            rand=np.array([1, 2, 3, 1, 2, 3], dtype=np.int64),
            method="RS",
        )
        second_cv = gamlss_cv_data(
            formula="y ~ x1",
            family=WEI(),
            data={
                "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64),
            },
            rand=np.array([1, 2, 3, 1, 2, 3], dtype=np.int64),
            method="RS",
        )

        self.assertTrue(np.isfinite(vgd(first_vgd)))
        self.assertEqual(len(vgd(first_vgd, second_vgd).rows), 2)
        self.assertTrue(np.isfinite(cv(first_cv)))
        self.assertEqual(len(cv(first_cv, second_cv).rows), 2)

    def test_drop1_tgd_returns_term_deletion_table(self) -> None:
        fit = gamlss(
            formula="y ~ x1 + x2",
            family=GA(),
            data={
                "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64),
                "x2": jnp.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=jnp.float64),
            },
            method="RS",
        )
        result = drop1_tgd(
            fit,
            newdata={
                "y": np.array([1.1, 2.1, 3.4], dtype=np.float64),
                "x1": np.array([0.5, 2.5, 4.5], dtype=np.float64),
                "x2": np.array([1.2, 2.2, 3.2], dtype=np.float64),
            },
            what="mu",
            sorted=True,
        )
        self.assertEqual(result.what, "mu")
        self.assertEqual(result.direction, "drop")
        self.assertEqual(result.rows[0].term, "<none>")
        self.assertGreaterEqual(len(result.rows), 2)
        self.assertTrue(all(np.isfinite(row.tgd) for row in result.rows))

    def test_add1_tgd_returns_term_addition_table(self) -> None:
        fit = gamlss(
            formula="y ~ x1",
            family=GA(),
            data={
                "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64),
                "x2": jnp.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=jnp.float64),
            },
            method="RS",
        )
        result = add1_tgd(
            fit,
            scope="~ x2",
            newdata={
                "y": np.array([1.1, 2.1, 3.4], dtype=np.float64),
                "x1": np.array([0.5, 2.5, 4.5], dtype=np.float64),
                "x2": np.array([1.2, 2.2, 3.2], dtype=np.float64),
            },
            what="mu",
            sorted=True,
        )
        self.assertEqual(result.what, "mu")
        self.assertEqual(result.direction, "add")
        self.assertEqual(result.rows[0].term, "<none>")
        self.assertGreaterEqual(len(result.rows), 2)
        self.assertTrue(all(np.isfinite(row.tgd) for row in result.rows))

    def test_step_tgd_returns_search_path_and_final_model(self) -> None:
        fit = gamlss(
            formula="y ~ x1",
            family=GA(),
            data={
                "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64),
                "x2": jnp.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=jnp.float64),
            },
            method="RS",
        )
        result = step_tgd(
            fit,
            scope={"upper": "~ x1 + x2"},
            newdata={
                "y": np.array([1.1, 2.1, 3.4], dtype=np.float64),
                "x1": np.array([0.5, 2.5, 4.5], dtype=np.float64),
                "x2": np.array([1.2, 2.2, 3.2], dtype=np.float64),
            },
            what="mu",
            direction="both",
            steps=3,
        )
        self.assertEqual(result.what, "mu")
        self.assertEqual(result.direction, "both")
        self.assertGreaterEqual(len(result.steps), 1)
        self.assertEqual(result.steps[0].step, 0)
        self.assertTrue(all(np.isfinite(step.tgd) for step in result.steps))
        self.assertTrue(is_gamlss(result.model))

    def test_drop1_tgd_all_and_add1_tgd_all_return_multi_parameter_tables(self) -> None:
        fit = gamlss(
            formula="y ~ x1 + x2",
            sigma_formula="~ x1",
            family=GA(),
            data={
                "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64),
                "x2": jnp.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=jnp.float64),
            },
            method="RS",
        )
        newdata = {
            "y": np.array([1.1, 2.1, 3.4], dtype=np.float64),
            "x1": np.array([0.5, 2.5, 4.5], dtype=np.float64),
            "x2": np.array([1.2, 2.2, 3.2], dtype=np.float64),
        }
        drop_result = drop1_tgd_all(
            fit,
            newdata=newdata,
            scope={"mu": {"lower": "~ x1"}, "sigma": {"lower": "~ 1"}},
            parameters=("mu", "sigma"),
        )
        add_result = add1_tgd_all(
            fit,
            newdata=newdata,
            scope={"mu": {"upper": "~ x1 + x2"}, "sigma": {"upper": "~ x1 + x2"}},
            parameters=("mu", "sigma"),
        )
        self.assertEqual(drop_result.direction, "drop")
        self.assertEqual(add_result.direction, "add")
        self.assertGreaterEqual(len(drop_result.rows), 1)
        self.assertGreaterEqual(len(add_result.rows), 1)
        self.assertTrue(all(np.isfinite(row.tgd) for row in drop_result.rows))
        self.assertTrue(all(np.isfinite(row.tgd) for row in add_result.rows))

    def test_drop1_tgdp_and_add1_tgdp_alias_multi_parameter_tgd_tables(self) -> None:
        fit = gamlss(
            formula="y ~ x1 + x2",
            sigma_formula="~ x1",
            family=GA(),
            data={
                "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64),
                "x2": jnp.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=jnp.float64),
            },
            method="RS",
        )
        newdata = {
            "y": np.array([1.1, 2.1, 3.4], dtype=np.float64),
            "x1": np.array([0.5, 2.5, 4.5], dtype=np.float64),
            "x2": np.array([1.2, 2.2, 3.2], dtype=np.float64),
        }
        drop_alias = drop1_tgdp(
            fit,
            newdata=newdata,
            scope={"mu": {"lower": "~ x1"}, "sigma": {"lower": "~ 1"}},
            parameters=("mu", "sigma"),
        )
        add_alias = add1_tgdp(
            fit,
            newdata=newdata,
            scope={"mu": {"upper": "~ x1 + x2"}, "sigma": {"upper": "~ x1 + x2"}},
            parameters=("mu", "sigma"),
        )
        self.assertEqual(drop_alias.direction, "drop")
        self.assertEqual(add_alias.direction, "add")
        self.assertGreaterEqual(len(drop_alias.rows), 1)
        self.assertGreaterEqual(len(add_alias.rows), 1)
        self.assertTrue(all(np.isfinite(row.tgd) for row in drop_alias.rows))
        self.assertTrue(all(np.isfinite(row.tgd) for row in add_alias.rows))

    def test_step_tgd_all_rotates_across_mu_and_sigma(self) -> None:
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ 1",
            family=GA(),
            data={
                "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64),
                "x2": jnp.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=jnp.float64),
            },
            method="RS",
        )
        result = step_tgd_all(
            fit,
            newdata={
                "y": np.array([1.1, 2.1, 3.4], dtype=np.float64),
                "x1": np.array([0.5, 2.5, 4.5], dtype=np.float64),
                "x2": np.array([1.2, 2.2, 3.2], dtype=np.float64),
            },
            scope={
                "mu": {"upper": "~ x1 + x2"},
                "sigma": {"upper": "~ x1 + x2"},
            },
            parameters=("mu", "sigma"),
            direction="both",
            steps=3,
        )
        self.assertEqual(result.parameters, ("mu", "sigma"))
        self.assertEqual(result.direction, "both")
        self.assertGreaterEqual(len(result.steps), 1)
        self.assertEqual(result.steps[0].step, 0)
        self.assertTrue(all(np.isfinite(step.tgd) for step in result.steps))
        self.assertTrue(is_gamlss(result.model))

    def test_r_aligned_step_tgd_module_hosts_runtime_implementation(self) -> None:
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ 1",
            family=GA(),
            data={
                "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64),
                "x2": jnp.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5], dtype=jnp.float64),
            },
            method="RS",
        )
        newdata = {
            "y": np.array([1.1, 2.1, 3.4], dtype=np.float64),
            "x1": np.array([0.5, 2.5, 4.5], dtype=np.float64),
            "x2": np.array([1.2, 2.2, 3.2], dtype=np.float64),
        }
        extracted = step_tgd_module.extract_tgd_data(fit, newdata=newdata)
        self.assertIsInstance(extracted, step_tgd_module.ExtractTGDResult)

        scoped = step_tgd_module.drop1_tgd(fit, newdata=newdata, what="mu")
        self.assertIsInstance(scoped, step_tgd_module.TGDScopeResult)

        stepped = step_tgd_module.step_tgd(
            fit,
            newdata=newdata,
            scope={"lower": "~ 1", "upper": "~ x1 + x2"},
            what="mu",
            direction="both",
            steps=1,
        )
        self.assertIsInstance(stepped, step_tgd_module.StepTGDResult)

    def test_choose_dist_data_returns_gaic_matrix(self) -> None:
        result = choose_dist_data(
            self.model,
            k=(2.0, 3.84),
            type="realAll",
        )
        self.assertEqual(result.type, "realAll")
        self.assertEqual(result.penalties, (2.0, 3.84))
        self.assertEqual(result.matrix.shape, (len(result.families), 2))
        self.assertGreaterEqual(len(result.minima), 1)
        self.assertTrue(any(name in result.families for name in result.minima.values()))

    def test_choose_dist_data_supports_counts_type(self) -> None:
        data = {
            "y": jnp.array([0, 1, 1, 2, 3, 5], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            family=PO(),
            data=data,
            method="RS",
        )
        result = choose_dist_data(
            fit,
            k=(2.0,),
            type="counts",
        )
        self.assertEqual(result.type, "counts")
        self.assertEqual(result.matrix.shape[1], 1)
        self.assertIn(result.minima[2.0], {"PO", "GEOM", "NBI", "ZIP"})

    def test_get_order_returns_sorted_choose_dist_column(self) -> None:
        result = choose_dist_data(
            self.model,
            k=(2.0, 3.84),
            type="realAll",
        )
        ordered = get_order(result, column=1)
        self.assertEqual(ordered.column_index, 1)
        self.assertEqual(ordered.penalty, 2.0)
        self.assertEqual(len(ordered.rows), len(result.families))
        finite_values = [row.value for row in ordered.rows if np.isfinite(row.value)]
        non_finite_values = [row.value for row in ordered.rows if not np.isfinite(row.value)]
        self.assertTrue(
            all(finite_values[i] <= finite_values[i + 1] for i in range(len(finite_values) - 1))
        )
        if non_finite_values:
            self.assertTrue(all(not np.isfinite(value) for value in non_finite_values))
            self.assertTrue(all(np.isfinite(row.value) for row in ordered.rows[: len(finite_values)]))

    def test_get_order_supports_penalty_lookup(self) -> None:
        result = choose_dist_data(
            self.model,
            k=(2.0, 3.84),
            type="realAll",
        )
        ordered = get_order(result, column=3.84)
        self.assertEqual(ordered.column_index, 2)
        self.assertAlmostEqual(ordered.penalty, 3.84)
        self.assertEqual(len(ordered.rows), len(result.families))
        finite_values = [row.value for row in ordered.rows if np.isfinite(row.value)]
        self.assertGreaterEqual(len(finite_values), 1)
        self.assertTrue(all(np.isfinite(value) for value in finite_values))

    def test_choose_dist_pred_data_supports_rand_split(self) -> None:
        data = {
            "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ 1",
            family=GA(),
            data=data,
            method="RS",
        )
        result = choose_dist_pred_data(
            fit,
            type="realplus",
            rand=np.array([1, 1, 1, 2, 2, 2], dtype=np.int64),
        )
        self.assertEqual(result.type, "realplus")
        self.assertEqual(result.validation_size, 3)
        self.assertEqual(result.scores.shape[0], len(result.families))
        self.assertIn(result.best_family, result.families)
        self.assertTrue(np.isfinite(result.scores[np.isfinite(result.scores)]).all())

    def test_choose_dist_pred_data_supports_newdata_path(self) -> None:
        fit = gamlss(
            formula="y ~ 1",
            family=GA(),
            data={"y": jnp.array([0.8, 1.0, 1.4, 2.0], dtype=jnp.float64)},
            method="RS",
        )
        result = choose_dist_pred_data(
            fit,
            type="realplus",
            newdata={"y": np.array([2.8, 3.9], dtype=np.float64)},
        )
        self.assertEqual(result.validation_size, 2)
        self.assertEqual(result.scores.shape[0], len(result.families))
        self.assertIn(result.best_family, result.families)
        self.assertTrue(np.isfinite(result.scores[np.isfinite(result.scores)]).all())

    def test_choose_dist_pred_data_supports_counts_type(self) -> None:
        fit = gamlss(
            formula="y ~ 1",
            family=PO(),
            data={"y": jnp.array([0, 1, 1, 2, 3, 5], dtype=jnp.float64)},
            method="RS",
        )
        result = choose_dist_pred_data(
            fit,
            type="counts",
            rand=np.array([1, 1, 1, 2, 2, 2], dtype=np.int64),
        )
        self.assertEqual(result.type, "counts")
        self.assertEqual(result.validation_size, 3)
        self.assertEqual(result.scores.shape[0], len(result.families))
        self.assertIn(result.best_family, {"PO", "GEOM", "NBI", "ZIP"})

    def test_r_aligned_choose_dist_module_hosts_runtime_implementation(self) -> None:
        result = choose_dist_parallel_module.choose_dist_data(self.model, k=(2.0, 3.84), type="realAll")
        self.assertIsInstance(result, choose_dist_parallel_module.ChooseDistResult)

        ordered = choose_dist_parallel_module.get_order(result, column=1)
        self.assertIsInstance(ordered, choose_dist_parallel_module.ChooseDistOrderResult)

        fit = gamlss(
            formula="y ~ 1",
            family=GA(),
            data={"y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64)},
            method="RS",
        )
        pred = choose_dist_parallel_module.choose_dist_pred_data(
            fit,
            type="realplus",
            rand=np.array([1, 1, 1, 2, 2, 2], dtype=np.int64),
        )
        self.assertIsInstance(pred, choose_dist_parallel_module.ChooseDistPredResult)

    def test_get_pef_data_returns_partial_effect_curve_and_derivative(self) -> None:
        pef = get_pef_data(self.model, term="x1", what="mu", type="response", n_points=5, how="median")
        self.assertEqual(pef.term, "x1")
        self.assertEqual(pef.what, "mu")
        self.assertEqual(pef.type, "response")
        self.assertEqual(pef.x.shape, (5,))
        self.assertEqual(pef.effect.shape, (5,))
        self.assertEqual(pef.derivative.shape, (5,))
        self.assertIn("x2", pef.fixed_at)
        self.assertTrue(np.all(pef.x[:-1] <= pef.x[1:]))
        self.assertTrue(np.isfinite(pef.effect).all())
        self.assertTrue(np.isfinite(pef.derivative).all())

    def test_get_quantile_data_returns_sorted_quantile_curves(self) -> None:
        curves = get_quantile_data(self.model, xvar="x1", probabilities=(0.1, 0.5, 0.9), n_points=5, how="median")
        self.assertEqual(curves.family, "TEST")
        self.assertEqual(curves.xvar, "x1")
        self.assertEqual(curves.probabilities, (0.1, 0.5, 0.9))
        self.assertEqual(len(curves.entries), 3)
        first = curves.entries[0]
        self.assertEqual(first.probability, 0.1)
        self.assertEqual(first.x.shape, (5,))
        self.assertEqual(first.quantile.shape, (5,))
        self.assertTrue(np.all(first.x[:-1] <= first.x[1:]))
        self.assertIn("x2", curves.fixed_at)
        self.assertTrue(np.isfinite(first.quantile).all())

    def test_centiles_data_returns_curve_bundle_with_observed_points(self) -> None:
        result = centiles_data(self.model, xvar="x1", cent=(10.0, 50.0, 90.0), n_points=5, how="median")
        self.assertEqual(result.family, "TEST")
        self.assertEqual(result.xvar, "x1")
        self.assertEqual(result.centiles, (10.0, 50.0, 90.0))
        self.assertEqual(len(result.entries), 3)
        first = result.entries[0]
        self.assertEqual(first.centile, 10.0)
        self.assertEqual(first.x.shape, (5,))
        self.assertEqual(first.y.shape, (5,))
        self.assertTrue(np.all(first.x[:-1] <= first.x[1:]))
        self.assertIsNotNone(result.observed_x)
        self.assertIsNotNone(result.observed_y)
        self.assertEqual(np.asarray(result.observed_x).shape, (3,))
        self.assertEqual(np.asarray(result.observed_y).shape, (3,))

    def test_get_quantile_data_supports_jsu_family(self) -> None:
        data = {
            "y": jnp.array([-0.8, -0.2, 0.1, 0.7, 1.4, 2.1], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=JSU(),
            data=data,
            method="RS",
        )
        curves = get_quantile_data(fit, xvar="x1", probabilities=(0.1, 0.5, 0.9), n_points=7, how="median")
        self.assertEqual(curves.family, "JSU")
        self.assertEqual(len(curves.entries), 3)
        low = curves.entries[0].quantile
        med = curves.entries[1].quantile
        high = curves.entries[2].quantile
        self.assertEqual(low.shape, (7,))
        self.assertTrue(np.isfinite(low).all())
        self.assertTrue(np.isfinite(med).all())
        self.assertTrue(np.isfinite(high).all())
        self.assertTrue(np.all(low <= med))
        self.assertTrue(np.all(med <= high))

    def test_centiles_data_supports_jsu_family(self) -> None:
        data = {
            "y": jnp.array([-0.8, -0.2, 0.1, 0.7, 1.4, 2.1], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=JSU(),
            data=data,
            method="RS",
        )
        result = centiles_data(fit, xvar="x1", cent=(10.0, 50.0, 90.0), n_points=7, how="median")
        self.assertEqual(result.family, "JSU")
        self.assertEqual(result.centiles, (10.0, 50.0, 90.0))
        self.assertEqual(len(result.entries), 3)
        low = result.entries[0].y
        med = result.entries[1].y
        high = result.entries[2].y
        self.assertTrue(np.isfinite(low).all())
        self.assertTrue(np.isfinite(med).all())
        self.assertTrue(np.isfinite(high).all())
        self.assertTrue(np.all(low <= med))
        self.assertTrue(np.all(med <= high))

    def test_centiles_coverage_data_returns_saved_centile_table(self) -> None:
        result = centiles_coverage_data(
            self.model,
            xvar="x1",
            cent=(10.0, 50.0, 90.0),
            n_points=5,
            how="median",
        )
        self.assertEqual(result.family, "TEST")
        self.assertEqual(result.xvar, "x1")
        self.assertEqual(len(result.rows), 3)
        self.assertEqual(result.rows[0].centile, 10.0)
        self.assertTrue(all(0.0 <= row.percent_below <= 100.0 for row in result.rows))
        self.assertIsNotNone(result.observed_x)
        self.assertIsNotNone(result.observed_y)

    def test_centiles_coverage_data_supports_jsu_family(self) -> None:
        data = {
            "y": jnp.array([-0.8, -0.2, 0.1, 0.7, 1.4, 2.1], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=JSU(),
            data=data,
            method="RS",
        )
        result = centiles_coverage_data(
            fit,
            xvar="x1",
            cent=(10.0, 50.0, 90.0),
            n_points=7,
            how="median",
        )
        self.assertEqual(result.family, "JSU")
        self.assertEqual(len(result.rows), 3)
        self.assertTrue(all(0.0 <= row.percent_below <= 100.0 for row in result.rows))

    def test_centiles_fan_data_returns_symmetric_bands(self) -> None:
        result = centiles_fan_data(
            self.model,
            xvar="x1",
            cent=(10.0, 25.0, 50.0, 75.0, 90.0),
            n_points=5,
            how="median",
            include_observed=True,
        )
        self.assertEqual(result.family, "TEST")
        self.assertEqual(result.xvar, "x1")
        self.assertEqual(len(result.bands), 2)
        first = result.bands[0]
        self.assertEqual((first.lower_centile, first.upper_centile), (10.0, 90.0))
        self.assertEqual(first.color_index, 0)
        self.assertEqual(first.x.shape, (5,))
        self.assertEqual(first.lower.shape, (5,))
        self.assertEqual(first.upper.shape, (5,))
        self.assertTrue(np.all(first.lower <= first.upper))
        self.assertIsNotNone(result.median)
        self.assertEqual(result.median.centile, 50.0)
        self.assertEqual(result.median.x.shape, (5,))
        self.assertEqual(result.median.y.shape, (5,))
        self.assertIsNotNone(result.observed_x)
        self.assertIsNotNone(result.observed_y)

    def test_centiles_fan_data_supports_jsu_family(self) -> None:
        data = {
            "y": jnp.array([-0.8, -0.2, 0.1, 0.7, 1.4, 2.1], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=JSU(),
            data=data,
            method="RS",
        )
        result = centiles_fan_data(
            fit,
            xvar="x1",
            cent=(0.4, 10.0, 50.0, 90.0, 99.6),
            n_points=7,
            how="median",
            include_observed=False,
        )
        self.assertEqual(result.family, "JSU")
        self.assertEqual(len(result.bands), 2)
        self.assertIsNotNone(result.median)
        self.assertEqual(result.median.centile, 50.0)
        for band in result.bands:
            self.assertEqual(band.x.shape, (7,))
            self.assertTrue(np.all(np.isfinite(band.lower)))
            self.assertTrue(np.all(np.isfinite(band.upper)))
            self.assertTrue(np.all(band.lower <= band.upper))
        self.assertIsNone(result.observed_x)
        self.assertIsNone(result.observed_y)

    def test_centiles_split_data_returns_interval_panels(self) -> None:
        result = centiles_split_data(
            self.model,
            xvar="x1",
            xcut_points=(1.5,),
            cent=(10.0, 50.0, 90.0),
            n_points=4,
            how="median",
            include_observed=True,
        )
        self.assertEqual(result.family, "TEST")
        self.assertEqual(result.xvar, "x1")
        self.assertEqual(len(result.intervals), 2)
        self.assertEqual(len(result.panels), 2)
        first_panel = result.panels[0]
        self.assertEqual(first_panel.centiles, (10.0, 50.0, 90.0))
        self.assertEqual(len(first_panel.entries), 3)
        self.assertEqual(first_panel.entries[0].x.shape, (4,))
        self.assertTrue(np.all(first_panel.entries[0].x >= first_panel.interval[0]))
        self.assertTrue(np.all(first_panel.entries[0].x <= first_panel.interval[1]))
        self.assertIsNotNone(first_panel.observed_x)
        self.assertTrue(np.all(np.asarray(first_panel.observed_x) <= first_panel.interval[1]))

    def test_centiles_split_data_supports_jsu_family(self) -> None:
        data = {
            "y": jnp.array([-0.8, -0.2, 0.1, 0.7, 1.4, 2.1], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=JSU(),
            data=data,
            method="RS",
        )
        result = centiles_split_data(
            fit,
            xvar="x1",
            n_inter=3,
            cent=(10.0, 50.0, 90.0),
            n_points=5,
            how="median",
            include_observed=False,
        )
        self.assertEqual(result.family, "JSU")
        self.assertEqual(len(result.panels), 3)
        for panel in result.panels:
            self.assertEqual(len(panel.entries), 3)
            low = panel.entries[0].y
            med = panel.entries[1].y
            high = panel.entries[2].y
            self.assertTrue(np.all(np.isfinite(low)))
            self.assertTrue(np.all(np.isfinite(med)))
            self.assertTrue(np.all(np.isfinite(high)))
            self.assertTrue(np.all(low <= med))
            self.assertTrue(np.all(med <= high))
            self.assertIsNone(panel.observed_x)
            self.assertIsNone(panel.observed_y)

    def test_centiles_split_coverage_data_returns_interval_matrix(self) -> None:
        result = centiles_split_coverage_data(
            self.model,
            xvar="x1",
            xcut_points=(1.5,),
            cent=(10.0, 50.0, 90.0),
            n_points=4,
            how="median",
        )
        self.assertEqual(result.family, "TEST")
        self.assertEqual(result.xvar, "x1")
        self.assertEqual(result.centiles, (10.0, 50.0, 90.0))
        self.assertEqual(result.matrix.shape, (3, 2))
        self.assertEqual(len(result.intervals), 2)
        self.assertTrue(np.all(np.isfinite(result.matrix)))
        self.assertTrue(np.all(result.matrix >= 0.0))
        self.assertTrue(np.all(result.matrix <= 100.0))

    def test_centiles_split_coverage_data_supports_jsu_family(self) -> None:
        data = {
            "y": jnp.array([-0.8, -0.2, 0.1, 0.7, 1.4, 2.1], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=JSU(),
            data=data,
            method="RS",
        )
        result = centiles_split_coverage_data(
            fit,
            xvar="x1",
            n_inter=3,
            cent=(10.0, 50.0, 90.0),
            n_points=5,
            how="median",
        )
        self.assertEqual(result.family, "JSU")
        self.assertEqual(result.matrix.shape, (3, 3))
        self.assertEqual(len(result.intervals), 3)
        self.assertTrue(np.all(np.isfinite(result.matrix)))
        self.assertTrue(np.all(result.matrix >= 0.0))
        self.assertTrue(np.all(result.matrix <= 100.0))

    def test_centiles_comparison_data_returns_multi_model_bundle(self) -> None:
        second = replace(
            self.model,
            fitted_values={
                "mu": np.asarray(self.model.fitted_values["mu"], dtype=np.float64) + 0.15,
                "sigma": np.asarray(self.model.fitted_values["sigma"], dtype=np.float64),
            },
        )
        result = centiles_comparison_data(
            self.model,
            second,
            xvar="x1",
            cent=(10.0, 50.0, 90.0),
            n_points=5,
            how="median",
        )
        self.assertEqual(result.xvar, "x1")
        self.assertEqual(result.centiles, (10.0, 50.0, 90.0))
        self.assertEqual(len(result.models), 2)
        self.assertIsNotNone(result.observed_x)
        self.assertIsNotNone(result.observed_y)
        first = result.models[0]
        self.assertEqual(first.index, 1)
        self.assertEqual(first.family, "TEST")
        self.assertEqual(len(first.entries), 3)
        self.assertEqual(set(first.percent_below.keys()), {10.0, 50.0, 90.0})
        self.assertTrue(all(0.0 <= value <= 100.0 for value in first.percent_below.values()))

    def test_centiles_comparison_data_supports_real_families(self) -> None:
        data = {
            "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit_ga = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            family=GA(),
            data=data,
            method="RS",
        )
        fit_wei = gamlss(
            formula="y ~ x1 + x2",
            sigma_formula="~ x1",
            family=WEI(),
            data=data,
            method="RS",
        )
        result = centiles_comparison_data(
            fit_ga,
            fit_wei,
            xvar="x1",
            cent=(10.0, 50.0, 90.0),
            n_points=7,
            how="median",
        )
        self.assertEqual(len(result.models), 2)
        self.assertEqual(result.models[0].family, "GA")
        self.assertEqual(result.models[1].family, "WEI")
        for model in result.models:
            self.assertEqual(len(model.entries), 3)
            low = model.entries[0].y
            med = model.entries[1].y
            high = model.entries[2].y
            self.assertTrue(np.all(np.isfinite(low)))
            self.assertTrue(np.all(np.isfinite(med)))
            self.assertTrue(np.all(np.isfinite(high)))
            self.assertTrue(np.all(low <= med))
            self.assertTrue(np.all(med <= high))
            self.assertTrue(all(0.0 <= value <= 100.0 for value in model.percent_below.values()))

    def test_centiles_comparison_coverage_data_returns_model_matrix(self) -> None:
        second = replace(
            self.model,
            family=FamilyDefinition(
                name="TEST2",
                parameters=("mu",),
                g_dev_inc=lambda y, mu: (jnp.asarray(y, dtype=jnp.float64) - mu) ** 2,
            ),
            fitted_values={
                "mu": np.asarray(self.model.fitted_values["mu"], dtype=np.float64) + 0.15,
            },
        )
        result = centiles_comparison_coverage_data(
            self.model,
            second,
            xvar="x1",
            cent=(10.0, 50.0, 90.0),
            n_points=5,
            how="median",
        )
        self.assertEqual(result.xvar, "x1")
        self.assertEqual(result.centiles, (10.0, 50.0, 90.0))
        self.assertEqual(result.families, ("TEST", "TEST2"))
        self.assertEqual(result.matrix.shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(result.matrix)))
        self.assertTrue(np.all(result.matrix >= 0.0))
        self.assertTrue(np.all(result.matrix <= 100.0))

    def test_centiles_comparison_coverage_data_supports_real_families(self) -> None:
        data = {
            "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit_ga = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            family=GA(),
            data=data,
            method="RS",
        )
        fit_wei = gamlss(
            formula="y ~ x1 + x2",
            sigma_formula="~ x1",
            family=WEI(),
            data=data,
            method="RS",
        )
        result = centiles_comparison_coverage_data(
            fit_ga,
            fit_wei,
            xvar="x1",
            cent=(10.0, 50.0, 90.0),
            n_points=7,
            how="median",
        )
        self.assertEqual(result.families, ("GA", "WEI"))
        self.assertEqual(result.matrix.shape, (2, 3))
        self.assertTrue(np.all(np.isfinite(result.matrix)))
        self.assertTrue(np.all(result.matrix >= 0.0))
        self.assertTrue(np.all(result.matrix <= 100.0))

    def test_centile_pred_data_returns_centile_curves(self) -> None:
        result = centile_pred_data(
            self.model,
            type="centiles",
            xname="x1",
            xvalues=(1.0, 1.5, 2.0),
            cent=(10.0, 50.0, 90.0),
        )
        self.assertEqual(result.type, "centiles")
        self.assertEqual(result.xname, "x1")
        self.assertEqual(result.x.shape, (3,))
        self.assertEqual(len(result.entries), 3)
        low = result.entries[0].values
        med = result.entries[1].values
        high = result.entries[2].values
        self.assertTrue(np.all(np.isfinite(low)))
        self.assertTrue(np.all(low <= med))
        self.assertTrue(np.all(med <= high))

    def test_centile_pred_data_returns_standard_centiles_and_z_scores(self) -> None:
        centiles_result = centile_pred_data(
            self.model,
            type="standard-centiles",
            xname="x1",
            xvalues=(1.0, 1.5, 2.0),
            dev=(-1.0, 0.0, 1.0),
        )
        self.assertEqual(centiles_result.type, "standard-centiles")
        self.assertEqual([entry.label for entry in centiles_result.entries], ["-1", "0", "1"])
        z_result = centile_pred_data(
            self.model,
            type="z-scores",
            xname="x1",
            xvalues=(1.0, 1.5, 2.0),
            yval=(1.1, 1.4, 2.0),
        )
        self.assertEqual(z_result.type, "z-scores")
        self.assertEqual(len(z_result.entries), 1)
        self.assertEqual(z_result.entries[0].label, "z_scores")
        self.assertEqual(z_result.entries[0].values.shape, (3,))
        self.assertTrue(np.all(np.isfinite(z_result.entries[0].values)))

    def test_centile_pred_data_supports_real_family_and_power_transform(self) -> None:
        data = {
            "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
            "x1": jnp.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1 + x2",
            sigma_formula="~ x1",
            family=GA(),
            data=data,
            method="RS",
        )
        result = centile_pred_data(
            fit,
            type="centiles",
            xname="x1",
            xvalues=(1.0, 1.1, 1.2),
            power=2.0,
            cent=(10.0, 50.0, 90.0),
        )
        self.assertEqual(result.type, "centiles")
        self.assertTrue(np.all(result.x == np.array([1.0, 1.1, 1.2])))
        for entry in result.entries:
            self.assertTrue(np.all(np.isfinite(entry.values)))
            self.assertTrue(np.all(entry.values > 0.0))

    def test_centile_pred_data_supports_jsu_centiles(self) -> None:
        data = {
            "y": jnp.array([-0.8, -0.2, 0.1, 0.7, 1.4, 2.1], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=JSU(),
            data=data,
            method="RS",
        )
        result = centile_pred_data(
            fit,
            type="centiles",
            xname="x1",
            xvalues=(0.2, 0.8, 1.4),
            cent=(10.0, 50.0, 90.0),
        )
        self.assertEqual(result.type, "centiles")
        low = result.entries[0].values
        med = result.entries[1].values
        high = result.entries[2].values
        self.assertTrue(np.all(np.isfinite(low)))
        self.assertTrue(np.all(np.isfinite(med)))
        self.assertTrue(np.all(np.isfinite(high)))
        self.assertTrue(np.all(low <= med))
        self.assertTrue(np.all(med <= high))

    def test_centile_pred_data_supports_bct_standard_centiles(self) -> None:
        data = {
            "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=BCT(),
            data=data,
            method="RS",
        )
        result = centile_pred_data(
            fit,
            type="standard-centiles",
            xname="x1",
            xvalues=(0.2, 0.8, 1.4),
            dev=(-1.0, 0.0, 1.0),
        )
        self.assertEqual(result.type, "standard-centiles")
        self.assertEqual(len(result.entries), 3)
        low = result.entries[0].values
        med = result.entries[1].values
        high = result.entries[2].values
        self.assertTrue(np.all(low > 0.0))
        self.assertTrue(np.all(low <= med))
        self.assertTrue(np.all(med <= high))

    def test_centile_pred_data_supports_ga_z_scores(self) -> None:
        data = {
            "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
            "x1": jnp.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1 + x2",
            sigma_formula="~ x1",
            family=GA(),
            data=data,
            method="RS",
        )
        result = centile_pred_data(
            fit,
            type="z-scores",
            xname="x1",
            xvalues=(1.0, 1.1, 1.2),
            yval=(1.1, 1.4, 1.8),
        )
        self.assertEqual(result.type, "z-scores")
        self.assertEqual(len(result.entries), 1)
        scores = result.entries[0].values
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertEqual(scores.shape, (3,))

    def test_centile_pred_data_supports_ga_standard_centiles(self) -> None:
        data = {
            "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
            "x1": jnp.array([1.0, 1.2, 1.4, 1.6, 1.8, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1 + x2",
            sigma_formula="~ x1",
            family=GA(),
            data=data,
            method="RS",
        )
        result = centile_pred_data(
            fit,
            type="standard-centiles",
            xname="x1",
            xvalues=(1.0, 1.1, 1.2),
            dev=(-1.0, 0.0, 1.0),
        )
        self.assertEqual(result.type, "standard-centiles")
        low = result.entries[0].values
        med = result.entries[1].values
        high = result.entries[2].values
        self.assertTrue(np.all(low > 0.0))
        self.assertTrue(np.all(low <= med))
        self.assertTrue(np.all(med <= high))

    def test_centile_pred_data_supports_bccg_centiles(self) -> None:
        data = {
            "y": jnp.array([0.8, 1.1, 1.6, 2.2, 3.0, 4.1], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1"},
            family=BCCG(),
            data=data,
            method="RS",
        )
        result = centile_pred_data(
            fit,
            type="centiles",
            xname="x1",
            xvalues=(0.2, 0.8, 1.4),
            cent=(10.0, 50.0, 90.0),
        )
        self.assertEqual(result.type, "centiles")
        low = result.entries[0].values
        med = result.entries[1].values
        high = result.entries[2].values
        self.assertTrue(np.all(low > 0.0))
        self.assertTrue(np.all(low <= med))
        self.assertTrue(np.all(med <= high))

    def test_centile_pred_data_supports_bcpe_standard_centiles(self) -> None:
        data = {
            "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=BCPE(),
            data=data,
            method="RS",
        )
        result = centile_pred_data(
            fit,
            type="standard-centiles",
            xname="x1",
            xvalues=(0.2, 0.8, 1.4),
            dev=(-1.0, 0.0, 1.0),
        )
        self.assertEqual(result.type, "standard-centiles")
        low = result.entries[0].values
        med = result.entries[1].values
        high = result.entries[2].values
        self.assertTrue(np.all(np.isfinite(low)))
        self.assertTrue(np.all(low > 0.0))
        self.assertTrue(np.all(low <= med))
        self.assertTrue(np.all(med <= high))

    def test_centile_pred_data_supports_logno_centiles(self) -> None:
        data = {
            "y": jnp.array([1.1, 1.4, 1.8, 2.3, 3.0, 4.2], dtype=jnp.float64),
            "x1": jnp.array([0.2, 0.5, 0.8, 1.1, 1.4, 1.7], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1 + x2",
            sigma_formula="~ x1",
            family=LOGNO(),
            data=data,
            method="RS",
        )
        result = centile_pred_data(
            fit,
            type="centiles",
            xname="x1",
            xvalues=(0.3, 0.9, 1.5),
            cent=(10.0, 50.0, 90.0),
        )
        self.assertEqual(result.type, "centiles")
        low = result.entries[0].values
        med = result.entries[1].values
        high = result.entries[2].values
        self.assertTrue(np.all(low > 0.0))
        self.assertTrue(np.all(low <= med))
        self.assertTrue(np.all(med <= high))

    def test_centile_pred_data_supports_logno_standard_centiles(self) -> None:
        data = {
            "y": jnp.array([1.1, 1.4, 1.8, 2.3, 3.0, 4.2], dtype=jnp.float64),
            "x1": jnp.array([0.2, 0.5, 0.8, 1.1, 1.4, 1.7], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1 + x2",
            sigma_formula="~ x1",
            family=LOGNO(),
            data=data,
            method="RS",
        )
        result = centile_pred_data(
            fit,
            type="standard-centiles",
            xname="x1",
            xvalues=(0.3, 0.9, 1.5),
            dev=(-1.0, 0.0, 1.0),
        )
        self.assertEqual(result.type, "standard-centiles")
        low = result.entries[0].values
        med = result.entries[1].values
        high = result.entries[2].values
        self.assertTrue(np.all(low > 0.0))
        self.assertTrue(np.all(low <= med))
        self.assertTrue(np.all(med <= high))

    def test_centile_pred_data_supports_jsu_z_scores(self) -> None:
        data = {
            "y": jnp.array([-0.8, -0.2, 0.1, 0.7, 1.4, 2.1], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=JSU(),
            data=data,
            method="RS",
        )
        result = centile_pred_data(
            fit,
            type="z-scores",
            xname="x1",
            xvalues=(0.2, 0.8, 1.4),
            yval=(-0.1, 0.5, 1.0),
        )
        self.assertEqual(result.type, "z-scores")
        self.assertEqual(len(result.entries), 1)
        scores = result.entries[0].values
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertEqual(scores.shape, (3,))

    def test_centile_pred_data_supports_jsu_standard_centiles(self) -> None:
        data = {
            "y": jnp.array([-0.8, -0.2, 0.1, 0.7, 1.4, 2.1], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=JSU(),
            data=data,
            method="RS",
        )
        result = centile_pred_data(
            fit,
            type="standard-centiles",
            xname="x1",
            xvalues=(0.2, 0.8, 1.4),
            dev=(-1.0, 0.0, 1.0),
        )
        self.assertEqual(result.type, "standard-centiles")
        low = result.entries[0].values
        med = result.entries[1].values
        high = result.entries[2].values
        self.assertTrue(np.all(np.isfinite(low)))
        self.assertTrue(np.all(np.isfinite(med)))
        self.assertTrue(np.all(np.isfinite(high)))
        self.assertTrue(np.all(low <= med))
        self.assertTrue(np.all(med <= high))

    def test_centile_pred_data_supports_bccg_z_scores(self) -> None:
        data = {
            "y": jnp.array([0.8, 1.1, 1.6, 2.2, 3.0, 4.1], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1"},
            family=BCCG(),
            data=data,
            method="RS",
        )
        result = centile_pred_data(
            fit,
            type="z-scores",
            xname="x1",
            xvalues=(0.2, 0.8, 1.4),
            yval=(1.0, 1.8, 2.7),
        )
        self.assertEqual(result.type, "z-scores")
        scores = result.entries[0].values
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertEqual(scores.shape, (3,))

    def test_centile_pred_data_supports_bct_z_scores(self) -> None:
        data = {
            "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=BCT(),
            data=data,
            method="RS",
        )
        result = centile_pred_data(
            fit,
            type="z-scores",
            xname="x1",
            xvalues=(0.2, 0.8, 1.4),
            yval=(1.0, 1.8, 2.7),
        )
        self.assertEqual(result.type, "z-scores")
        scores = result.entries[0].values
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertEqual(scores.shape, (3,))

    def test_centile_pred_data_supports_bcpe_z_scores(self) -> None:
        data = {
            "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=BCPE(),
            data=data,
            method="RS",
        )
        result = centile_pred_data(
            fit,
            type="z-scores",
            xname="x1",
            xvalues=(0.2, 0.8, 1.4),
            yval=(1.0, 1.8, 2.7),
        )
        self.assertEqual(result.type, "z-scores")
        scores = result.entries[0].values
        self.assertTrue(np.all(np.isfinite(scores)))
        self.assertEqual(scores.shape, (3,))

    def test_centile_pred_data_marks_calibration_on_centiles_path(self) -> None:
        result = centile_pred_data(
            self.model,
            type="centiles",
            xname="x1",
            xvalues=(1.0, 1.5, 2.0),
            cent=(10.0, 50.0, 90.0),
            calibration=True,
        )
        self.assertEqual(result.type, "centiles")
        self.assertTrue(result.calibration_applied)
        self.assertEqual(len(result.entries), 3)
        self.assertTrue(all(entry.probability is not None for entry in result.entries))
        self.assertTrue(all(0.0 < float(entry.probability) < 1.0 for entry in result.entries))

    def test_centile_pred_data_supports_calibration_for_jsu(self) -> None:
        data = {
            "y": jnp.array([-0.8, -0.2, 0.1, 0.7, 1.4, 2.1], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=JSU(),
            data=data,
            method="RS",
        )
        result = centile_pred_data(
            fit,
            type="centiles",
            xname="x1",
            xvalues=(0.2, 0.8, 1.4),
            cent=(10.0, 50.0, 90.0),
            calibration=True,
        )
        self.assertTrue(result.calibration_applied)
        self.assertEqual(len(result.entries), 3)
        low = result.entries[0].values
        med = result.entries[1].values
        high = result.entries[2].values
        self.assertTrue(np.all(np.isfinite(low)))
        self.assertTrue(np.all(np.isfinite(med)))
        self.assertTrue(np.all(np.isfinite(high)))
        self.assertTrue(np.all(low <= med))
        self.assertTrue(np.all(med <= high))

    def test_centile_pred_data_supports_calibration_for_logno(self) -> None:
        data = {
            "y": jnp.array([1.1, 1.4, 1.8, 2.3, 3.0, 4.2], dtype=jnp.float64),
            "x1": jnp.array([0.2, 0.5, 0.8, 1.1, 1.4, 1.7], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1 + x2",
            sigma_formula="~ x1",
            family=LOGNO(),
            data=data,
            method="RS",
        )
        result = centile_pred_data(
            fit,
            type="centiles",
            xname="x1",
            xvalues=(0.3, 0.9, 1.5),
            cent=(10.0, 50.0, 90.0),
            calibration=True,
        )
        self.assertTrue(result.calibration_applied)
        self.assertEqual(len(result.entries), 3)
        low = result.entries[0].values
        med = result.entries[1].values
        high = result.entries[2].values
        self.assertTrue(np.all(low > 0.0))
        self.assertTrue(np.all(low <= med))
        self.assertTrue(np.all(med <= high))

    def test_centile_pred_data_rejects_calibration_for_z_scores(self) -> None:
        with self.assertRaisesRegex(ValueError, "not implemented for z-scores"):
            centile_pred_data(
                self.model,
                type="z-scores",
                xname="x1",
                xvalues=(1.0, 1.5, 2.0),
                yval=(1.1, 1.4, 2.0),
                calibration=True,
            )

    def test_get_quantile_data_supports_bct_family(self) -> None:
        data = {
            "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=BCT(),
            data=data,
            method="RS",
        )
        curves = get_quantile_data(fit, xvar="x1", probabilities=(0.1, 0.5, 0.9), n_points=7, how="last")
        self.assertEqual(curves.family, "BCT")
        self.assertEqual(len(curves.entries), 3)
        low = curves.entries[0].quantile
        med = curves.entries[1].quantile
        high = curves.entries[2].quantile
        self.assertEqual(low.shape, (7,))
        self.assertTrue(np.isfinite(low).all())
        self.assertTrue(np.isfinite(med).all())
        self.assertTrue(np.isfinite(high).all())
        self.assertTrue(np.all(low > 0.0))
        self.assertTrue(np.all(low <= med))
        self.assertTrue(np.all(med <= high))

    def test_get_quantile_data_supports_bccg_family(self) -> None:
        data = {
            "y": jnp.array([0.8, 1.1, 1.6, 2.2, 3.0, 4.1], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1"},
            family=BCCG(),
            data=data,
            method="RS",
        )
        curves = get_quantile_data(fit, xvar="x1", probabilities=(0.1, 0.5, 0.9), n_points=7, how="median")
        self.assertEqual(curves.family, "BCCG")
        self.assertEqual(len(curves.entries), 3)
        low = curves.entries[0].quantile
        med = curves.entries[1].quantile
        high = curves.entries[2].quantile
        self.assertEqual(low.shape, (7,))
        self.assertTrue(np.isfinite(low).all())
        self.assertTrue(np.isfinite(med).all())
        self.assertTrue(np.isfinite(high).all())
        self.assertTrue(np.all(low > 0.0))
        self.assertTrue(np.all(low <= med))
        self.assertTrue(np.all(med <= high))

    def test_get_quantile_data_supports_bcpe_family(self) -> None:
        data = {
            "y": jnp.array([0.9, 1.1, 1.5, 2.1, 2.9, 4.0], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=BCPE(),
            data=data,
            method="RS",
        )
        curves = get_quantile_data(fit, xvar="x1", probabilities=(0.1, 0.5, 0.9), n_points=7, how="last")
        self.assertEqual(curves.family, "BCPE")
        self.assertEqual(len(curves.entries), 3)
        low = curves.entries[0].quantile
        med = curves.entries[1].quantile
        high = curves.entries[2].quantile
        self.assertEqual(low.shape, (7,))
        self.assertTrue(np.isfinite(low).all())
        self.assertTrue(np.isfinite(med).all())
        self.assertTrue(np.isfinite(high).all())
        self.assertTrue(np.all(low > 0.0))
        self.assertTrue(np.all(low <= med))
        self.assertTrue(np.all(med <= high))

    def test_get_quantile_data_supports_logno_family(self) -> None:
        data = {
            "y": jnp.array([0.8, 1.1, 1.6, 2.4, 3.3, 4.5], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            family=LOGNO(),
            data=data,
            method="RS",
        )
        curves = get_quantile_data(fit, xvar="x1", probabilities=(0.1, 0.5, 0.9), n_points=7, how="median")
        self.assertEqual(curves.family, "LOGNO")
        self.assertEqual(len(curves.entries), 3)
        low = curves.entries[0].quantile
        med = curves.entries[1].quantile
        high = curves.entries[2].quantile
        self.assertEqual(low.shape, (7,))
        self.assertTrue(np.isfinite(low).all())
        self.assertTrue(np.isfinite(med).all())
        self.assertTrue(np.isfinite(high).all())
        self.assertTrue(np.all(low > 0.0))
        self.assertTrue(np.all(low <= med))
        self.assertTrue(np.all(med <= high))

    def test_get_quantile_data_supports_tf_family(self) -> None:
        data = {
            "y": jnp.array([1.2, 1.8, 2.7, 3.3, 4.6, 5.1], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 1.1, 0.9, 1.2, 1.0, 1.3], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1"},
            family=TF(),
            data=data,
            method="RS",
        )
        curves = get_quantile_data(fit, xvar="x1", probabilities=(0.1, 0.5, 0.9), n_points=7, how="last")
        self.assertEqual(curves.family, "TF")
        self.assertEqual(len(curves.entries), 3)
        low = curves.entries[0].quantile
        med = curves.entries[1].quantile
        high = curves.entries[2].quantile
        self.assertEqual(low.shape, (7,))
        self.assertTrue(np.isfinite(low).all())
        self.assertTrue(np.isfinite(med).all())
        self.assertTrue(np.isfinite(high).all())
        self.assertTrue(np.all(low <= med))
        self.assertTrue(np.all(med <= high))

    def test_get_quantile_data_supports_no_family(self) -> None:
        data = {
            "y": jnp.array([1.0, 1.6, 2.2, 3.0, 3.7, 4.5], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            family=NO(),
            data=data,
            method="RS",
        )
        curves = get_quantile_data(fit, xvar="x1", probabilities=(0.1, 0.5, 0.9), n_points=7, how="median")
        self.assertEqual(curves.family, "NO")
        low = curves.entries[0].quantile
        med = curves.entries[1].quantile
        high = curves.entries[2].quantile
        self.assertTrue(np.isfinite(low).all())
        self.assertTrue(np.isfinite(med).all())
        self.assertTrue(np.isfinite(high).all())
        self.assertTrue(np.all(low <= med))
        self.assertTrue(np.all(med <= high))

    def test_get_quantile_data_supports_ga_family(self) -> None:
        data = {
            "y": jnp.array([0.8, 1.1, 1.5, 2.0, 2.6, 3.4], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            family=GA(),
            data=data,
            method="RS",
        )
        curves = get_quantile_data(fit, xvar="x1", probabilities=(0.1, 0.5, 0.9), n_points=7, how="median")
        self.assertEqual(curves.family, "GA")
        low = curves.entries[0].quantile
        med = curves.entries[1].quantile
        high = curves.entries[2].quantile
        self.assertTrue(np.isfinite(low).all())
        self.assertTrue(np.isfinite(med).all())
        self.assertTrue(np.isfinite(high).all())
        self.assertTrue(np.all(low > 0.0))
        self.assertTrue(np.all(low <= med))
        self.assertTrue(np.all(med <= high))

    def test_get_quantile_data_supports_ig_family(self) -> None:
        data = {
            "y": jnp.array([0.9, 1.2, 1.7, 2.3, 3.1, 4.2], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            family=IG(),
            data=data,
            method="RS",
        )
        curves = get_quantile_data(fit, xvar="x1", probabilities=(0.1, 0.5, 0.9), n_points=7, how="last")
        self.assertEqual(curves.family, "IG")
        low = curves.entries[0].quantile
        med = curves.entries[1].quantile
        high = curves.entries[2].quantile
        self.assertTrue(np.isfinite(low).all())
        self.assertTrue(np.isfinite(med).all())
        self.assertTrue(np.isfinite(high).all())
        self.assertTrue(np.all(low > 0.0))
        self.assertTrue(np.all(low <= med))
        self.assertTrue(np.all(med <= high))

    def test_get_quantile_data_supports_wei_family(self) -> None:
        data = {
            "y": jnp.array([0.7, 1.0, 1.3, 1.9, 2.7, 3.8], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            family=WEI(),
            data=data,
            method="RS",
        )
        curves = get_quantile_data(fit, xvar="x1", probabilities=(0.1, 0.5, 0.9), n_points=7, how="median")
        self.assertEqual(curves.family, "WEI")
        low = curves.entries[0].quantile
        med = curves.entries[1].quantile
        high = curves.entries[2].quantile
        self.assertTrue(np.isfinite(low).all())
        self.assertTrue(np.isfinite(med).all())
        self.assertTrue(np.isfinite(high).all())
        self.assertTrue(np.all(low > 0.0))
        self.assertTrue(np.all(low <= med))
        self.assertTrue(np.all(med <= high))

    def test_plot2way_data_returns_joint_contribution_matrix(self) -> None:
        plot2 = plot2way_data(self.model, terms=("x1", "x2"), what="mu")
        self.assertEqual(plot2.what, "mu")
        self.assertEqual(plot2.terms, ("x1", "x2"))
        self.assertEqual(plot2.x_levels.shape, (3,))
        self.assertEqual(plot2.y_levels.shape, (3,))
        self.assertEqual(plot2.contribution.shape, (3, 3))
        self.assertTrue(np.isfinite(np.diag(plot2.contribution)).all())

    def test_q_stats_returns_interval_diagnostic_matrices(self) -> None:
        diag_model = GAMLSSModel(
            par=("mu",),
            family=self.model.family,
            df_fit=1.0,
            g_dev=5.0,
            n=8,
            y=jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=jnp.float64),
            fitted_values={"mu": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], dtype=jnp.float64)},
            coefficients={"mu": jnp.array([0.1], dtype=jnp.float64)},
            linear_predictors={"mu": jnp.array([0.0], dtype=jnp.float64)},
            residuals=jnp.array([-1.2, -0.6, -0.1, 0.0, 0.2, 0.5, 0.9, 1.4], dtype=jnp.float64),
            weights=jnp.ones(8, dtype=jnp.float64),
            type="Continuous",
            class_name="gamlss",
        )
        result = q_stats(diag_model, n_inter=4)
        self.assertEqual(len(result.interval_labels), 4)
        self.assertEqual(result.z_matrix.shape, (4, 6))
        self.assertEqual(result.q_matrix.shape, (4, 6))
        self.assertEqual(set(result.totals.keys()), {"Q_stats", "df", "p_value"})
        self.assertEqual(result.totals["Q_stats"].shape, (6,))

    def test_dropterm_and_addterm_return_structured_selection_tables(self) -> None:
        data = {
            "y": jnp.array([1.0, 2.0, 1.5, 3.2, 3.8, 5.1], dtype=jnp.float64),
            "x1": jnp.array([0.0, 1.0, 0.5, 1.5, 2.0, 2.5], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.5, 1.5, 1.0, 2.0, 2.5], dtype=jnp.float64),
        }
        fit = gamlss_ml(formula="y ~ x1 + x2", family=NO(), data=data)

        dropped = dropterm(fit, what="mu", k=2.0)
        self.assertEqual(dropped.what, "mu")
        self.assertEqual(dropped.direction, "drop")
        self.assertEqual(dropped.rows[0].term, "<none>")
        self.assertEqual(len(dropped.rows), 3)
        self.assertTrue(all(np.isfinite(row.criterion) for row in dropped.rows))
        self.assertTrue(any(row.term == "- x1" for row in dropped.rows[1:]))
        self.assertTrue(any(row.term == "- x2" for row in dropped.rows[1:]))

        base_fit = gamlss_ml(formula="y ~ x1", family=NO(), data=data)
        added = addterm(base_fit, scope=["x2"], what="mu", k=2.0)
        self.assertEqual(added.what, "mu")
        self.assertEqual(added.direction, "add")
        self.assertEqual(added.rows[0].term, "<none>")
        self.assertEqual(len(added.rows), 2)
        self.assertEqual(added.rows[1].term, "+ x2")
        self.assertGreaterEqual(added.rows[1].df_fit, added.rows[0].df_fit)
        self.assertTrue(np.isfinite(added.rows[1].criterion))

    def test_step_gaic_returns_search_path_and_final_model(self) -> None:
        data = {
            "y": jnp.array([1.0, 2.0, 1.5, 3.2, 3.8, 5.1], dtype=jnp.float64),
            "x1": jnp.array([0.0, 1.0, 0.5, 1.5, 2.0, 2.5], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.5, 1.5, 1.0, 2.0, 2.5], dtype=jnp.float64),
        }
        fit = gamlss_ml(formula="y ~ 1", family=NO(), data=data)
        result = step_gaic(
            fit,
            scope={"upper": "~ x1 + x2"},
            what="mu",
            direction="forward",
            k=2.0,
            steps=5,
        )
        self.assertEqual(result.what, "mu")
        self.assertEqual(result.direction, "forward")
        self.assertGreaterEqual(len(result.steps), 1)
        self.assertEqual(result.steps[0].change, "")
        self.assertTrue(np.isfinite(result.steps[0].criterion))
        self.assertTrue(is_gamlss(result.model))
        self.assertIn("step_gaic_path", result.model.additional_slots)
        self.assertTrue(all(np.isfinite(step.criterion) for step in result.steps))

    def test_r_aligned_step_gaic_modules_host_runtime_implementation(self) -> None:
        data = {
            "y": jnp.array([1.0, 2.0, 1.5, 3.2, 3.8, 5.1], dtype=jnp.float64),
            "x1": jnp.array([0.0, 1.0, 0.5, 1.5, 2.0, 2.5], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.5, 1.5, 1.0, 2.0, 2.5], dtype=jnp.float64),
        }
        fit = gamlss_ml(formula="y ~ x1 + x2", family=NO(), data=data)

        aic_result = step_gaic_a_module.extract_aic(fit, k=2.0)
        self.assertIsInstance(aic_result, step_gaic_a_module.ExtractAICResult)

        dropped = drop_add_step_gaic_module.dropterm(fit, what="mu", k=2.0)
        self.assertIsInstance(dropped, drop_add_step_gaic_module.ScopeSelectionResult)

        stepped = drop_add_step_gaic_module.step_gaic(
            gamlss_ml(formula="y ~ 1", family=NO(), data=data),
            scope={"upper": "~ x1 + x2"},
            what="mu",
            direction="forward",
            k=2.0,
            steps=2,
        )
        self.assertIsInstance(stepped, step_gaic_a_module.StepGAICResult)

    def test_sigma_formula_and_sigma_step_gaic_work_for_two_parameter_family(self) -> None:
        data = {
            "y": jnp.array([1.2, 1.8, 2.6, 3.1, 4.4, 5.2], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.6, 1.2, 0.9, 1.5, 1.1], dtype=jnp.float64),
        }
        fit = gamlss_ml(
            formula="y ~ x1",
            sigma_formula="~ x1",
            family=GA(),
            data=data,
        )
        self.assertIn("sigma", fit.formulas)
        self.assertEqual(fit.formulas["sigma"], "y ~ x1")
        self.assertIn("sigma", fit.design_matrices)
        self.assertEqual(np.asarray(fit.design_matrices["sigma"]).shape, (6, 2))
        self.assertEqual(np.asarray(fit.coefficients["sigma"]).shape[0], 2)

        sigma_added = addterm(fit, scope=["x2"], what="sigma", k=2.0)
        self.assertEqual(sigma_added.what, "sigma")
        self.assertEqual(sigma_added.rows[1].term, "+ x2")
        self.assertTrue(np.isfinite(sigma_added.rows[1].criterion))

        sigma_step = step_gaic(
            fit,
            scope={"upper": "~ x1 + x2"},
            what="sigma",
            direction="forward",
            k=2.0,
            steps=3,
        )
        self.assertEqual(sigma_step.what, "sigma")
        self.assertEqual(sigma_step.direction, "forward")
        self.assertTrue(is_gamlss(sigma_step.model))
        self.assertTrue(all(np.isfinite(step.criterion) for step in sigma_step.steps))

    def test_step_gaic_all_rotates_across_mu_and_sigma(self) -> None:
        data = {
            "y": jnp.array([1.2, 1.8, 2.6, 3.1, 4.4, 5.2], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.6, 1.2, 0.9, 1.5, 1.1], dtype=jnp.float64),
        }
        fit = gamlss_ml(
            formula="y ~ 1",
            sigma_formula="~ 1",
            family=GA(),
            data=data,
        )
        result = step_gaic_all(
            fit,
            scope={"mu": {"upper": "~ x1 + x2"}, "sigma": {"upper": "~ x1 + x2"}},
            parameters=("mu", "sigma"),
            direction="forward",
            k=2.0,
            steps=4,
        )
        self.assertEqual(result.parameters, ("mu", "sigma"))
        self.assertEqual(result.direction, "forward")
        self.assertGreaterEqual(len(result.steps), 1)
        self.assertTrue(is_gamlss(result.model))
        self.assertIn("step_gaic_all_path", result.model.additional_slots)
        self.assertTrue(all(np.isfinite(step.criterion) for step in result.steps))

    def test_dropterm_all_and_addterm_all_return_multi_parameter_tables(self) -> None:
        data = {
            "y": jnp.array([1.2, 1.8, 2.6, 3.1, 4.4, 5.2], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.6, 1.2, 0.9, 1.5, 1.1], dtype=jnp.float64),
        }
        fit = gamlss_ml(
            formula="y ~ x1 + x2",
            sigma_formula="~ x1 + x2",
            family=GA(),
            data=data,
        )
        dropped = dropterm_all(
            fit,
            scope={"mu": ["x2"], "sigma": ["x2"]},
            parameters=("mu", "sigma"),
            k=2.0,
        )
        self.assertEqual(dropped.direction, "drop")
        self.assertEqual(dropped.rows[0].term, "<none>")
        self.assertTrue(any(row.parameter == "mu" and row.term == "- x2" for row in dropped.rows[1:]))
        self.assertTrue(any(row.parameter == "sigma" and row.term == "- x2" for row in dropped.rows[1:]))

        base_fit = gamlss_ml(
            formula="y ~ x1",
            sigma_formula="~ x1",
            family=GA(),
            data=data,
        )
        added = addterm_all(
            base_fit,
            scope={"mu": ["x2"], "sigma": ["x2"]},
            parameters=("mu", "sigma"),
            k=2.0,
        )
        self.assertEqual(added.direction, "add")
        self.assertEqual(added.rows[0].term, "<none>")
        self.assertTrue(any(row.parameter == "mu" and row.term == "+ x2" for row in added.rows[1:]))
        self.assertTrue(any(row.parameter == "sigma" and row.term == "+ x2" for row in added.rows[1:]))

    def test_step_gaic_supports_nu_parameter_for_tf_family(self) -> None:
        data = {
            "y": jnp.array([1.2, 1.8, 2.7, 3.3, 4.6, 5.1], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 1.1, 0.9, 1.2, 1.0, 1.3], dtype=jnp.float64),
        }
        fit = gamlss_ml(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1"},
            family=TF(),
            data=data,
        )
        dropped = dropterm(fit, what="nu", k=2.0)
        self.assertEqual(dropped.what, "nu")
        self.assertTrue(any(row.term == "- x1" for row in dropped.rows[1:]))

        added = addterm(
            gamlss_ml(
                formula="y ~ x1",
                sigma_formula="~ x1",
                parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ 1"},
                family=TF(),
                data=data,
            ),
            scope=["x1"],
            what="nu",
            k=2.0,
        )
        self.assertEqual(added.what, "nu")
        self.assertTrue(any(row.term == "+ x1" for row in added.rows[1:]))

        stepped = step_gaic(
            gamlss_ml(
                formula="y ~ x1",
                sigma_formula="~ x1",
                parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ 1"},
                family=TF(),
                data=data,
            ),
            scope={"upper": "~ x1"},
            what="nu",
            direction="forward",
            k=2.0,
            steps=2,
        )
        self.assertEqual(stepped.what, "nu")
        self.assertTrue(is_gamlss(stepped.model))
        self.assertTrue(all(np.isfinite(step.criterion) for step in stepped.steps))

    def test_step_gaic_all_supports_tau_parameter_for_jsu_family(self) -> None:
        data = {
            "y": jnp.array([-0.8, -0.2, 0.1, 0.7, 1.4, 2.1], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit = gamlss_ml(
            formula="y ~ 1",
            sigma_formula="~ 1",
            parameter_formulas={"mu": "~ 1", "sigma": "~ 1", "nu": "~ 1", "tau": "~ 1"},
            family=JSU(),
            data=data,
        )
        dropped = dropterm_all(
            gamlss_ml(
                formula="y ~ x1",
                sigma_formula="~ x1",
                parameter_formulas={"mu": "~ x1", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
                family=JSU(),
                data=data,
            ),
            scope={"tau": ["x1"]},
            parameters=("tau",),
            k=2.0,
        )
        self.assertTrue(any(row.parameter == "tau" and row.term == "- x1" for row in dropped.rows[1:]))

        added = addterm_all(
            fit,
            scope={"tau": {"upper": "~ x1"}},
            parameters=("tau",),
            k=2.0,
        )
        self.assertTrue(any(row.parameter == "tau" and row.term == "+ x1" for row in added.rows[1:]))

        stepped = step_gaic_all(
            fit,
            scope={"tau": {"upper": "~ x1"}},
            parameters=("tau",),
            direction="forward",
            k=2.0,
            steps=2,
        )
        self.assertEqual(stepped.parameters, ("tau",))
        self.assertTrue(is_gamlss(stepped.model))
        self.assertTrue(all(np.isfinite(step.criterion) for step in stepped.steps))

    def test_rsq_returns_pseudo_r_squared_metrics(self) -> None:
        fit = gamlss_ml(
            formula="y ~ x1",
            family=PO(),
            data={
                "y": jnp.array([1.0, 2.0, 1.0, 3.0, 4.0], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 0.0, 1.0, 2.0], dtype=jnp.float64),
            },
        )
        both = rsq(fit, type="both")
        self.assertTrue(np.isfinite(both.cox_snell))
        self.assertTrue(np.isfinite(both.cragg_uhler))
        self.assertGreaterEqual(both.cox_snell, 0.0)
        self.assertGreaterEqual(both.cragg_uhler, 0.0)
        self.assertAlmostEqual(rsq(fit, type="Cox Snell"), both.cox_snell)
        self.assertAlmostEqual(rsq(fit, type="Cragg Uhler"), both.cragg_uhler)

    def test_likelihood_ratio_test_returns_structured_result(self) -> None:
        null_fit = gamlss_ml(
            formula="y ~ 1",
            family=PO(),
            data={
                "y": jnp.array([1.0, 2.0, 1.0, 3.0, 4.0], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 0.0, 1.0, 2.0], dtype=jnp.float64),
            },
        )
        alt_fit = gamlss_ml(
            formula="y ~ x1",
            family=PO(),
            data={
                "y": jnp.array([1.0, 2.0, 1.0, 3.0, 4.0], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 0.0, 1.0, 2.0], dtype=jnp.float64),
            },
        )
        result = likelihood_ratio_test(null_fit, alt_fit)
        self.assertGreaterEqual(result.chi_square, 0.0)
        self.assertGreaterEqual(result.df, 0.0)
        self.assertTrue(np.isfinite(result.p_value))
        self.assertEqual(result.null_df_fit, null_fit.df_fit)
        self.assertEqual(result.alternative_df_fit, alt_fit.df_fit)

    def test_vuong_clarke_test_returns_structured_result(self) -> None:
        model1 = gamlss_ml(
            formula="y ~ x1",
            family=PO(),
            data={
                "y": jnp.array([1.0, 2.0, 1.0, 3.0, 4.0], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 0.0, 1.0, 2.0], dtype=jnp.float64),
            },
        )
        model2 = gamlss_ml(
            formula="y ~ 1",
            family=PO(),
            data={
                "y": jnp.array([1.0, 2.0, 1.0, 3.0, 4.0], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 0.0, 1.0, 2.0], dtype=jnp.float64),
            },
        )
        result = vuong_clarke_test(model1, model2)
        self.assertTrue(np.isfinite(result.vuong_statistic))
        self.assertIn(result.vuong_preferred, {"model1", "model2", "tie"})
        self.assertGreaterEqual(result.clarke_b, 0)
        self.assertLessEqual(result.clarke_b, result.nobs)
        self.assertTrue(np.isfinite(result.clarke_p_value))
        self.assertIn(result.clarke_preferred, {"model1", "model2", "tie"})

    def test_compare_models_returns_structured_table(self) -> None:
        model1 = gamlss_ml(
            formula="y ~ 1",
            family=PO(),
            data={
                "y": jnp.array([1.0, 2.0, 1.0, 3.0, 4.0], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 0.0, 1.0, 2.0], dtype=jnp.float64),
            },
        )
        model2 = gamlss_ml(
            formula="y ~ x1",
            family=PO(),
            data={
                "y": jnp.array([1.0, 2.0, 1.0, 3.0, 4.0], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 0.0, 1.0, 2.0], dtype=jnp.float64),
            },
        )
        result = compare_models(model1, model2)
        self.assertEqual(len(result.rows), 2)
        self.assertEqual(result.rows[0].index, 1)
        self.assertEqual(result.rows[1].index, 2)
        self.assertEqual(result.rows[1].family, "PO")
        self.assertIsNotNone(result.rows[1].delta_df)
        self.assertIsNotNone(result.rows[1].delta_deviance)

    def test_extract_aic_returns_edf_and_aic(self) -> None:
        fit = gamlss_ml(
            formula="y ~ x1",
            family=PO(),
            data={
                "y": jnp.array([1.0, 2.0, 1.0, 3.0, 4.0], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 0.0, 1.0, 2.0], dtype=jnp.float64),
            },
        )
        result = extract_aic(fit, k=2.0, c=False)
        self.assertEqual(result.edf, fit.df_fit)
        self.assertTrue(np.isfinite(result.aic))

    def test_gaic_weights_returns_normalized_model_weights(self) -> None:
        model1 = gamlss_ml(
            formula="y ~ 1",
            family=PO(),
            data={
                "y": jnp.array([1.0, 2.0, 1.0, 3.0, 4.0], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 0.0, 1.0, 2.0], dtype=jnp.float64),
            },
        )
        model2 = gamlss_ml(
            formula="y ~ x1",
            family=PO(),
            data={
                "y": jnp.array([1.0, 2.0, 1.0, 3.0, 4.0], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 0.0, 1.0, 2.0], dtype=jnp.float64),
            },
        )
        result = gaic_weights(model1, model2, k=2.0)
        self.assertEqual(len(result.rows), 2)
        total = sum(row.weight for row in result.rows)
        self.assertAlmostEqual(total, 1.0)
        self.assertTrue(all(row.delta >= 0.0 for row in result.rows))

    def test_vcov_can_fall_back_to_numerical_hessian(self) -> None:
        fit = gamlss_ml(
            formula="y ~ x1",
            family=NO(),
            data={
                "y": jnp.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64),
            },
        )
        fit.additional_slots.pop("vcov", None)
        covariance = vcov(fit, type="vcov")
        self.assertEqual(covariance.shape[0], covariance.shape[1])
        self.assertFalse(np.isnan(covariance).all())
        se = vcov(fit, type="se")
        self.assertEqual(se.shape[0], covariance.shape[0])

    def test_confint_and_newdata_predict(self) -> None:
        mu_ci = confint(self.model, what="mu")
        self.assertEqual(mu_ci.shape, (3, 2))
        self.assertLess(mu_ci[0, 0], 0.2)
        self.assertGreater(mu_ci[0, 1], 0.2)

        all_ci = confint(self.model, what="all")
        self.assertIn("mu", all_ci)
        self.assertIn("sigma", all_ci)

        new_pred_link = predict(
            self.model,
            what="mu",
            type="link",
            newdata={
                "x1": jnp.array([4.0, 5.0], dtype=jnp.float64),
                "x2": jnp.array([40.0, 50.0], dtype=jnp.float64),
            },
        )
        np.testing.assert_allclose(np.asarray(new_pred_link), [16.8, 21.0])

        new_pred_terms = predict(
            self.model,
            what="mu",
            type="terms",
            terms=("x2",),
            newdata={
                "x1": jnp.array([4.0, 5.0], dtype=jnp.float64),
                "x2": jnp.array([40.0, 50.0], dtype=jnp.float64),
            },
        )
        np.testing.assert_allclose(np.asarray(new_pred_terms), [[16.0], [20.0]])

        new_pred_with_se = predict(
            self.model,
            what="mu",
            type="link",
            se_fit=True,
            newdata={
                "x1": jnp.array([4.0, 5.0], dtype=jnp.float64),
                "x2": jnp.array([40.0, 50.0], dtype=jnp.float64),
            },
        )
        self.assertIn("fit", new_pred_with_se)
        self.assertIn("se.fit", new_pred_with_se)
        self.assertEqual(np.asarray(new_pred_with_se["fit"]).shape, (2,))
        self.assertEqual(np.asarray(new_pred_with_se["se.fit"]).shape, (2,))

        sigma_pred = predict(
            self.model,
            what="sigma",
            type="response",
            newdata={
                "x1": jnp.array([4.0, 5.0], dtype=jnp.float64),
                "x2": jnp.array([40.0, 50.0], dtype=jnp.float64),
            },
        )
        self.assertEqual(np.asarray(sigma_pred).shape, (2,))
        self.assertTrue(np.all(np.asarray(sigma_pred) > 0.0))

        all_pred = predict(
            self.model,
            what="all",
            type="response",
            newdata={
                "x1": jnp.array([4.0, 5.0], dtype=jnp.float64),
                "x2": jnp.array([40.0, 50.0], dtype=jnp.float64),
            },
        )
        self.assertIn("mu", all_pred)
        self.assertIn("sigma", all_pred)

    def test_gaic_single_and_multi_model_outputs(self) -> None:
        other = GAMLSSModel(
            par=("mu",),
            family=self.model.family,
            df_fit=2.0,
            g_dev=9.0,
            n=20,
            y=self.model.y,
            fitted_values={"mu": jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64)},
            coefficients={"mu": jnp.array([0.3], dtype=jnp.float64)},
            linear_predictors={"mu": jnp.array([0.2, 0.2, 0.2], dtype=jnp.float64)},
        )
        self.assertEqual(gaic(self.model, k=2.0), 16.0)
        rows = gaic(self.model, other, k=2.0)
        self.assertEqual(rows[0]["model"], "model_2")
        self.assertEqual(rows[0]["AIC"], 13.0)

    def test_gaic_table_and_scaled_helpers(self) -> None:
        other = GAMLSSModel(
            par=("mu",),
            family=self.model.family,
            df_fit=2.0,
            g_dev=9.0,
            n=20,
            y=self.model.y,
            fitted_values={"mu": jnp.array([1.0, 1.0, 1.0], dtype=jnp.float64)},
            coefficients={"mu": jnp.array([0.3], dtype=jnp.float64)},
            linear_predictors={"mu": jnp.array([0.2, 0.2, 0.2], dtype=jnp.float64)},
        )
        table = gaic_table(self.model, other, k=(2.0, 4.0), text_to_show=("fit1", "fit2"))
        self.assertEqual(table[0]["model"], "fit1")
        self.assertEqual(table[1]["k=4.0"], 17.0)

        scaled = gaic_scaled(self.model, other, k=2.0, text_to_show=("fit1", "fit2"))
        self.assertEqual(scaled[0]["model"], "fit1")
        self.assertEqual(scaled[1]["scaled"], 1.0)

        matrix_scaled = gaic_scaled(((16.0, 30.0), (13.0, 20.0)), which=1)
        self.assertEqual(matrix_scaled[1]["scaled"], 1.0)

    def test_hat_wx_matches_linear_algebra_definition(self) -> None:
        weights = jnp.array([1.0, 4.0, 9.0], dtype=jnp.float64)
        x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float64)
        hat_diag = np.asarray(hat_wx(weights, x))
        self.assertEqual(hat_diag.shape, (3,))
        self.assertTrue(np.all(hat_diag >= 0.0))
        self.assertTrue(np.all(hat_diag <= 1.0 + 1e-6))
        self.assertEqual(np.asarray(hat_diag).dtype, np.float64)

    def test_formula_terms_model_matrix_and_model_frame(self) -> None:
        self.assertEqual(formula(self.model, "mu"), "y ~ x1 + x2")
        self.assertEqual(formula(self.model, "sigma"), "y ~ x1 + x2")
        self.assertEqual(terms(self.model, "mu")["term_labels"], ["x1", "x2"])
        np.testing.assert_allclose(
            np.asarray(model_matrix(self.model, "mu")),
            [[1.0, 1.0, 10.0], [1.0, 2.0, 20.0], [1.0, 3.0, 30.0]],
        )
        frame = model_frame(self.model, "mu")
        self.assertEqual(set(frame.keys()), {"y", "x1", "x2"})

    def test_lpred_terms_and_residuals_follow_stored_term_protocol(self) -> None:
        term_prediction = lpred(self.model, what="mu", type="terms")
        np.testing.assert_allclose(
            np.asarray(term_prediction),
            [[0.2, 0.3], [0.4, 0.5], [0.6, 0.7]],
        )
        term_subset = lpred(self.model, what="mu", type="terms", terms=("x2",), se_fit=True)
        np.testing.assert_allclose(np.asarray(term_subset["fit"]), [[0.3], [0.5], [0.7]])
        np.testing.assert_allclose(np.asarray(term_subset["se.fit"]), [[0.03], [0.05], [0.07]])
        self.assertEqual(term_subset["constant"], 0.15)

        simple = residuals(self.model, what="mu", type="simple")
        weighted = residuals(self.model, what="mu", type="weighted")
        partial = residuals(self.model, what="mu", type="partial", terms=("x2",))
        np.testing.assert_allclose(np.asarray(simple), [0.4, 0.5, 0.6])
        np.testing.assert_allclose(np.asarray(weighted), [0.8, 1.0, 1.2])
        np.testing.assert_allclose(np.asarray(partial), [0.7, 1.0, 1.3])

    def test_z_score_residuals_support_frequency_weights(self) -> None:
        weighted_model = GAMLSSModel(
            par=self.model.par,
            family=self.model.family,
            df_fit=self.model.df_fit,
            g_dev=self.model.g_dev,
            n=self.model.n,
            y=self.model.y,
            fitted_values=self.model.fitted_values,
            coefficients=self.model.coefficients,
            linear_predictors=self.model.linear_predictors,
            weights=jnp.array([1.0, 2.0, 1.0], dtype=jnp.float64),
            residuals=jnp.array([0.1, 0.2, 0.3], dtype=jnp.float64),
            type="Continuous",
            class_name="gamlss",
        )
        repeated = residuals(weighted_model, what="z-scores")
        np.testing.assert_allclose(np.asarray(repeated), [0.1, 0.2, 0.2, 0.3])

    def test_discrete_models_store_rqres_for_z_score_residuals(self) -> None:
        data = {
            "y": jnp.array([0.0, 1.0, 0.0, 2.0, 3.0], dtype=jnp.float64),
            "x1": jnp.array([0.0, 1.0, 0.5, 1.5, 2.0], dtype=jnp.float64),
        }
        fit = gamlss(formula="y ~ x1", family=PO(), data=data, method="RS")
        self.assertIsNotNone(fit.rqres)
        zres = residuals(fit, what="z-scores")
        self.assertEqual(np.asarray(zres).dtype, np.float64)
        self.assertEqual(np.asarray(zres).shape[0], data["y"].shape[0])
        self.assertTrue(np.isfinite(np.asarray(zres)).all())

    def test_weighted_discrete_residuals_use_stored_rqres_callable(self) -> None:
        data = {
            "y": jnp.array([0.0, 1.0, 2.0], dtype=jnp.float64),
            "x1": jnp.array([0.0, 1.0, 2.0], dtype=jnp.float64),
        }
        fit = gamlss(formula="y ~ x1", family=GEOM(), data=data, method="RS")
        fit.weights = jnp.array([2.0, 1.0, 2.0], dtype=jnp.float64)
        repeated = residuals(fit, what="z-scores")
        self.assertEqual(np.asarray(repeated).shape[0], 5)
        self.assertTrue(np.isfinite(np.asarray(repeated)).all())

    def test_numeric_deriv_returns_value_and_gradient(self) -> None:
        def expr(alpha: float, beta: float) -> jnp.ndarray:
            return jnp.array([alpha**2 + beta, alpha + beta**2], dtype=jnp.float64)

        value, gradient = numeric_deriv(expr, theta=("alpha", "beta"), rho={"alpha": 2.0, "beta": 3.0})
        np.testing.assert_allclose(np.asarray(value), [7.0, 11.0], rtol=1e-5, atol=1e-5)
        expected = np.array([[4.0, 1.0], [1.0, 6.0]])
        np.testing.assert_allclose(np.asarray(gradient), expected, rtol=5e-3, atol=5e-3)
        self.assertEqual(np.asarray(value).dtype, np.float64)
        self.assertEqual(np.asarray(gradient).dtype, np.float64)

    def test_deviance_increment_uses_family_parameter_order(self) -> None:
        increment = deviance_increment(self.model)
        np.testing.assert_allclose(np.asarray(increment), [0.0625, 0.0625, 0.0625])

        newdata_increment = deviance_increment(
            self.model, newdata={"y": jnp.array([2.5, 2.5, 2.5], dtype=jnp.float64)}
        )
        np.testing.assert_allclose(np.asarray(newdata_increment), [0.25, 0.0, 0.25])

    def test_invalid_parameter_raises_clear_error(self) -> None:
        with self.assertRaises(ValueError):
            fitted(self.model, "tau")

    def test_control_helpers_match_r_style_clamping(self) -> None:
        outer = gamlss_control(c_crit=-1.0, n_cyc=0, mu_step=2.0, gd_tol=-2.0, iter=-1)
        inner = glim_control(cc=-1.0, cyc=0, bf_cyc=0, bf_tol=-1.0)
        self.assertIsInstance(outer, GAMLSSControl)
        self.assertIsInstance(inner, GLIMControl)
        self.assertEqual(outer.c_crit, 0.001)
        self.assertEqual(outer.n_cyc, 20)
        self.assertEqual(outer.mu_step, 1.0)
        self.assertTrue(np.isinf(outer.gd_tol))
        self.assertEqual(inner.cc, 0.001)
        self.assertEqual(inner.cyc, 20)
        self.assertEqual(inner.bf_cyc, 30)
        self.assertEqual(inner.bf_tol, 0.001)

    def test_gamlss_ml_fits_initial_normal_model_in_float64(self) -> None:
        fit = gamlss_ml(
            formula="y ~ x1 + x2",
            family=NO(),
            data={
                "y": jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 2.0, 3.0], dtype=jnp.float64),
                "x2": jnp.array([1.0, 1.0, 1.0, 1.0], dtype=jnp.float64),
            },
        )
        self.assertTrue(is_gamlss(fit))
        self.assertEqual(fit.family.name, "NO")
        self.assertEqual(fit.par, ("mu", "sigma"))
        self.assertEqual(np.asarray(fit.coefficients["mu"]).dtype, np.float64)
        self.assertEqual(np.asarray(fit.fitted_values["mu"]).dtype, np.float64)
        self.assertEqual(model_matrix(fit, "mu").shape, (4, 3))
        self.assertEqual(formula(fit, "mu"), "y ~ x1 + x2")
        self.assertGreater(float(fit.coefficients["sigma"][0]), 0.0)
        self.assertTrue(fit.additional_slots["converged"])
        self.assertEqual(len(fit.additional_slots["deviance_history"]), 1)

    def test_parameter_formulas_mapping_supports_mu_sigma_and_is_stored(self) -> None:
        data = {
            "y": jnp.array([1.0, 2.0, 3.2, 4.1, 5.3], dtype=jnp.float64),
            "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.5, 1.5, 1.0, 2.0], dtype=jnp.float64),
        }
        fit_ml = gamlss_ml(
            formula="y ~ x1",
            sigma_formula="~ 1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1"},
            family=NO(),
            data=data,
        )
        self.assertEqual(formula(fit_ml, "mu"), "y ~ x1 + x2")
        self.assertEqual(formula(fit_ml, "sigma"), "y ~ x1")
        self.assertEqual(fit_ml.call["parameter_formulas"]["mu"], "y ~ x1 + x2")
        self.assertEqual(fit_ml.call["parameter_formulas"]["sigma"], "y ~ x1")
        self.assertEqual(model_matrix(fit_ml, "mu").shape, (5, 3))
        self.assertEqual(model_matrix(fit_ml, "sigma").shape, (5, 2))

        fit_rs = gamlss(
            formula="y ~ x1",
            sigma_formula="~ 1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1"},
            family=NO(),
            data=data,
            method="RS",
        )
        self.assertEqual(formula(fit_rs, "mu"), "y ~ x1 + x2")
        self.assertEqual(formula(fit_rs, "sigma"), "y ~ x1")
        self.assertEqual(fit_rs.call["parameter_formulas"]["mu"], "y ~ x1 + x2")
        self.assertEqual(fit_rs.call["parameter_formulas"]["sigma"], "y ~ x1")

    def test_parameter_formulas_reject_unknown_or_unsupported_parameters(self) -> None:
        data = {
            "y": jnp.array([1.0, 2.0, 3.0, 4.0], dtype=jnp.float64),
            "x1": jnp.array([0.0, 1.0, 2.0, 3.0], dtype=jnp.float64),
        }
        with self.assertRaises(ValueError):
            gamlss_ml(
                formula="y ~ x1",
                parameter_formulas={"omega": "~ x1"},
                family=PO(),
                data=data,
            )
        with self.assertRaises(ValueError):
            gamlss(
                formula="y ~ x1",
                parameter_formulas={"nu": "~ x1"},
                family=NO(),
                data=data,
                method="RS",
            )

    def test_no_family_exposes_link_and_derivative_protocol(self) -> None:
        family = NO()
        eta = jnp.array([0.0, 1.0], dtype=jnp.float64)
        mu = family.link_inverses["mu"](eta)
        sigma = family.link_inverses["sigma"](eta)
        np.testing.assert_allclose(np.asarray(mu), [0.0, 1.0])
        np.testing.assert_allclose(np.asarray(sigma), np.exp([0.0, 1.0]))
        np.testing.assert_allclose(np.asarray(family.link_derivatives["mu"](eta)), [1.0, 1.0])
        np.testing.assert_allclose(np.asarray(family.link_derivatives["sigma"](eta)), np.exp([0.0, 1.0]))

    def test_tf_family_and_resolution_work(self) -> None:
        family = resolve_family("TF")
        self.assertEqual(family.name, "TF")
        eta = jnp.array([0.0, 1.0], dtype=jnp.float64)
        mu = family.link_inverses["mu"](eta)
        sigma = family.link_inverses["sigma"](eta)
        nu = family.link_inverses["nu"](eta)
        np.testing.assert_allclose(np.asarray(mu), [0.0, 1.0])
        np.testing.assert_allclose(np.asarray(sigma), np.exp([0.0, 1.0]))
        np.testing.assert_allclose(np.asarray(nu), np.exp([0.0, 1.0]))

    def test_jsu_family_and_resolution_work(self) -> None:
        family = resolve_family("JSU")
        self.assertEqual(family.name, "JSU")
        eta = jnp.array([0.0, 1.0], dtype=jnp.float64)
        mu = family.link_inverses["mu"](eta)
        sigma = family.link_inverses["sigma"](eta)
        nu = family.link_inverses["nu"](eta)
        tau = family.link_inverses["tau"](eta)
        np.testing.assert_allclose(np.asarray(mu), [0.0, 1.0])
        np.testing.assert_allclose(np.asarray(sigma), np.exp([0.0, 1.0]))
        np.testing.assert_allclose(np.asarray(nu), [0.0, 1.0])
        np.testing.assert_allclose(np.asarray(tau), np.exp([0.0, 1.0]))

    def test_bccg_family_and_resolution_work(self) -> None:
        family = resolve_family("BCCG")
        self.assertEqual(family.name, "BCCG")
        eta = jnp.array([0.0, 1.0], dtype=jnp.float64)
        mu = family.link_inverses["mu"](eta)
        sigma = family.link_inverses["sigma"](eta)
        nu = family.link_inverses["nu"](eta)
        np.testing.assert_allclose(np.asarray(mu), np.exp([0.0, 1.0]))
        np.testing.assert_allclose(np.asarray(sigma), np.exp([0.0, 1.0]))
        np.testing.assert_allclose(np.asarray(nu), [0.0, 1.0])
        self.assertIsInstance(BCCG(), BoxCoxColeGreenFamily)

    def test_bct_family_and_resolution_work(self) -> None:
        family = resolve_family("BCT")
        self.assertEqual(family.name, "BCT")
        eta = jnp.array([0.0, 1.0], dtype=jnp.float64)
        mu = family.link_inverses["mu"](eta)
        sigma = family.link_inverses["sigma"](eta)
        nu = family.link_inverses["nu"](eta)
        tau = family.link_inverses["tau"](eta)
        np.testing.assert_allclose(np.asarray(mu), np.exp([0.0, 1.0]))
        np.testing.assert_allclose(np.asarray(sigma), np.exp([0.0, 1.0]))
        np.testing.assert_allclose(np.asarray(nu), [0.0, 1.0])
        self.assertTrue(np.all(np.asarray(tau) > 2.0))
        self.assertIsInstance(BCT(), BoxCoxTFamily)

    def test_bcpe_family_and_resolution_work(self) -> None:
        family = resolve_family("BCPE")
        self.assertEqual(family.name, "BCPE")
        eta = jnp.array([0.0, 1.0], dtype=jnp.float64)
        mu = family.link_inverses["mu"](eta)
        sigma = family.link_inverses["sigma"](eta)
        nu = family.link_inverses["nu"](eta)
        tau = family.link_inverses["tau"](eta)
        np.testing.assert_allclose(np.asarray(mu), np.exp([0.0, 1.0]))
        np.testing.assert_allclose(np.asarray(sigma), np.exp([0.0, 1.0]))
        np.testing.assert_allclose(np.asarray(nu), [0.0, 1.0])
        self.assertTrue(np.all(np.asarray(tau) > 0.0))
        self.assertIsInstance(BCPE(), BoxCoxPowerExponentialFamily)

    def test_gamlss_ml_and_gamlss_support_bccg_family(self) -> None:
        data = {
            "y": jnp.array([0.8, 1.1, 1.6, 2.2, 3.0, 4.1], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit_ml = gamlss_ml(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1"},
            family=BCCG(),
            data=data,
        )
        self.assertEqual(fit_ml.par, ("mu", "sigma", "nu"))
        self.assertIn("nu", fit_ml.fitted_values)
        self.assertEqual(formula(fit_ml, "nu"), "y ~ x1")
        self.assertEqual(np.asarray(model_matrix(fit_ml, "nu")).shape, (6, 2))
        self.assertTrue(np.all(np.asarray(fitted(fit_ml, "mu")) > 0.0))
        self.assertTrue(np.all(np.asarray(fitted(fit_ml, "sigma")) > 0.0))

        fit_rs = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1"},
            family=BCCG(),
            data=data,
            method="RS",
        )
        self.assertEqual(fit_rs.par, ("mu", "sigma", "nu"))
        self.assertIn("nu", fit_rs.working_vectors)
        self.assertIn("nu", fit_rs.iterative_weights)
        prediction = predict_all(fit_rs, type="response", output="data.frame")
        self.assertIn("nu", prediction)
        self.assertEqual(np.asarray(prediction["nu"]).shape, (6,))

    def test_gamlss_ml_and_gamlss_support_bct_family(self) -> None:
        data = {
            "y": jnp.array([0.8, 1.0, 1.4, 2.0, 2.8, 3.9], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit_ml = gamlss_ml(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=BCT(),
            data=data,
        )
        self.assertEqual(fit_ml.par, ("mu", "sigma", "nu", "tau"))
        self.assertIn("tau", fit_ml.fitted_values)
        self.assertEqual(formula(fit_ml, "tau"), "y ~ x1")
        self.assertEqual(np.asarray(model_matrix(fit_ml, "tau")).shape, (6, 2))
        self.assertTrue(np.all(np.asarray(fitted(fit_ml, "mu")) > 0.0))
        self.assertTrue(np.all(np.asarray(fitted(fit_ml, "sigma")) > 0.0))
        self.assertTrue(np.all(np.asarray(fitted(fit_ml, "tau")) > 2.0))

        fit_rs = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=BCT(),
            data=data,
            method="RS",
        )
        self.assertEqual(fit_rs.par, ("mu", "sigma", "nu", "tau"))
        self.assertIn("tau", fit_rs.working_vectors)
        self.assertIn("tau", fit_rs.iterative_weights)
        prediction = predict_all(fit_rs, type="response", output="data.frame")
        self.assertIn("tau", prediction)
        self.assertEqual(np.asarray(prediction["tau"]).shape, (6,))

        covariance = vcov(fit_rs, type="vcov")
        self.assertEqual(covariance.shape[0], covariance.shape[1])
        self.assertEqual(covariance.shape[0], sum(np.asarray(coef(fit_rs, parameter)).size for parameter in fit_rs.par))
        self.assertTrue(np.isfinite(np.diag(covariance)).all())

        summary_result = summary(fit_rs)
        self.assertIn("nu", summary_result.coefficients)
        self.assertIn("tau", summary_result.coefficients)
        self.assertEqual(np.asarray(summary_result.coefficients["tau"]["estimate"]).shape, (2,))
        self.assertEqual(np.asarray(summary_result.coefficients["tau"]["std_error"]).shape, (2,))

        confidence = confint(fit_rs, what="all")
        self.assertIn("tau", confidence)
        self.assertEqual(np.asarray(confidence["tau"]).shape, (2, 2))

        prediction_se = predict_all(fit_rs, type="link", output="data.frame", se_fit=True)
        self.assertIn("nu_se", prediction_se)
        self.assertIn("tau_se", prediction_se)
        self.assertEqual(np.asarray(prediction_se["tau"]).shape, (6,))
        self.assertEqual(np.asarray(prediction_se["tau_se"]).shape, (6,))

    def test_gamlss_ml_and_gamlss_support_bcpe_family(self) -> None:
        data = {
            "y": jnp.array([0.9, 1.1, 1.5, 2.1, 2.9, 4.0], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit_ml = gamlss_ml(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=BCPE(),
            data=data,
        )
        self.assertEqual(fit_ml.par, ("mu", "sigma", "nu", "tau"))
        self.assertIn("tau", fit_ml.fitted_values)
        self.assertEqual(formula(fit_ml, "tau"), "y ~ x1")
        self.assertEqual(np.asarray(model_matrix(fit_ml, "tau")).shape, (6, 2))
        self.assertTrue(np.all(np.asarray(fitted(fit_ml, "mu")) > 0.0))
        self.assertTrue(np.all(np.asarray(fitted(fit_ml, "sigma")) > 0.0))
        self.assertTrue(np.all(np.asarray(fitted(fit_ml, "tau")) > 0.0))

        fit_rs = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=BCPE(),
            data=data,
            method="RS",
        )
        self.assertEqual(fit_rs.par, ("mu", "sigma", "nu", "tau"))
        self.assertIn("tau", fit_rs.working_vectors)
        self.assertIn("tau", fit_rs.iterative_weights)
        prediction = predict_all(fit_rs, type="response", output="data.frame")
        self.assertIn("tau", prediction)
        self.assertEqual(np.asarray(prediction["tau"]).shape, (6,))

        covariance = vcov(fit_rs, type="vcov")
        self.assertEqual(covariance.shape[0], covariance.shape[1])
        self.assertEqual(covariance.shape[0], sum(np.asarray(coef(fit_rs, parameter)).size for parameter in fit_rs.par))
        self.assertTrue(np.isfinite(np.diag(covariance)).all())

        summary_result = summary(fit_rs)
        self.assertIn("nu", summary_result.coefficients)
        self.assertIn("tau", summary_result.coefficients)
        self.assertEqual(np.asarray(summary_result.coefficients["tau"]["estimate"]).shape, (2,))
        self.assertEqual(np.asarray(summary_result.coefficients["tau"]["std_error"]).shape, (2,))

        confidence = confint(fit_rs, what="all")
        self.assertIn("tau", confidence)
        self.assertEqual(np.asarray(confidence["tau"]).shape, (2, 2))

        prediction_se = predict_all(fit_rs, type="link", output="data.frame", se_fit=True)
        self.assertIn("nu_se", prediction_se)
        self.assertIn("tau_se", prediction_se)
        self.assertEqual(np.asarray(prediction_se["tau"]).shape, (6,))
        self.assertEqual(np.asarray(prediction_se["tau_se"]).shape, (6,))

    def test_gamlss_ml_and_gamlss_support_jsu_family(self) -> None:
        data = {
            "y": jnp.array([-0.8, -0.2, 0.1, 0.7, 1.4, 2.1], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }
        fit_ml = gamlss_ml(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=JSU(),
            data=data,
        )
        self.assertEqual(fit_ml.par, ("mu", "sigma", "nu", "tau"))
        self.assertIn("tau", fit_ml.fitted_values)
        self.assertEqual(formula(fit_ml, "tau"), "y ~ x1")
        self.assertEqual(np.asarray(model_matrix(fit_ml, "tau")).shape, (6, 2))
        self.assertTrue(np.all(np.asarray(fitted(fit_ml, "tau")) > 0.0))

        fit_rs = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=JSU(),
            data=data,
            method="RS",
        )
        self.assertEqual(fit_rs.par, ("mu", "sigma", "nu", "tau"))
        self.assertIn("tau", fit_rs.working_vectors)
        self.assertIn("tau", fit_rs.iterative_weights)
        prediction = predict_all(fit_rs, type="response", output="data.frame")
        self.assertIn("tau", prediction)
        self.assertEqual(np.asarray(prediction["tau"]).shape, (6,))

        covariance = vcov(fit_rs, type="vcov")
        self.assertEqual(covariance.shape[0], covariance.shape[1])
        self.assertEqual(covariance.shape[0], sum(np.asarray(coef(fit_rs, parameter)).size for parameter in fit_rs.par))
        self.assertTrue(np.isfinite(np.diag(covariance)).all())

        summary_result = summary(fit_rs)
        self.assertIn("nu", summary_result.coefficients)
        self.assertIn("tau", summary_result.coefficients)
        self.assertEqual(np.asarray(summary_result.coefficients["tau"]["estimate"]).shape, (2,))
        self.assertEqual(np.asarray(summary_result.coefficients["tau"]["std_error"]).shape, (2,))

        confidence = confint(fit_rs, what="all")
        self.assertIn("tau", confidence)
        self.assertEqual(np.asarray(confidence["tau"]).shape, (2, 2))

        prediction_se = predict_all(fit_rs, type="link", output="data.frame", se_fit=True)
        self.assertIn("nu_se", prediction_se)
        self.assertIn("tau_se", prediction_se)
        self.assertEqual(np.asarray(prediction_se["tau"]).shape, (6,))
        self.assertEqual(np.asarray(prediction_se["tau_se"]).shape, (6,))

    def test_gamlss_ml_and_gamlss_support_tf_family_with_nu(self) -> None:
        data = {
            "y": jnp.array([1.2, 1.8, 2.7, 3.3, 4.6, 5.1], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=jnp.float64),
            "x2": jnp.array([1.0, 1.1, 0.9, 1.2, 1.0, 1.3], dtype=jnp.float64),
        }
        fit_ml = gamlss_ml(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ 1"},
            family=TF(),
            data=data,
        )
        self.assertEqual(fit_ml.par, ("mu", "sigma", "nu"))
        self.assertIn("nu", fit_ml.fitted_values)
        self.assertIn("nu", fit_ml.coefficients)
        self.assertIn("nu", fit_ml.formulas)
        self.assertEqual(formula(fit_ml, "nu"), "y ~ 1")
        self.assertTrue(np.all(np.asarray(fitted(fit_ml, "nu")) > 2.0))
        self.assertEqual(np.asarray(model_matrix(fit_ml, "nu")).shape, (6, 1))

        fit_rs = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1"},
            family=TF(),
            data=data,
            method="RS",
        )
        self.assertEqual(fit_rs.par, ("mu", "sigma", "nu"))
        self.assertIn("nu", fit_rs.fitted_values)
        self.assertEqual(formula(fit_rs, "nu"), "y ~ x1")
        self.assertEqual(np.asarray(model_matrix(fit_rs, "nu")).shape, (6, 2))
        self.assertEqual(np.asarray(coef(fit_rs, "nu")).shape, (2,))
        self.assertIn("nu", fit_rs.working_vectors)
        self.assertIn("nu", fit_rs.iterative_weights)
        self.assertGreaterEqual(len(fit_rs.additional_slots["deviance_history"]), 2)
        prediction = predict_all(fit_rs, type="response", output="data.frame")
        self.assertIn("nu", prediction)
        self.assertEqual(np.asarray(prediction["nu"]).shape, (6,))

    def test_gamlss_and_gamlss_ml_support_generic_tau_path(self) -> None:
        def g_dev_inc(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, nu: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
            y = jnp.asarray(y, dtype=jnp.float64)
            mu = jnp.asarray(mu, dtype=jnp.float64)
            sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), jnp.finfo(jnp.float64).eps)
            nu = jnp.asarray(nu, dtype=jnp.float64)
            tau = jnp.maximum(jnp.asarray(tau, dtype=jnp.float64), jnp.finfo(jnp.float64).eps)
            return (
                jnp.square((y - mu) / sigma)
                + 2.0 * jnp.log(sigma)
                + jnp.square(nu)
                + jnp.square(tau - 1.0)
            )

        def dldm(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, nu: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
            y = jnp.asarray(y, dtype=jnp.float64)
            mu = jnp.asarray(mu, dtype=jnp.float64)
            sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), jnp.finfo(jnp.float64).eps)
            return (y - mu) / jnp.square(sigma)

        def d2ldm2(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, nu: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
            sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), jnp.finfo(jnp.float64).eps)
            return -1.0 / jnp.square(sigma)

        def dlds(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, nu: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
            y = jnp.asarray(y, dtype=jnp.float64)
            mu = jnp.asarray(mu, dtype=jnp.float64)
            sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), jnp.finfo(jnp.float64).eps)
            return -1.0 / sigma + jnp.square(y - mu) / jnp.power(sigma, 3.0)

        def d2lds2(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, nu: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
            y = jnp.asarray(y, dtype=jnp.float64)
            mu = jnp.asarray(mu, dtype=jnp.float64)
            sigma = jnp.maximum(jnp.asarray(sigma, dtype=jnp.float64), jnp.finfo(jnp.float64).eps)
            return 1.0 / jnp.square(sigma) - 3.0 * jnp.square(y - mu) / jnp.power(sigma, 4.0)

        def dldnu(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, nu: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
            nu = jnp.asarray(nu, dtype=jnp.float64)
            return -nu

        def d2ldnu2(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, nu: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
            return -jnp.ones_like(jnp.asarray(nu, dtype=jnp.float64))

        def dldtau(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, nu: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
            tau = jnp.maximum(jnp.asarray(tau, dtype=jnp.float64), jnp.finfo(jnp.float64).eps)
            return -(tau - 1.0)

        def d2ldtau2(y: jnp.ndarray, mu: jnp.ndarray, sigma: jnp.ndarray, nu: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
            return -jnp.ones_like(jnp.asarray(tau, dtype=jnp.float64))

        four_parameter_family = FamilyDefinition(
            name="TEST4",
            parameters=("mu", "sigma", "nu", "tau"),
            g_dev_inc=g_dev_inc,
            type="Continuous",
            links={"mu": "identity", "sigma": "log", "nu": "identity", "tau": "log"},
            link_functions={
                "mu": lambda x: jnp.asarray(x, dtype=jnp.float64),
                "sigma": lambda x: jnp.log(jnp.maximum(jnp.asarray(x, dtype=jnp.float64), jnp.finfo(jnp.float64).eps)),
                "nu": lambda x: jnp.asarray(x, dtype=jnp.float64),
                "tau": lambda x: jnp.log(jnp.maximum(jnp.asarray(x, dtype=jnp.float64), jnp.finfo(jnp.float64).eps)),
            },
            link_inverses={
                "mu": lambda eta: jnp.asarray(eta, dtype=jnp.float64),
                "sigma": lambda eta: jnp.exp(jnp.asarray(eta, dtype=jnp.float64)),
                "nu": lambda eta: jnp.asarray(eta, dtype=jnp.float64),
                "tau": lambda eta: jnp.exp(jnp.asarray(eta, dtype=jnp.float64)),
            },
            link_derivatives={
                "mu": lambda eta: jnp.ones_like(jnp.asarray(eta, dtype=jnp.float64)),
                "sigma": lambda eta: jnp.exp(jnp.asarray(eta, dtype=jnp.float64)),
                "nu": lambda eta: jnp.ones_like(jnp.asarray(eta, dtype=jnp.float64)),
                "tau": lambda eta: jnp.exp(jnp.asarray(eta, dtype=jnp.float64)),
            },
            score_functions={"mu": dldm, "sigma": dlds, "nu": dldnu, "tau": dldtau},
            hessian_functions={"mu": d2ldm2, "sigma": d2lds2, "nu": d2ldnu2, "tau": d2ldtau2},
        )

        data = {
            "y": jnp.array([1.0, 1.6, 2.4, 3.5, 4.1, 5.0], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=jnp.float64),
            "x2": jnp.array([1.0, 0.9, 1.1, 1.0, 1.2, 1.1], dtype=jnp.float64),
        }

        fit_ml = gamlss_ml(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=four_parameter_family,
            data=data,
        )
        self.assertEqual(fit_ml.par, ("mu", "sigma", "nu", "tau"))
        self.assertIn("tau", fit_ml.fitted_values)
        self.assertEqual(formula(fit_ml, "tau"), "y ~ x1")
        self.assertEqual(np.asarray(model_matrix(fit_ml, "tau")).shape, (6, 2))

        fit_rs = gamlss(
            formula="y ~ x1",
            sigma_formula="~ x1",
            parameter_formulas={"mu": "~ x1 + x2", "sigma": "~ x1", "nu": "~ x1", "tau": "~ x1"},
            family=four_parameter_family,
            data=data,
            method="RS",
        )
        self.assertEqual(fit_rs.par, ("mu", "sigma", "nu", "tau"))
        self.assertIn("tau", fit_rs.fitted_values)
        self.assertIn("tau", fit_rs.working_vectors)
        self.assertIn("tau", fit_rs.iterative_weights)
        self.assertEqual(np.asarray(coef(fit_rs, "tau")).shape, (2,))
        self.assertEqual(np.asarray(model_matrix(fit_rs, "tau")).shape, (6, 2))
        prediction = predict_all(fit_rs, type="response", output="data.frame")
        self.assertIn("tau", prediction)
        self.assertEqual(np.asarray(prediction["tau"]).shape, (6,))

    def test_po_family_and_resolution_work(self) -> None:
        family = resolve_family("PO")
        self.assertEqual(family.name, "PO")
        eta = jnp.array([0.0, 1.0], dtype=jnp.float64)
        mu = family.link_inverses["mu"](eta)
        np.testing.assert_allclose(np.asarray(mu), np.exp([0.0, 1.0]))
        self.assertEqual(family.parameters, ("mu",))
        direct = PO()
        self.assertEqual(direct.name, "PO")

    def test_geom_family_and_resolution_work(self) -> None:
        family = resolve_family("GEOM")
        self.assertEqual(family.name, "GEOM")
        eta = jnp.array([0.0, 1.0], dtype=jnp.float64)
        mu = family.link_inverses["mu"](eta)
        np.testing.assert_allclose(np.asarray(mu), np.exp([0.0, 1.0]))
        self.assertEqual(family.parameters, ("mu",))
        direct = GEOM()
        self.assertIsInstance(direct, GeometricFamily)
        self.assertEqual(direct.name, "GEOM")

    def test_zip_family_and_resolution_work(self) -> None:
        family = resolve_family("ZIP")
        self.assertEqual(family.name, "ZIP")
        eta = jnp.array([0.0, 1.0], dtype=jnp.float64)
        mu = family.link_inverses["mu"](eta)
        sigma = family.link_inverses["sigma"](eta)
        np.testing.assert_allclose(np.asarray(mu), np.exp([0.0, 1.0]))
        self.assertTrue(np.all(np.asarray(sigma) > 0.0))
        self.assertTrue(np.all(np.asarray(sigma) < 1.0))
        self.assertEqual(family.parameters, ("mu", "sigma"))
        direct = ZIP()
        self.assertIsInstance(direct, ZeroInflatedPoissonFamily)
        self.assertEqual(direct.name, "ZIP")

    def test_bi_family_and_resolution_work(self) -> None:
        family = resolve_family("BI")
        self.assertEqual(family.name, "BI")
        eta = jnp.array([0.0, 1.0], dtype=jnp.float64)
        mu = family.link_inverses["mu"](eta)
        np.testing.assert_allclose(np.asarray(mu), 1.0 / (1.0 + np.exp(-np.array([0.0, 1.0]))))
        self.assertEqual(family.parameters, ("mu",))
        direct = BI()
        self.assertIsInstance(direct, BinomialFamily)
        self.assertEqual(direct.name, "BI")

    def test_be_family_and_resolution_work(self) -> None:
        family = resolve_family("BE")
        self.assertEqual(family.name, "BE")
        eta = jnp.array([0.0, 1.0], dtype=jnp.float64)
        mu = family.link_inverses["mu"](eta)
        sigma = family.link_inverses["sigma"](eta)
        np.testing.assert_allclose(np.asarray(mu), 1.0 / (1.0 + np.exp(-np.array([0.0, 1.0]))))
        np.testing.assert_allclose(np.asarray(sigma), 1.0 / (1.0 + np.exp(-np.array([0.0, 1.0]))))
        self.assertEqual(family.parameters, ("mu", "sigma"))
        direct = BE()
        self.assertIsInstance(direct, BetaFamily)
        self.assertEqual(direct.name, "BE")

    def test_ga_family_and_resolution_work(self) -> None:
        family = resolve_family("GA")
        self.assertEqual(family.name, "GA")
        eta = jnp.array([0.0, 1.0], dtype=jnp.float64)
        mu = family.link_inverses["mu"](eta)
        sigma = family.link_inverses["sigma"](eta)
        np.testing.assert_allclose(np.asarray(mu), np.exp([0.0, 1.0]))
        np.testing.assert_allclose(np.asarray(sigma), np.exp([0.0, 1.0]))
        self.assertEqual(family.parameters, ("mu", "sigma"))
        self.assertEqual(GA().name, "GA")

    def test_wei_family_and_resolution_work(self) -> None:
        family = resolve_family("WEI")
        self.assertEqual(family.name, "WEI")
        eta = jnp.array([0.0, 1.0], dtype=jnp.float64)
        mu = family.link_inverses["mu"](eta)
        sigma = family.link_inverses["sigma"](eta)
        np.testing.assert_allclose(np.asarray(mu), np.exp([0.0, 1.0]))
        np.testing.assert_allclose(np.asarray(sigma), np.exp([0.0, 1.0]))
        self.assertEqual(family.parameters, ("mu", "sigma"))
        direct = WEI()
        self.assertIsInstance(direct, WeibullFamily)
        self.assertEqual(direct.name, "WEI")

    def test_exp_family_and_resolution_work(self) -> None:
        family = resolve_family("EXP")
        self.assertEqual(family.name, "EXP")
        eta = jnp.array([0.0, 1.0], dtype=jnp.float64)
        mu = family.link_inverses["mu"](eta)
        np.testing.assert_allclose(np.asarray(mu), np.exp([0.0, 1.0]))
        self.assertEqual(family.parameters, ("mu",))
        direct = EXP()
        self.assertIsInstance(direct, ExponentialFamily)
        self.assertEqual(direct.name, "EXP")

    def test_logno_family_and_resolution_work(self) -> None:
        family = resolve_family("LOGNO")
        self.assertEqual(family.name, "LOGNO")
        eta = jnp.array([0.0, 1.0], dtype=jnp.float64)
        mu = family.link_inverses["mu"](eta)
        sigma = family.link_inverses["sigma"](eta)
        np.testing.assert_allclose(np.asarray(mu), [0.0, 1.0])
        np.testing.assert_allclose(np.asarray(sigma), np.exp([0.0, 1.0]))
        self.assertEqual(family.parameters, ("mu", "sigma"))
        direct = LOGNO()
        self.assertIsInstance(direct, LogNormalFamily)
        self.assertEqual(direct.name, "LOGNO")

    def test_nbi_family_and_resolution_work(self) -> None:
        family = resolve_family("NBI")
        self.assertEqual(family.name, "NBI")
        eta = jnp.array([0.0, 1.0], dtype=jnp.float64)
        mu = family.link_inverses["mu"](eta)
        sigma = family.link_inverses["sigma"](eta)
        np.testing.assert_allclose(np.asarray(mu), np.exp([0.0, 1.0]))
        np.testing.assert_allclose(np.asarray(sigma), np.exp([0.0, 1.0]))
        self.assertEqual(family.parameters, ("mu", "sigma"))
        direct = NBI()
        self.assertIsInstance(direct, NegativeBinomialFamily)
        self.assertEqual(direct.name, "NBI")

    def test_ig_family_and_resolution_work(self) -> None:
        family = resolve_family("IG")
        self.assertEqual(family.name, "IG")
        eta = jnp.array([0.0, 1.0], dtype=jnp.float64)
        mu = family.link_inverses["mu"](eta)
        sigma = family.link_inverses["sigma"](eta)
        np.testing.assert_allclose(np.asarray(mu), np.exp([0.0, 1.0]))
        np.testing.assert_allclose(np.asarray(sigma), np.exp([0.0, 1.0]))
        self.assertEqual(family.parameters, ("mu", "sigma"))
        direct = IG()
        self.assertIsInstance(direct, InverseGaussianFamily)
        self.assertEqual(direct.name, "IG")

    def test_lo_family_and_resolution_work(self) -> None:
        family = resolve_family("LO")
        self.assertEqual(family.name, "LO")
        eta = jnp.array([0.0, 1.0], dtype=jnp.float64)
        mu = family.link_inverses["mu"](eta)
        sigma = family.link_inverses["sigma"](eta)
        np.testing.assert_allclose(np.asarray(mu), [0.0, 1.0])
        np.testing.assert_allclose(np.asarray(sigma), np.exp([0.0, 1.0]))
        self.assertEqual(family.parameters, ("mu", "sigma"))
        direct = LO()
        self.assertIsInstance(direct, LogisticFamily)
        self.assertEqual(direct.name, "LO")

    def test_gamlss_rs_style_fit_runs_for_normal_family(self) -> None:
        fit = gamlss(
            formula="y ~ x1",
            family=NO(),
            data={
                "y": jnp.array([1.0, 2.1, 2.9, 4.2, 5.1], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64),
            },
            control=gamlss_control(c_crit=1e-8, n_cyc=20, trace=False),
        )
        self.assertTrue(is_gamlss(fit))
        self.assertEqual(fit.par, ("mu", "sigma"))
        self.assertGreaterEqual(fit.iter, 1)
        self.assertLess(float(fit.g_dev), 30.0)
        self.assertEqual(np.asarray(fit.fitted_values["mu"]).dtype, np.float64)
        self.assertEqual(np.asarray(fit.working_vectors["mu"]).dtype, np.float64)
        self.assertEqual(np.asarray(fit.iterative_weights["sigma"]).dtype, np.float64)

    def test_gamlss_and_gamlss_ml_support_poisson_family(self) -> None:
        data = {
            "y": jnp.array([1.0, 2.0, 1.0, 3.0, 4.0], dtype=jnp.float64),
            "x1": jnp.array([0.0, 1.0, 0.0, 1.0, 2.0], dtype=jnp.float64),
        }
        ml_fit = gamlss_ml(formula="y ~ x1", family=PO(), data=data)
        rs_fit = gamlss(formula="y ~ x1", family=PO(), data=data)
        self.assertEqual(ml_fit.par, ("mu",))
        self.assertEqual(rs_fit.par, ("mu",))
        self.assertEqual(ml_fit.family.name, "PO")
        self.assertEqual(rs_fit.family.name, "PO")
        self.assertEqual(np.asarray(ml_fit.fitted_values["mu"]).dtype, np.float64)
        self.assertEqual(np.asarray(rs_fit.fitted_values["mu"]).dtype, np.float64)
        self.assertTrue(np.all(np.asarray(rs_fit.fitted_values["mu"]) > 0.0))

    def test_gamlss_and_gamlss_ml_support_geometric_family(self) -> None:
        data = {
            "y": jnp.array([0.0, 1.0, 2.0, 1.0, 3.0, 4.0], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5], dtype=jnp.float64),
        }
        ml_fit = gamlss_ml(formula="y ~ x1", family=GEOM(), data=data)
        rs_fit = gamlss(formula="y ~ x1", family=GEOM(), data=data)
        self.assertEqual(ml_fit.par, ("mu",))
        self.assertEqual(rs_fit.par, ("mu",))
        self.assertEqual(ml_fit.family.name, "GEOM")
        self.assertEqual(rs_fit.family.name, "GEOM")
        self.assertEqual(np.asarray(ml_fit.fitted_values["mu"]).dtype, np.float64)
        self.assertTrue(np.all(np.asarray(rs_fit.fitted_values["mu"]) > 0.0))

    def test_gamlss_and_gamlss_ml_support_zip_family(self) -> None:
        data = {
            "y": jnp.array([0.0, 0.0, 1.0, 0.0, 3.0, 2.0, 0.0], dtype=jnp.float64),
            "x1": jnp.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0], dtype=jnp.float64),
        }
        ml_fit = gamlss_ml(formula="y ~ x1", family=ZIP(), data=data)
        rs_fit = gamlss(formula="y ~ x1", family=ZIP(), data=data, method="CG")
        self.assertEqual(ml_fit.par, ("mu", "sigma"))
        self.assertEqual(rs_fit.par, ("mu", "sigma"))
        self.assertEqual(ml_fit.family.name, "ZIP")
        self.assertEqual(rs_fit.family.name, "ZIP")
        self.assertEqual(rs_fit.additional_slots["method"], "CG")
        self.assertTrue(np.all(np.asarray(rs_fit.fitted_values["mu"]) > 0.0))
        self.assertTrue(np.all(np.asarray(rs_fit.fitted_values["sigma"]) > 0.0))
        self.assertTrue(np.all(np.asarray(rs_fit.fitted_values["sigma"]) < 1.0))

    def test_gamlss_and_gamlss_ml_support_binomial_family(self) -> None:
        data = {
            "y": jnp.array([0.0, 0.0, 1.0, 1.0, 1.0], dtype=jnp.float64),
            "x1": jnp.array([0.0, 1.0, 0.0, 1.0, 2.0], dtype=jnp.float64),
        }
        ml_fit = gamlss_ml(formula="y ~ x1", family=BI(), data=data)
        rs_fit = gamlss(formula="y ~ x1", family=BI(), data=data)
        self.assertEqual(ml_fit.par, ("mu",))
        self.assertEqual(rs_fit.par, ("mu",))
        self.assertEqual(ml_fit.family.name, "BI")
        self.assertEqual(rs_fit.family.name, "BI")
        self.assertEqual(np.asarray(ml_fit.fitted_values["mu"]).dtype, np.float64)
        self.assertEqual(np.asarray(rs_fit.fitted_values["mu"]).dtype, np.float64)
        self.assertTrue(np.all(np.asarray(rs_fit.fitted_values["mu"]) > 0.0))
        self.assertTrue(np.all(np.asarray(rs_fit.fitted_values["mu"]) < 1.0))

    def test_gamlss_and_gamlss_ml_support_gamma_family(self) -> None:
        data = {
            "y": jnp.array([1.0, 1.5, 2.0, 2.8, 3.5], dtype=jnp.float64),
            "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64),
        }
        ml_fit = gamlss_ml(formula="y ~ x1", family=GA(), data=data)
        rs_fit = gamlss(formula="y ~ x1", family=GA(), data=data, method="CG")
        self.assertEqual(ml_fit.par, ("mu", "sigma"))
        self.assertEqual(rs_fit.par, ("mu", "sigma"))
        self.assertEqual(ml_fit.family.name, "GA")
        self.assertEqual(rs_fit.family.name, "GA")
        self.assertEqual(rs_fit.additional_slots["method"], "CG")
        self.assertEqual(np.asarray(ml_fit.fitted_values["mu"]).dtype, np.float64)
        self.assertEqual(np.asarray(rs_fit.fitted_values["sigma"]).dtype, np.float64)
        self.assertTrue(np.all(np.asarray(rs_fit.fitted_values["mu"]) > 0.0))
        self.assertTrue(np.all(np.asarray(rs_fit.fitted_values["sigma"]) > 0.0))

    def test_gamlss_and_gamlss_ml_support_weibull_family(self) -> None:
        data = {
            "y": jnp.array([0.9, 1.4, 2.0, 2.7, 3.6], dtype=jnp.float64),
            "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64),
        }
        ml_fit = gamlss_ml(formula="y ~ x1", family=WEI(), data=data)
        rs_fit = gamlss(formula="y ~ x1", family=WEI(), data=data, method="RS")
        self.assertEqual(ml_fit.par, ("mu", "sigma"))
        self.assertEqual(rs_fit.par, ("mu", "sigma"))
        self.assertEqual(ml_fit.family.name, "WEI")
        self.assertEqual(rs_fit.family.name, "WEI")
        self.assertTrue(np.all(np.asarray(rs_fit.fitted_values["mu"]) > 0.0))
        self.assertTrue(np.all(np.asarray(rs_fit.fitted_values["sigma"]) > 0.0))

    def test_gamlss_method_selector_accepts_mixed_alias(self) -> None:
        fit = gamlss(
            formula="y ~ x1",
            family=PO(),
            method="mixed",
            data={
                "y": jnp.array([1.0, 2.0, 1.0, 3.0], dtype=jnp.float64),
                "x1": jnp.array([0.0, 1.0, 0.0, 1.0], dtype=jnp.float64),
            },
        )
        self.assertEqual(fit.additional_slots["method"], "MIXED")

    def test_gamlss_and_gamlss_ml_support_exponential_family(self) -> None:
        data = {
            "y": jnp.array([0.8, 1.1, 1.6, 2.0, 2.6], dtype=jnp.float64),
            "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64),
        }
        ml_fit = gamlss_ml(formula="y ~ x1", family=EXP(), data=data)
        rs_fit = gamlss(formula="y ~ x1", family=EXP(), data=data, method="RS")
        mixed_fit = gamlss(formula="y ~ x1", family=EXP(), data=data, method="mixed")
        self.assertEqual(ml_fit.par, ("mu",))
        self.assertEqual(rs_fit.par, ("mu",))
        self.assertEqual(rs_fit.family.name, "EXP")
        self.assertEqual(mixed_fit.additional_slots["method"], "MIXED")
        self.assertEqual(np.asarray(ml_fit.fitted_values["mu"]).dtype, np.float64)
        self.assertTrue(np.all(np.asarray(rs_fit.fitted_values["mu"]) > 0.0))
        self.assertTrue(np.all(np.asarray(mixed_fit.fitted_values["mu"]) > 0.0))

    def test_method_specific_updates_change_fit_path(self) -> None:
        data = {
            "y": jnp.array([1.0, 1.4, 2.1, 2.5, 3.3], dtype=jnp.float64),
            "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64),
        }
        rs_fit = gamlss(formula="y ~ x1", family=EXP(), data=data, method="RS")
        cg_fit = gamlss(formula="y ~ x1", family=EXP(), data=data, method="CG")
        self.assertFalse(
            np.allclose(
                np.asarray(rs_fit.coefficients["mu"]),
                np.asarray(cg_fit.coefficients["mu"]),
            )
        )

    def test_gamlss_and_gamlss_ml_support_lognormal_family(self) -> None:
        data = {
            "y": jnp.array([1.2, 1.5, 2.1, 2.9, 4.0], dtype=jnp.float64),
            "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64),
        }
        ml_fit = gamlss_ml(formula="y ~ x1", family=LOGNO(), data=data)
        rs_fit = gamlss(formula="y ~ x1", family=LOGNO(), data=data, method="RS")
        self.assertEqual(ml_fit.par, ("mu", "sigma"))
        self.assertEqual(rs_fit.par, ("mu", "sigma"))
        self.assertEqual(ml_fit.family.name, "LOGNO")
        self.assertEqual(rs_fit.family.name, "LOGNO")
        self.assertEqual(np.asarray(ml_fit.fitted_values["mu"]).dtype, np.float64)
        self.assertTrue(np.all(np.asarray(rs_fit.fitted_values["sigma"]) > 0.0))

    def test_gamlss_and_gamlss_ml_support_negative_binomial_family(self) -> None:
        data = {
            "y": jnp.array([0.0, 1.0, 3.0, 2.0, 6.0, 4.0], dtype=jnp.float64),
            "x1": jnp.array([0.0, 1.0, 0.0, 1.0, 2.0, 2.0], dtype=jnp.float64),
        }
        ml_fit = gamlss_ml(formula="y ~ x1", family=NBI(), data=data)
        rs_fit = gamlss(formula="y ~ x1", family=NBI(), data=data, method="CG")
        self.assertEqual(ml_fit.par, ("mu", "sigma"))
        self.assertEqual(rs_fit.par, ("mu", "sigma"))
        self.assertEqual(ml_fit.family.name, "NBI")
        self.assertEqual(rs_fit.family.name, "NBI")
        self.assertEqual(rs_fit.additional_slots["method"], "CG")
        self.assertTrue(np.all(np.asarray(rs_fit.fitted_values["mu"]) > 0.0))
        self.assertTrue(np.all(np.asarray(rs_fit.fitted_values["sigma"]) > 0.0))

    def test_gamlss_and_gamlss_ml_support_inverse_gaussian_family(self) -> None:
        data = {
            "y": jnp.array([0.9, 1.3, 1.8, 2.4, 3.1], dtype=jnp.float64),
            "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64),
        }
        ml_fit = gamlss_ml(formula="y ~ x1", family=IG(), data=data)
        rs_fit = gamlss(formula="y ~ x1", family=IG(), data=data, method="RS")
        self.assertEqual(ml_fit.par, ("mu", "sigma"))
        self.assertEqual(rs_fit.par, ("mu", "sigma"))
        self.assertEqual(ml_fit.family.name, "IG")
        self.assertEqual(rs_fit.family.name, "IG")
        self.assertTrue(rs_fit.additional_slots["converged"])
        self.assertGreaterEqual(len(rs_fit.additional_slots["deviance_history"]), 2)
        self.assertTrue(np.all(np.asarray(rs_fit.fitted_values["mu"]) > 0.0))
        self.assertTrue(np.all(np.asarray(rs_fit.fitted_values["sigma"]) > 0.0))

    def test_gamlss_and_gamlss_ml_support_logistic_family(self) -> None:
        data = {
            "y": jnp.array([-1.2, -0.3, 0.5, 1.1, 2.0], dtype=jnp.float64),
            "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64),
        }
        ml_fit = gamlss_ml(formula="y ~ x1", family=LO(), data=data)
        rs_fit = gamlss(formula="y ~ x1", family=LO(), data=data, method="RS")
        self.assertEqual(ml_fit.par, ("mu", "sigma"))
        self.assertEqual(rs_fit.par, ("mu", "sigma"))
        self.assertEqual(ml_fit.family.name, "LO")
        self.assertEqual(rs_fit.family.name, "LO")
        self.assertEqual(rs_fit.additional_slots["method"], "RS")
        self.assertIn("converged", rs_fit.additional_slots)
        self.assertGreaterEqual(len(rs_fit.additional_slots["deviance_history"]), 2)
        self.assertTrue(np.all(np.asarray(rs_fit.fitted_values["sigma"]) > 0.0))

    def test_gamlss_and_gamlss_ml_support_beta_family(self) -> None:
        data = {
            "y": jnp.array([0.12, 0.28, 0.44, 0.63, 0.79], dtype=jnp.float64),
            "x1": jnp.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=jnp.float64),
        }
        ml_fit = gamlss_ml(formula="y ~ x1", family=BE(), data=data)
        rs_fit = gamlss(formula="y ~ x1", family=BE(), data=data, method="RS")
        self.assertEqual(ml_fit.par, ("mu", "sigma"))
        self.assertEqual(rs_fit.par, ("mu", "sigma"))
        self.assertEqual(ml_fit.family.name, "BE")
        self.assertEqual(rs_fit.family.name, "BE")
        self.assertTrue(np.all(np.asarray(rs_fit.fitted_values["mu"]) > 0.0))
        self.assertTrue(np.all(np.asarray(rs_fit.fitted_values["mu"]) < 1.0))
        self.assertTrue(np.all(np.asarray(rs_fit.fitted_values["sigma"]) > 0.0))


if __name__ == "__main__":
    unittest.main()
