"""测试预测功能

测试 GAMLSS 模型的预测功能，包括：
- 参数预测 (predict_params)
- 分位数预测 (predict_quantiles)
- Centile curves (centiles)
- 响应变量预测 (predict)
"""

import pytest
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "omnilss/src")

from omnilss import gamlss


class TestPredictParams:
    """测试参数预测功能"""
    
    def test_predict_params_simple(self):
        """测试简单线性模型的参数预测"""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 + 3*x + np.random.randn(n)
        data = {"y": y, "x": x}
        
        # 拟合模型
        model = gamlss("y ~ x", family="NO", data=data)
        
        # 预测新数据
        newdata = {"x": np.array([0, 1, 2])}
        params = model.predict_params(newdata)
        
        # 检查返回的参数
        assert "mu" in params
        assert "sigma" in params
        
        # 检查形状
        assert len(params["mu"]) == 3
        assert len(params["sigma"]) == 3
        
        # 检查值的合理性
        assert np.all(np.isfinite(params["mu"]))
        assert np.all(np.isfinite(params["sigma"]))
        assert np.all(params["sigma"] > 0)
    
    def test_predict_params_which(self):
        """测试选择性参数预测"""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 + 3*x + np.random.randn(n)
        data = {"y": y, "x": x}
        
        model = gamlss("y ~ x", family="NO", data=data)
        
        # 只预测 mu
        newdata = {"x": np.array([0, 1, 2])}
        params = model.predict_params(newdata, which=["mu"])
        
        assert "mu" in params
        assert len(params) == 1  # 只有 mu
    
    def test_predict_params_multiple_predictors(self):
        """测试多个预测变量"""
        np.random.seed(42)
        n = 100
        x1 = np.random.randn(n)
        x2 = np.random.randn(n)
        y = 2 + 3*x1 - 2*x2 + np.random.randn(n)
        data = {"y": y, "x1": x1, "x2": x2}
        
        model = gamlss("y ~ x1 + x2", family="NO", data=data)
        
        newdata = {"x1": np.array([0, 1]), "x2": np.array([0, 1])}
        params = model.predict_params(newdata)
        
        assert len(params["mu"]) == 2
        assert np.all(np.isfinite(params["mu"]))


class TestPredictQuantiles:
    """测试分位数预测功能"""
    
    def test_predict_quantiles_basic(self):
        """测试基本分位数预测"""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 + 3*x + np.random.randn(n)
        data = {"y": y, "x": x}
        
        model = gamlss("y ~ x", family="NO", data=data)
        
        # 预测分位数
        newdata = {"x": np.array([0, 1, 2])}
        quantiles = model.predict_quantiles(
            newdata,
            quantiles=[0.05, 0.5, 0.95]
        )
        
        # 检查返回的分位数
        assert 0.05 in quantiles
        assert 0.5 in quantiles
        assert 0.95 in quantiles
        
        # 检查形状
        assert len(quantiles[0.05]) == 3
        assert len(quantiles[0.5]) == 3
        assert len(quantiles[0.95]) == 3
        
        # 检查值的合理性
        assert np.all(np.isfinite(quantiles[0.05]))
        assert np.all(np.isfinite(quantiles[0.5]))
        assert np.all(np.isfinite(quantiles[0.95]))
    
    def test_predict_quantiles_ordering(self):
        """测试分位数的顺序性"""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 + 3*x + np.random.randn(n)
        data = {"y": y, "x": x}
        
        model = gamlss("y ~ x", family="NO", data=data)
        
        newdata = {"x": np.array([0, 1, 2])}
        quantiles = model.predict_quantiles(
            newdata,
            quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
        )
        
        # 检查顺序：q5 < q25 < q50 < q75 < q95
        assert np.all(quantiles[0.05] < quantiles[0.25])
        assert np.all(quantiles[0.25] < quantiles[0.5])
        assert np.all(quantiles[0.5] < quantiles[0.75])
        assert np.all(quantiles[0.75] < quantiles[0.95])
    
    def test_predict_quantiles_median_vs_mean(self):
        """测试中位数与均值的关系（对于正态分布应该接近）"""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 + 3*x + np.random.randn(n)
        data = {"y": y, "x": x}
        
        model = gamlss("y ~ x", family="NO", data=data)
        
        newdata = {"x": np.array([0, 1, 2])}
        
        # 预测均值（通过参数）
        params = model.predict_params(newdata)
        mean_pred = params["mu"]
        
        # 预测中位数
        quantiles = model.predict_quantiles(newdata, quantiles=[0.5])
        median_pred = quantiles[0.5]
        
        # 对于正态分布，均值 = 中位数
        np.testing.assert_allclose(mean_pred, median_pred, rtol=1e-5)
    
    def test_predict_quantiles_invalid_range(self):
        """测试无效的分位数范围"""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 + 3*x + np.random.randn(n)
        data = {"y": y, "x": x}
        
        model = gamlss("y ~ x", family="NO", data=data)
        
        newdata = {"x": np.array([0, 1, 2])}
        
        # 测试 q = 0（无效）
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            model.predict_quantiles(newdata, quantiles=[0.0])
        
        # 测试 q = 1（无效）
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            model.predict_quantiles(newdata, quantiles=[1.0])
        
        # 测试 q < 0（无效）
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            model.predict_quantiles(newdata, quantiles=[-0.1])
        
        # 测试 q > 1（无效）
        with pytest.raises(ValueError, match="must be in \\(0, 1\\)"):
            model.predict_quantiles(newdata, quantiles=[1.5])


class TestCentiles:
    """测试 centile curves 功能"""
    
    def test_centiles_basic(self):
        """测试基本 centile curves 生成"""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 + 3*x + np.random.randn(n)
        data = {"y": y, "x": x}
        
        model = gamlss("y ~ x", family="NO", data=data)
        
        # 生成 centile curves
        curves = model.centiles(xvar="x", cent=[5, 50, 95])
        
        # 检查返回类型
        assert isinstance(curves, pd.DataFrame)
        
        # 检查列
        assert "x" in curves.columns
        assert "C5" in curves.columns
        assert "C50" in curves.columns
        assert "C95" in curves.columns
        
        # 检查默认点数
        assert len(curves) == 100
        
        # 检查值的合理性
        assert np.all(np.isfinite(curves["C5"]))
        assert np.all(np.isfinite(curves["C50"]))
        assert np.all(np.isfinite(curves["C95"]))
    
    def test_centiles_ordering(self):
        """测试 centile curves 的顺序性"""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 + 3*x + np.random.randn(n)
        data = {"y": y, "x": x}
        
        model = gamlss("y ~ x", family="NO", data=data)
        
        curves = model.centiles(xvar="x", cent=[5, 25, 50, 75, 95])
        
        # 检查顺序：C5 < C25 < C50 < C75 < C95
        assert np.all(curves["C5"] < curves["C25"])
        assert np.all(curves["C25"] < curves["C50"])
        assert np.all(curves["C50"] < curves["C75"])
        assert np.all(curves["C75"] < curves["C95"])
    
    def test_centiles_custom_xvalues(self):
        """测试自定义 X 值"""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 + 3*x + np.random.randn(n)
        data = {"y": y, "x": x}
        
        model = gamlss("y ~ x", family="NO", data=data)
        
        # 自定义 X 值
        xvalues = np.array([0, 1, 2, 3, 4])
        curves = model.centiles(xvar="x", xvalues=xvalues, cent=[50])
        
        # 检查 X 值
        np.testing.assert_array_equal(curves["x"], xvalues)
        assert len(curves) == 5
    
    def test_centiles_custom_n_points(self):
        """测试自定义点数"""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 + 3*x + np.random.randn(n)
        data = {"y": y, "x": x}
        
        model = gamlss("y ~ x", family="NO", data=data)
        
        # 自定义点数
        curves = model.centiles(xvar="x", cent=[50], n_points=50)
        
        assert len(curves) == 50
    
    def test_centiles_invalid_cent(self):
        """测试无效的百分位数"""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 + 3*x + np.random.randn(n)
        data = {"y": y, "x": x}
        
        model = gamlss("y ~ x", family="NO", data=data)
        
        # 测试 cent = 0（无效）
        with pytest.raises(ValueError, match="must be in \\(0, 100\\)"):
            model.centiles(xvar="x", cent=[0])
        
        # 测试 cent = 100（无效）
        with pytest.raises(ValueError, match="must be in \\(0, 100\\)"):
            model.centiles(xvar="x", cent=[100])
        
        # 测试 cent < 0（无效）
        with pytest.raises(ValueError, match="must be in \\(0, 100\\)"):
            model.centiles(xvar="x", cent=[-5])
        
        # 测试 cent > 100（无效）
        with pytest.raises(ValueError, match="must be in \\(0, 100\\)"):
            model.centiles(xvar="x", cent=[105])


class TestPredictResponse:
    """测试响应变量预测功能"""
    
    def test_predict_response_type(self):
        """测试响应变量预测（response 类型）"""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 + 3*x + np.random.randn(n)
        data = {"y": y, "x": x}
        
        model = gamlss("y ~ x", family="NO", data=data)
        
        newdata = {"x": np.array([0, 1, 2])}
        y_pred = model.predict(newdata, type="response")
        
        # 检查形状
        assert len(y_pred) == 3
        
        # 检查值的合理性
        assert np.all(np.isfinite(y_pred))
        
        # 对于正态分布，response 预测应该等于 mu
        params = model.predict_params(newdata)
        np.testing.assert_allclose(y_pred, params["mu"], rtol=1e-10)
    
    def test_predict_link_type(self):
        """测试线性预测值（link 类型）"""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 + 3*x + np.random.randn(n)
        data = {"y": y, "x": x}
        
        model = gamlss("y ~ x", family="NO", data=data)
        
        newdata = {"x": np.array([0, 1, 2])}
        eta_pred = model.predict(newdata, type="link")
        
        # 检查形状
        assert len(eta_pred) == 3
        
        # 检查值的合理性
        assert np.all(np.isfinite(eta_pred))
    
    def test_predict_invalid_type(self):
        """测试无效的预测类型"""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 + 3*x + np.random.randn(n)
        data = {"y": y, "x": x}
        
        model = gamlss("y ~ x", family="NO", data=data)
        
        newdata = {"x": np.array([0, 1, 2])}
        
        with pytest.raises(ValueError, match="Unknown type"):
            model.predict(newdata, type="invalid")


class TestPredictionIntegration:
    """测试预测功能的集成"""
    
    def test_prediction_workflow(self):
        """测试完整的预测工作流"""
        np.random.seed(42)
        n = 200
        age = np.random.uniform(0, 100, n)
        # 模拟生长曲线：均值和方差都随年龄变化
        mu = 170 + 0.5*age - 0.005*age**2
        sigma = 5 + 0.1*age
        y = np.random.normal(mu, sigma)
        data = {"y": y, "age": age}
        
        # 拟合模型
        model = gamlss("y ~ age", family="NO", data=data)
        
        # 1. 预测参数
        newdata = {"age": np.array([20, 40, 60, 80])}
        params = model.predict_params(newdata)
        
        assert "mu" in params
        assert "sigma" in params
        assert len(params["mu"]) == 4
        
        # 2. 预测分位数
        quantiles = model.predict_quantiles(
            newdata,
            quantiles=[0.05, 0.5, 0.95]
        )
        
        assert len(quantiles) == 3
        assert np.all(quantiles[0.05] < quantiles[0.5])
        assert np.all(quantiles[0.5] < quantiles[0.95])
        
        # 3. 生成 centile curves
        curves = model.centiles(xvar="age", cent=[5, 50, 95])
        
        assert isinstance(curves, pd.DataFrame)
        assert len(curves) == 100
        assert np.all(curves["C5"] < curves["C50"])
        assert np.all(curves["C50"] < curves["C95"])
        
        # 4. 预测响应变量
        y_pred = model.predict(newdata, type="response")
        
        assert len(y_pred) == 4
        assert np.all(np.isfinite(y_pred))
    
    def test_prediction_consistency(self):
        """测试预测结果的一致性"""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 + 3*x + np.random.randn(n)
        data = {"y": y, "x": x}
        
        model = gamlss("y ~ x", family="NO", data=data)
        
        newdata = {"x": np.array([0, 1, 2])}
        
        # 通过不同方法获取 mu 预测
        params = model.predict_params(newdata)
        mu_from_params = params["mu"]
        
        y_pred = model.predict(newdata, type="response")
        mu_from_predict = y_pred
        
        quantiles = model.predict_quantiles(newdata, quantiles=[0.5])
        mu_from_quantiles = quantiles[0.5]
        
        # 三种方法应该给出一致的结果
        np.testing.assert_allclose(mu_from_params, mu_from_predict, rtol=1e-10)
        np.testing.assert_allclose(mu_from_params, mu_from_quantiles, rtol=1e-5)


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])
