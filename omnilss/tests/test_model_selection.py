"""测试模型选择功能

测试自动分布选择、分布比较等功能。
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, "omnilss/src")

from omnilss.model_selection import (
    compare_distributions,
    select_best_distribution,
    quick_distribution_search,
    COMMON_FAMILIES
)


class TestCompareDistributions:
    """测试分布比较功能"""
    
    def test_compare_distributions_basic(self):
        """测试基本的分布比较"""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 + 3*x + np.random.randn(n)
        data = {"y": y, "x": x}
        
        results = compare_distributions(
            "y ~ x",
            families=["NO", "GA", "LOGNO"],
            data=data,
            criterion="AIC"
        )
        
        # 检查结果
        assert len(results) == 3
        assert "family" in results.columns
        assert "AIC" in results.columns
        assert "BIC" in results.columns
        assert "deviance" in results.columns
        assert "df" in results.columns
        assert "converged" in results.columns
        
        # 检查排序（AIC 最小的在前）
        assert results.iloc[0]["AIC"] <= results.iloc[1]["AIC"]
        assert results.iloc[1]["AIC"] <= results.iloc[2]["AIC"]
    
    def test_compare_distributions_with_failure(self):
        """测试当某些分布拟合失败时的处理"""
        np.random.seed(42)
        n = 50
        x = np.random.randn(n)
        y = 2 + 3*x + np.random.randn(n)
        data = {"y": y, "x": x}
        
        # 包含一些可能失败的分布
        results = compare_distributions(
            "y ~ x",
            families=["NO", "GA"],
            data=data,
            criterion="AIC"
        )
        
        # 应该至少有一些成功的
        assert len(results) >= 1
        assert results.iloc[0]["converged"] == True
    
    def test_compare_distributions_verbose(self):
        """测试 verbose 模式"""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 + 3*x + np.random.randn(n)
        data = {"y": y, "x": x}
        
        results = compare_distributions(
            "y ~ x",
            families=["NO", "GA"],
            data=data,
            criterion="AIC",
            verbose=True  # 打印详细信息
        )
        
        assert len(results) == 2


class TestSelectBestDistribution:
    """测试自动选择最佳分布"""
    
    def test_select_best_distribution_basic(self):
        """测试基本的自动选择"""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 + 3*x + np.random.randn(n)
        data = {"y": y, "x": x}
        
        best_model, comparison = select_best_distribution(
            "y ~ x",
            families=["NO", "GA"],
            data=data,
            criterion="AIC"
        )
        
        # 检查返回值
        assert best_model is not None
        assert len(comparison) == 2
        
        # 检查最佳模型的分布族
        assert best_model.family.name == comparison.iloc[0]["family"]
        
        # 检查模型可以用于预测
        newdata = {"x": np.array([0, 1, 2])}
        params = best_model.predict_params(newdata)
        assert "mu" in params
    
    def test_select_best_distribution_with_bic(self):
        """测试使用 BIC 准则"""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 + 3*x + np.random.randn(n)
        data = {"y": y, "x": x}
        
        best_model, comparison = select_best_distribution(
            "y ~ x",
            families=["NO", "GA", "LOGNO"],
            data=data,
            criterion="BIC"
        )
        
        # 检查排序
        assert comparison.iloc[0]["BIC"] <= comparison.iloc[1]["BIC"]
        assert best_model.family.name == comparison.iloc[0]["family"]


class TestQuickDistributionSearch:
    """测试快速分布搜索"""
    
    def test_quick_search_auto_continuous_real(self):
        """测试自动检测连续实数数据"""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 + 3*x + np.random.randn(n)  # 可以是负数
        data = {"y": y, "x": x}
        
        best_model, comparison = quick_distribution_search(
            "y ~ x",
            data=data,
            data_type="auto"
        )
        
        # 应该选择适合实数的分布
        assert best_model is not None
        assert len(comparison) > 0
    
    def test_quick_search_auto_continuous_positive(self):
        """测试自动检测连续正数数据"""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = np.exp(2 + 0.5*x + np.random.randn(n)*0.3)  # 正数
        data = {"y": y, "x": x}
        
        best_model, comparison = quick_distribution_search(
            "y ~ x",
            data=data,
            data_type="auto"
        )
        
        # 应该选择适合正数的分布（如 GA, LOGNO）
        assert best_model is not None
        assert best_model.family.name in ["GA", "LOGNO", "WEI", "IG"]
    
    def test_quick_search_auto_count(self):
        """测试自动检测计数数据"""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        lambda_ = np.exp(1 + 0.5*x)
        y = np.random.poisson(lambda_)  # 计数数据
        data = {"y": y, "x": x}
        
        best_model, comparison = quick_distribution_search(
            "y ~ x",
            data=data,
            data_type="auto"
        )
        
        # 应该选择适合计数的分布（如 PO, NBI）
        assert best_model is not None
        assert best_model.family.name in ["PO", "NBI", "ZIP"]
    
    def test_quick_search_manual_type(self):
        """测试手动指定数据类型"""
        np.random.seed(42)
        n = 100
        x = np.random.randn(n)
        y = 2 + 3*x + np.random.randn(n)
        data = {"y": y, "x": x}
        
        best_model, comparison = quick_distribution_search(
            "y ~ x",
            data=data,
            data_type="continuous_real"
        )
        
        assert best_model is not None
        # 应该只比较连续实数分布
        assert all(f in ["NO", "TF", "LO"] for f in comparison["family"])


class TestCommonFamilies:
    """测试预定义的分布组合"""
    
    def test_common_families_exist(self):
        """测试预定义的分布组合存在"""
        assert "continuous_positive" in COMMON_FAMILIES
        assert "continuous_real" in COMMON_FAMILIES
        assert "count" in COMMON_FAMILIES
        assert "proportion" in COMMON_FAMILIES
        assert "all_common" in COMMON_FAMILIES
    
    def test_common_families_content(self):
        """测试预定义的分布组合内容"""
        # 连续正数
        assert "GA" in COMMON_FAMILIES["continuous_positive"]
        assert "LOGNO" in COMMON_FAMILIES["continuous_positive"]
        
        # 连续实数
        assert "NO" in COMMON_FAMILIES["continuous_real"]
        
        # 计数
        assert "PO" in COMMON_FAMILIES["count"]
        assert "NBI" in COMMON_FAMILIES["count"]
        
        # 比例
        assert "BE" in COMMON_FAMILIES["proportion"]


class TestIntegration:
    """集成测试"""
    
    def test_full_workflow(self):
        """测试完整的工作流"""
        np.random.seed(42)
        n = 200
        x = np.random.randn(n)
        # 生成 Gamma 分布数据
        shape = 2.0
        scale = np.exp(1 + 0.5*x)
        y = np.random.gamma(shape, scale)
        data = {"y": y, "x": x}
        
        # 1. 比较分布
        comparison = compare_distributions(
            "y ~ x",
            families=["NO", "GA", "LOGNO", "WEI"],
            data=data,
            criterion="AIC"
        )
        
        # 2. 选择最佳分布
        best_model, _ = select_best_distribution(
            "y ~ x",
            families=["NO", "GA", "LOGNO", "WEI"],
            data=data,
            criterion="AIC"
        )
        
        # 3. 使用最佳模型预测
        newdata = {"x": np.array([0, 1, 2])}
        quantiles = best_model.predict_quantiles(
            newdata,
            quantiles=[0.05, 0.5, 0.95]
        )
        
        # 检查结果
        assert len(comparison) == 4
        assert best_model is not None
        assert 0.05 in quantiles
        assert 0.5 in quantiles
        assert 0.95 in quantiles
        
        # Gamma 应该是最佳或接近最佳
        # （因为数据是从 Gamma 生成的）
        top_3_families = comparison.head(3)["family"].tolist()
        assert "GA" in top_3_families or "LOGNO" in top_3_families


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "--tb=short"])
