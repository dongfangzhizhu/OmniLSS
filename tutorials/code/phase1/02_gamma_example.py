"""
Gamma 分布族示例代码
Tutorial 02: Gamma Distributions (GA, GG, IGAMMA, IG)

这个脚本包含了教程中所有的示例代码，可以直接运行。
"""

import sys
from pathlib import Path

import jax
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "omnilss" / "src"))
jax.config.update("jax_enable_x64", True)

from omnilss import GA, IG, gamlss  # noqa: E402

# 设置绘图样式
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def _data_dict(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Convert tutorial DataFrames to the dict format used by current OmniLSS."""
    return {column: df[column].to_numpy() for column in df.columns}


def _fitted(model, parameter: str = "mu") -> np.ndarray:
    """Return fitted values for one distribution parameter."""
    return np.asarray(model.fitted_values[parameter], dtype=np.float64)


def _coef(model, parameter: str = "mu") -> np.ndarray:
    """Return coefficient vector for one distribution parameter."""
    return np.asarray(model.coefficients[parameter], dtype=np.float64)


def _aic(model) -> float:
    """Return AIC from the current model diagnostics slot."""
    return float(model.additional_slots["aic"])


def _bic(model) -> float:
    """Return BIC/SBC from the current model diagnostics slot."""
    return float(model.additional_slots["sbc"])


def _print_model(model) -> None:
    """Print a compact summary using current GAMLSSModel fields."""
    print(f"Family: {model.family.name}")
    print(f"Global deviance: {model.g_dev:.4f}")
    print(f"AIC: {_aic(model):.2f}")
    for parameter, beta in model.coefficients.items():
        print(f"{parameter} coefficients: {np.asarray(beta)}")


def _predict_param(model, newdata: pd.DataFrame, parameter: str) -> np.ndarray:
    """Predict one distribution parameter for new data."""
    return np.asarray(model.predict_params(_data_dict(newdata))[parameter])


def _predict_quantile(model, newdata: pd.DataFrame, quantile: float) -> np.ndarray:
    """Predict one response quantile for new data."""
    return np.asarray(model.predict_quantiles(_data_dict(newdata), [quantile])[quantile])


def example1_simple_gamma():
    """示例 1: 简单 Gamma 回归"""
    print("=" * 60)
    print("示例 1: 简单 Gamma 回归")
    print("=" * 60)
    
    # 设置随机种子
    np.random.seed(42)
    
    # 生成示例数据
    n = 300
    x = np.linspace(0, 5, n)
    
    # Gamma 分布参数
    shape = 2.0
    scale = 1.0 + 0.3 * x  # 尺度随 x 增加
    
    # 生成 Gamma 数据
    y = np.random.gamma(shape, scale)
    
    # 创建 DataFrame
    data = pd.DataFrame({'x': x, 'y': y})
    
    print(f"\n数据维度: {data.shape}")
    print(f"y 的统计:\n{data['y'].describe()}")
    
    # 拟合模型
    print("\n拟合 Gamma 模型...")
    model = gamlss(
        formula="y ~ x",
        family=GA(),
        data=_data_dict(data)
    )
    
    # 查看结果
    print("\n模型摘要:")
    _print_model(model)
    
    print(f"\nMu 系数: {_coef(model, 'mu')}")
    print(f"Sigma 系数: {_coef(model, 'sigma')}")
    print(f"AIC: {_aic(model):.2f}")
    
    # 可视化
    plt.figure(figsize=(12, 5))
    
    # 左图：原始数据和拟合均值
    plt.subplot(1, 2, 1)
    plt.scatter(data['x'], data['y'], alpha=0.5, s=20, label='Observed')
    plt.plot(data['x'], _fitted(model), 'r-', linewidth=2, label='Fitted mean')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Gamma Regression: Data and Fitted Mean', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 右图：残差
    plt.subplot(1, 2, 2)
    residuals = data['y'] - _fitted(model)
    plt.scatter(_fitted(model), residuals, alpha=0.5, s=20)
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Fitted values', fontsize=12)
    plt.ylabel('Residuals', fontsize=12)
    plt.title('Residual Plot', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../images/02_gamma_basic.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存: tutorials/images/02_gamma_basic.png")
    plt.show()
    
    return model, data


def example2_inverse_gaussian():
    """示例 2: 逆高斯分布（IG）"""
    print("\n" + "=" * 60)
    print("示例 2: 逆高斯分布（IG）")
    print("=" * 60)
    
    # 生成逆高斯数据
    np.random.seed(123)
    n = 250
    x = np.random.uniform(0, 10, n)
    
    # 逆高斯参数
    mu_true = 5 + 0.5 * x
    lambda_param = 10  # 形状参数
    
    # 生成 IG 数据（使用 Wald 分布）
    from scipy.stats import invgauss
    y = invgauss.rvs(mu=mu_true/mu_true.mean(), scale=lambda_param, size=n) * mu_true.mean()
    
    data = pd.DataFrame({'x': x, 'y': y})
    
    print(f"\n数据维度: {data.shape}")
    print(f"y 的统计:\n{data['y'].describe()}")
    
    # 拟合模型
    print("\n拟合 IG 模型...")
    model = gamlss(
        formula="y ~ x",
        family=IG(),
        data=_data_dict(data)
    )
    
    print("\n模型摘要:")
    _print_model(model)
    print(f"AIC: {_aic(model):.2f}")
    
    # 可视化
    plt.figure(figsize=(12, 5))
    
    # 左图：数据和拟合
    plt.subplot(1, 2, 1)
    plt.scatter(data['x'], data['y'], alpha=0.5, s=20, label='Observed')
    plt.plot(data['x'], _fitted(model), 'r-', linewidth=2, label='Fitted')
    plt.xlabel('x', fontsize=12)
    plt.ylabel('y', fontsize=12)
    plt.title('Inverse Gaussian Regression', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 右图：Q-Q 图
    plt.subplot(1, 2, 2)
    from scipy import stats
    stats.probplot(data['y'] - _fitted(model), dist="norm", plot=plt)
    plt.title('Q-Q Plot', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../images/02_inverse_gaussian.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存: tutorials/images/02_inverse_gaussian.png")
    plt.show()
    
    return model, data


def example3_insurance_claims():
    """示例 3: 保险索赔金额建模"""
    print("\n" + "=" * 60)
    print("示例 3: 保险索赔金额建模")
    print("=" * 60)
    
    # 生成模拟保险索赔数据
    np.random.seed(2024)
    n = 1000
    
    # 特征
    age = np.random.uniform(20, 70, n)
    vehicle_age = np.random.randint(0, 15, n)
    region = np.random.choice(['Urban', 'Suburban', 'Rural'], n, p=[0.4, 0.35, 0.25])
    
    # 索赔金额模型
    log_mu = (7.0 + 
              0.01 * age + 
              0.05 * vehicle_age + 
              0.3 * (region == 'Urban') +
              np.random.normal(0, 0.3, n))
    
    claim_amount = np.exp(log_mu)
    
    # 创建 DataFrame
    data = pd.DataFrame({
        'age': age,
        'vehicle_age': vehicle_age,
        'region': region,
        'claim_amount': claim_amount
    })
    
    print(f"\n数据维度: {data.shape}")
    print("\n索赔金额统计:")
    print(data['claim_amount'].describe())
    print("\n按地区统计:")
    print(data.groupby('region')['claim_amount'].describe())
    
    # 数据可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 索赔金额分布
    axes[0, 0].hist(data['claim_amount'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Claim Amount ($)', fontsize=11)
    axes[0, 0].set_ylabel('Frequency', fontsize=11)
    axes[0, 0].set_title('Distribution of Claim Amounts', fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 索赔金额 vs 年龄
    axes[0, 1].scatter(data['age'], data['claim_amount'], alpha=0.5, s=20)
    axes[0, 1].set_xlabel('Age', fontsize=11)
    axes[0, 1].set_ylabel('Claim Amount ($)', fontsize=11)
    axes[0, 1].set_title('Claim Amount vs Age', fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 索赔金额 vs 车龄
    axes[1, 0].scatter(data['vehicle_age'], data['claim_amount'], alpha=0.5, s=20)
    axes[1, 0].set_xlabel('Vehicle Age (years)', fontsize=11)
    axes[1, 0].set_ylabel('Claim Amount ($)', fontsize=11)
    axes[1, 0].set_title('Claim Amount vs Vehicle Age', fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    
    # 4. 按地区的箱线图
    data.boxplot(column='claim_amount', by='region', ax=axes[1, 1])
    axes[1, 1].set_xlabel('Region', fontsize=11)
    axes[1, 1].set_ylabel('Claim Amount ($)', fontsize=11)
    axes[1, 1].set_title('Claim Amount by Region', fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    plt.suptitle('')  # 移除默认标题
    
    plt.tight_layout()
    plt.savefig('../../images/02_claims_eda.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存: tutorials/images/02_claims_eda.png")
    plt.show()
    
    # 模型构建
    print("\n构建模型...")
    
    # 模型 1: 简单模型
    print("\n模型 1: 基础模型")
    model1 = gamlss(
        formula="claim_amount ~ age + vehicle_age",
        family=GA(),
        data=_data_dict(data)
    )
    print(f"Model 1 AIC: {_aic(model1):.2f}")
    
    # 模型 2: 添加地区因素
    print("\n模型 2: 添加地区因素")
    data['region_Urban'] = (data['region'] == 'Urban').astype(int)
    data['region_Suburban'] = (data['region'] == 'Suburban').astype(int)
    
    model2 = gamlss(
        formula="claim_amount ~ age + vehicle_age + region_Urban + region_Suburban",
        family=GA(),
        data=_data_dict(data)
    )
    print(f"Model 2 AIC: {_aic(model2):.2f}")
    print(f"AIC improvement: {_aic(model1) - _aic(model2):.2f}")
    
    # 模型 3: 建模方差
    print("\n模型 3: 建模方差")
    model3 = gamlss(
        formula="claim_amount ~ age + vehicle_age + region_Urban + region_Suburban",
        sigma_formula="~ age",
        family=GA(),
        data=_data_dict(data)
    )
    print(f"Model 3 AIC: {_aic(model3):.2f}")
    print(f"AIC improvement: {_aic(model2) - _aic(model3):.2f}")
    
    # 模型比较
    models = {
        'Model 1 (Basic)': model1,
        'Model 2 (+ Region)': model2,
        'Model 3 (+ Sigma)': model3
    }
    
    comparison = pd.DataFrame({
        'Model': list(models.keys()),
        'AIC': [_aic(m) for m in models.values()],
        'BIC': [_bic(m) for m in models.values()],
        'Deviance': [m.deviance for m in models.values()]
    })
    
    print("\n模型比较:")
    print(comparison)
    
    # 可视化模型比较
    plt.figure(figsize=(10, 6))
    x_pos = np.arange(len(comparison))
    plt.bar(x_pos, comparison['AIC'], alpha=0.7, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.xlabel('Model', fontsize=12)
    plt.ylabel('AIC', fontsize=12)
    plt.title('Model Comparison by AIC (Lower is Better)', fontsize=14)
    plt.xticks(x_pos, comparison['Model'], rotation=15, ha='right')
    plt.grid(True, alpha=0.3, axis='y')
    
    # 添加数值标签
    for i, v in enumerate(comparison['AIC']):
        plt.text(i, v + 20, f'{v:.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig('../../images/02_model_comparison.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存: tutorials/images/02_model_comparison.png")
    plt.show()
    
    # 结果解释
    print("\n最终模型系数解释:")
    print("\nMu 参数（均值模型）:")
    print(f"  截距: {_coef(model3, "mu")[0]:.4f}")
    print(f"  年龄效应: {_coef(model3, "mu")[1]:.4f}")
    print(f"  车龄效应: {_coef(model3, "mu")[2]:.4f}")
    print(f"  城市地区效应: {_coef(model3, "mu")[3]:.4f}")
    print(f"  郊区效应: {_coef(model3, "mu")[4]:.4f}")
    
    print("\nSigma 参数（方差模型）:")
    print(f"  截距: {_coef(model3, "sigma")[0]:.4f}")
    print(f"  年龄效应: {_coef(model3, "sigma")[1]:.4f}")
    
    # 业务解释
    print("\n业务解释:")
    print("1. 车龄每增加1年，索赔金额平均增加约 {:.1f}%".format(
        (np.exp(_coef(model3, "mu")[2]) - 1) * 100))
    print("2. 城市地区的索赔金额比农村地区高约 {:.1f}%".format(
        (np.exp(_coef(model3, "mu")[3]) - 1) * 100))
    print("3. 年龄越大，索赔金额的变异性越大")
    
    # 预测应用
    print("\n预测应用:")
    new_data = pd.DataFrame({
        'age': [30, 30, 50, 50],
        'vehicle_age': [2, 10, 2, 10],
        'region_Urban': [1, 1, 0, 0],
        'region_Suburban': [0, 0, 1, 1]
    })
    
    pred_mu = _predict_param(model3, new_data, "mu")
    pred_sigma = _predict_param(model3, new_data, "sigma")
    pred_lower = _predict_quantile(model3, new_data, 0.05)
    pred_upper = _predict_quantile(model3, new_data, 0.95)
    
    pred_results = pd.DataFrame({
        'Age': new_data['age'],
        'Vehicle_Age': new_data['vehicle_age'],
        'Region': ['Urban', 'Urban', 'Suburban', 'Suburban'],
        'Predicted_Mean': pred_mu,
        'Predicted_SD': pred_sigma,
        'Lower_5%': pred_lower,
        'Upper_95%': pred_upper
    })
    
    print("\n预测结果:")
    print(pred_results.to_string(index=False))
    
    return model1, model2, model3, data


def benchmark_performance():
    """性能测试"""
    print("\n" + "=" * 60)
    print("性能测试")
    print("=" * 60)
    
    import time
    
    n_samples = [100, 500, 1000, 5000, 10000]
    results = []
    
    print("\n运行性能测试...")
    print(f"{'样本量':<10} {'时间 (秒)':<15} {'每秒样本数':<15}")
    print("-" * 40)
    
    for n in n_samples:
        # 生成数据
        np.random.seed(42)
        x = np.random.randn(n)
        y = np.random.gamma(2, 1 + 0.5*np.abs(x))
        data = pd.DataFrame({'x': x, 'y': y})
        
        # 测试
        times = []
        for _ in range(3):  # 重复 3 次
            start = time.time()
            _ = gamlss(formula="y ~ x", family=GA(), data=_data_dict(data))
            times.append(time.time() - start)
        
        mean_time = np.mean(times)
        samples_per_sec = n / mean_time
        
        results.append({
            'n': n,
            'time': mean_time,
            'samples_per_sec': samples_per_sec
        })
        
        print(f"{n:<10} {mean_time:<15.4f} {samples_per_sec:<15.0f}")
    
    # 可视化
    results_df = pd.DataFrame(results)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(results_df['n'], results_df['time'], 'o-', linewidth=2, markersize=8)
    plt.xlabel('Sample Size', fontsize=12)
    plt.ylabel('Time (seconds)', fontsize=12)
    plt.title('Computation Time vs Sample Size', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(results_df['n'], results_df['samples_per_sec'], 'o-', 
             linewidth=2, markersize=8, color='green')
    plt.xlabel('Sample Size', fontsize=12)
    plt.ylabel('Samples per Second', fontsize=12)
    plt.title('Throughput vs Sample Size', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../images/02_performance.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存: tutorials/images/02_performance.png")
    plt.show()
    
    return results_df


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("OmniLSS 教程 02: Gamma 分布族")
    print("=" * 60)
    
    # 创建图像目录
    import os
    os.makedirs('../../images', exist_ok=True)
    
    # 运行示例
    try:
        model1, data1 = example1_simple_gamma()
        model2, data2 = example2_inverse_gaussian()
        m1, m2, m3, data3 = example3_insurance_claims()
        _results = benchmark_performance()
        
        print("\n" + "=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)
        print("\n生成的文件:")
        print("  - tutorials/images/02_gamma_basic.png")
        print("  - tutorials/images/02_inverse_gaussian.png")
        print("  - tutorials/images/02_claims_eda.png")
        print("  - tutorials/images/02_model_comparison.png")
        print("  - tutorials/images/02_performance.png")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
