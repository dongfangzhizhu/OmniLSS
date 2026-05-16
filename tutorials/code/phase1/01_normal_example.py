"""
正态分布族示例代码
Tutorial 01: Normal Distributions (NO, NO2, LOGNO, LOGNO2)

这个脚本包含了教程中所有的示例代码，可以直接运行。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import jax.numpy as jnp
import sys
sys.path.insert(0, '../../../omnilss/src')

import omnilss as om
from omnilss import NO, LOGNO

# 设置绘图样式
plt.style.use('seaborn-v0_8-darkgrid')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10


def example1_simple_linear():
    """示例 1: 简单线性回归（NO 分布）"""
    print("=" * 60)
    print("示例 1: 简单线性回归（NO 分布）")
    print("=" * 60)
    
    # 设置随机种子
    np.random.seed(42)
    
    # 生成示例数据
    n = 200
    x = np.linspace(0, 10, n)
    mu_true = 2 + 0.5 * x
    sigma_true = 0.5 + 0.05 * x  # 异方差
    
    y = np.random.normal(mu_true, sigma_true)
    
    # 创建 DataFrame
    data = pd.DataFrame({'x': x, 'y': y})
    
    print(f"\n数据维度: {data.shape}")
    print(f"数据预览:\n{data.head()}")
    
    # 拟合模型
    print("\n拟合模型...")
    model = jg.gamlss(
        formula="y ~ x",
        sigma_formula="~ x",
        family=NO(),
        data=data
    )
    
    # 查看结果
    print("\n模型摘要:")
    print(model.summary())
    
    print(f"\nMu 系数: {model.coef_mu}")
    print(f"Sigma 系数: {model.coef_sigma}")
    print(f"AIC: {model.aic:.2f}")
    
    # 可视化
    plt.figure(figsize=(12, 5))
    
    # 左图：数据和拟合线
    plt.subplot(1, 2, 1)
    plt.scatter(data['x'], data['y'], alpha=0.5, label='Observed')
    plt.plot(data['x'], model.fitted_values, 'r-', linewidth=2, label='Fitted')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Data and Fitted Values')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 右图：残差
    plt.subplot(1, 2, 2)
    residuals = data['y'] - model.fitted_values
    plt.scatter(model.fitted_values, residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residual Plot')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../images/01_example1_results.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存: tutorials/images/01_example1_results.png")
    plt.show()
    
    return model, data


def example2_lognormal():
    """示例 2: 对数正态分布（LOGNO）"""
    print("\n" + "=" * 60)
    print("示例 2: 对数正态分布（LOGNO）")
    print("=" * 60)
    
    # 生成对数正态数据
    np.random.seed(123)
    n = 300
    x = np.random.uniform(0, 5, n)
    
    # 真实参数（对数尺度）
    log_mu_true = 1.0 + 0.3 * x
    log_sigma_true = 0.5
    
    # 生成数据
    y = np.exp(np.random.normal(log_mu_true, log_sigma_true))
    
    data = pd.DataFrame({'x': x, 'y': y})
    
    print(f"\n数据维度: {data.shape}")
    print(f"数据统计:\n{data.describe()}")
    
    # 拟合模型
    print("\n拟合 LOGNO 模型...")
    model = jg.gamlss(
        formula="y ~ x",
        family=LOGNO(),
        data=data
    )
    
    print("\n模型摘要:")
    print(model.summary())
    
    print(f"\nMu 系数: {model.coef_mu}")
    print(f"Sigma 系数: {model.coef_sigma}")
    print(f"AIC: {model.aic:.2f}")
    
    # 可视化
    plt.figure(figsize=(12, 5))
    
    # 左图：原始尺度
    plt.subplot(1, 2, 1)
    plt.scatter(data['x'], data['y'], alpha=0.5)
    
    # 预测曲线
    x_pred = np.linspace(0, 5, 100)
    data_pred = pd.DataFrame({'x': x_pred})
    y_pred = model.predict(data_pred, what='mu')
    plt.plot(x_pred, y_pred, 'r-', linewidth=2, label='Fitted')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Log-Normal Data (Original Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 右图：对数尺度
    plt.subplot(1, 2, 2)
    plt.scatter(data['x'], np.log(data['y']), alpha=0.5)
    plt.plot(x_pred, np.log(y_pred), 'r-', linewidth=2, label='Fitted')
    plt.xlabel('x')
    plt.ylabel('log(y)')
    plt.title('Log-Normal Data (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../images/01_example2_lognormal.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存: tutorials/images/01_example2_lognormal.png")
    plt.show()
    
    return model, data


def example3_height_data():
    """示例 3: 身高数据建模（实际案例）"""
    print("\n" + "=" * 60)
    print("示例 3: 身高数据建模")
    print("=" * 60)
    
    # 生成模拟身高数据
    np.random.seed(2024)
    n = 500
    
    # 模拟儿童身高数据（年龄 2-18 岁）
    age = np.random.uniform(2, 18, n)
    
    # 身高随年龄增长，方差也增加
    height_mean = 80 + 8 * age - 0.15 * age**2
    height_sd = 3 + 0.2 * age
    
    height = np.random.normal(height_mean, height_sd)
    
    data = pd.DataFrame({
        'age': age,
        'height': height
    })
    
    print(f"\n数据维度: {data.shape}")
    print(f"年龄范围: {data['age'].min():.1f} - {data['age'].max():.1f}")
    print(f"身高范围: {data['height'].min():.1f} - {data['height'].max():.1f}")
    
    # 模型 1: 简单线性模型
    print("\n拟合模型 1: 线性模型...")
    model1 = jg.gamlss(
        formula="height ~ age",
        family=NO(),
        data=data
    )
    
    print(f"模型 1 AIC: {model1.aic:.2f}")
    
    # 模型 2: 二次模型
    print("\n拟合模型 2: 二次模型...")
    data['age2'] = data['age'] ** 2
    
    model2 = jg.gamlss(
        formula="height ~ age + age2",
        sigma_formula="~ age",
        family=NO(),
        data=data
    )
    
    print(f"模型 2 AIC: {model2.aic:.2f}")
    print(f"\nAIC 改进: {model1.aic - model2.aic:.2f}")
    
    # 可视化
    plt.figure(figsize=(14, 6))
    
    # 左图：两个模型对比
    plt.subplot(1, 2, 1)
    plt.scatter(data['age'], data['height'], alpha=0.3, s=20, label='Observed')
    
    # 预测
    age_pred = np.linspace(2, 18, 100)
    data_pred = pd.DataFrame({
        'age': age_pred,
        'age2': age_pred ** 2
    })
    
    # 模型 1 预测
    data_pred1 = pd.DataFrame({'age': age_pred})
    pred1 = model1.predict(data_pred1, what='mu')
    plt.plot(age_pred, pred1, 'g--', linewidth=2, label='Model 1 (Linear)')
    
    # 模型 2 预测
    pred2_mu = model2.predict(data_pred, what='mu')
    plt.plot(age_pred, pred2_mu, 'r-', linewidth=2, label='Model 2 (Quadratic)')
    
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('Height (cm)', fontsize=12)
    plt.title('Model Comparison', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 右图：模型 2 带预测区间
    plt.subplot(1, 2, 2)
    plt.scatter(data['age'], data['height'], alpha=0.3, s=20, label='Observed')
    
    pred2_sigma = model2.predict(data_pred, what='sigma')
    
    plt.plot(age_pred, pred2_mu, 'r-', linewidth=2, label='Predicted mean')
    plt.fill_between(age_pred,
                     pred2_mu - 2*pred2_sigma,
                     pred2_mu + 2*pred2_sigma,
                     alpha=0.2, color='red', label='95% prediction interval')
    
    plt.xlabel('Age (years)', fontsize=12)
    plt.ylabel('Height (cm)', fontsize=12)
    plt.title('Height Growth Curve with Prediction Intervals', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('../../images/01_example3_height.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存: tutorials/images/01_example3_height.png")
    plt.show()
    
    return model1, model2, data


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
        y = 2 + 0.5 * x + np.random.randn(n) * 0.5
        data = pd.DataFrame({'x': x, 'y': y})
        
        # 测试
        times = []
        for _ in range(3):  # 重复 3 次
            start = time.time()
            model = jg.gamlss(formula="y ~ x", family=NO(), data=data)
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
    plt.savefig('../../images/01_performance.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存: tutorials/images/01_performance.png")
    plt.show()
    
    return results_df


def main():
    """运行所有示例"""
    print("\n" + "=" * 60)
    print("OmniLSS 教程 01: 正态分布族")
    print("=" * 60)
    
    # 创建图像目录
    import os
    os.makedirs('../../images', exist_ok=True)
    
    # 运行示例
    try:
        model1, data1 = example1_simple_linear()
        model2, data2 = example2_lognormal()
        model_h1, model_h2, data_h = example3_height_data()
        results = benchmark_performance()
        
        print("\n" + "=" * 60)
        print("所有示例运行完成！")
        print("=" * 60)
        print("\n生成的文件:")
        print("  - tutorials/images/01_example1_results.png")
        print("  - tutorials/images/01_example2_lognormal.png")
        print("  - tutorials/images/01_example3_height.png")
        print("  - tutorials/images/01_performance.png")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
