"""预测功能演示

展示 OmniLSS 的预测功能，包括：
1. 参数预测
2. 分位数预测
3. Centile curves 生成
4. 可视化

这是 GAMLSS 的核心价值：预测完整的条件分布，而不仅仅是条件均值。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "omnilss/src")

from omnilss import gamlss


def demo_basic_prediction():
    """演示基本预测功能"""
    print("=" * 70)
    print("演示 1: 基本预测功能")
    print("=" * 70)
    
    # 生成数据
    np.random.seed(42)
    n = 200
    x = np.random.uniform(0, 10, n)
    # 真实模型：y ~ N(2 + 3*x, 1 + 0.2*x)
    mu_true = 2 + 3*x
    sigma_true = 1 + 0.2*x
    y = np.random.normal(mu_true, sigma_true)
    
    data = {"y": y, "x": x}
    
    # 拟合模型
    print("\n拟合模型: y ~ x, sigma ~ x")
    model = gamlss("y ~ x", sigma_formula="~ x", family="NO", data=data)
    print(f"  Deviance: {model.g_dev:.2f}")
    print(f"  AIC: {model.additional_slots['aic']:.2f}")
    
    # 预测新数据
    print("\n预测新数据:")
    newdata = {"x": np.array([2, 5, 8])}
    
    # 1. 参数预测
    print("\n1. 参数预测 (predict_params):")
    params = model.predict_params(newdata)
    print(f"   x    |   μ (预测)  |  σ (预测)")
    print("   " + "-" * 40)
    for i, x_val in enumerate(newdata["x"]):
        print(f"   {x_val:3.0f}  |  {params['mu'][i]:8.2f}  |  {params['sigma'][i]:8.2f}")
    
    # 2. 分位数预测
    print("\n2. 分位数预测 (predict_quantiles):")
    quantiles = model.predict_quantiles(newdata, quantiles=[0.05, 0.5, 0.95])
    print(f"   x    |   5%    |   50%   |   95%")
    print("   " + "-" * 45)
    for i, x_val in enumerate(newdata["x"]):
        print(f"   {x_val:3.0f}  | {quantiles[0.05][i]:7.2f} | {quantiles[0.5][i]:7.2f} | {quantiles[0.95][i]:7.2f}")
    
    # 3. 响应变量预测
    print("\n3. 响应变量预测 (predict):")
    y_pred = model.predict(newdata, type="response")
    print(f"   x    |  y (预测)")
    print("   " + "-" * 25)
    for i, x_val in enumerate(newdata["x"]):
        print(f"   {x_val:3.0f}  |  {y_pred[i]:8.2f}")


def demo_centile_curves():
    """演示 centile curves 生成"""
    print("\n" + "=" * 70)
    print("演示 2: Centile Curves")
    print("=" * 70)
    
    # 生成数据（模拟生长曲线）
    np.random.seed(42)
    n = 300
    age = np.random.uniform(0, 100, n)
    # 真实模型：身高随年龄变化，均值和方差都变化
    mu_true = 170 + 0.5*age - 0.005*age**2
    sigma_true = 5 + 0.1*age
    height = np.random.normal(mu_true, sigma_true)
    
    data = {"height": height, "age": age}
    
    # 拟合模型
    print("\n拟合生长曲线模型: height ~ age, sigma ~ age")
    model = gamlss("height ~ age", sigma_formula="~ age", family="NO", data=data)
    print(f"  Deviance: {model.g_dev:.2f}")
    
    # 生成 centile curves
    print("\n生成 centile curves...")
    curves = model.centiles(xvar="age", cent=[5, 25, 50, 75, 95], n_points=100)
    print(f"  生成了 {len(curves)} 个点")
    print(f"  列: {list(curves.columns)}")
    
    # 显示部分数据
    print("\n前 5 行:")
    print(curves.head())
    
    # 可视化
    print("\n生成可视化...")
    plt.figure(figsize=(12, 7))
    
    # 绘制原始数据
    plt.scatter(age, height, alpha=0.3, s=20, color='gray', label='Data')
    
    # 绘制 centile curves
    colors = ['#e74c3c', '#f39c12', '#2ecc71', '#f39c12', '#e74c3c']
    linestyles = ['--', '-.', '-', '-.', '--']
    linewidths = [2, 2, 3, 2, 2]
    
    for i, c in enumerate([5, 25, 50, 75, 95]):
        plt.plot(
            curves["age"], 
            curves[f"C{c}"], 
            label=f"{c}%", 
            color=colors[i],
            linestyle=linestyles[i],
            linewidth=linewidths[i]
        )
    
    plt.xlabel("Age", fontsize=12)
    plt.ylabel("Height", fontsize=12)
    plt.title("Growth Centile Curves", fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图表
    output_file = "centile_curves_demo.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存到: {output_file}")
    
    return curves


def demo_prediction_intervals():
    """演示预测区间"""
    print("\n" + "=" * 70)
    print("演示 3: 预测区间")
    print("=" * 70)
    
    # 生成数据
    np.random.seed(42)
    n = 150
    x = np.random.uniform(0, 10, n)
    y = 5 + 2*x + np.random.normal(0, 2, n)
    
    data = {"y": y, "x": x}
    
    # 拟合模型
    print("\n拟合模型: y ~ x")
    model = gamlss("y ~ x", family="NO", data=data)
    
    # 生成预测点
    x_new = np.linspace(0, 10, 50)
    newdata = {"x": x_new}
    
    # 预测多个分位数
    print("\n计算预测区间...")
    quantiles = model.predict_quantiles(
        newdata,
        quantiles=[0.025, 0.05, 0.25, 0.5, 0.75, 0.95, 0.975]
    )
    
    # 可视化
    print("生成可视化...")
    plt.figure(figsize=(12, 7))
    
    # 绘制原始数据
    plt.scatter(x, y, alpha=0.5, s=30, color='steelblue', label='Data', zorder=3)
    
    # 绘制中位数
    plt.plot(x_new, quantiles[0.5], 'r-', linewidth=3, label='Median (50%)', zorder=4)
    
    # 绘制预测区间
    plt.fill_between(
        x_new, 
        quantiles[0.025], 
        quantiles[0.975],
        alpha=0.2, 
        color='red', 
        label='95% Prediction Interval',
        zorder=1
    )
    plt.fill_between(
        x_new, 
        quantiles[0.05], 
        quantiles[0.95],
        alpha=0.3, 
        color='orange', 
        label='90% Prediction Interval',
        zorder=2
    )
    plt.fill_between(
        x_new, 
        quantiles[0.25], 
        quantiles[0.75],
        alpha=0.4, 
        color='yellow', 
        label='50% Prediction Interval (IQR)',
        zorder=2
    )
    
    plt.xlabel("X", fontsize=12)
    plt.ylabel("Y", fontsize=12)
    plt.title("Prediction Intervals", fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # 保存图表
    output_file = "prediction_intervals_demo.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存到: {output_file}")


def demo_comparison_with_traditional():
    """对比传统回归与 GAMLSS"""
    print("\n" + "=" * 70)
    print("演示 4: GAMLSS vs 传统回归")
    print("=" * 70)
    
    # 生成异方差数据
    np.random.seed(42)
    n = 200
    x = np.random.uniform(0, 10, n)
    # 方差随 x 增加
    sigma = 1 + 0.5*x
    y = 5 + 2*x + np.random.normal(0, sigma, n)
    
    data = {"y": y, "x": x}
    
    # 1. 传统回归（假设同方差）
    print("\n1. 传统回归（假设 σ 恒定）:")
    model_traditional = gamlss("y ~ x", family="NO", data=data)
    print(f"   Deviance: {model_traditional.g_dev:.2f}")
    print(f"   AIC: {model_traditional.additional_slots['aic']:.2f}")
    
    # 2. GAMLSS（允许异方差）
    print("\n2. GAMLSS（允许 σ 随 x 变化）:")
    model_gamlss = gamlss("y ~ x", sigma_formula="~ x", family="NO", data=data)
    print(f"   Deviance: {model_gamlss.g_dev:.2f}")
    print(f"   AIC: {model_gamlss.additional_slots['aic']:.2f}")
    
    # 比较
    print("\n模型比较:")
    print(f"   Deviance 改进: {model_traditional.g_dev - model_gamlss.g_dev:.2f}")
    print(f"   AIC 改进: {model_traditional.additional_slots['aic'] - model_gamlss.additional_slots['aic']:.2f}")
    
    # 可视化对比
    print("\n生成对比可视化...")
    x_new = np.linspace(0, 10, 50)
    newdata = {"x": x_new}
    
    # 传统回归的预测区间
    q_trad = model_traditional.predict_quantiles(newdata, quantiles=[0.05, 0.5, 0.95])
    
    # GAMLSS 的预测区间
    q_gamlss = model_gamlss.predict_quantiles(newdata, quantiles=[0.05, 0.5, 0.95])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 左图：传统回归
    ax1.scatter(x, y, alpha=0.5, s=20, color='gray')
    ax1.plot(x_new, q_trad[0.5], 'r-', linewidth=2, label='Median')
    ax1.fill_between(x_new, q_trad[0.05], q_trad[0.95], alpha=0.3, color='red', label='90% PI')
    ax1.set_xlabel("X", fontsize=12)
    ax1.set_ylabel("Y", fontsize=12)
    ax1.set_title("Traditional Regression\n(Constant σ)", fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 右图：GAMLSS
    ax2.scatter(x, y, alpha=0.5, s=20, color='gray')
    ax2.plot(x_new, q_gamlss[0.5], 'b-', linewidth=2, label='Median')
    ax2.fill_between(x_new, q_gamlss[0.05], q_gamlss[0.95], alpha=0.3, color='blue', label='90% PI')
    ax2.set_xlabel("X", fontsize=12)
    ax2.set_ylabel("Y", fontsize=12)
    ax2.set_title("GAMLSS\n(σ varies with x)", fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图表
    output_file = "gamlss_vs_traditional_demo.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n图表已保存到: {output_file}")


def main():
    """运行所有演示"""
    print("\n" + "=" * 70)
    print("OmniLSS 预测功能演示")
    print("=" * 70)
    print("\n这个演示展示了 GAMLSS 的核心价值：")
    print("  预测完整的条件分布，而不仅仅是条件均值")
    print("\n包括:")
    print("  1. 基本预测功能（参数、分位数、响应变量）")
    print("  2. Centile curves 生成")
    print("  3. 预测区间")
    print("  4. 与传统回归的对比")
    
    # 运行演示
    demo_basic_prediction()
    curves = demo_centile_curves()
    demo_prediction_intervals()
    demo_comparison_with_traditional()
    
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    print("\n生成的文件:")
    print("  - centile_curves_demo.png")
    print("  - prediction_intervals_demo.png")
    print("  - gamlss_vs_traditional_demo.png")
    print("\n关键要点:")
    print("  ✓ GAMLSS 预测完整分布，不仅仅是均值")
    print("  ✓ 可以获得任意分位数的预测")
    print("  ✓ Centile curves 是强大的可视化工具")
    print("  ✓ 预测区间更准确（特别是异方差情况）")
    print("  ✓ 对标 R gamlss 的核心功能")


if __name__ == "__main__":
    main()
