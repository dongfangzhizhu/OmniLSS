"""模型选择演示

展示 OmniLSS 的自动分布选择功能，包括：
1. 比较多个分布族
2. 自动选择最佳分布
3. 快速分布搜索
4. 可视化比较结果
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "omnilss/src")

from omnilss.model_selection import (
    compare_distributions,
    select_best_distribution,
    quick_distribution_search
)


def demo_compare_distributions():
    """演示 1: 比较多个分布族"""
    print("=" * 70)
    print("演示 1: 比较多个分布族")
    print("=" * 70)
    
    # 生成数据（Log-normal 分布）
    np.random.seed(42)
    n = 200
    x = np.random.uniform(0, 10, n)
    # 真实模型：log(y) ~ N(1 + 0.5*x, 0.3)
    log_y = 1 + 0.5*x + np.random.randn(n) * 0.3
    y = np.exp(log_y)
    data = {"y": y, "x": x}
    
    print(f"\n数据: n={n}, y 是 Log-normal 分布")
    print(f"  y 范围: [{y.min():.2f}, {y.max():.2f}]")
    print(f"  y 均值: {y.mean():.2f}, 标准差: {y.std():.2f}")
    
    # 比较多个分布
    print("\n比较 4 个分布族...")
    results = compare_distributions(
        "y ~ x",
        families=["NO", "GA", "LOGNO", "WEI"],
        data=data,
        criterion="AIC",
        verbose=True
    )
    
    print("\n详细结果:")
    print(results[["family", "deviance", "df", "AIC", "BIC", "converged"]])
    
    # 可视化
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # AIC 比较
    ax1.bar(results["family"], results["AIC"], color=['green' if i == 0 else 'gray' for i in range(len(results))])
    ax1.set_xlabel("Distribution Family")
    ax1.set_ylabel("AIC")
    ax1.set_title("AIC Comparison (Lower is Better)")
    ax1.grid(True, alpha=0.3)
    
    # BIC 比较
    ax2.bar(results["family"], results["BIC"], color=['blue' if i == 0 else 'gray' for i in range(len(results))])
    ax2.set_xlabel("Distribution Family")
    ax2.set_ylabel("BIC")
    ax2.set_title("BIC Comparison (Lower is Better)")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("model_selection_comparison.png", dpi=300)
    print("\n图表已保存: model_selection_comparison.png")
    
    return results


def demo_select_best_distribution():
    """演示 2: 自动选择最佳分布"""
    print("\n" + "=" * 70)
    print("演示 2: 自动选择最佳分布")
    print("=" * 70)
    
    # 生成数据（Gamma 分布）
    np.random.seed(42)
    n = 200
    x = np.random.randn(n)
    shape = 2.0
    scale = np.exp(1 + 0.5*x)
    y = np.random.gamma(shape, scale)
    data = {"y": y, "x": x}
    
    print(f"\n数据: n={n}, y 是 Gamma 分布")
    print(f"  y 范围: [{y.min():.2f}, {y.max():.2f}]")
    
    # 自动选择最佳分布
    print("\n自动选择最佳分布...")
    best_model, comparison = select_best_distribution(
        "y ~ x",
        families=["NO", "GA", "LOGNO", "WEI", "IG"],
        data=data,
        criterion="AIC",
        verbose=True
    )
    
    print(f"\n最佳分布: {best_model.family.name}")
    print(f"  Deviance: {best_model.g_dev:.2f}")
    print(f"  AIC: {best_model.additional_slots['aic']:.2f}")
    print(f"  BIC: {best_model.additional_slots['sbc']:.2f}")
    
    # 使用最佳模型进行预测
    print("\n使用最佳模型预测...")
    newdata = {"x": np.array([-1, 0, 1])}
    quantiles = best_model.predict_quantiles(
        newdata,
        quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
    )
    
    print("\n预测结果:")
    print(f"  x    |   5%    |  25%    |  50%    |  75%    |  95%")
    print("  " + "-" * 60)
    for i, x_val in enumerate(newdata["x"]):
        print(f"  {x_val:3.0f}  | {quantiles[0.05][i]:7.2f} | {quantiles[0.25][i]:7.2f} | "
              f"{quantiles[0.5][i]:7.2f} | {quantiles[0.75][i]:7.2f} | {quantiles[0.95][i]:7.2f}")
    
    return best_model, comparison


def demo_quick_distribution_search():
    """演示 3: 快速分布搜索"""
    print("\n" + "=" * 70)
    print("演示 3: 快速分布搜索（自动检测数据类型）")
    print("=" * 70)
    
    # 生成不同类型的数据
    np.random.seed(42)
    n = 150
    x = np.random.randn(n)
    
    # 3.1 连续正数数据
    print("\n3.1 连续正数数据")
    y_positive = np.exp(2 + 0.5*x + np.random.randn(n)*0.3)
    data_positive = {"y": y_positive, "x": x}
    
    best_model_pos, comparison_pos = quick_distribution_search(
        "y ~ x",
        data=data_positive,
        data_type="auto",
        verbose=True
    )
    
    print(f"\n  检测到: 连续正数")
    print(f"  最佳分布: {best_model_pos.family.name}")
    
    # 3.2 计数数据
    print("\n3.2 计数数据")
    lambda_ = np.exp(1 + 0.5*x)
    y_count = np.random.poisson(lambda_)
    data_count = {"y": y_count, "x": x}
    
    best_model_count, comparison_count = quick_distribution_search(
        "y ~ x",
        data=data_count,
        data_type="auto",
        verbose=True
    )
    
    print(f"\n  检测到: 计数数据")
    print(f"  最佳分布: {best_model_count.family.name}")
    
    # 3.3 连续实数数据
    print("\n3.3 连续实数数据")
    y_real = 2 + 3*x + np.random.randn(n)
    data_real = {"y": y_real, "x": x}
    
    best_model_real, comparison_real = quick_distribution_search(
        "y ~ x",
        data=data_real,
        data_type="auto",
        verbose=True
    )
    
    print(f"\n  检测到: 连续实数")
    print(f"  最佳分布: {best_model_real.family.name}")


def demo_visualization():
    """演示 4: 可视化不同分布的拟合效果"""
    print("\n" + "=" * 70)
    print("演示 4: 可视化不同分布的拟合效果")
    print("=" * 70)
    
    # 生成数据
    np.random.seed(42)
    n = 200
    x = np.random.uniform(0, 10, n)
    log_y = 1 + 0.5*x + np.random.randn(n) * 0.3
    y = np.exp(log_y)
    data = {"y": y, "x": x}
    
    # 比较分布
    families = ["NO", "GA", "LOGNO", "WEI"]
    results = compare_distributions(
        "y ~ x",
        families=families,
        data=data,
        criterion="AIC",
        verbose=False
    )
    
    # 为每个分布拟合模型并预测
    from omnilss import gamlss
    
    x_new = np.linspace(0, 10, 100)
    newdata = {"x": x_new}
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, family_name in enumerate(families):
        ax = axes[i]
        
        # 拟合模型
        model = gamlss("y ~ x", family=family_name, data=data)
        
        # 预测
        quantiles = model.predict_quantiles(
            newdata,
            quantiles=[0.05, 0.25, 0.5, 0.75, 0.95]
        )
        
        # 绘图
        ax.scatter(x, y, alpha=0.3, s=20, color='gray', label='Data')
        ax.plot(x_new, quantiles[0.5], 'r-', linewidth=2, label='Median')
        ax.fill_between(x_new, quantiles[0.05], quantiles[0.95], 
                        alpha=0.2, color='red', label='90% PI')
        ax.fill_between(x_new, quantiles[0.25], quantiles[0.75], 
                        alpha=0.3, color='orange', label='50% PI')
        
        # 标题
        aic = results[results["family"] == family_name]["AIC"].values[0]
        rank = results[results["family"] == family_name].index[0] + 1
        ax.set_title(f"{family_name} (Rank #{rank}, AIC={aic:.1f})", 
                    fontweight='bold')
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("model_selection_fits.png", dpi=300)
    print("\n图表已保存: model_selection_fits.png")


def main():
    """运行所有演示"""
    print("\n" + "=" * 70)
    print("OmniLSS 模型选择功能演示")
    print("=" * 70)
    print("\n这个演示展示了如何自动选择最适合数据的分布族")
    print("\n包括:")
    print("  1. 比较多个分布族")
    print("  2. 自动选择最佳分布")
    print("  3. 快速分布搜索（自动检测数据类型）")
    print("  4. 可视化不同分布的拟合效果")
    
    # 运行演示
    results = demo_compare_distributions()
    best_model, comparison = demo_select_best_distribution()
    demo_quick_distribution_search()
    demo_visualization()
    
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)
    print("\n生成的文件:")
    print("  - model_selection_comparison.png")
    print("  - model_selection_fits.png")
    print("\n关键要点:")
    print("  ✓ 可以自动比较多个分布族")
    print("  ✓ 使用 AIC/BIC 选择最佳分布")
    print("  ✓ 自动检测数据类型并推荐分布")
    print("  ✓ 可视化不同分布的拟合效果")
    print("  ✓ 对标 R gamlss 的分布选择功能")


if __name__ == "__main__":
    main()
