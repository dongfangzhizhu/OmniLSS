"""
演示新实现的功能：Probit Link 和 Cook's Distance

本示例展示如何使用：
1. Probit link function
2. Cook's distance 诊断

作者: Kiro AI Assistant
日期: 2026-05-03
"""

import numpy as np
import jax.numpy as jnp
from omnilss import gamlss, NO, BI
from omnilss.links import probit_link, probit_inverse, probit_derivative
from omnilss.diagnostics import cooks_distance, comprehensive_diagnostics


def demo_probit_link():
    """演示 Probit Link Function"""
    print("=" * 80)
    print("演示 1: Probit Link Function")
    print("=" * 80)
    print()
    
    # 1. 基础使用
    print("1. 基础使用:")
    print("-" * 40)
    mu = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
    print(f"概率 μ: {mu}")
    
    # Probit link: 将概率映射到实数线
    eta = probit_link(mu)
    print(f"线性预测器 η = probit(μ): {eta}")
    
    # Inverse probit: 将实数线映射回概率
    mu_recovered = probit_inverse(eta)
    print(f"恢复的概率 μ = probit^(-1)(η): {mu_recovered}")
    
    # 验证逆关系
    print(f"误差: {jnp.max(jnp.abs(mu - mu_recovered)):.2e}")
    print()
    
    # 2. 导数计算
    print("2. 导数计算:")
    print("-" * 40)
    eta_test = jnp.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    deriv = probit_derivative(eta_test)
    print(f"η: {eta_test}")
    print(f"dμ/dη = φ(η): {deriv}")
    print()
    
    # 3. 与 Logit Link 对比
    print("3. Probit vs Logit:")
    print("-" * 40)
    from omnilss.links import logit_link
    
    mu_compare = jnp.array([0.1, 0.3, 0.5, 0.7, 0.9])
    eta_probit = probit_link(mu_compare)
    eta_logit = logit_link(mu_compare)
    
    print(f"{'μ':<8} {'Probit':<12} {'Logit':<12} {'差异':<12}")
    print("-" * 48)
    for m, p, l in zip(mu_compare, eta_probit, eta_logit):
        diff = abs(p - l)
        print(f"{m:<8.2f} {p:<12.4f} {l:<12.4f} {diff:<12.4f}")
    print()
    
    # 4. 对称性
    print("4. 对称性验证:")
    print("-" * 40)
    print("Probit link 满足: probit(1-p) = -probit(p)")
    mu_sym = jnp.array([0.1, 0.2, 0.3, 0.4])
    eta_low = probit_link(mu_sym)
    eta_high = probit_link(1.0 - mu_sym)
    print(f"μ: {mu_sym}")
    print(f"probit(μ): {eta_low}")
    print(f"probit(1-μ): {eta_high}")
    print(f"-probit(μ): {-eta_low}")
    print(f"误差: {jnp.max(jnp.abs(eta_high + eta_low)):.2e}")
    print()


def demo_cooks_distance():
    """演示 Cook's Distance"""
    print("=" * 80)
    print("演示 2: Cook's Distance")
    print("=" * 80)
    print()
    
    # 1. 生成数据（包含异常值）
    print("1. 生成数据:")
    print("-" * 40)
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 10, n)
    y = 5 + 0.5 * x + np.random.normal(0, 1, n)
    
    # 添加异常值
    outlier_indices = [25, 50, 75]
    for idx in outlier_indices:
        y[idx] = 50
    
    print(f"样本量: {n}")
    print(f"真实模型: y = 5 + 0.5*x + ε, ε ~ N(0, 1)")
    print(f"异常值位置: {outlier_indices}")
    print(f"异常值: y = 50")
    print()
    
    # 2. 拟合模型
    print("2. 拟合模型:")
    print("-" * 40)
    data = {'x': x, 'y': y}
    model = gamlss(formula="y ~ x", family=NO(), data=data)
    print("模型拟合完成")
    print()
    
    # 3. 计算 Cook's distance
    print("3. 计算 Cook's Distance:")
    print("-" * 40)
    result = cooks_distance(model)
    
    print(f"观测数量: {len(result.cooks_distance)}")
    print(f"阈值 (4/n): {result.threshold:.4f}")
    print(f"有影响的观测数量: {result.n_influential}")
    print()
    
    # 4. 显示有影响的观测
    print("4. 有影响的观测:")
    print("-" * 40)
    influential_idx = result.index[result.influential] - 1  # 转换为 0-based
    influential_cooks = result.cooks_distance[result.influential]
    
    print(f"{'索引':<8} {'Cook D':<12} {'x':<12} {'y':<12}")
    print("-" * 48)
    for idx, cd in zip(influential_idx, influential_cooks):
        idx_int = int(idx)
        print(f"{idx_int:<8} {cd:<12.4f} {x[idx_int]:<12.2f} {y[idx_int]:<12.2f}")
    print()
    
    # 5. 统计信息
    print("5. Cook's Distance 统计:")
    print("-" * 40)
    print(f"最小值: {np.min(result.cooks_distance):.4f}")
    print(f"中位数: {np.median(result.cooks_distance):.4f}")
    print(f"平均值: {np.mean(result.cooks_distance):.4f}")
    print(f"最大值: {np.max(result.cooks_distance):.4f}")
    print(f"标准差: {np.std(result.cooks_distance):.4f}")
    print()
    
    # 6. 验证异常值检测
    print("6. 异常值检测验证:")
    print("-" * 40)
    detected = set(influential_idx.astype(int))
    expected = set(outlier_indices)
    
    print(f"预期异常值: {expected}")
    print(f"检测到的异常值: {detected}")
    print(f"正确检测: {detected == expected}")
    print()


def demo_comprehensive_diagnostics():
    """演示综合诊断（包含 Cook's distance）"""
    print("=" * 80)
    print("演示 3: 综合诊断")
    print("=" * 80)
    print()
    
    # 生成数据
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 10, n)
    y = 5 + 0.5 * x + np.random.normal(0, 1, n)
    y[50] = 50  # 添加一个异常值
    
    data = {'x': x, 'y': y}
    model = gamlss(formula="y ~ x", family=NO(), data=data)
    
    # 运行综合诊断
    print("运行综合诊断...")
    diag = comprehensive_diagnostics(model)
    
    # 显示结果
    print()
    print("诊断结果:")
    print("-" * 40)
    print(f"1. 分位数残差:")
    print(f"   均值: {diag.quantile_residuals.mean:.4f} (应接近 0)")
    print(f"   方差: {diag.quantile_residuals.variance:.4f} (应接近 1)")
    print(f"   偏度: {diag.quantile_residuals.skewness:.4f} (应接近 0)")
    print(f"   峰度: {diag.quantile_residuals.kurtosis:.4f} (应接近 0)")
    print()
    
    print(f"2. Q-Q Plot:")
    print(f"   Filliben 相关系数: {diag.qq_plot.correlation:.4f} (应接近 1)")
    print()
    
    print(f"3. Worm Plot:")
    n_outside = np.sum((diag.worm_plot.deviations < diag.worm_plot.lower_band) |
                       (diag.worm_plot.deviations > diag.worm_plot.upper_band))
    pct_outside = 100.0 * n_outside / diag.worm_plot.n
    print(f"   超出 95% CI 的点: {n_outside}/{diag.worm_plot.n} ({pct_outside:.1f}%)")
    print(f"   预期超出: ~{int(0.05 * diag.worm_plot.n)} (5%)")
    print()
    
    print(f"4. Cook's Distance:")
    print(f"   有影响的观测: {diag.cooks_distance.n_influential}")
    print(f"   阈值: {diag.cooks_distance.threshold:.4f}")
    if diag.cooks_distance.n_influential > 0:
        influential_idx = diag.cooks_distance.index[diag.cooks_distance.influential]
        print(f"   索引: {influential_idx}")
    print()


def demo_custom_threshold():
    """演示自定义 Cook's distance 阈值"""
    print("=" * 80)
    print("演示 4: 自定义 Cook's Distance 阈值")
    print("=" * 80)
    print()
    
    # 生成数据
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 10, n)
    y = 5 + 0.5 * x + np.random.normal(0, 1, n)
    y[50] = 20  # 添加一个中等异常值
    
    data = {'x': x, 'y': y}
    model = gamlss(formula="y ~ x", family=NO(), data=data)
    
    # 使用不同的阈值
    thresholds = [4/n, 0.05, 0.1, 0.2]
    
    print(f"{'阈值':<12} {'有影响的观测':<20} {'百分比':<12}")
    print("-" * 48)
    
    for threshold in thresholds:
        result = cooks_distance(model, threshold=threshold)
        pct = 100.0 * result.n_influential / n
        print(f"{threshold:<12.4f} {result.n_influential:<20} {pct:<12.1f}%")
    
    print()
    print("说明:")
    print("- 阈值越高，检测到的有影响观测越少")
    print("- 常用阈值: 4/n (默认)")
    print("- 可根据具体情况调整阈值")
    print()


def main():
    """运行所有演示"""
    print("\n")
    print("=" * 80)
    print("OmniLSS 新功能演示")
    print("=" * 80)
    print()
    print("本演示展示以下新功能:")
    print("1. Probit Link Function")
    print("2. Cook's Distance")
    print("3. 综合诊断（包含 Cook's distance）")
    print("4. 自定义 Cook's distance 阈值")
    print()
    
    try:
        # 演示 1: Probit Link
        demo_probit_link()
        
        # 演示 2: Cook's Distance
        demo_cooks_distance()
        
        # 演示 3: 综合诊断
        demo_comprehensive_diagnostics()
        
        # 演示 4: 自定义阈值
        demo_custom_threshold()
        
        print("=" * 80)
        print("所有演示完成！")
        print("=" * 80)
        print()
        print("下一步:")
        print("1. 查看文档: docs/GAMLSS_COMPLETENESS_CHECKLIST.md")
        print("2. 运行测试: python -m pytest omnilss/tests/ -v")
        print("3. 查看更多示例: examples/")
        print()
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
