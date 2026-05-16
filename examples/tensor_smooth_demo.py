"""Tensor Product Smooth 演示

演示如何使用 tensor product smooth (te, ti) 进行多维平滑。

Tensor product smooth 是处理多变量交互的关键技术，允许对多个变量
的联合效应进行平滑建模。
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from omnilss.smoothers.tensor_smooth import create_tensor_basis, te, ti


def demo_2d_tensor_smooth():
    """演示 2D tensor product smooth"""
    print("=" * 60)
    print("演示 1: 2D Tensor Product Smooth")
    print("=" * 60)
    
    # 生成数据
    np.random.seed(42)
    n = 300
    x1 = np.random.uniform(-2, 2, n)
    x2 = np.random.uniform(-2, 2, n)
    
    # 真实函数: f(x1, x2) = x1^2 + x2^2 + x1*x2
    # 这个函数有明显的交互效应
    y_true = x1**2 + x2**2 + x1*x2
    y = y_true + np.random.randn(n) * 0.3
    
    print(f"\n数据: n={n}, x1 ∈ [{x1.min():.2f}, {x1.max():.2f}], x2 ∈ [{x2.min():.2f}, {x2.max():.2f}]")
    print(f"真实函数: f(x1, x2) = x1² + x2² + x1*x2")
    
    # 创建 tensor product basis
    print("\n创建 tensor product basis (k=10)...")
    basis, penalty = create_tensor_basis(x1, x2, k=10)
    print(f"Basis shape: {basis.shape}")
    print(f"Penalty shape: {penalty.shape}")
    
    # 拟合模型（简单的岭回归）
    print("\n拟合模型...")
    lambda_ = 0.1
    XtX = basis.T @ basis + lambda_ * penalty
    Xty = basis.T @ y
    coefficients = np.linalg.solve(XtX, Xty)
    
    # 预测
    y_pred = basis @ coefficients
    
    # 计算 R²
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - ss_res / ss_tot
    
    print(f"\n拟合结果:")
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {np.sqrt(np.mean((y - y_pred) ** 2)):.4f}")
    
    # 可视化
    fig = plt.figure(figsize=(15, 5))
    
    # 子图 1: 真实值 vs 预测值
    ax1 = fig.add_subplot(131)
    ax1.scatter(y_true, y_pred, alpha=0.5)
    ax1.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    ax1.set_xlabel('True Values')
    ax1.set_ylabel('Predicted Values')
    ax1.set_title(f'True vs Predicted (R² = {r2:.4f})')
    ax1.grid(True, alpha=0.3)
    
    # 子图 2: 残差图
    ax2 = fig.add_subplot(132)
    residuals = y - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted Values')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)
    
    # 子图 3: 3D 表面图
    ax3 = fig.add_subplot(133, projection='3d')
    
    # 创建网格用于绘制表面
    x1_grid = np.linspace(-2, 2, 30)
    x2_grid = np.linspace(-2, 2, 30)
    X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)
    
    # 在网格上预测
    x1_flat = X1_grid.ravel()
    x2_flat = X2_grid.ravel()
    basis_grid, _ = create_tensor_basis(x1_flat, x2_flat, k=10)
    y_grid = basis_grid @ coefficients
    Y_grid = y_grid.reshape(X1_grid.shape)
    
    # 绘制表面
    surf = ax3.plot_surface(X1_grid, X2_grid, Y_grid, cmap='viridis', alpha=0.8)
    ax3.scatter(x1[:50], x2[:50], y[:50], c='red', marker='o', s=20, alpha=0.6)
    ax3.set_xlabel('x1')
    ax3.set_ylabel('x2')
    ax3.set_zlabel('y')
    ax3.set_title('Fitted Surface')
    fig.colorbar(surf, ax=ax3, shrink=0.5)
    
    plt.tight_layout()
    plt.savefig('tensor_smooth_2d_demo.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存: tensor_smooth_2d_demo.png")
    plt.close()


def demo_tensor_vs_additive():
    """演示 tensor product smooth 与加性模型的对比"""
    print("\n" + "=" * 60)
    print("演示 2: Tensor Product Smooth vs 加性模型")
    print("=" * 60)
    
    # 生成数据（有强交互效应）
    np.random.seed(42)
    n = 300
    x1 = np.random.uniform(-2, 2, n)
    x2 = np.random.uniform(-2, 2, n)
    
    # 真实函数有强交互效应
    y_true = x1**2 + x2**2 + 2*x1*x2
    y = y_true + np.random.randn(n) * 0.3
    
    print(f"\n数据: n={n}")
    print(f"真实函数: f(x1, x2) = x1² + x2² + 2*x1*x2 (强交互效应)")
    
    # Tensor product smooth
    print("\n拟合 Tensor Product Smooth...")
    basis_tensor, penalty_tensor = create_tensor_basis(x1, x2, k=10)
    lambda_ = 0.1
    XtX = basis_tensor.T @ basis_tensor + lambda_ * penalty_tensor
    Xty = basis_tensor.T @ y
    coef_tensor = np.linalg.solve(XtX, Xty)
    y_pred_tensor = basis_tensor @ coef_tensor
    
    r2_tensor = 1 - np.sum((y - y_pred_tensor) ** 2) / np.sum((y - np.mean(y)) ** 2)
    print(f"  Tensor Product R² = {r2_tensor:.4f}")
    
    # 加性模型（分别对 x1 和 x2 平滑）
    print("\n拟合加性模型 (s(x1) + s(x2))...")
    from omnilss.smoothers.bsplines import bspline_basis
    from omnilss.smoothers.penalties import penalty_matrix
    
    # 为 x1 创建基
    x1_min, x1_max = np.min(x1), np.max(x1)
    x1_range = x1_max - x1_min
    n_interior1 = 10 - 4
    interior_knots1 = np.linspace(x1_min, x1_max, n_interior1 + 2)[1:-1]
    knots1 = np.concatenate([
        np.repeat(x1_min - 0.01 * x1_range, 4),
        interior_knots1,
        np.repeat(x1_max + 0.01 * x1_range, 4)
    ])
    basis1 = np.array(bspline_basis(np.array(x1), np.array(knots1), degree=3))
    actual_k1 = basis1.shape[1]
    penalty1 = penalty_matrix(actual_k1, order=2)
    
    # 为 x2 创建基
    x2_min, x2_max = np.min(x2), np.max(x2)
    x2_range = x2_max - x2_min
    n_interior2 = 10 - 4
    interior_knots2 = np.linspace(x2_min, x2_max, n_interior2 + 2)[1:-1]
    knots2 = np.concatenate([
        np.repeat(x2_min - 0.01 * x2_range, 4),
        interior_knots2,
        np.repeat(x2_max + 0.01 * x2_range, 4)
    ])
    basis2 = np.array(bspline_basis(np.array(x2), np.array(knots2), degree=3))
    actual_k2 = basis2.shape[1]
    penalty2 = penalty_matrix(actual_k2, order=2)
    
    basis_additive = np.hstack([basis1, basis2])
    penalty_additive = np.block([
        [penalty1, np.zeros((actual_k1, actual_k2))],
        [np.zeros((actual_k2, actual_k1)), penalty2]
    ])
    
    XtX_add = basis_additive.T @ basis_additive + lambda_ * penalty_additive
    Xty_add = basis_additive.T @ y
    coef_add = np.linalg.solve(XtX_add, Xty_add)
    y_pred_add = basis_additive @ coef_add
    
    r2_additive = 1 - np.sum((y - y_pred_add) ** 2) / np.sum((y - np.mean(y)) ** 2)
    print(f"  加性模型 R² = {r2_additive:.4f}")
    
    print(f"\n对比:")
    print(f"  Tensor Product R² = {r2_tensor:.4f}")
    print(f"  加性模型 R² = {r2_additive:.4f}")
    print(f"  改进 = {(r2_tensor - r2_additive) * 100:.2f}%")
    print(f"\n结论: Tensor Product Smooth 能够捕捉交互效应，性能更好！")
    
    # 可视化对比
    fig = plt.figure(figsize=(15, 5))
    
    # 子图 1: Tensor Product
    ax1 = fig.add_subplot(131, projection='3d')
    x1_grid = np.linspace(-2, 2, 30)
    x2_grid = np.linspace(-2, 2, 30)
    X1_grid, X2_grid = np.meshgrid(x1_grid, x2_grid)
    x1_flat = X1_grid.ravel()
    x2_flat = X2_grid.ravel()
    basis_grid, _ = create_tensor_basis(x1_flat, x2_flat, k=10)
    y_grid = basis_grid @ coef_tensor
    Y_grid = y_grid.reshape(X1_grid.shape)
    surf1 = ax1.plot_surface(X1_grid, X2_grid, Y_grid, cmap='viridis', alpha=0.8)
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_zlabel('y')
    ax1.set_title(f'Tensor Product (R²={r2_tensor:.4f})')
    
    # 子图 2: 加性模型
    ax2 = fig.add_subplot(132, projection='3d')
    # 为网格创建加性模型预测
    basis1_grid = np.array(bspline_basis(np.array(x1_flat), np.array(knots1), degree=3))
    basis2_grid = np.array(bspline_basis(np.array(x2_flat), np.array(knots2), degree=3))
    basis_add_grid = np.hstack([basis1_grid, basis2_grid])
    y_add_grid = basis_add_grid @ coef_add
    Y_add_grid = y_add_grid.reshape(X1_grid.shape)
    surf2 = ax2.plot_surface(X1_grid, X2_grid, Y_add_grid, cmap='plasma', alpha=0.8)
    ax2.set_xlabel('x1')
    ax2.set_ylabel('x2')
    ax2.set_zlabel('y')
    ax2.set_title(f'Additive Model (R²={r2_additive:.4f})')
    
    # 子图 3: 残差对比
    ax3 = fig.add_subplot(133)
    residuals_tensor = y - y_pred_tensor
    residuals_add = y - y_pred_add
    ax3.scatter(y_pred_tensor, residuals_tensor, alpha=0.5, label='Tensor Product')
    ax3.scatter(y_pred_add, residuals_add, alpha=0.5, label='Additive')
    ax3.axhline(y=0, color='r', linestyle='--', lw=2)
    ax3.set_xlabel('Predicted Values')
    ax3.set_ylabel('Residuals')
    ax3.set_title('Residual Comparison')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tensor_vs_additive_demo.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存: tensor_vs_additive_demo.png")
    plt.close()


def demo_3d_tensor_smooth():
    """演示 3D tensor product smooth"""
    print("\n" + "=" * 60)
    print("演示 3: 3D Tensor Product Smooth")
    print("=" * 60)
    
    # 生成数据
    np.random.seed(42)
    n = 500
    x1 = np.random.uniform(-1, 1, n)
    x2 = np.random.uniform(-1, 1, n)
    x3 = np.random.uniform(-1, 1, n)
    
    # 真实函数: f(x1, x2, x3) = x1^2 + x2^2 + x3^2 + x1*x2 + x2*x3
    y_true = x1**2 + x2**2 + x3**2 + x1*x2 + x2*x3
    y = y_true + np.random.randn(n) * 0.2
    
    print(f"\n数据: n={n}, 3个变量")
    print(f"真实函数: f(x1, x2, x3) = x1² + x2² + x3² + x1*x2 + x2*x3")
    
    # 创建 3D tensor product basis
    print("\n创建 3D tensor product basis (k=6)...")
    basis, penalty = create_tensor_basis(x1, x2, x3, k=6)
    print(f"Basis shape: {basis.shape}")
    print(f"Penalty shape: {penalty.shape}")
    
    # 拟合模型
    print("\n拟合模型...")
    lambda_ = 0.1
    XtX = basis.T @ basis + lambda_ * penalty
    Xty = basis.T @ y
    coefficients = np.linalg.solve(XtX, Xty)
    
    # 预测
    y_pred = basis @ coefficients
    
    # 计算 R²
    r2 = 1 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2)
    
    print(f"\n拟合结果:")
    print(f"  R² = {r2:.4f}")
    print(f"  RMSE = {np.sqrt(np.mean((y - y_pred) ** 2)):.4f}")
    print(f"  系数数量 = {len(coefficients)}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 子图 1: 真实值 vs 预测值
    axes[0].scatter(y_true, y_pred, alpha=0.5)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('True Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title(f'3D Tensor Product (R² = {r2:.4f})')
    axes[0].grid(True, alpha=0.3)
    
    # 子图 2: 残差图
    residuals = y - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.5)
    axes[1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tensor_smooth_3d_demo.png', dpi=150, bbox_inches='tight')
    print("\n图表已保存: tensor_smooth_3d_demo.png")
    plt.close()


def demo_te_ti_specification():
    """演示 te() 和 ti() 规格"""
    print("\n" + "=" * 60)
    print("演示 4: te() 和 ti() 规格")
    print("=" * 60)
    
    # te() 规格
    print("\nte() - Tensor Product Smooth (包含所有效应):")
    spec_te = te("x1", "x2", k=10)
    print(f"  类型: {spec_te['type']}")
    print(f"  变量: {spec_te['variables']}")
    print(f"  基函数数量: {spec_te['k']}")
    print(f"  基函数类型: {spec_te['bs']}")
    print("\n  用法: gamlss('y ~ te(x1, x2)', family='NO', data=data)")
    print("  包含: 主效应 + 交互效应")
    
    # ti() 规格
    print("\nti() - Tensor Product Interaction (只包含交互效应):")
    spec_ti = ti("x1", "x2", k=10)
    print(f"  类型: {spec_ti['type']}")
    print(f"  变量: {spec_ti['variables']}")
    print(f"  基函数数量: {spec_ti['k']}")
    print(f"  基函数类型: {spec_ti['bs']}")
    print("\n  用法: gamlss('y ~ ti(x1) + ti(x2) + ti(x1, x2)', family='NO', data=data)")
    print("  等价于: gamlss('y ~ te(x1, x2)', family='NO', data=data)")
    print("  优势: 可以分别控制主效应和交互效应的平滑程度")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Tensor Product Smooth 演示")
    print("=" * 60)
    
    # 运行所有演示
    demo_2d_tensor_smooth()
    demo_tensor_vs_additive()
    demo_3d_tensor_smooth()
    demo_te_ti_specification()
    
    print("\n" + "=" * 60)
    print("所有演示完成！")
    print("=" * 60)
    print("\n生成的图表:")
    print("  - tensor_smooth_2d_demo.png")
    print("  - tensor_vs_additive_demo.png")
    print("  - tensor_smooth_3d_demo.png")
    print("\n关键要点:")
    print("  1. Tensor Product Smooth 能够捕捉多变量的交互效应")
    print("  2. 相比加性模型，Tensor Product 在有交互效应时性能更好")
    print("  3. 支持 2D、3D 及更高维度的平滑")
    print("  4. te() 包含所有效应，ti() 只包含交互效应")
    print("=" * 60)
