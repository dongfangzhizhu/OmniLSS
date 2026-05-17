"""优化器性能对比示例

这个脚本展示了如何使用 OmniLSS 的不同优化器，并对比它们的性能。

支持的优化器:
1. RS - Rigby-Stasinopoulos 算法（传统坐标下降）
2. CG_FULL_HESSIAN - Cole-Green 完整 Hessian 后端（默认 method="CG"）
3. CG_IRLS_CROSS - 实验性 eta 尺度交叉导数 IRLS 后端
4. Joint - 联合优化（Optax: Adam, SGD, RMSprop, Adagrad）
5. L-BFGS - 准牛顿法

使用方法:
    python examples/optimizer_comparison.py
"""

# ruff: noqa: E402

import numpy as np
import pandas as pd
import time
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "omnilss" / "src"))

import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)

from omnilss.controls import gamlss_control  # noqa: E402
from omnilss.fitting import gamlss  # noqa: E402


def _convergence_flag(model):
    """Return the most specific convergence flag recorded by a backend."""
    slots = model.additional_slots
    return slots.get(
        "cg_converged",
        slots.get("rs_converged", slots.get("converged", False)),
    )


def _iteration_count(model):
    """Return the most specific iteration count recorded by a backend."""
    slots = model.additional_slots
    return slots.get("cg_iterations", slots.get("rs_iterations", model.iter))


def generate_test_data(n=1000, seed=42):
    """生成测试数据

    Parameters
    ----------
    n : int
        样本数量
    seed : int
        随机种子

    Returns
    -------
    data : dict
        包含 y 和 x 的数据字典
    """
    np.random.seed(seed)
    x = np.random.randn(n)
    y = 2 + 3 * x + np.random.randn(n)
    return {"y": y, "x": x}


def test_optimizer(name, method_kwargs, data, verbose=False):
    """测试单个优化器

    Parameters
    ----------
    name : str
        优化器名称
    method_kwargs : dict
        优化器参数
    data : dict
        数据
    verbose : bool
        是否打印详细信息

    Returns
    -------
    result : dict
        测试结果
    """
    print(f"\n{'=' * 70}")
    print(f"Testing: {name}")
    print(f"{'=' * 70}")

    start_time = time.time()

    try:
        model = gamlss(
            "y ~ x", family="NO", data=data, verbose=verbose, **method_kwargs
        )

        elapsed = time.time() - start_time

        slots = model.additional_slots
        result = {
            "Method": name,
            "Backend": slots.get(
                "cg_backend", slots.get("method", method_kwargs.get("method"))
            ),
            "Cross derivs": slots.get("cg_cross_derivatives", "n/a"),
            "Deviance": f"{model.g_dev:.6f}",
            "Iterations": _iteration_count(model),
            "Time (s)": f"{elapsed:.3f}",
            "Converged": "✓" if _convergence_flag(model) else "✗",
            "AIC": f"{model.additional_slots['aic']:.2f}",
            "BIC": f"{model.additional_slots['sbc']:.2f}",
        }

        # 打印系数
        mu_coef = np.asarray(model.coefficients["mu"])
        print("\nCoefficients:")
        print(f"  Intercept: {mu_coef[0]:.4f}")
        print(f"  x:         {mu_coef[1]:.4f}")

        if "sigma" in model.coefficients:
            sigma_coef = np.asarray(model.coefficients["sigma"])
            print(f"  sigma:     {sigma_coef[0]:.4f}")

        print("\nModel fit:")
        print(f"  Deviance:  {result['Deviance']}")
        print(f"  AIC:       {result['AIC']}")
        print(f"  BIC:       {result['BIC']}")
        print(f"  Iterations: {result['Iterations']}")
        print(f"  Backend:   {result['Backend']}")
        print(f"  Cross derivs: {result['Cross derivs']}")
        print(f"  Time:      {result['Time (s)']}s")
        print(f"  Converged: {result['Converged']}")

        return result

    except Exception as e:
        print(f"✗ Error: {e}")
        return {
            "Method": name,
            "Backend": method_kwargs.get(
                "cg_backend", method_kwargs.get("method", "-")
            ),
            "Cross derivs": "-",
            "Deviance": "Error",
            "Iterations": "-",
            "Time (s)": "-",
            "Converged": "✗",
            "AIC": "-",
            "BIC": "-",
        }


def main():
    """主函数"""
    print("=" * 70)
    print("OmniLSS 优化器性能对比")
    print("=" * 70)

    # 生成测试数据
    print("\n生成测试数据...")
    n = 1000
    data = generate_test_data(n=n)
    control = gamlss_control(n_cyc=20, c_crit=1e-4, trace=False)
    print(f"样本数量: {n}")
    print("真实参数: Intercept=2.0, x=3.0, sigma=1.0")

    # 定义要测试的优化器
    optimizers = {
        "RS (默认)": {
            "method": "RS",
            "control": control,
        },
        "CG_FULL_HESSIAN": {
            "method": "CG",
            "cg_backend": "full_hessian",
            "control": control,
        },
        "CG_IRLS_CROSS (实验)": {
            "method": "CG",
            "cg_backend": "irls_cross",
            "control": control,
        },
        "Joint-Adam": {
            "method": "joint",
            "optimizer": "adam",
            "learning_rate": 0.01,
            "max_iter": 1000,
        },
        "Joint-SGD": {
            "method": "joint",
            "optimizer": "sgd",
            "learning_rate": 0.1,
            "max_iter": 1000,
        },
        "Joint-RMSprop": {
            "method": "joint",
            "optimizer": "rmsprop",
            "learning_rate": 0.01,
            "max_iter": 1000,
        },
        "L-BFGS": {"method": "lbfgs", "max_iter": 100, "history_size": 10},
    }

    # 测试所有优化器
    results = []
    for name, kwargs in optimizers.items():
        result = test_optimizer(name, kwargs, data, verbose=False)
        results.append(result)
        time.sleep(0.5)  # 短暂暂停，便于阅读输出

    # 打印汇总表
    print(f"\n{'=' * 70}")
    print("性能对比汇总")
    print(f"{'=' * 70}\n")

    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    # 保存结果
    output_file = "optimizer_comparison_results.csv"
    df.to_csv(output_file, index=False)
    print(f"\n结果已保存到: {output_file}")

    # 打印建议
    print(f"\n{'=' * 70}")
    print("优化器选择建议")
    print(f"{'=' * 70}")
    print("""
1. RS (默认)
   - 适用场景: 小到中等数据集，简单模型
   - 优点: 稳定，快速，无需调参
   - 缺点: 收敛较慢

2. CG_FULL_HESSIAN (method="CG", 默认 CG 后端)
   - 适用场景: 需要可审计 Cole-Green 交叉导数更新的场景
   - 优点: 保留完整 coefficient-level Hessian 交叉块，并记录 cg_backend / cg_cross_derivatives 诊断
   - 缺点: 相比 RS 计算更重；性能结论需以具体模型和 benchmark 为准

3. CG_IRLS_CROSS (method="CG", cg_backend="irls_cross")
   - 适用场景: 试验 eta 尺度交叉导数修正和 RS 风格加权最小二乘结构
   - 优点: 暴露 cg_cross_derivatives="eta_correction"，便于与 full-Hessian reference 对照
   - 缺点: 当前仍为实验后端，正确性敏感流程应对照 CG_FULL_HESSIAN

4. Joint-Adam
   - 适用场景: 大数据集，复杂模型
   - 优点: 最快收敛，支持 GPU
   - 缺点: 需要调参（学习率）

5. Joint-SGD
   - 适用场景: 简单模型，快速原型
   - 优点: 简单快速
   - 缺点: 可能需要更多迭代

6. Joint-RMSprop
   - 适用场景: 非平稳数据
   - 优点: 自适应学习率
   - 缺点: 内存占用稍大

7. L-BFGS
   - 适用场景: 中等数据集，需要高精度
   - 优点: 准牛顿法，收敛快
   - 缺点: 内存占用较大

推荐选择:
- 日常 CPU 拟合: RS
- 需要 Cole-Green 交叉导数诊断: CG_FULL_HESSIAN
- 试验 eta-level CG 修正: CG_IRLS_CROSS，并与 CG_FULL_HESSIAN 对照
- 中等数据且需要准牛顿: L-BFGS
- 大数据或需要 GPU 加速: Joint-Adam / RS_JAX（按硬件 benchmark 决定）
    """)


if __name__ == "__main__":
    main()
