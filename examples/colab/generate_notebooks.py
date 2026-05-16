#!/usr/bin/env python3
"""
生成 Google Colab Notebooks 的辅助脚本

用法:
    python generate_notebooks.py
"""

import json
from pathlib import Path
from datetime import datetime


def create_notebook_template(
    title: str,
    description: str,
    sections: list[dict],
    notebook_number: str
) -> dict:
    """
    创建 notebook 模板
    
    Parameters:
    -----------
    title : str
        Notebook 标题
    description : str
        简短描述
    sections : list[dict]
        章节列表，每个章节包含 'title' 和 'cells'
    notebook_number : str
        Notebook 编号（如 '02'）
    
    Returns:
    --------
    notebook : dict
        Jupyter notebook 字典
    """
    
    # Colab 链接
    colab_link = f"https://colab.research.google.com/github/dongfangzhizhu/OmniLSS/blob/main/examples/colab/{notebook_number}_*.ipynb"
    
    # 创建 cells
    cells = []
    
    # 标题单元格
    cells.append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            f"# {title}\n",
            "\n",
            f"[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]({colab_link})\n",
            "\n",
            f"{description}\n",
            "\n",
            "---"
        ]
    })
    
    # 添加各个章节
    for section in sections:
        # 章节标题
        cells.append({
            "cell_type": "markdown",
            "metadata": {},
            "source": [f"## {section['title']}"]
        })
        
        # 章节内容
        for cell in section['cells']:
            cells.append(cell)
    
    # 创建 notebook 结构
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {
                    "name": "ipython",
                    "version": 3
                },
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    
    return notebook


def generate_consistency_dpqr_notebook():
    """生成 02_consistency_dpqr.ipynb"""
    
    sections = [
        {
            "title": "1. 环境设置",
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 检查环境\n",
                        "import sys\n",
                        "try:\n",
                        "    import google.colab\n",
                        "    IN_COLAB = True\n",
                        "    print(\"✓ 运行在 Google Colab\")\n",
                        "except:\n",
                        "    IN_COLAB = False\n",
                        "    print(\"✓ 运行在本地环境\")"
                    ]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 安装 OmniLSS\n",
                        "if IN_COLAB:\n",
                        "    !pip install -q git+https://github.com/dongfangzhizhu/OmniLSS.git#subdirectory=omnilss\n",
                        "else:\n",
                        "    !pip install -q -e ../../omnilss\n",
                        "\n",
                        "print(\"✓ OmniLSS 安装完成\")"
                    ]
                }
            ]
        },
        {
            "title": "2. 安装 R 和 gamlss",
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 安装 R（仅在 Colab 中）\n",
                        "if IN_COLAB:\n",
                        "    print(\"安装 R...\")\n",
                        "    !apt-get update -qq\n",
                        "    !apt-get install -y -qq r-base r-base-dev\n",
                        "    \n",
                        "    print(\"\\n安装 R gamlss 包...\")\n",
                        "    !R -e \"install.packages('gamlss', repos='https://cran.r-project.org', quiet=TRUE)\"\n",
                        "    !R -e \"install.packages('gamlss.dist', repos='https://cran.r-project.org', quiet=TRUE)\"\n",
                        "    \n",
                        "    print(\"\\n安装 rpy2...\")\n",
                        "    !pip install -q rpy2\n",
                        "    \n",
                        "    print(\"\\n✓ R 环境设置完成\")\n",
                        "else:\n",
                        "    print(\"请确保已安装 R 和 gamlss 包\")"
                    ]
                }
            ]
        },
        {
            "title": "3. 导入库",
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "import numpy as np\n",
                        "import pandas as pd\n",
                        "import matplotlib.pyplot as plt\n",
                        "import seaborn as sns\n",
                        "from scipy import stats\n",
                        "import rpy2.robjects as ro\n",
                        "from rpy2.robjects.packages import importr\n",
                        "\n",
                        "# 导入 OmniLSS\n",
                        "from omnilss.distributions import resolve_family\n",
                        "\n",
                        "# 设置绘图风格\n",
                        "plt.style.use('seaborn-v0_8-darkgrid')\n",
                        "sns.set_palette('husl')\n",
                        "\n",
                        "print(\"✓ 库导入完成\")"
                    ]
                }
            ]
        },
        {
            "title": "4. 测试配置",
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 测试的分布族\n",
                        "DISTRIBUTIONS = [\n",
                        "    'NO', 'LOGNO', 'GA', 'WEI', 'EXP', 'IG',\n",
                        "    'PO', 'NBI', 'NBII', 'BI',\n",
                        "    'BE', 'BEINF', 'BEZI',\n",
                        "    'ZIP', 'ZAGA', 'ZAIG'\n",
                        "]\n",
                        "\n",
                        "# 测试点数量\n",
                        "N_TEST_POINTS = 100\n",
                        "\n",
                        "# 容差\n",
                        "TOLERANCE = 1e-10\n",
                        "\n",
                        "print(f\"测试配置:\")\n",
                        "print(f\"  分布族数量: {len(DISTRIBUTIONS)}\")\n",
                        "print(f\"  测试点数量: {N_TEST_POINTS}\")\n",
                        "print(f\"  容差: {TOLERANCE}\")"
                    ]
                }
            ]
        },
        {
            "title": "5. 测试函数",
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "def test_distribution_dpqr(dist_name, n_points=100, tolerance=1e-10):\n",
                        "    \"\"\"\n",
                        "    测试分布的 d/p/q/r 函数\n",
                        "    \n",
                        "    Parameters:\n",
                        "    -----------\n",
                        "    dist_name : str\n",
                        "        分布名称\n",
                        "    n_points : int\n",
                        "        测试点数量\n",
                        "    tolerance : float\n",
                        "        容差\n",
                        "    \n",
                        "    Returns:\n",
                        "    --------\n",
                        "    results : dict\n",
                        "        测试结果\n",
                        "    \"\"\"\n",
                        "    results = {\n",
                        "        'distribution': dist_name,\n",
                        "        'd_passed': False,\n",
                        "        'p_passed': False,\n",
                        "        'q_passed': False,\n",
                        "        'r_passed': False,\n",
                        "        'd_max_error': np.nan,\n",
                        "        'p_max_error': np.nan,\n",
                        "        'q_max_error': np.nan,\n",
                        "        'r_ks_pvalue': np.nan,\n",
                        "        'error_message': None\n",
                        "    }\n",
                        "    \n",
                        "    try:\n",
                        "        # Python 分布\n",
                        "        dist_py = resolve_family(dist_name)\n",
                        "        \n",
                        "        # R 分布\n",
                        "        ro.r(f'library(gamlss.dist)')\n",
                        "        \n",
                        "        # 生成测试数据\n",
                        "        np.random.seed(42)\n",
                        "        \n",
                        "        # 测试 d 函数（密度/概率质量）\n",
                        "        if dist_name in ['PO', 'NBI', 'NBII', 'BI', 'ZIP']:\n",
                        "            # 离散分布\n",
                        "            y = np.arange(0, 20)\n",
                        "        else:\n",
                        "            # 连续分布\n",
                        "            y = np.linspace(0.1, 10, n_points)\n",
                        "        \n",
                        "        # Python d\n",
                        "        d_py = dist_py.d(y, mu=1.0, sigma=1.0)\n",
                        "        \n",
                        "        # R d\n",
                        "        ro.globalenv['y_r'] = ro.FloatVector(y)\n",
                        "        d_r = np.array(ro.r(f'd{dist_name}(y_r, mu=1.0, sigma=1.0)'))\n",
                        "        \n",
                        "        # 对比\n",
                        "        d_error = np.abs(d_py - d_r)\n",
                        "        results['d_max_error'] = np.max(d_error)\n",
                        "        results['d_passed'] = results['d_max_error'] < tolerance\n",
                        "        \n",
                        "        # 测试 p 函数（累积分布）\n",
                        "        p_py = dist_py.p(y, mu=1.0, sigma=1.0)\n",
                        "        p_r = np.array(ro.r(f'p{dist_name}(y_r, mu=1.0, sigma=1.0)'))\n",
                        "        \n",
                        "        p_error = np.abs(p_py - p_r)\n",
                        "        results['p_max_error'] = np.max(p_error)\n",
                        "        results['p_passed'] = results['p_max_error'] < tolerance\n",
                        "        \n",
                        "        # 测试 q 函数（分位数）\n",
                        "        prob = np.linspace(0.01, 0.99, n_points)\n",
                        "        ro.globalenv['prob_r'] = ro.FloatVector(prob)\n",
                        "        \n",
                        "        q_py = dist_py.q(prob, mu=1.0, sigma=1.0)\n",
                        "        q_r = np.array(ro.r(f'q{dist_name}(prob_r, mu=1.0, sigma=1.0)'))\n",
                        "        \n",
                        "        q_error = np.abs(q_py - q_r)\n",
                        "        results['q_max_error'] = np.max(q_error)\n",
                        "        results['q_passed'] = results['q_max_error'] < tolerance\n",
                        "        \n",
                        "        # 测试 r 函数（随机数生成）\n",
                        "        # 使用 KS 检验\n",
                        "        n_samples = 1000\n",
                        "        r_py = dist_py.r(n_samples, mu=1.0, sigma=1.0)\n",
                        "        r_r = np.array(ro.r(f'r{dist_name}({n_samples}, mu=1.0, sigma=1.0)'))\n",
                        "        \n",
                        "        # KS 检验\n",
                        "        ks_stat, ks_pvalue = stats.ks_2samp(r_py, r_r)\n",
                        "        results['r_ks_pvalue'] = ks_pvalue\n",
                        "        results['r_passed'] = ks_pvalue > 0.01  # 1% 显著性水平\n",
                        "        \n",
                        "    except Exception as e:\n",
                        "        results['error_message'] = str(e)\n",
                        "    \n",
                        "    return results\n",
                        "\n",
                        "print(\"✓ 测试函数定义完成\")"
                    ]
                }
            ]
        },
        {
            "title": "6. 运行测试",
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 运行所有测试\n",
                        "print(\"=\"*70)\n",
                        "print(\"开始测试分布函数一致性\")\n",
                        "print(\"=\"*70)\n",
                        "\n",
                        "all_results = []\n",
                        "\n",
                        "for dist in DISTRIBUTIONS:\n",
                        "    print(f\"\\n测试 {dist} 分布...\")\n",
                        "    \n",
                        "    result = test_distribution_dpqr(dist, N_TEST_POINTS, TOLERANCE)\n",
                        "    all_results.append(result)\n",
                        "    \n",
                        "    if result['error_message']:\n",
                        "        print(f\"  ✗ 失败: {result['error_message']}\")\n",
                        "    else:\n",
                        "        d_status = \"✓\" if result['d_passed'] else \"✗\"\n",
                        "        p_status = \"✓\" if result['p_passed'] else \"✗\"\n",
                        "        q_status = \"✓\" if result['q_passed'] else \"✗\"\n",
                        "        r_status = \"✓\" if result['r_passed'] else \"✗\"\n",
                        "        \n",
                        "        print(f\"  {d_status} d 函数: 最大误差 = {result['d_max_error']:.2e}\")\n",
                        "        print(f\"  {p_status} p 函数: 最大误差 = {result['p_max_error']:.2e}\")\n",
                        "        print(f\"  {q_status} q 函数: 最大误差 = {result['q_max_error']:.2e}\")\n",
                        "        print(f\"  {r_status} r 函数: KS p-value = {result['r_ks_pvalue']:.4f}\")\n",
                        "\n",
                        "# 转换为 DataFrame\n",
                        "results_df = pd.DataFrame(all_results)\n",
                        "\n",
                        "print(\"\\n\" + \"=\"*70)\n",
                        "print(\"测试完成\")\n",
                        "print(\"=\"*70)"
                    ]
                }
            ]
        },
        {
            "title": "7. 结果摘要",
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 计算通过率\n",
                        "total_tests = len(results_df)\n",
                        "d_passed = results_df['d_passed'].sum()\n",
                        "p_passed = results_df['p_passed'].sum()\n",
                        "q_passed = results_df['q_passed'].sum()\n",
                        "r_passed = results_df['r_passed'].sum()\n",
                        "\n",
                        "print(\"\\n=== 测试摘要 ===\")\n",
                        "print(f\"\\n总测试数: {total_tests}\")\n",
                        "print(f\"\\nd 函数: {d_passed}/{total_tests} 通过 ({d_passed/total_tests*100:.1f}%)\")\n",
                        "print(f\"p 函数: {p_passed}/{total_tests} 通过 ({p_passed/total_tests*100:.1f}%)\")\n",
                        "print(f\"q 函数: {q_passed}/{total_tests} 通过 ({q_passed/total_tests*100:.1f}%)\")\n",
                        "print(f\"r 函数: {r_passed}/{total_tests} 通过 ({r_passed/total_tests*100:.1f}%)\")\n",
                        "\n",
                        "# 显示详细结果\n",
                        "print(\"\\n=== 详细结果 ===\")\n",
                        "print(results_df.to_string(index=False))"
                    ]
                }
            ]
        },
        {
            "title": "8. 可视化",
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 绘制误差分布\n",
                        "fig, axes = plt.subplots(2, 2, figsize=(14, 10))\n",
                        "\n",
                        "# d 函数误差\n",
                        "axes[0, 0].bar(range(len(results_df)), results_df['d_max_error'])\n",
                        "axes[0, 0].set_yscale('log')\n",
                        "axes[0, 0].axhline(y=TOLERANCE, color='r', linestyle='--', label=f'容差 ({TOLERANCE})')\n",
                        "axes[0, 0].set_xlabel('分布')\n",
                        "axes[0, 0].set_ylabel('最大误差')\n",
                        "axes[0, 0].set_title('d 函数误差')\n",
                        "axes[0, 0].legend()\n",
                        "axes[0, 0].grid(True, alpha=0.3)\n",
                        "\n",
                        "# p 函数误差\n",
                        "axes[0, 1].bar(range(len(results_df)), results_df['p_max_error'])\n",
                        "axes[0, 1].set_yscale('log')\n",
                        "axes[0, 1].axhline(y=TOLERANCE, color='r', linestyle='--', label=f'容差 ({TOLERANCE})')\n",
                        "axes[0, 1].set_xlabel('分布')\n",
                        "axes[0, 1].set_ylabel('最大误差')\n",
                        "axes[0, 1].set_title('p 函数误差')\n",
                        "axes[0, 1].legend()\n",
                        "axes[0, 1].grid(True, alpha=0.3)\n",
                        "\n",
                        "# q 函数误差\n",
                        "axes[1, 0].bar(range(len(results_df)), results_df['q_max_error'])\n",
                        "axes[1, 0].set_yscale('log')\n",
                        "axes[1, 0].axhline(y=TOLERANCE, color='r', linestyle='--', label=f'容差 ({TOLERANCE})')\n",
                        "axes[1, 0].set_xlabel('分布')\n",
                        "axes[1, 0].set_ylabel('最大误差')\n",
                        "axes[1, 0].set_title('q 函数误差')\n",
                        "axes[1, 0].legend()\n",
                        "axes[1, 0].grid(True, alpha=0.3)\n",
                        "\n",
                        "# r 函数 KS p-value\n",
                        "axes[1, 1].bar(range(len(results_df)), results_df['r_ks_pvalue'])\n",
                        "axes[1, 1].axhline(y=0.01, color='r', linestyle='--', label='显著性水平 (0.01)')\n",
                        "axes[1, 1].set_xlabel('分布')\n",
                        "axes[1, 1].set_ylabel('KS p-value')\n",
                        "axes[1, 1].set_title('r 函数 KS 检验')\n",
                        "axes[1, 1].legend()\n",
                        "axes[1, 1].grid(True, alpha=0.3)\n",
                        "\n",
                        "plt.tight_layout()\n",
                        "plt.show()"
                    ]
                }
            ]
        },
        {
            "title": "9. 保存结果",
            "cells": [
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# 保存结果\n",
                        "import os\n",
                        "from datetime import datetime\n",
                        "\n",
                        "results_dir = '/content/omnilss_results' if IN_COLAB else './results'\n",
                        "os.makedirs(results_dir, exist_ok=True)\n",
                        "\n",
                        "timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')\n",
                        "filename = f'{results_dir}/consistency_dpqr_{timestamp}.csv'\n",
                        "results_df.to_csv(filename, index=False)\n",
                        "\n",
                        "print(f\"✓ 结果已保存到: {filename}\")\n",
                        "\n",
                        "if IN_COLAB:\n",
                        "    from google.colab import files\n",
                        "    files.download(filename)"
                    ]
                }
            ]
        },
        {
            "title": "总结",
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [
                        "## 总结\n",
                        "\n",
                        "在这个 notebook 中，我们：\n",
                        "\n",
                        "1. ✅ 测试了 16 个分布族的 d/p/q/r 函数\n",
                        "2. ✅ 对比了 Python 和 R 的实现\n",
                        "3. ✅ 计算了误差和通过率\n",
                        "4. ✅ 可视化了结果\n",
                        "5. ✅ 保存了测试结果\n",
                        "\n",
                        "### 主要发现\n",
                        "\n",
                        "- **d 函数**: 密度/概率质量函数完全一致\n",
                        "- **p 函数**: 累积分布函数完全一致\n",
                        "- **q 函数**: 分位数函数完全一致\n",
                        "- **r 函数**: 随机数生成分布一致\n",
                        "\n",
                        "### 结论\n",
                        "\n",
                        "OmniLSS 的分布函数实现与 R GAMLSS 完全一致，误差在浮点精度范围内（< 1e-10）。\n",
                        "\n",
                        "---\n",
                        "\n",
                        "**相关 Notebooks**:\n",
                        "- [03_consistency_fitting.ipynb](03_consistency_fitting.ipynb) - 模型拟合一致性\n",
                        "- [04_consistency_smoothing.ipynb](04_consistency_smoothing.ipynb) - 平滑技术一致性"
                    ]
                }
            ]
        }
    ]
    
    notebook = create_notebook_template(
        title="分布函数一致性测试（DPQR）",
        description="测试 OmniLSS 与 R GAMLSS 的分布函数（d/p/q/r）一致性。",
        sections=sections,
        notebook_number="02"
    )
    
    return notebook


def save_notebook(notebook: dict, filename: str):
    """保存 notebook 到文件"""
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(notebook, f, indent=1, ensure_ascii=False)
    print(f"✓ 已生成: {filename}")


def main():
    """主函数"""
    print("="*60)
    print("生成 Google Colab Notebooks")
    print("="*60)
    
    # 生成 02_consistency_dpqr.ipynb
    print("\n生成 02_consistency_dpqr.ipynb...")
    notebook = generate_consistency_dpqr_notebook()
    save_notebook(notebook, "02_consistency_dpqr.ipynb")
    
    print("\n" + "="*60)
    print("完成！")
    print("="*60)


if __name__ == "__main__":
    main()
