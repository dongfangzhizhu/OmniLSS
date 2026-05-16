"""
生成正态分布族教程的示例数据集

这个脚本生成教程中使用的所有数据集，并保存为 CSV 文件。
"""

import numpy as np
import pandas as pd
import os


def generate_simple_linear_data(n=200, seed=42):
    """生成简单线性回归数据"""
    np.random.seed(seed)
    
    x = np.linspace(0, 10, n)
    mu_true = 2 + 0.5 * x
    sigma_true = 0.5 + 0.05 * x  # 异方差
    
    y = np.random.normal(mu_true, sigma_true)
    
    data = pd.DataFrame({
        'x': x,
        'y': y,
        'mu_true': mu_true,
        'sigma_true': sigma_true
    })
    
    return data


def generate_lognormal_data(n=300, seed=123):
    """生成对数正态分布数据"""
    np.random.seed(seed)
    
    x = np.random.uniform(0, 5, n)
    
    # 真实参数（对数尺度）
    log_mu_true = 1.0 + 0.3 * x
    log_sigma_true = 0.5
    
    # 生成数据
    y = np.exp(np.random.normal(log_mu_true, log_sigma_true))
    
    data = pd.DataFrame({
        'x': x,
        'y': y,
        'log_mu_true': log_mu_true,
        'log_sigma_true': log_sigma_true
    })
    
    return data


def generate_height_data(n=500, seed=2024):
    """生成身高数据（儿童生长曲线）"""
    np.random.seed(seed)
    
    # 模拟儿童身高数据（年龄 2-18 岁）
    age = np.random.uniform(2, 18, n)
    
    # 身高随年龄增长，方差也增加
    height_mean = 80 + 8 * age - 0.15 * age**2
    height_sd = 3 + 0.2 * age
    
    height = np.random.normal(height_mean, height_sd)
    
    # 添加性别变量
    gender = np.random.choice(['M', 'F'], size=n)
    
    # 男孩平均稍高
    height = np.where(gender == 'M', height + 2, height)
    
    data = pd.DataFrame({
        'age': age,
        'height': height,
        'gender': gender,
        'height_mean_true': height_mean,
        'height_sd_true': height_sd
    })
    
    return data


def generate_income_data(n=1000, seed=456):
    """生成收入数据（对数正态分布）"""
    np.random.seed(seed)
    
    # 特征
    education = np.random.choice([12, 14, 16, 18, 20], size=n, 
                                  p=[0.2, 0.25, 0.3, 0.15, 0.1])
    experience = np.random.uniform(0, 30, n)
    
    # 收入模型（对数尺度）
    log_income = (8.0 + 
                  0.08 * education + 
                  0.05 * experience - 
                  0.001 * experience**2 +
                  np.random.normal(0, 0.4, n))
    
    income = np.exp(log_income)
    
    data = pd.DataFrame({
        'education': education,
        'experience': experience,
        'income': income,
        'log_income': log_income
    })
    
    return data


def generate_measurement_error_data(n=150, seed=789):
    """生成测量误差数据"""
    np.random.seed(seed)
    
    # 真实值
    true_value = np.linspace(10, 100, n)
    
    # 测量误差随真实值增加
    measurement_error = 0.5 + 0.02 * true_value
    
    # 观测值
    observed_value = true_value + np.random.normal(0, measurement_error)
    
    # 重复测量
    observed_value2 = true_value + np.random.normal(0, measurement_error)
    observed_value3 = true_value + np.random.normal(0, measurement_error)
    
    data = pd.DataFrame({
        'true_value': true_value,
        'observed_1': observed_value,
        'observed_2': observed_value2,
        'observed_3': observed_value3,
        'measurement_error_true': measurement_error
    })
    
    return data


def generate_all_datasets():
    """生成所有数据集并保存"""
    
    # 创建数据集目录
    output_dir = '../../datasets'
    os.makedirs(output_dir, exist_ok=True)
    
    print("生成数据集...")
    print("=" * 60)
    
    # 1. 简单线性数据
    print("\n1. 生成简单线性回归数据...")
    data1 = generate_simple_linear_data()
    filename1 = os.path.join(output_dir, 'simple_linear_data.csv')
    data1.to_csv(filename1, index=False)
    print(f"   保存到: {filename1}")
    print(f"   维度: {data1.shape}")
    print(f"   预览:\n{data1.head()}")
    
    # 2. 对数正态数据
    print("\n2. 生成对数正态分布数据...")
    data2 = generate_lognormal_data()
    filename2 = os.path.join(output_dir, 'lognormal_data.csv')
    data2.to_csv(filename2, index=False)
    print(f"   保存到: {filename2}")
    print(f"   维度: {data2.shape}")
    print(f"   预览:\n{data2.head()}")
    
    # 3. 身高数据
    print("\n3. 生成身高数据...")
    data3 = generate_height_data()
    filename3 = os.path.join(output_dir, 'height_data.csv')
    data3.to_csv(filename3, index=False)
    print(f"   保存到: {filename3}")
    print(f"   维度: {data3.shape}")
    print(f"   预览:\n{data3.head()}")
    
    # 4. 收入数据
    print("\n4. 生成收入数据...")
    data4 = generate_income_data()
    filename4 = os.path.join(output_dir, 'income_data.csv')
    data4.to_csv(filename4, index=False)
    print(f"   保存到: {filename4}")
    print(f"   维度: {data4.shape}")
    print(f"   预览:\n{data4.head()}")
    
    # 5. 测量误差数据
    print("\n5. 生成测量误差数据...")
    data5 = generate_measurement_error_data()
    filename5 = os.path.join(output_dir, 'measurement_error_data.csv')
    data5.to_csv(filename5, index=False)
    print(f"   保存到: {filename5}")
    print(f"   维度: {data5.shape}")
    print(f"   预览:\n{data5.head()}")
    
    print("\n" + "=" * 60)
    print("所有数据集生成完成！")
    print("=" * 60)
    
    # 生成数据集说明文档
    generate_dataset_readme(output_dir)


def generate_dataset_readme(output_dir):
    """生成数据集说明文档"""
    
    readme_content = """# 正态分布族教程数据集

本目录包含教程 01（正态分布族）使用的所有数据集。

## 数据集列表

### 1. simple_linear_data.csv

**描述**: 简单线性回归数据，带异方差

**变量**:
- `x`: 自变量（0 到 10）
- `y`: 响应变量（正态分布）
- `mu_true`: 真实均值
- `sigma_true`: 真实标准差

**样本量**: 200

**用途**: 演示基本的 GAMLSS 拟合，包括异方差建模

---

### 2. lognormal_data.csv

**描述**: 对数正态分布数据

**变量**:
- `x`: 自变量（0 到 5）
- `y`: 响应变量（对数正态分布）
- `log_mu_true`: 对数尺度的真实均值
- `log_sigma_true`: 对数尺度的真实标准差

**样本量**: 300

**用途**: 演示 LOGNO 分布的使用

---

### 3. height_data.csv

**描述**: 儿童身高数据（模拟）

**变量**:
- `age`: 年龄（2-18 岁）
- `height`: 身高（厘米）
- `gender`: 性别（M/F）
- `height_mean_true`: 真实均值
- `height_sd_true`: 真实标准差

**样本量**: 500

**用途**: 演示生长曲线建模，包括非线性关系和异方差

---

### 4. income_data.csv

**描述**: 收入数据（模拟）

**变量**:
- `education`: 教育年限（12, 14, 16, 18, 20）
- `experience`: 工作经验（0-30 年）
- `income`: 收入（美元）
- `log_income`: 对数收入

**样本量**: 1000

**用途**: 演示对数正态分布在经济数据中的应用

---

### 5. measurement_error_data.csv

**描述**: 测量误差数据

**变量**:
- `true_value`: 真实值（10-100）
- `observed_1`: 第一次观测值
- `observed_2`: 第二次观测值
- `observed_3`: 第三次观测值
- `measurement_error_true`: 真实测量误差

**样本量**: 150

**用途**: 演示测量误差建模，误差随真实值增加

---

## 使用方法

### Python

```python
import pandas as pd

# 读取数据
data = pd.read_csv('tutorials/datasets/height_data.csv')

# 查看数据
print(data.head())
print(data.describe())
```

### R

```r
# 读取数据
data <- read.csv('tutorials/datasets/height_data.csv')

# 查看数据
head(data)
summary(data)
```

---

## 数据生成

所有数据集都是使用 `generate_normal_data.py` 脚本生成的。

要重新生成数据集：

```bash
cd tutorials/code/phase1
python generate_normal_data.py
```

---

## 许可

这些数据集是为教育目的生成的模拟数据，可以自由使用。

---

*生成日期: 2026-04-26*
"""
    
    readme_path = os.path.join(output_dir, 'README_normal.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n数据集说明文档已保存: {readme_path}")


if __name__ == "__main__":
    generate_all_datasets()
