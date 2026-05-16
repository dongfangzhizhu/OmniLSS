"""
生成 Gamma 分布族教程的示例数据集

这个脚本生成教程中使用的所有数据集，并保存为 CSV 文件。
"""

import numpy as np
import pandas as pd
import os
from scipy.stats import invgauss


def generate_simple_gamma_data(n=300, seed=42):
    """生成简单 Gamma 回归数据"""
    np.random.seed(seed)
    
    x = np.linspace(0, 5, n)
    
    # Gamma 分布参数
    shape = 2.0
    scale = 1.0 + 0.3 * x  # 尺度随 x 增加
    
    # 生成 Gamma 数据
    y = np.random.gamma(shape, scale)
    
    data = pd.DataFrame({
        'x': x,
        'y': y,
        'shape_true': shape,
        'scale_true': scale
    })
    
    return data


def generate_inverse_gaussian_data(n=250, seed=123):
    """生成逆高斯分布数据"""
    np.random.seed(seed)
    
    x = np.random.uniform(0, 10, n)
    
    # 逆高斯参数
    mu_true = 5 + 0.5 * x
    lambda_param = 10  # 形状参数
    
    # 生成 IG 数据（使用 Wald 分布）
    y = invgauss.rvs(mu=mu_true/mu_true.mean(), scale=lambda_param, size=n) * mu_true.mean()
    
    data = pd.DataFrame({
        'x': x,
        'y': y,
        'mu_true': mu_true,
        'lambda_param': lambda_param
    })
    
    return data


def generate_insurance_claims_data(n=1000, seed=2024):
    """生成保险索赔数据"""
    np.random.seed(seed)
    
    # 特征
    age = np.random.uniform(20, 70, n)
    vehicle_age = np.random.randint(0, 15, n)
    region = np.random.choice(['Urban', 'Suburban', 'Rural'], n, p=[0.4, 0.35, 0.25])
    
    # 索赔金额模型
    # 均值随年龄、车龄增加
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
        'claim_amount': claim_amount,
        'log_claim_amount': np.log(claim_amount)
    })
    
    return data


def generate_rainfall_data(n=500, seed=456):
    """生成降雨量数据（Gamma 分布的经典应用）"""
    np.random.seed(seed)
    
    # 特征
    month = np.random.randint(1, 13, n)
    temperature = np.random.uniform(10, 35, n)
    humidity = np.random.uniform(30, 90, n)
    
    # 降雨量模型
    # 夏季（6-8月）降雨多，温度和湿度影响降雨
    shape = 2.0
    scale = (1.0 + 
             0.5 * ((month >= 6) & (month <= 8)) +
             0.02 * temperature +
             0.01 * humidity)
    
    rainfall = np.random.gamma(shape, scale)
    
    data = pd.DataFrame({
        'month': month,
        'temperature': temperature,
        'humidity': humidity,
        'rainfall': rainfall
    })
    
    return data


def generate_survival_time_data(n=400, seed=789):
    """生成生存时间数据（Weibull/Gamma 分布）"""
    np.random.seed(seed)
    
    # 特征
    age = np.random.uniform(40, 80, n)
    treatment = np.random.choice(['A', 'B', 'Control'], n, p=[0.35, 0.35, 0.3])
    stage = np.random.choice([1, 2, 3, 4], n, p=[0.2, 0.3, 0.3, 0.2])
    
    # 生存时间模型（月）
    # 治疗和分期影响生存时间
    shape = 1.5
    scale = (12.0 - 
             0.1 * age +
             5.0 * (treatment == 'A') +
             3.0 * (treatment == 'B') -
             2.0 * stage)
    
    scale = np.maximum(scale, 1.0)  # 确保 scale > 0
    
    survival_time = np.random.gamma(shape, scale)
    
    # 审查指示器（1 = 事件发生，0 = 审查）
    censored = np.random.binomial(1, 0.7, n)
    
    data = pd.DataFrame({
        'age': age,
        'treatment': treatment,
        'stage': stage,
        'survival_time': survival_time,
        'censored': censored
    })
    
    return data


def generate_waiting_time_data(n=600, seed=321):
    """生成等待时间数据（指数/Gamma 分布）"""
    np.random.seed(seed)
    
    # 特征
    time_of_day = np.random.choice(['Morning', 'Afternoon', 'Evening'], n, p=[0.3, 0.4, 0.3])
    day_of_week = np.random.choice(['Weekday', 'Weekend'], n, p=[0.7, 0.3])
    service_type = np.random.choice(['Standard', 'Express', 'Premium'], n, p=[0.5, 0.3, 0.2])
    
    # 等待时间模型（分钟）
    shape = 1.2
    scale = (5.0 +
             3.0 * (time_of_day == 'Afternoon') +
             2.0 * (day_of_week == 'Weekend') -
             2.0 * (service_type == 'Express') -
             3.0 * (service_type == 'Premium'))
    
    waiting_time = np.random.gamma(shape, scale)
    
    data = pd.DataFrame({
        'time_of_day': time_of_day,
        'day_of_week': day_of_week,
        'service_type': service_type,
        'waiting_time': waiting_time
    })
    
    return data


def generate_all_datasets():
    """生成所有数据集并保存"""
    
    # 创建数据集目录
    output_dir = '../../datasets'
    os.makedirs(output_dir, exist_ok=True)
    
    print("生成 Gamma 分布族数据集...")
    print("=" * 60)
    
    # 1. 简单 Gamma 数据
    print("\n1. 生成简单 Gamma 回归数据...")
    data1 = generate_simple_gamma_data()
    filename1 = os.path.join(output_dir, 'gamma_simple_data.csv')
    data1.to_csv(filename1, index=False)
    print(f"   保存到: {filename1}")
    print(f"   维度: {data1.shape}")
    print(f"   预览:\n{data1.head()}")
    
    # 2. 逆高斯数据
    print("\n2. 生成逆高斯分布数据...")
    data2 = generate_inverse_gaussian_data()
    filename2 = os.path.join(output_dir, 'inverse_gaussian_data.csv')
    data2.to_csv(filename2, index=False)
    print(f"   保存到: {filename2}")
    print(f"   维度: {data2.shape}")
    print(f"   预览:\n{data2.head()}")
    
    # 3. 保险索赔数据
    print("\n3. 生成保险索赔数据...")
    data3 = generate_insurance_claims_data()
    filename3 = os.path.join(output_dir, 'insurance_claims_data.csv')
    data3.to_csv(filename3, index=False)
    print(f"   保存到: {filename3}")
    print(f"   维度: {data3.shape}")
    print(f"   预览:\n{data3.head()}")
    
    # 4. 降雨量数据
    print("\n4. 生成降雨量数据...")
    data4 = generate_rainfall_data()
    filename4 = os.path.join(output_dir, 'rainfall_data.csv')
    data4.to_csv(filename4, index=False)
    print(f"   保存到: {filename4}")
    print(f"   维度: {data4.shape}")
    print(f"   预览:\n{data4.head()}")
    
    # 5. 生存时间数据
    print("\n5. 生成生存时间数据...")
    data5 = generate_survival_time_data()
    filename5 = os.path.join(output_dir, 'survival_time_data.csv')
    data5.to_csv(filename5, index=False)
    print(f"   保存到: {filename5}")
    print(f"   维度: {data5.shape}")
    print(f"   预览:\n{data5.head()}")
    
    # 6. 等待时间数据
    print("\n6. 生成等待时间数据...")
    data6 = generate_waiting_time_data()
    filename6 = os.path.join(output_dir, 'waiting_time_data.csv')
    data6.to_csv(filename6, index=False)
    print(f"   保存到: {filename6}")
    print(f"   维度: {data6.shape}")
    print(f"   预览:\n{data6.head()}")
    
    print("\n" + "=" * 60)
    print("所有数据集生成完成！")
    print("=" * 60)
    
    # 生成数据集说明文档
    generate_dataset_readme(output_dir)


def generate_dataset_readme(output_dir):
    """生成数据集说明文档"""
    
    readme_content = """# Gamma 分布族教程数据集

本目录包含教程 02（Gamma 分布族）使用的所有数据集。

## 数据集列表

### 1. gamma_simple_data.csv

**描述**: 简单 Gamma 回归数据

**变量**:
- `x`: 自变量（0 到 5）
- `y`: 响应变量（Gamma 分布）
- `shape_true`: 真实形状参数
- `scale_true`: 真实尺度参数

**样本量**: 300

**用途**: 演示基本的 Gamma 回归

---

### 2. inverse_gaussian_data.csv

**描述**: 逆高斯分布数据

**变量**:
- `x`: 自变量（0 到 10）
- `y`: 响应变量（逆高斯分布）
- `mu_true`: 真实均值参数
- `lambda_param`: 真实形状参数

**样本量**: 250

**用途**: 演示逆高斯分布的使用

---

### 3. insurance_claims_data.csv

**描述**: 保险索赔金额数据（模拟）

**变量**:
- `age`: 投保人年龄（20-70 岁）
- `vehicle_age`: 车龄（0-15 年）
- `region`: 地区（Urban/Suburban/Rural）
- `claim_amount`: 索赔金额（美元）
- `log_claim_amount`: 对数索赔金额

**样本量**: 1000

**用途**: 演示 Gamma 分布在保险领域的应用，包括多参数建模

---

### 4. rainfall_data.csv

**描述**: 降雨量数据（模拟）

**变量**:
- `month`: 月份（1-12）
- `temperature`: 温度（摄氏度）
- `humidity`: 湿度（百分比）
- `rainfall`: 降雨量（毫米）

**样本量**: 500

**用途**: 演示 Gamma 分布在气象数据中的应用

---

### 5. survival_time_data.csv

**描述**: 生存时间数据（模拟）

**变量**:
- `age`: 年龄（40-80 岁）
- `treatment`: 治疗方案（A/B/Control）
- `stage`: 疾病分期（1-4）
- `survival_time`: 生存时间（月）
- `censored`: 审查指示器（1=事件发生，0=审查）

**样本量**: 400

**用途**: 演示 Gamma 分布在生存分析中的应用

---

### 6. waiting_time_data.csv

**描述**: 等待时间数据（模拟）

**变量**:
- `time_of_day`: 时段（Morning/Afternoon/Evening）
- `day_of_week`: 工作日/周末（Weekday/Weekend）
- `service_type`: 服务类型（Standard/Express/Premium）
- `waiting_time`: 等待时间（分钟）

**样本量**: 600

**用途**: 演示 Gamma 分布在服务时间建模中的应用

---

## 使用方法

### Python

```python
import pandas as pd

# 读取数据
data = pd.read_csv('tutorials/datasets/insurance_claims_data.csv')

# 查看数据
print(data.head())
print(data.describe())

# 按地区分组统计
print(data.groupby('region')['claim_amount'].describe())
```

### R

```r
# 读取数据
data <- read.csv('tutorials/datasets/insurance_claims_data.csv')

# 查看数据
head(data)
summary(data)

# 按地区分组统计
aggregate(claim_amount ~ region, data = data, summary)
```

---

## 数据生成

所有数据集都是使用 `generate_gamma_data.py` 脚本生成的。

要重新生成数据集：

```bash
cd tutorials/code/phase1
python generate_gamma_data.py
```

---

## 数据特点

### Gamma 分布的适用性

这些数据集展示了 Gamma 分布的典型应用场景：

1. **正值数据**: 所有响应变量都是正值
2. **右偏分布**: 数据呈现右偏特征
3. **异方差**: 方差随均值变化
4. **实际应用**: 涵盖保险、气象、医疗等领域

### 数据质量

- ✅ 无缺失值
- ✅ 无异常值（除非有意设置）
- ✅ 变量类型正确
- ✅ 符合 Gamma 分布特征

---

## 许可

这些数据集是为教育目的生成的模拟数据，可以自由使用。

---

*生成日期: 2026-04-26*
"""
    
    readme_path = os.path.join(output_dir, 'README_gamma.md')
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(readme_content)
    
    print(f"\n数据集说明文档已保存: {readme_path}")


if __name__ == "__main__":
    generate_all_datasets()
