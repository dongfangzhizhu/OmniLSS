# Gamma 分布族教程数据集

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
