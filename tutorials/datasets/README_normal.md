# 正态分布族教程数据集

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
