# OmniLSS 模型 Artifact Schema 与验证

[English version](model-artifact-schema.md)

本文描述[六个月执行计划](../development/six-month-execution-plan-2026-05-17_cn.md)中第 1/2 周可信 Core 工作引入的面向生产的 JSON 模型 artifact 边界。

## Artifact 布局

`.omnilss` JSON artifact 是一个 ZIP archive，包含两个必需成员：

- `meta.json`：JSON metadata、公式、schema snapshot、diagnostics 和 capability snapshot。
- `arrays.npz`：NumPy 数组，包含系数、拟合值和线性预测器。训练响应数据默认省略。

## 必需 Metadata

`meta.json` 应包含：

- `omnilss_version`，当前 loader 兼容 `0.3.x`。
- `parameters`，列出 `mu`、`sigma` 等模型参数。
- `design_matrix_schema.version == 2` 和 `design_matrix_schema.artifact_version == 2`。
- `design_matrix_schema.parameters.<parameter>` 条目，包含公式、term 顺序、截距状态、列数、factor levels、数值变换 AST metadata，以及必要的 smooth basis metadata。
- `family_capability`，紧凑的 family capability snapshot。
- `smooth_infos`，当 smooth prediction 需要重建时必须存在。
- `diagnostics`，紧凑的标量模型诊断信息。

## 训练数据策略

`save_model_json(model, path)` 默认省略完整训练数据。只有在受控工作流确实需要训练数组时，才使用 `save_model_json(model, path, include_training_data=True)`。Validator 会把 `training_data_included` 报告为 warning，而不是 error。

## 程序化验证

```python
from omnilss import validate_model_json

report = validate_model_json("model.omnilss")
if not report["ok"]:
    for error in report["errors"]:
        print(error["code"], error["path"], error["message"])
```

同样的验证也可通过开发 CLI 使用：

```bash
PYTHONPATH=src python tools/validate_model_artifact.py model.omnilss
PYTHONPATH=src python tools/validate_model_artifact.py model.omnilss --fail-on-warning
```

## 结构化验证问题

验证报告使用 versioned envelope，使客户端可以基于稳定 issue 字段进行路由，而不是解析人类可读文本：

```json
{
  "type": "artifact_validation_report",
  "version": 1,
  "artifact": "model.omnilss",
  "ok": false,
  "error_count": 1,
  "warning_count": 0,
  "errors": [
    {
      "type": "artifact_validation_issue",
      "severity": "error",
      "code": "coefficient_schema_mismatch",
      "path": "arrays.coef__mu",
      "message": "Coefficient count 2 does not match schema n_columns 99"
    }
  ],
  "warnings": []
}
```

重要 issue code 包括：

- `missing_meta`、`invalid_meta`、`missing_arrays`、`invalid_arrays`、`invalid_zip`。
- `unsupported_version`、`unsupported_schema_version`、`unsupported_artifact_schema_version`。
- `missing_parameter_schema`、`missing_parameter_formula`、`invalid_term_order`。
- `coefficient_schema_mismatch`。
- `missing_smooth_metadata`、`invalid_smooth_metadata_entry`、`missing_smooth_knots`。
- `training_data_included`，作为 `severity == "warning"` 的 warning。

## Artifact 与预测错误示例

下面的简化 `meta.json` 片段展示了 categorical predictor 进行 schema-safe prediction 所需的最小 schema 字段：

```json
{
  "omnilss_version": "0.3.0",
  "parameters": ["mu"],
  "design_matrix_schema": {
    "version": 2,
    "artifact_version": 2,
    "parameters": {
      "mu": {
        "formula": "y ~ factor(grp)",
        "term_order": ["factor(grp)"],
        "has_intercept": true,
        "factor_levels": {"grp": ["a", "b"]},
        "n_columns": 2,
        "coefficient_count": 2
      }
    }
  }
}
```

如果客户端通过默认 `model.predict_params()` 接口或 legacy R 对齐的 `predict()` / `predict_all()` 接口使用未知 level 进行预测，runtime 会抛出同一个结构化 envelope：

```python
from omnilss.prediction import PredictionSchemaError
from omnilss.predict_gamlss_23_12_21 import predict

try:
    predict(model, what="mu", newdata={"grp": ["c", "a"]})
except PredictionSchemaError as exc:
    print(exc.to_dict())
```

示例输出：

```json
{
  "code": "unseen_factor_levels",
  "parameter": "mu",
  "term": "factor(grp)",
  "reason": "unseen factor levels ['c']",
  "message": "Factor term 'factor(grp)' contains unseen levels ['c']"
}
```

validator CLI 仍然是 runtime 之前的 artifact gate。有效的 categorical artifact 应无 error：

```bash
PYTHONPATH=src python tools/validate_model_artifact.py categorical.omnilss
```

```json
{
  "ok": true,
  "errors": [],
  "warnings": []
}
```


## 预测错误边界

运行时 prediction schema 失败会抛出带有 `code`、`parameter`、`term` 和 `reason` 字段的 `PredictionSchemaError`。客户端应基于 `code` 路由，而不是解析面向人类的异常消息。
