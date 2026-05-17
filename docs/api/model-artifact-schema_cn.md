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

验证报告格式如下：

```json
{
  "ok": false,
  "errors": [
    {
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
- `training_data_included`，作为 warning。

## 预测错误边界

运行时 prediction schema 失败会抛出带有 `code`、`parameter`、`term` 和 `reason` 字段的 `PredictionSchemaError`。客户端应基于 `code` 路由，而不是解析面向人类的异常消息。
