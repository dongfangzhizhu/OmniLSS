# 模型构造参数校验与加载一致性

提取 `_family_constructor_issue`，供 validate_model_json 和 load_model_json
共同使用。校验阶段不构造家族，不执行模型代码；加载失败消息与校验报告一致。

错误代码：
- invalid_family_constructor_parameters：构造参数不是对象。
- unsupported_family_constructor_parameters：不支持的家族或选项。
- invalid_family_denominator：bd 类型、形状或数值不合法。

错误路径为 meta.family_constructor_parameters 或其 .bd 字段。
ZABB 保持正静态整数；ZIBB 支持非负整数值标量/规则数组。
转换 NumPy 前检查叶子类型，拒绝混合布尔/整数、字符串、空或不规则数组，
避免隐式类型转换误接受参数。缺省/空对象保留历史默认构造行为。

项目 .venv 验证：

```text
python -m pytest omnilss/tests/test_zero_bb_serialization.py omnilss/tests/test_serialization_json.py omnilss/tests/test_serialization.py -q
43 passed
```

新增12组非法元数据一致性测试、4组有效ZIBB参数测试；原有非默认bd恢复、
旧格式兼容及实际NO拟合/预测序列化回归通过。此次未涉及分布数值变更，
未执行R/GPU测试，不宣称所有artifact字段均已有完整校验。

下一项建议：非默认bd的ZABB/ZIBB真实拟合、预测与保存恢复端到端验证。
目前新族roundtrip基于构造的模型状态，仍需检验优化器收敛、参数初始化
和固定分母在新数据预测中的语义。
