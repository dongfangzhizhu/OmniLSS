# ZABB/ZIBB 构造参数持久化

确认原 JSON 加载器只按 family 名称重建，非默认 bd 会静默恢复为10。
本次 FamilyDefinition 增加可选 constructor_parameters，AD 构造器及兼容
克隆透传该状态；ZABB/ZIBB 显式记录固定 bd。

JSON 保存新增 family_constructor_parameters；加载按支持的构造参数
重建家族，未知选项不再被静默忽略。ZABB 恢复正静态整数，ZIBB 可恢复
非负整数值标量或数组。旧文件无此字段仍按旧默认行为加载。
历史文件未记录的自定义 bd 无法凭空恢复。

验证（项目 .venv）：

```text
python -m pytest omnilss/tests/test_zero_bb_serialization.py omnilss/tests/test_serialization_json.py omnilss/tests/test_zabb_contract.py omnilss/tests/test_zibb_contract.py -q
33 passed
```

新增测试构造含完整模型状态的 GAMLSSModel，验证非默认bd=3/17、
ZIBB bd=0、向量分母、重复保存、旧格式兼容和非法元数据拒绝。
比较恢复前后的 d/p/q 和固定key采样；既有 NO 实际拟合/预测序列化
回归也通过。不声称 ZABB/ZIBB 实际优化器拟合或GPU验证已完成。

下一项建议：补齐模型 artifact 校验器对构造参数的检查，使 validate_model_json
与 load_model_json 对非法 family_constructor_parameters 给出一致诊断；
随后执行非默认bd的真实拟合、预测和保存加载端到端验证。
