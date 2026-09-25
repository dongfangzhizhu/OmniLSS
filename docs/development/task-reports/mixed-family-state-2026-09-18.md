# MIXED 家族状态与自动路由修复

- gamlss 的 MIXED 路径保留 family 实例，不再仅按名称重建而丢失固定bd。
- mixed_fit 接受 FamilyDefinition；自动选择使用 family.name，保证 BCT/BCPE
  等稳定性规则在传对象时仍生效。
- 大样本自动选择原先返回cg，而执行分支只接受lbfgs，导致未知算法错误；
  统一为现有lbfgs兼容后端标识，不改变显式cg别名的兼容行为。

项目 .venv 验证：
`python -m pytest omnilss/tests/test_mixed_family_state.py omnilss/tests/test_mixed_algorithm.py omnilss/tests/test_max_iter_contract.py -q`
结果 **51 passed**。

两族bd=3真实MIXED→RS短迭代拟合与直接RS比较，通过保存加载、新数据参数
预测及概率一致性验证；自动选择边界499/500、BCT和自定义family透传通过。
短迭代用例只证明状态/路由一致，不证明收敛；eta裁剪警告仍可能出现。
大样本自动分支验证为调用边界测试，未宣称真实大样本或GPU拟合性能。

下一项建议：MIXED 入口还未传递 weights 和 parameter_formulas，可能静默
忽略加权拟合及nu/tau公式。应补齐跨调度参数传递并用实际加权拟合验证。
现有lbfgs命名的兼容后端内部仍调用CG，该历史命名问题另行审计。
