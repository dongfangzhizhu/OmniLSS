# max_iter 路由契约修复

公开 gamlss 的 max_iter 原先仅在 JOINT/LBFGS 分支生效。现在显式正整数
max_iter 优先于 control.n_cyc，覆盖 RS、auto、CG(full_hessian/irls_cross)、
MIXED、RS_JAX，并保留 JOINT/LBFGS 行为。
有 control 时通过 dataclasses.replace 复制，不修改调用者对象；无control时
仅在需要控制对象的路径创建它，保持 RS 等路径原有 tol 默认1e-4。
未提供 max_iter 时，所有既有默认迭代数保持不变。

输入拒绝0、负数、布尔、浮点、字符串、NaN/Inf，接受 Python/NumPy 整数。
max_iter 是所选求解器的迭代预算，不包含额外 warm-start 阶段的迭代总和。

项目 .venv：
`python -m pytest omnilss/tests/test_max_iter_contract.py omnilss/tests/test_method_routing_integration.py -q`
结果 **47 passed**。

覆盖求解器调用边界处的预算/容差、显式覆盖、默认保持、control不变性、
非法输入，以及真实 NO RS 一轮拟合与等价 control 设置结果一致。
RS_JAX 验证的是透传，无实际 GPU 数值验证；未运行全库或R拟合对照。

下一项建议：MIXED 路由仍将 family 实例转为名称，可能丢失 ZABB/ZIBB 的
自定义 bd。应保留完整 family 对象并验证实际混合调度和保存恢复路径。
