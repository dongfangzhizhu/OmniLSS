# Zero-BB 真实拟合与恢复

修复 ZABB/ZIBB 初始化：nu 从加权零频率获得内点概率，不再使用通用形状
参数初值1；mu 的初始链接预测器使用固定bd归一化后的半计数调整；sigma
从0.1开始。初始化阶段明确检查正整数bd与[0,bd]内整数响应。
采样允许的退化bd=0不作为可估计三参数模型进行拟合。

新增真实 RS 端到端回归：固定key、200条观测、bd=5，分别拟合两族。
验证deviance等于公开密度计算、与独立Nelder-Mead优化器的deviance差<0.05，
新数据参数预测、模型保存恢复、自定义bd及概率结果一致。
使用 GAMLSSControl(n_cyc=200,c_crit=1e-7)。默认20轮的ZIBB测试尚未达到
该优化精度；此次没有修改全局默认迭代数。迭代仍可能触发eta裁剪警告。

项目 .venv 分开运行结果：
- test_zero_bb_fit_roundtrip.py：3 passed。
- test_zero_bb_serialization.py + test_zabb_contract.py + test_zibb_contract.py：32 passed。

组合命令曾在300秒超时，随后拆分完成。没有宣称全库测试或R拟合等价。
独立优化器检查只覆盖同一Python似然的截距模型，尚未覆盖协变量或GPU。

下一项建议：修复 gamlss(method='RS', max_iter=...) 的参数透传。当前公开
签名声明max_iter覆盖control.n_cyc，但RS路由仍仅读取control/default20；
应统一各路由优先级并回归验证，避免用户增加迭代数却未生效。
