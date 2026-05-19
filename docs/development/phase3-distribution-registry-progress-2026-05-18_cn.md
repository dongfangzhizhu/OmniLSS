# 第三阶段进展：权威分布注册表

English version: [phase3-distribution-registry-progress-2026-05-18.md](phase3-distribution-registry-progress-2026-05-18.md)

## 本步骤已完成

- 围绕一个统一的大写 family factory 字典重构 `distribution_registry.py`。
- 新增公共函数 `register(name, factory)`、`resolve(name_or_family)` 和 `list_families()`。
- 将现有 `DistributionRegistry`、`create_default_registry()` 与 `get_default_registry()` 兼容 API 保留为权威注册表的快照。
- 让 `distributions.resolve_family()` 委托给 `distribution_registry.resolve()`，同时保留 `None -> NO()` 行为。
- 增加测试，覆盖大小写不敏感查找、已实例化 family 透传、动态注册、排序后的 family 列表以及未知 family 诊断。

## 说明

注册表现在是公共查找入口。内置 family 通过单一 `_BUILTIN_FAMILY_FACTORIES` 字典 bootstrap，`distributions.py` 中不再保留旧的私有 legacy resolver chain。
