# 第三阶段进展：基于字典的内置注册表

English version: [phase3-dictionary-registry-progress-2026-05-18.md](phase3-dictionary-registry-progress-2026-05-18.md)

## 本步骤已完成

- 将依赖 legacy resolver 的注册表 bootstrap 路径替换为单一 `_BUILTIN_FAMILY_FACTORIES` 字典。
- 将当前已注册的每个内置 family 名称映射到其模块与零参数 factory 属性。
- 从该字典派生 `_REGISTERED_FAMILIES`，使 capability matrix、registry snapshot 与 family listing 不会偏离内置注册表表格。
- 增加字典注册表契约的回归测试，包括 `EXGAUS` -> `exGAUS` 这类大小写不完全一致的 factory 名称。

## 说明

`distributions.resolve_family()` 仍保留公共函数名，并委托给权威注册表。旧的私有 legacy resolver chain 现已移除，因此分布查找只保留单一的基于字典的实现路径。
