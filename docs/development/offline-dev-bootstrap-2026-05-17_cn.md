# 中文版本：OmniLSS 离线开发环境引导（2026-05-17）

[English version](offline-dev-bootstrap-2026-05-17.md)

> 该页面是英文原文的中文入口。完整中文翻译将在后续文档本地化工作中补齐；在此之前，请以英文版本为准。

## 概要

英文版 runbook 说明如何在有网络时构建一次 devcontainer 镜像，并在无网络的开发会话中离线复用：
镜像内置 Python wheelhouse、CRAN 包缓存（离线安装 R 包的权威来源）以及
`.codex/setup.sh` 的幂等初始化流程。`R/` 为 gitignored 的 vendored 参考源码、
fresh clone 中不存在，故镜像构建期不复制它；仅在 post-create 阶段若工作区存在 `R/`
才由 `setup.sh` 尽力打包进 `/opt/omnilss-r-source`（缺失不致命）。详见英文版 P0-1。

## 快速命令

```bash
# 有网络：构建并启动一次
devcontainer build --workspace-folder .
devcontainer up --workspace-folder .

# 容器内：离线初始化与验证门禁
bash .codex/setup.sh
python benchmarks/run_local_validation.py --quick
```

## 翻译状态

- [ ] 完成全文中文翻译
- [ ] 与英文版本保持同步
