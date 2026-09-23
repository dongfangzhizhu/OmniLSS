# TLS 配置

## gRPC

设置以下环境变量后启动 gRPC 服务：

```text
OMNILSS_API_TOKEN=<required-token>
OMNILSS_TLS_CERT=/run/secrets/server.crt
OMNILSS_TLS_KEY=/run/secrets/server.key
```

证书和私钥必须同时存在、非空，并使用 PEM 格式。只配置其中一个或配置不存在/损坏的文件时，服务启动会立即失败。

## FastAPI/Uvicorn

通过 `OMNILSS_UVICORN_TLS_ARGS` 传递 Uvicorn TLS 参数：

```text
OMNILSS_UVICORN_TLS_ARGS=--ssl-keyfile /run/secrets/server.key --ssl-certfile /run/secrets/server.crt
```

HTTP 服务同时要求 `OMNILSS_API_TOKEN`，客户端必须发送 `Authorization: Bearer <token>`。

## Compose TLS overlay

准备 `secrets/server.crt` 和 `secrets/server.key` 后运行：

```powershell
docker compose -f docker-compose.yml -f docker-compose.tls.yml up
```

也可以通过 `OMNILSS_TLS_DIR` 指定证书目录。部署前运行 readiness gate：

```powershell
python tools/tls_readiness.py --cert secrets/server.crt --key secrets/server.key --min-days 7
```

该命令只输出证书路径、过期时间和剩余天数，不读取或输出私钥内容。

发布门禁中显式传入证书参数即可启用 fail-closed 校验：

```powershell
python omnilss/tools/release_check.py --preflight-only `
  --tls-cert secrets/server.crt --tls-key secrets/server.key --tls-min-days 7
```
