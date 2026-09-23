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
