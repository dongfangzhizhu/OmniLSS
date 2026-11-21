# gRPC & HTTP Serving

[中文版本](tutorial-grpc-serving_cn.md)

OmniLSS exposes both a gRPC serving layer (`omnilss.api.grpc` + `omnilss-pro`
client) and a FastAPI HTTP server (`omnilss-server/main.py`) protected by the
same fail-closed family capability gate.

## gRPC Services & Endpoints

| Service | RPCs | Purpose |
|---|---|---|
| `FitService` | `Fit`, `BatchFit` | Single and batch model fitting |
| `PredictService` | `Predict` | Out-of-sample parameter / quantile prediction |
| `SampleService` | `Sample` | Predictive distribution sampling |
| `CapabilityService` | `CapabilityMatrix`, `RouteCapability`, `Version`, `Health` | Runtime capability matrix, route check, version, health |

Start the gRPC server:

```bash
python -m omnilss.api.grpc.server --help
```

Python client (`omnilss_pro.client`):

```python
from omnilss_pro.client import OmniLSSClient

client = OmniLSSClient("localhost:50051")
fit = client.fit(data, "y ~ x", family="GA")
pred = client.predict(fit.model_id, newdata)
client.health()
client.version()
```

## FastAPI HTTP Service (`omnilss-server`)

`omnilss-server/main.py` provides `/health`, `/fit`, `/predict`,
`/distributions`, `/distributions/select`, `/models`, and `/models/{id}` with:

- Optional Bearer-token auth (`OMNILSS_API_TOKEN`) and per-IP sliding-window
  rate limiting (with bounded client table pruning).
- Fail-closed `require_production_family_route` enforcement on `/fit` and
  `/distributions/select` (unproven `experimental` / `compat_alias` families
  return HTTP `400`).
- Input DoS guards (`MAX_ROWS`, `MAX_COLS`, `max_iter` bounded to `1..1000`).

## Security & Contract Freeze

- Enable TLS (`OMNILSS_GRPC_TLS_CERT` / `OMNILSS_GRPC_TLS_KEY`) before exposing
  gRPC endpoints outside localhost; verify certificates via
  `tools/tls_readiness.py`.
- Protobuf contracts in `omnilss/src/omnilss/api/grpc/proto/*.proto` are frozen;
  stubs are regenerated with `tools/generate_grpc_stubs.py`.

## Next Steps

- [OmniLSS Documentation Home](../index.md)
