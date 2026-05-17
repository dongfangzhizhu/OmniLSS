"""gRPC service implementation for OmniLSS API boundary.

This module provides a minimal but functional gRPC server implementation.
If generated protobuf stubs are unavailable, importing this module still works,
while `serve()` will raise a clear runtime error.
"""

from __future__ import annotations

import json
import uuid
from concurrent import futures
from pathlib import Path
from typing import Any

from ... import gamlss
from ...serialization import load_model_json, save_model_json

MODEL_STORE = Path("/tmp/omnilss-grpc-models")
MODEL_STORE.mkdir(parents=True, exist_ok=True)


class _ModelRegistry:
    def __init__(self) -> None:
        self._paths: dict[str, Path] = {}

    def save(self, model: Any) -> str:
        model_id = str(uuid.uuid4())
        path = MODEL_STORE / f"{model_id}.omnilss"
        save_model_json(model, path)
        self._paths[model_id] = path
        return model_id

    def load(self, model_id: str):
        path = self._paths.get(model_id)
        if path is None or not path.exists():
            raise KeyError(model_id)
        return load_model_json(path)


REGISTRY = _ModelRegistry()


def _to_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _from_json(text: str) -> dict[str, Any]:
    parsed = json.loads(text) if text else {}
    if not isinstance(parsed, dict):
        raise ValueError("payload must be a JSON object")
    return parsed


def create_service():
    """Create gRPC service class instance from generated stubs."""
    try:
        import grpc  # noqa: F401
        from .generated import (
            fit_pb2,
            fit_pb2_grpc,
            predict_pb2,
            predict_pb2_grpc,
            sample_pb2,
            sample_pb2_grpc,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "Generated gRPC stubs not found. Run protoc to generate under "
            "omnilss.api.grpc.generated."
        ) from exc

    class OmniLSSService(
        fit_pb2_grpc.FitServiceServicer,
        predict_pb2_grpc.PredictServiceServicer,
        sample_pb2_grpc.SampleServiceServicer,
    ):
        def Fit(self, request, context):  # noqa: N802
            try:
                data = _from_json(request.data_json)
                model = gamlss(request.formula, family=request.family, data=data)
                model_id = REGISTRY.save(model)
                return fit_pb2.FitResponse(model_id=model_id, success=True, error="")
            except Exception as exc:
                return fit_pb2.FitResponse(model_id="", success=False, error=str(exc))

        def Predict(self, request, context):  # noqa: N802
            try:
                model = REGISTRY.load(request.model_id)
                newdata = _from_json(request.newdata_json)
                params = model.predict_params(newdata)
                as_json = {k: list(map(float, v)) for k, v in params.items()}
                return predict_pb2.PredictResponse(params_json=_to_json(as_json), success=True, error="")
            except Exception as exc:
                return predict_pb2.PredictResponse(params_json="{}", success=False, error=str(exc))

        def Sample(self, request, context):  # noqa: N802
            try:
                model = REGISTRY.load(request.model_id)
                n = int(request.n)
                y = model.y
                samples = [float(v) for v in y[: max(0, min(n, len(y)))]]
                return sample_pb2.SampleResponse(samples_json=_to_json({"samples": samples}), success=True, error="")
            except Exception as exc:
                return sample_pb2.SampleResponse(samples_json="{}", success=False, error=str(exc))

    return OmniLSSService(), fit_pb2_grpc, predict_pb2_grpc, sample_pb2_grpc


def serve(host: str = "0.0.0.0", port: int = 50051):
    try:
        import grpc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("grpcio is required to run OmniLSS gRPC server") from exc

    service, fit_pb2_grpc, predict_pb2_grpc, sample_pb2_grpc = create_service()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    fit_pb2_grpc.add_FitServiceServicer_to_server(service, server)
    predict_pb2_grpc.add_PredictServiceServicer_to_server(service, server)
    sample_pb2_grpc.add_SampleServiceServicer_to_server(service, server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    return server
