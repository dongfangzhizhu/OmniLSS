"""gRPC service implementation for OmniLSS API boundary.

This module provides a minimal but functional gRPC server implementation.
If generated protobuf stubs are unavailable, importing this module still works,
while `serve()` will raise a clear runtime error.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sqlite3
import time
import uuid
from concurrent import futures
from pathlib import Path
from typing import Any

from ... import gamlss
from ...prediction import PredictionSchemaError
from ...serialization import load_model_json, save_model_json

MODEL_STORE = Path(os.getenv("OMNILSS_MODEL_STORE_DIR", "/tmp/omnilss-grpc-models"))
MODEL_STORE.mkdir(parents=True, exist_ok=True)
MODEL_DB = Path(os.getenv("OMNILSS_MODEL_DB_PATH", str(MODEL_STORE / "registry.sqlite3")))


class _ModelRegistry:
    """Bounded local model registry for the gRPC service.

    The server stores fitted model artifacts on disk and keeps an in-memory
    index for lookup.  The index is intentionally bounded to avoid unbounded
    memory and /tmp growth in long-running processes.
    """

    MAX_MODELS = 200
    TTL_SECONDS = 3600

    def __init__(self) -> None:
        self._paths: dict[str, tuple[Path, float]] = {}
        self._db = sqlite3.connect(MODEL_DB, check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS models (id TEXT PRIMARY KEY, path TEXT NOT NULL, created_at REAL NOT NULL)")
        self._db.commit()
        for model_id, path, created_at in self._db.execute("SELECT id, path, created_at FROM models"):
            self._paths[str(model_id)] = (Path(path), float(created_at))

    def save(self, model: Any) -> str:
        self._evict_expired()
        if len(self._paths) >= self.MAX_MODELS:
            oldest_id = min(self._paths, key=lambda key: self._paths[key][1])
            self._remove(oldest_id)

        model_id = str(uuid.uuid4())
        path = MODEL_STORE / f"{model_id}.omnilss"
        save_model_json(model, path)
        created_at = time.time()
        self._paths[model_id] = (path, created_at)
        self._db.execute("INSERT INTO models(id, path, created_at) VALUES (?, ?, ?)", (model_id, str(path), created_at))
        self._db.commit()
        return model_id

    def load(self, model_id: str):
        self._evict_expired()
        entry = self._paths.get(model_id)
        if entry is None:
            raise KeyError(model_id)
        path, _created_at = entry
        if not path.exists():
            self._paths.pop(model_id, None)
            self._db.execute("DELETE FROM models WHERE id = ?", (model_id,))
            self._db.commit()
            raise KeyError(model_id)
        return load_model_json(path)

    def list_ids(self) -> list[str]:
        self._evict_expired()
        return sorted(self._paths)

    def delete(self, model_id: str) -> bool:
        if model_id not in self._paths:
            return False
        self._remove(model_id)
        return True

    def _evict_expired(self) -> None:
        now = time.time()
        expired = [
            model_id
            for model_id, (_path, created_at) in self._paths.items()
            if now - created_at > self.TTL_SECONDS
        ]
        for model_id in expired:
            self._remove(model_id)

    def _remove(self, model_id: str) -> None:
        path, _created_at = self._paths.pop(model_id, (None, 0.0))
        if path is not None:
            path.unlink(missing_ok=True)
        self._db.execute("DELETE FROM models WHERE id = ?", (model_id,))
        self._db.commit()


REGISTRY = _ModelRegistry()


def _to_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False)


def _from_json(text: str) -> dict[str, Any]:
    parsed = json.loads(text) if text else {}
    if not isinstance(parsed, dict):
        raise ValueError("payload must be a JSON object")
    return parsed


def _error_text(exc: Exception) -> str:
    """Return a machine-readable error string for gRPC response fields."""

    if isinstance(exc, PredictionSchemaError):
        return _to_json({"type": "prediction_schema_error", **exc.to_dict()})
    return str(exc)


def _grpc_runtime_gaps() -> list[str]:
    """Return missing runtime pieces required by generated gRPC stubs."""
    gaps: list[str] = []
    if importlib.util.find_spec("grpc") is None:
        gaps.append("grpcio")
    if importlib.util.find_spec("google") is None:
        gaps.append("protobuf")
    elif importlib.util.find_spec("google.protobuf") is None:
        gaps.append("protobuf")
    for module in (
        "omnilss.api.grpc.generated.capability_pb2",
        "omnilss.api.grpc.generated.capability_pb2_grpc",
        "omnilss.api.grpc.generated.fit_pb2",
        "omnilss.api.grpc.generated.fit_pb2_grpc",
        "omnilss.api.grpc.generated.predict_pb2",
        "omnilss.api.grpc.generated.predict_pb2_grpc",
        "omnilss.api.grpc.generated.sample_pb2",
        "omnilss.api.grpc.generated.sample_pb2_grpc",
    ):
        if importlib.util.find_spec(module) is None:
            gaps.append(module)
    return gaps


def create_service():
    """Create gRPC service class instance from generated stubs."""
    gaps = _grpc_runtime_gaps()
    if gaps:
        raise RuntimeError(
            "OmniLSS gRPC runtime is incomplete. Missing: "
            + ", ".join(gaps)
            + ". Install the grpc extra (`pip install 'omnilss[grpc]'`) and "
            "ensure generated stubs exist under omnilss.api.grpc.generated."
        )

    from .generated import (
        capability_pb2,
        capability_pb2_grpc,
        fit_pb2,
        fit_pb2_grpc,
        predict_pb2,
        predict_pb2_grpc,
        sample_pb2,
        sample_pb2_grpc,
    )

    class OmniLSSService(
        fit_pb2_grpc.FitServiceServicer,
        predict_pb2_grpc.PredictServiceServicer,
        sample_pb2_grpc.SampleServiceServicer,
        capability_pb2_grpc.CapabilityServiceServicer,
    ):
        def CapabilityMatrix(self, request, context):  # noqa: N802, ARG002
            try:
                from ...family_capabilities import capability_matrix

                return capability_pb2.CapabilityMatrixResponse(
                    matrix_json=_to_json(capability_matrix()),
                    success=True,
                    error="",
                )
            except Exception as exc:
                return capability_pb2.CapabilityMatrixResponse(
                    matrix_json="{}", success=False, error=_error_text(exc)
                )

        def RouteCapability(self, request, context):  # noqa: N802, ARG002
            try:
                from ...family_capabilities import method_route_capability_report

                family = str(request.family).strip()
                method = str(request.method).strip()
                if not family or not method:
                    raise ValueError(
                        "route capability checks require non-empty family and method"
                    )
                return capability_pb2.RouteCapabilityResponse(
                    report_json=_to_json(
                        method_route_capability_report(
                            family, method, strict=bool(request.strict)
                        )
                    ),
                    success=True,
                    error="",
                )
            except Exception as exc:
                return capability_pb2.RouteCapabilityResponse(
                    report_json="{}", success=False, error=_error_text(exc)
                )

        def ListModels(self, request, context):  # noqa: N802, ARG002
            try:
                return fit_pb2.ListModelsResponse(
                    model_ids=REGISTRY.list_ids(), success=True, error=""
                )
            except Exception as exc:
                return fit_pb2.ListModelsResponse(
                    model_ids=[], success=False, error=_error_text(exc)
                )

        def DeleteModel(self, request, context):  # noqa: N802
            try:
                deleted = REGISTRY.delete(str(request.model_id))
                return fit_pb2.DeleteModelResponse(
                    deleted=bool(deleted), success=True, error=""
                )
            except Exception as exc:
                return fit_pb2.DeleteModelResponse(
                    deleted=False, success=False, error=_error_text(exc)
                )

        def _fit_one(self, request):
            data = _from_json(request.data_json)
            kwargs: dict[str, Any] = {
                "formula": request.formula,
                "family": request.family,
                "data": data,
            }
            if request.sigma_formula:
                kwargs["sigma_formula"] = request.sigma_formula
            if request.nu_formula:
                kwargs.setdefault("parameter_formulas", {})["nu"] = request.nu_formula
            if request.tau_formula:
                kwargs.setdefault("parameter_formulas", {})["tau"] = request.tau_formula
            if request.method:
                kwargs["method"] = request.method
            if request.max_iter > 0:
                kwargs["max_iter"] = int(request.max_iter)
            kwargs["verbose"] = bool(request.verbose)

            model = gamlss(**kwargs)
            model_id = REGISTRY.save(model)
            slots = dict(getattr(model, "additional_slots", {}) or {})
            converged = slots.get("rs_converged", slots.get("cg_converged", True))
            return fit_pb2.FitResponse(
                model_id=model_id,
                success=True,
                error="",
                deviance=float(model.g_dev),
                iterations=int(model.iter),
                converged=bool(converged),
            )

        def Fit(self, request, context):  # noqa: N802, ARG002
            try:
                return self._fit_one(request)
            except Exception as exc:
                return fit_pb2.BatchFitResponse(
                    responses=[], success=False, error=_error_text(exc)
                )

        def BatchFit(self, request, context):  # noqa: N802, ARG002
            try:
                responses = []
                for fit_request in request.requests:
                    try:
                        responses.append(self._fit_one(fit_request))
                    except Exception as exc:
                        responses.append(
                            fit_pb2.FitResponse(
                                model_id="", success=False, error=_error_text(exc)
                            )
                        )
                failed = [resp for resp in responses if not resp.success]
                error = ""
                if failed:
                    error = f"{len(failed)} of {len(responses)} fits failed"
                return fit_pb2.BatchFitResponse(
                    responses=responses,
                    success=not failed,
                    error=error,
                )
            except Exception as exc:
                return fit_pb2.BatchFitResponse(
                    responses=[], success=False, error=_error_text(exc)
                )

        def Predict(self, request, context):  # noqa: N802
            try:
                model = REGISTRY.load(request.model_id)
                if request.newdata_json:
                    newdata = _from_json(request.newdata_json)
                else:
                    newdata = {col.name: list(map(float, col.values)) for col in request.newdata_columns}
                params = model.predict_params(newdata)
                as_json = {k: list(map(float, v)) for k, v in params.items()}
                param_vectors = [
                    predict_pb2.ParamVector(name=name, values=values)
                    for name, values in as_json.items()
                ]
                return predict_pb2.PredictResponse(
                    params_json=_to_json(as_json), params=param_vectors, success=True, error=""
                )
            except Exception as exc:
                return predict_pb2.PredictResponse(
                    params_json="{}", success=False, error=_error_text(exc)
                )

        def Sample(self, request, context):  # noqa: N802
            try:
                import inspect

                import jax
                import jax.numpy as jnp
                import numpy as np

                model = REGISTRY.load(request.model_id)
                n = max(0, int(request.n))
                family = model.family
                fitted = getattr(model, "fitted_values", {}) or {}
                param_means = {
                    p: float(jnp.mean(jnp.asarray(v)))
                    for p, v in fitted.items()
                    if p in family.parameters
                }

                rng_fn = getattr(family, "r", None)
                if rng_fn is not None:
                    sig = inspect.signature(rng_fn)
                    if "key" in sig.parameters:
                        samples_arr = rng_fn(jax.random.PRNGKey(42), n, **param_means)
                    else:
                        samples_arr = rng_fn(n, **param_means)
                    samples = [float(v) for v in np.asarray(samples_arr).reshape(-1)]
                else:
                    q_fn = getattr(family, "q", None)
                    if q_fn is None:
                        raise NotImplementedError(
                            f"Family {family.name} does not support sampling"
                        )
                    rng = np.random.default_rng(42)
                    u = rng.uniform(0.0, 1.0, n)
                    samples = [float(q_fn(float(ui), **param_means)) for ui in u]

                return sample_pb2.SampleResponse(
                    samples_json=_to_json({"samples": samples}), success=True, error=""
                )
            except Exception as exc:
                return sample_pb2.SampleResponse(
                    samples_json="{}", success=False, error=_error_text(exc)
                )

    return (
        OmniLSSService(),
        fit_pb2_grpc,
        predict_pb2_grpc,
        sample_pb2_grpc,
        capability_pb2_grpc,
    )


def serve(host: str = "0.0.0.0", port: int = 50051):
    try:
        import grpc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("grpcio is required to run OmniLSS gRPC server") from exc

    (
        service,
        fit_pb2_grpc,
        predict_pb2_grpc,
        sample_pb2_grpc,
        capability_pb2_grpc,
    ) = create_service()
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=8))
    fit_pb2_grpc.add_FitServiceServicer_to_server(service, server)
    predict_pb2_grpc.add_PredictServiceServicer_to_server(service, server)
    sample_pb2_grpc.add_SampleServiceServicer_to_server(service, server)
    capability_pb2_grpc.add_CapabilityServiceServicer_to_server(service, server)
    server.add_insecure_port(f"{host}:{port}")
    server.start()
    return server
