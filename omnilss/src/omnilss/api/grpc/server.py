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
import tempfile
import threading
import time
import uuid
import zipfile
from concurrent import futures
from pathlib import Path
from typing import Any

from ... import gamlss
from ...family_capabilities import (
    CAPABILITY_MATRIX_VERSION,
    ProductionFamilyRejectedError,
    require_production_family_route,
    method_route_capability_report,
)
from ...prediction import PredictionSchemaError
from ...serialization import compare_model_capability_snapshot, load_model_json, save_model_json

MODEL_STORE = Path(
    os.getenv(
        "OMNILSS_MODEL_STORE_DIR",
        str(Path(tempfile.gettempdir()) / "omnilss-grpc-models"),
    )
)
MODEL_STORE.mkdir(parents=True, exist_ok=True)
MODEL_DB = Path(os.getenv("OMNILSS_MODEL_DB_PATH", str(MODEL_STORE / "registry.sqlite3")))


_MODEL_REGISTRY_SCHEMA_VERSION = 2


def _artifact_family_capability(path: Path) -> dict[str, Any]:
    """Read the saved family capability snapshot from a model artifact."""

    with zipfile.ZipFile(path, "r") as zf:
        meta = json.loads(zf.read("meta.json").decode("utf-8"))
    capability = meta.get("family_capability", {})
    return dict(capability) if isinstance(capability, dict) else {}


def _artifact_capability_replay_report(path: Path) -> dict[str, Any]:
    """Replay capability validation for an artifact against this runtime."""

    return dict(compare_model_capability_snapshot(path))


class _ModelRegistry:
    """Bounded local model registry for the gRPC service.

    The server stores fitted model artifacts on disk and keeps an in-memory
    index for lookup.  The index is intentionally bounded to avoid unbounded
    memory and /tmp growth in long-running processes.
    """

    MAX_MODELS = 200
    TTL_SECONDS = 3600

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._paths: dict[str, tuple[Path, float]] = {}
        self._db = sqlite3.connect(MODEL_DB, check_same_thread=False)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._ensure_schema()
        self._db.commit()
        rows = self._db.execute("SELECT id, path, created_at FROM models")
        for model_id, path, created_at in rows:
            artifact_path = Path(path)
            self._paths[str(model_id)] = (artifact_path, float(created_at))
            if artifact_path.exists():
                self._refresh_capability_snapshot(str(model_id), artifact_path)
            else:
                self._db.execute("DELETE FROM models WHERE id = ?", (str(model_id),))
        self._db.commit()

    def _ensure_schema(self) -> None:
        """Create or migrate the persistent registry schema in place."""

        self._db.execute(
            "CREATE TABLE IF NOT EXISTS models ("
            "id TEXT PRIMARY KEY, "
            "path TEXT NOT NULL, "
            "created_at REAL NOT NULL, "
            "schema_version INTEGER NOT NULL DEFAULT 1, "
            "family TEXT, "
            "capability_matrix_version INTEGER, "
            "implementation_status TEXT, "
            "capability_json TEXT, "
            "capability_report_json TEXT, "
            "capability_ok INTEGER"
            ")"
        )
        existing = {row[1] for row in self._db.execute("PRAGMA table_info(models)")}
        migrations = {
            "schema_version": (
                "ALTER TABLE models ADD COLUMN "
                "schema_version INTEGER NOT NULL DEFAULT 1"
            ),
            "family": "ALTER TABLE models ADD COLUMN family TEXT",
            "capability_matrix_version": (
                "ALTER TABLE models ADD COLUMN capability_matrix_version INTEGER"
            ),
            "implementation_status": "ALTER TABLE models ADD COLUMN implementation_status TEXT",
            "capability_json": "ALTER TABLE models ADD COLUMN capability_json TEXT",
            "capability_report_json": "ALTER TABLE models ADD COLUMN capability_report_json TEXT",
            "capability_ok": "ALTER TABLE models ADD COLUMN capability_ok INTEGER",
        }
        for column, statement in migrations.items():
            if column not in existing:
                self._db.execute(statement)

    def _refresh_capability_snapshot(self, model_id: str, path: Path) -> None:
        """Persist artifact capability metadata and current replay status."""

        try:
            capability = _artifact_family_capability(path)
            report = _artifact_capability_replay_report(path)
        except Exception as exc:  # pragma: no cover - defensive for corrupt legacy rows
            capability = {}
            report = {
                "ok": False,
                "family": None,
                "artifact_capability": {},
                "runtime_capability": None,
                "changes": [
                    {
                        "code": "capability_replay_failed",
                        "message": str(exc),
                    }
                ],
            }
        self._db.execute(
            "UPDATE models SET "
            "schema_version = ?, "
            "family = ?, "
            "capability_matrix_version = ?, "
            "implementation_status = ?, "
            "capability_json = ?, "
            "capability_report_json = ?, "
            "capability_ok = ? "
            "WHERE id = ?",
            (
                _MODEL_REGISTRY_SCHEMA_VERSION,
                capability.get("name") or report.get("family"),
                CAPABILITY_MATRIX_VERSION,
                capability.get("implementation_status"),
                json.dumps(capability, ensure_ascii=False, sort_keys=True),
                json.dumps(report, ensure_ascii=False, sort_keys=True),
                1 if report.get("ok") else 0,
                model_id,
            ),
        )

    def save(self, model: Any) -> str:
        with self._lock:
            self._evict_expired()
            if len(self._paths) >= self.MAX_MODELS:
                oldest_id = min(self._paths, key=lambda key: self._paths[key][1])
                self._remove(oldest_id)

            model_id = str(uuid.uuid4())
            path = MODEL_STORE / f"{model_id}.omnilss"
            save_model_json(model, path)
            created_at = time.time()
            self._paths[model_id] = (path, created_at)
            self._db.execute(
                "INSERT INTO models(id, path, created_at, schema_version) "
                "VALUES (?, ?, ?, ?)",
                (model_id, str(path), created_at, _MODEL_REGISTRY_SCHEMA_VERSION),
            )
            self._refresh_capability_snapshot(model_id, path)
            self._db.commit()
            return model_id

    def load(self, model_id: str):
        with self._lock:
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
            self._refresh_capability_snapshot(model_id, path)
            self._db.commit()
            return load_model_json(path)

    def list_ids(self) -> list[str]:
        with self._lock:
            self._evict_expired()
            return sorted(self._paths)

    def delete(self, model_id: str) -> bool:
        with self._lock:
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


# Phase 5.3: DoS limits (gRPC service).
MAX_SAMPLE_N = 100_000
MAX_BATCH_FITS = 100
MAX_ITER = 1000
MAX_JSON_BODY_BYTES = 1_000_000  # 1 MiB
MAX_GRPC_MESSAGE_BYTES = 64 * 1024 * 1024  # 64 MiB


def _from_json(text: str) -> dict[str, Any]:
    # Phase 5.3: bound the JSON payload size before parsing.
    if len(text.encode("utf-8")) > MAX_JSON_BODY_BYTES:
        raise ValueError(f"JSON payload too large (>{MAX_JSON_BODY_BYTES} bytes)")
    parsed = json.loads(text) if text else {}
    if not isinstance(parsed, dict):
        raise ValueError("payload must be a JSON object")
    return parsed


def _error_text(exc: Exception) -> str:
    """Return a machine-readable error string for gRPC response fields."""

    if isinstance(exc, ProductionFamilyRejectedError):
        return _to_json(exc.to_dict())
    if isinstance(exc, PredictionSchemaError):
        return _to_json({"type": "prediction_schema_error", **exc.to_dict()})
    if isinstance(exc, ValueError) and "does not support penalized smooth terms" in str(exc):
        return _to_json({
            "type": "method_capability_error",
            "code": "smooth_terms_unsupported_for_method",
            "method": "CG",
            "message": str(exc),
        })
    # Phase 5.4: never leak raw exception text to clients; return the error
    # class name so operators can correlate logs without exposing internals.
    return f"{type(exc).__name__} error"


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
                family = str(request.family).strip()
                method = str(request.method).strip()
                formulas = {
                    key: value for key, value in {
                        "mu": str(request.formula).strip(),
                        "sigma": str(request.sigma_formula).strip(),
                        "nu": str(request.nu_formula).strip(),
                        "tau": str(request.tau_formula).strip(),
                    }.items() if value
                }
                if not family or not method:
                    raise ValueError(
                        "route capability checks require non-empty family and method"
                    )
                return capability_pb2.RouteCapabilityResponse(
                    report_json=_to_json(
                        method_route_capability_report(
                            family, method, strict=bool(request.strict), formulas=formulas
                        )
                    ),
                    success=True,
                    error="",
                )
            except Exception as exc:
                return capability_pb2.RouteCapabilityResponse(
                    report_json="{}", success=False, error=_error_text(exc)
                )

        def Version(self, request, context):  # noqa: N802, ARG002
            """Phase 8.1: report the server/runtime versions and x64 precision."""
            try:
                from ... import __version__
                import sys as _sys

                x64 = "true"
                try:
                    import jax

                    x64 = "true" if bool(jax.config.jax_enable_x64) else "false"
                except Exception:
                    x64 = "unknown"
                return capability_pb2.VersionResponse(
                    version=str(__version__),
                    python_version=f"{_sys.version_info.major}.{_sys.version_info.minor}.{_sys.version_info.micro}",
                    jax_version="",
                    x64_enabled=x64,
                    success=True,
                    error="",
                )
            except Exception as exc:
                return capability_pb2.VersionResponse(
                    version="", success=False, error=_error_text(exc)
                )

        def Health(self, request, context):  # noqa: N802, ARG002
            """Phase 8.1: health probe with the registered model count."""
            try:
                return capability_pb2.HealthResponse(
                    status="ok",
                    model_count=int(len(REGISTRY.list_ids())),
                    success=True,
                    error="",
                )
            except Exception as exc:
                return capability_pb2.HealthResponse(
                    status="degraded", model_count=0, success=False, error=_error_text(exc)
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
            method = str(request.method or "RS").strip() or "RS"
            require_production_family_route(
                request.family, method, route="service"
            )
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
            kwargs["method"] = method
            if request.max_iter > 0:
                kwargs["max_iter"] = min(int(request.max_iter), MAX_ITER)
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
                return fit_pb2.FitResponse(
                    model_id="", success=False, error=_error_text(exc)
                )

        def BatchFit(self, request, context):  # noqa: N802, ARG002
            try:
                if len(request.requests) > MAX_BATCH_FITS:
                    return fit_pb2.BatchFitResponse(
                        responses=[],
                        success=False,
                        error=f"batch too large: {len(request.requests)} > {MAX_BATCH_FITS}",
                    )
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
                n = min(max(0, int(request.n)), MAX_SAMPLE_N)
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


def _validate_api_token(host: str, token: str) -> None:
    """Phase 5.1 default-deny: without a token, refuse non-loopback binds."""
    if not token and host not in {"127.0.0.1", "localhost", "::1"}:
        raise RuntimeError(
            "OMNILSS_API_TOKEN is not set; refusing to start the gRPC server "
            "on a non-loopback interface (Phase 5.1 default-deny). Set "
            "OMNILSS_API_TOKEN or bind host to 127.0.0.1."
        )


def serve(host: str = "0.0.0.0", port: int = 50051):
    try:
        import grpc
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("grpcio is required to run OmniLSS gRPC server") from exc

    # Phase 5.1: authentication defaults to denied.  Without OMNILSS_API_TOKEN
    # the server refuses to start on a non-loopback interface; set the token
    # (or bind 127.0.0.1 explicitly) to run it.
    api_token = os.getenv("OMNILSS_API_TOKEN", "")
    _validate_api_token(host, api_token)

    (
        service,
        fit_pb2_grpc,
        predict_pb2_grpc,
        sample_pb2_grpc,
        capability_pb2_grpc,
    ) = create_service()

    class _TokenAuthInterceptor(grpc.ServerInterceptor):
        def __init__(self, token: str) -> None:
            self._token = token

        def intercept_service(self, continuation, handler_call_details):
            if not self._token:
                return continuation(handler_call_details)
            metadata = dict(handler_call_details.invocation_metadata or [])
            if metadata.get("authorization", "") != f"Bearer {self._token}":
                def _abort(_request, context):
                    context.abort(grpc.StatusCode.UNAUTHENTICATED, "Invalid token")
                return grpc.unary_unary_rpc_method_handler(_abort)
            return continuation(handler_call_details)

    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=8),
        interceptors=[_TokenAuthInterceptor(api_token)],
        options=[
            ("grpc.max_receive_message_length", MAX_GRPC_MESSAGE_BYTES),
            ("grpc.max_send_message_length", MAX_GRPC_MESSAGE_BYTES),
        ],
    )
    fit_pb2_grpc.add_FitServiceServicer_to_server(service, server)
    predict_pb2_grpc.add_PredictServiceServicer_to_server(service, server)
    sample_pb2_grpc.add_SampleServiceServicer_to_server(service, server)
    capability_pb2_grpc.add_CapabilityServiceServicer_to_server(service, server)

    # Phase 5.2: TLS is enabled when the certificate/key env vars are set;
    # otherwise the server binds insecurely (auth token still applies).
    tls_cert = os.getenv("OMNILSS_TLS_CERT", "")
    tls_key = os.getenv("OMNILSS_TLS_KEY", "")
    if tls_cert and tls_key:
        with open(tls_cert, "rb") as fh:
            cert_bytes = fh.read()
        with open(tls_key, "rb") as fh:
            key_bytes = fh.read()
        creds = grpc.ssl_server_credentials(((key_bytes, cert_bytes),))
        server.add_secure_port(f"{host}:{port}", creds)
    else:
        server.add_insecure_port(f"{host}:{port}")
    server.start()
    return server
