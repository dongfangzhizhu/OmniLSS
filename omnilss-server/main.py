"""FastAPI REST boundary for OmniLSS Core.

The REST server intentionally shares the same persistent model registry used by
``omnilss.api.grpc.server`` so HTTP and gRPC deployments have identical model
lifecycle behavior: artifacts live under ``OMNILSS_MODEL_STORE_DIR`` and the
SQLite index lives at ``OMNILSS_MODEL_DB_PATH``.
"""

from __future__ import annotations

import math
import os
import time
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, status
from pydantic import BaseModel

from omnilss import gamlss
from omnilss.api.grpc.server import REGISTRY
from omnilss.family_capabilities import (
    ProductionFamilyRejectedError,
    require_production_family_route,
)
from omnilss.formula_parser import has_smooth_terms
from omnilss.prediction import PredictionSchemaError

app = FastAPI(title="omnilss-server")

# Phase 5.3 / A4: DoS limits for the REST boundary.
MAX_JSON_BODY_BYTES = 1_000_000  # 1 MiB
RATE_LIMIT_WINDOW_SECONDS = 60
RATE_LIMIT_MAX_REQUESTS = 300
RATE_LIMIT_MAX_CLIENTS = 10_000

_RATE_LIMIT_STATE: dict[str, list[float]] = {}


def _prune_rate_limit_state(now: float) -> None:
    """Drop expired client buckets and enforce an upper bound on tracked IPs."""
    stale_clients = [
        key
        for key, timestamps in _RATE_LIMIT_STATE.items()
        if not timestamps or (now - timestamps[-1]) >= RATE_LIMIT_WINDOW_SECONDS
    ]
    for key in stale_clients:
        _RATE_LIMIT_STATE.pop(key, None)

    if len(_RATE_LIMIT_STATE) > RATE_LIMIT_MAX_CLIENTS:
        overflow = len(_RATE_LIMIT_STATE) - RATE_LIMIT_MAX_CLIENTS
        oldest = sorted(
            _RATE_LIMIT_STATE.items(),
            key=lambda item: item[1][-1] if item[1] else -1.0,
        )[:overflow]
        for key, _ in oldest:
            _RATE_LIMIT_STATE.pop(key, None)


def _check_rate_limit(client: str, now: float | None = None) -> bool:
    """Return True when the request is allowed, False when rate-limited."""
    ts = time.monotonic() if now is None else float(now)
    if len(_RATE_LIMIT_STATE) >= RATE_LIMIT_MAX_CLIENTS and client not in _RATE_LIMIT_STATE:
        _prune_rate_limit_state(ts)

    window = [t for t in _RATE_LIMIT_STATE.get(client, []) if ts - t < RATE_LIMIT_WINDOW_SECONDS]
    if len(window) >= RATE_LIMIT_MAX_REQUESTS:
        _RATE_LIMIT_STATE[client] = window
        return False
    window.append(ts)
    _RATE_LIMIT_STATE[client] = window

    if len(_RATE_LIMIT_STATE) > RATE_LIMIT_MAX_CLIENTS:
        _prune_rate_limit_state(ts)
    return True


def _check_content_length(content_length: str | None) -> bool:
    """Return False when Content-Length exceeds MAX_JSON_BODY_BYTES."""
    if content_length and content_length.isdigit():
        return int(content_length) <= MAX_JSON_BODY_BYTES
    return True


@app.middleware("http")
async def _dos_guard(request, call_next):
    """Reject oversized bodies and cap requests per client per window."""
    if not _check_content_length(request.headers.get("content-length")):
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content={"error": "request body too large"},
        )

    client = request.client.host if request.client else "unknown"
    if not _check_rate_limit(client):
        from fastapi.responses import JSONResponse

        return JSONResponse(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            content={"error": "rate limit exceeded"},
        )

    return await call_next(request)


# Phase 5.1: token authentication on every endpoint (same policy as
# omnilss.api.http.fastapi_server._check_auth). Production deployments must
# set OMNILSS_API_TOKEN (docker-compose does); with no token configured the
# API is open for local development only.
def _check_auth(authorization: str = Header(default="")) -> None:
    api_token = os.getenv("OMNILSS_API_TOKEN", "")
    if not api_token:
        return
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
    if authorization[7:] != api_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)


class FitRequest(BaseModel):
    formula: str
    family: str
    data: dict[str, Any]
    sigma_formula: str = "~ 1"
    nu_formula: str | None = None
    tau_formula: str | None = None
    method: str = "RS"
    max_iter: int | None = None
    verbose: bool = False


class PredictRequest(BaseModel):
    model_id: str
    newdata: dict[str, Any]


class DistributionSelectRequest(BaseModel):
    formula: str
    data: dict[str, Any]
    candidate_families: list[str] = ["NO"]
    sigma_formula: str = "~ 1"
    method: str = "RS"
    gaic_k: float = 2.0


def _model_kwargs(
    req: FitRequest | DistributionSelectRequest, family: str | None = None
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "formula": req.formula,
        "family": family or req.family,  # type: ignore[attr-defined]
        "data": req.data,
        "sigma_formula": req.sigma_formula,
        "method": req.method,
        "verbose": getattr(req, "verbose", False),
    }
    if isinstance(req, FitRequest):
        if req.nu_formula:
            kwargs.setdefault("parameter_formulas", {})["nu"] = req.nu_formula
        if req.tau_formula:
            kwargs.setdefault("parameter_formulas", {})["tau"] = req.tau_formula
        if req.max_iter is not None:
            kwargs["max_iter"] = req.max_iter
    return kwargs


def _reject_unsupported_smooth_method(req: FitRequest | DistributionSelectRequest) -> None:
    if str(req.method).upper() != "CG":
        return
    formulas = [req.formula, req.sigma_formula]
    if isinstance(req, FitRequest):
        formulas.extend(value for value in (req.nu_formula, req.tau_formula) if value)
    if any(has_smooth_terms(value) for value in formulas):
        raise HTTPException(
            status_code=400,
            detail={
                "type": "method_capability_error",
                "code": "smooth_terms_unsupported_for_method",
                "method": "CG",
                "message": "method='CG' does not support penalized smooth terms; use method='RS' or method='auto'.",
            },
        )


def _enforce_route_capability(family: str, method: str) -> None:
    """Apply the same fail-closed production route gate used by gRPC/FastAPI."""
    try:
        require_production_family_route(family, method, route="http")
    except ProductionFamilyRejectedError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "type": "production_family_rejected",
                "family": family,
                "method": method,
                "message": str(exc),
            },
        ) from exc


def _model_summary(model_id: str, model: Any) -> dict[str, Any]:
    slots = dict(getattr(model, "additional_slots", {}) or {})
    converged = slots.get("rs_converged", slots.get("cg_converged", True))
    return {
        "model_id": model_id,
        "deviance": float(model.g_dev),
        "iterations": int(model.iter),
        "converged": bool(converged),
    }


def _load_or_404(model_id: str) -> Any:
    try:
        return REGISTRY.load(model_id)
    except KeyError as exc:
        raise HTTPException(404, "model not found") from exc


@app.post("/fit")
def fit(req: FitRequest, _: None = Depends(_check_auth)):
    _reject_unsupported_smooth_method(req)
    _enforce_route_capability(req.family, req.method)
    if req.max_iter is not None and not (1 <= int(req.max_iter) <= 1000):
        raise HTTPException(status_code=400, detail="max_iter must be between 1 and 1000")
    try:
        model = gamlss(**_model_kwargs(req))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    model_id = REGISTRY.save(model)
    return _model_summary(model_id, model)


@app.post("/predict")
def predict(req: PredictRequest, _: None = Depends(_check_auth)):
    model = _load_or_404(req.model_id)
    try:
        params = model.predict_params(req.newdata)
        return {
            "params": {
                k: v.tolist() if hasattr(v, "tolist") else list(v)
                for k, v in params.items()
            }
        }
    except PredictionSchemaError as exc:
        raise HTTPException(400, exc.to_dict()) from exc
    except KeyError:
        raise HTTPException(400, "Missing variable in newdata") from None
    except Exception as exc:
        raise HTTPException(500, f"Prediction failed: {exc}") from exc


@app.get("/diagnostics/{model_id}")
def diagnostics(model_id: str, _: None = Depends(_check_auth)):
    model = _load_or_404(model_id)
    return {"deviance": float(model.g_dev), "iter": int(model.iter)}


@app.get("/models")
def list_models(_: None = Depends(_check_auth)):
    return {"model_ids": REGISTRY.list_ids()}


@app.delete("/models/{model_id}")
def delete_model(model_id: str, _: None = Depends(_check_auth)):
    deleted = REGISTRY.delete(model_id)
    if not deleted:
        raise HTTPException(404, "model not found")
    return {"deleted": True, "model_id": model_id}


@app.post("/distributions/select")
def distributions_select(
    req: DistributionSelectRequest, _: None = Depends(_check_auth)
):
    _reject_unsupported_smooth_method(req)
    n_obs = max(
        (len(v) for v in req.data.values() if hasattr(v, "__len__")), default=1
    )
    rows: list[dict[str, Any]] = []
    for family in req.candidate_families:
        try:
            require_production_family_route(family, req.method, route="http")
            model = gamlss(**_model_kwargs(req, family=family))
            deviance = float(model.g_dev)
            parameter_count = sum(
                len(v) for v in getattr(model, "coefficients", {}).values()
            )
            rows.append(
                {
                    "family": family,
                    "deviance": deviance,
                    "aic": deviance + 2.0 * parameter_count,
                    "bic": deviance + math.log(max(n_obs, 2)) * parameter_count,
                    "gaic": deviance + float(req.gaic_k) * parameter_count,
                    "parameter_count": parameter_count,
                    "iterations": int(model.iter),
                }
            )
        except ProductionFamilyRejectedError:
            rows.append({"family": family, "error": "family_not_admitted"})
        except Exception:
            # Phase 5.4: never leak raw exception text to clients.
            rows.append({"family": family, "error": "fit_failed"})

    successful = [row for row in rows if "deviance" in row]
    if not successful:
        raise HTTPException(
            400, {"error": "no candidate family fit succeeded", "candidates": rows}
        )
    ranked = sorted(successful, key=lambda row: row["gaic"])
    return {"selected_family": ranked[0]["family"], "best": ranked[0], "candidates": rows}
