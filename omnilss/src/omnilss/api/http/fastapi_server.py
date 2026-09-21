"""Optional FastAPI service layer sharing the gRPC model registry."""

from __future__ import annotations

import os
import json
from typing import Any

try:  # Keep importing this module safe when the optional extra is absent.
    from fastapi import Request as _FastAPIRequest
except Exception:  # pragma: no cover - exercised only without FastAPI installed
    _FastAPIRequest = Any  # type: ignore[assignment,misc]


def create_app():
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException, status
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "FastAPI runtime is unavailable. Install with `pip install fastapi uvicorn`."
        ) from exc

    from ... import gamlss
    from ...family_capabilities import capability_matrix, method_route_capability_report, require_production_family_route
    from ..grpc import server as grpc_server

    def _error_detail(exc: Exception) -> dict[str, Any] | str:
        """Convert the shared gRPC-safe error text into an HTTP JSON detail."""
        text = grpc_server._error_text(exc)
        try:
            payload = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return text
        return payload if isinstance(payload, dict) else text

    def _check_auth(authorization: str = Header(default="")) -> None:
        api_token = os.getenv("OMNILSS_API_TOKEN", "")
        if not api_token:
            return
        if not authorization.startswith("Bearer "):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)
        if authorization[7:] != api_token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED)

    app = FastAPI(title="OmniLSS REST API", version="v1-prototype")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "omnilss-fastapi"}

    @app.get("/models")
    def list_models(_: None = Depends(_check_auth)) -> dict[str, list[str]]:
        return {"model_ids": grpc_server.REGISTRY.list_ids()}

    @app.get("/capabilities")
    def capabilities(_: None = Depends(_check_auth)) -> dict[str, Any]:
        return capability_matrix()

    @app.get("/route-capability")
    def route_capability(
        family: str,
        method: str,
        strict: bool = False,
        formula: str | None = None,
        sigma_formula: str | None = None,
        nu_formula: str | None = None,
        tau_formula: str | None = None,
        _: None = Depends(_check_auth),
    ) -> dict[str, Any]:
        formulas = {
            key: value for key, value in {
                "mu": formula, "sigma": sigma_formula, "nu": nu_formula, "tau": tau_formula,
            }.items() if value
        }
        return method_route_capability_report(family, method, strict=strict, formulas=formulas)

    @app.delete("/models/{model_id}")
    def delete_model(model_id: str, _: None = Depends(_check_auth)) -> dict[str, Any]:
        deleted = grpc_server.REGISTRY.delete(model_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="model_id not found")
        return {"deleted": True, "model_id": model_id}

    @app.post("/fit")
    async def fit(request: _FastAPIRequest, _: None = Depends(_check_auth)) -> dict[str, Any]:
        """Fit a model through the same production admission as gRPC."""
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            family = str(payload.get("family", "")).strip()
            formula = str(payload.get("formula", "")).strip()
            if not family or not formula:
                raise ValueError("fit requires non-empty family and formula")
            method = str(payload.get("method", "RS")).strip() or "RS"
            require_production_family_route(family, method, route="http")
            data = payload.get("data")
            if not isinstance(data, dict):
                raise ValueError("fit requires data as a JSON object")
            kwargs: dict[str, Any] = {
                "formula": formula,
                "family": family,
                "data": dict(data),
                "method": method,
            }
            if "weights" in kwargs["data"]:
                kwargs["weights"] = kwargs["data"].pop("weights")
            if payload.get("sigma_formula"):
                kwargs["sigma_formula"] = str(payload["sigma_formula"])
            formulas = payload.get("parameter_formulas")
            if formulas is not None:
                if not isinstance(formulas, dict):
                    raise ValueError("parameter_formulas must be an object")
                kwargs["parameter_formulas"] = formulas
            if payload.get("max_iter") is not None:
                kwargs["max_iter"] = min(max(1, int(payload["max_iter"])), 1000)
            model = gamlss(**kwargs)
            model_id = grpc_server.REGISTRY.save(model)
            return {
                "success": True,
                "model_id": model_id,
                "deviance": float(model.g_dev),
                "iterations": int(model.iter),
                "converged": bool(getattr(model, "additional_slots", {}).get("converged", True)),
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=_error_detail(exc)) from exc

    @app.post("/models/{model_id}/predict")
    async def predict(model_id: str, request: _FastAPIRequest, _: None = Depends(_check_auth)) -> dict[str, Any]:
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            newdata = payload.get("newdata", payload)
            if not isinstance(newdata, dict):
                raise ValueError("predict requires newdata as a JSON object")
            model = grpc_server.REGISTRY.load(model_id)
            params = model.predict_params(newdata)
            return {"success": True, "model_id": model_id, "params": {k: list(map(float, v)) for k, v in params.items()}}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="model_id not found") from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=_error_detail(exc)) from exc

    return app
