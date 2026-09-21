"""Optional FastAPI service layer sharing the gRPC model registry."""

from __future__ import annotations

import os
from typing import Any


def create_app():
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException, Request, status
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "FastAPI runtime is unavailable. Install with `pip install fastapi uvicorn`."
        ) from exc

    from .. import gamlss
    from ...family_capabilities import require_production_family_route
    from ..grpc.server import REGISTRY, _error_text

    api_token = os.getenv("OMNILSS_API_TOKEN", "")

    def _check_auth(authorization: str = Header(default="")) -> None:
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
        return {"model_ids": REGISTRY.list_ids()}

    @app.delete("/models/{model_id}")
    def delete_model(model_id: str, _: None = Depends(_check_auth)) -> dict[str, Any]:
        deleted = REGISTRY.delete(model_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="model_id not found")
        return {"deleted": True, "model_id": model_id}

    @app.post("/fit")
    async def fit(request: Request, _: None = Depends(_check_auth)) -> dict[str, Any]:
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
            model_id = REGISTRY.save(model)
            return {
                "success": True,
                "model_id": model_id,
                "deviance": float(model.g_dev),
                "iterations": int(model.iter),
                "converged": bool(getattr(model, "additional_slots", {}).get("converged", True)),
            }
        except Exception as exc:
            raise HTTPException(status_code=400, detail=_error_text(exc)) from exc

    @app.post("/models/{model_id}/predict")
    async def predict(model_id: str, request: Request, _: None = Depends(_check_auth)) -> dict[str, Any]:
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("request body must be a JSON object")
            newdata = payload.get("newdata", payload)
            if not isinstance(newdata, dict):
                raise ValueError("predict requires newdata as a JSON object")
            model = REGISTRY.load(model_id)
            params = model.predict_params(newdata)
            return {"success": True, "model_id": model_id, "params": {k: list(map(float, v)) for k, v in params.items()}}
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="model_id not found") from exc
        except Exception as exc:
            raise HTTPException(status_code=400, detail=_error_text(exc)) from exc

    return app
