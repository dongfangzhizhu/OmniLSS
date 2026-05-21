"""Optional FastAPI service layer sharing the gRPC model registry."""

from __future__ import annotations

import os
from typing import Any


def create_app():
    try:
        from fastapi import Depends, FastAPI, Header, HTTPException, status
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "FastAPI runtime is unavailable. Install with `pip install fastapi uvicorn`."
        ) from exc

    from ..grpc.server import REGISTRY

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

    return app
