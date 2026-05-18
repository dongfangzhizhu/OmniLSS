"""Optional FastAPI service layer sharing the gRPC model registry."""

from __future__ import annotations

from typing import Any


def create_app():
    try:
        from fastapi import FastAPI, HTTPException
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "FastAPI runtime is unavailable. Install with `pip install fastapi uvicorn`."
        ) from exc

    from ..grpc.server import REGISTRY

    app = FastAPI(title="OmniLSS REST API", version="v1-prototype")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": "omnilss-fastapi"}

    @app.get("/models")
    def list_models() -> dict[str, list[str]]:
        return {"model_ids": REGISTRY.list_ids()}

    @app.delete("/models/{model_id}")
    def delete_model(model_id: str) -> dict[str, Any]:
        deleted = REGISTRY.delete(model_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="model_id not found")
        return {"deleted": True, "model_id": model_id}

    return app
