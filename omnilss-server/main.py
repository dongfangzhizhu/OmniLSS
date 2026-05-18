"""FastAPI REST boundary for OmniLSS Core.

The REST server intentionally shares the same persistent model registry used by
``omnilss.api.grpc.server`` so HTTP and gRPC deployments have identical model
lifecycle behavior: artifacts live under ``OMNILSS_MODEL_STORE_DIR`` and the
SQLite index lives at ``OMNILSS_MODEL_DB_PATH``.
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from omnilss import gamlss
from omnilss.api.grpc.server import REGISTRY
from omnilss.prediction import PredictionSchemaError

app = FastAPI(title="omnilss-server")


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
def fit(req: FitRequest):
    model = gamlss(**_model_kwargs(req))
    model_id = REGISTRY.save(model)
    return _model_summary(model_id, model)


@app.post("/predict")
def predict(req: PredictRequest):
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
    except KeyError as exc:
        raise HTTPException(400, f"Missing variable in newdata: {exc}") from exc
    except Exception as exc:
        raise HTTPException(500, f"Prediction failed: {exc}") from exc


@app.get("/diagnostics/{model_id}")
def diagnostics(model_id: str):
    model = _load_or_404(model_id)
    return {"deviance": float(model.g_dev), "iter": int(model.iter)}


@app.get("/models")
def list_models():
    return {"model_ids": REGISTRY.list_ids()}


@app.delete("/models/{model_id}")
def delete_model(model_id: str):
    deleted = REGISTRY.delete(model_id)
    if not deleted:
        raise HTTPException(404, "model not found")
    return {"deleted": True, "model_id": model_id}


@app.post("/distributions/select")
def distributions_select(req: DistributionSelectRequest):
    n_obs = max(
        (len(v) for v in req.data.values() if hasattr(v, "__len__")), default=1
    )
    rows: list[dict[str, Any]] = []
    for family in req.candidate_families:
        try:
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
                    "bic": deviance
                    + __import__("math").log(max(n_obs, 2)) * parameter_count,
                    "gaic": deviance + float(req.gaic_k) * parameter_count,
                    "parameter_count": parameter_count,
                    "iterations": int(model.iter),
                }
            )
        except Exception as exc:
            rows.append({"family": family, "error": str(exc)})

    successful = [row for row in rows if "deviance" in row]
    if not successful:
        raise HTTPException(
            400, {"error": "no candidate family fit succeeded", "candidates": rows}
        )
    ranked = sorted(successful, key=lambda row: row["gaic"])
    return {"selected_family": ranked[0]["family"], "best": ranked[0], "candidates": rows}
