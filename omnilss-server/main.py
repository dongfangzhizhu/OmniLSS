from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any
import uuid

from omnilss import gamlss
from omnilss.serialization import save_model_json, load_model_json

app = FastAPI(title="omnilss-server")
STORE = {}


class FitRequest(BaseModel):
    formula: str
    family: str
    data: dict[str, Any]


class PredictRequest(BaseModel):
    model_id: str
    newdata: dict[str, Any]


@app.post("/fit")
def fit(req: FitRequest):
    model = gamlss(req.formula, family=req.family, data=req.data)
    model_id = str(uuid.uuid4())
    path = f"/tmp/{model_id}.omnilss"
    save_model_json(model, path)
    STORE[model_id] = path
    return {"model_id": model_id}


@app.post("/predict")
def predict(req: PredictRequest):
    path = STORE.get(req.model_id)
    if not path:
        raise HTTPException(404, "model not found")
    try:
        model = load_model_json(path)
        params = model.predict_params(req.newdata)
        return {
            "params": {
                k: v.tolist() if hasattr(v, "tolist") else list(v)
                for k, v in params.items()
            }
        }
    except KeyError as exc:
        raise HTTPException(400, f"Missing variable in newdata: {exc}") from exc
    except Exception as exc:
        raise HTTPException(500, f"Prediction failed: {exc}") from exc


@app.get("/diagnostics/{model_id}")
def diagnostics(model_id: str):
    path = STORE.get(model_id)
    if not path:
        raise HTTPException(404, "model not found")
    model = load_model_json(path)
    return {"deviance": float(model.g_dev), "iter": int(model.iter)}


@app.post("/distributions/select")
def distributions_select(payload: dict[str, Any]):
    # placeholder for AutoML selection route
    return {"selected_family": payload.get("candidate_families", ["NO"])[0]}
