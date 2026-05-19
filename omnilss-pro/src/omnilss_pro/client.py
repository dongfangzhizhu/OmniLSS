"""OmniLSS Pro gRPC client.

This module communicates with OmniLSS Core via gRPC only. It intentionally
imports no ``omnilss`` package modules and uses an independently generated copy
of the protobuf stubs as the IPC contract.
"""

from __future__ import annotations

import json
from typing import Any

import grpc

from .api.grpc.generated import (
    fit_pb2,
    fit_pb2_grpc,
    predict_pb2,
    predict_pb2_grpc,
    sample_pb2,
    sample_pb2_grpc,
)


def _jsonable_mapping(data: dict[str, Any]) -> dict[str, Any]:
    """Convert array-like mapping values to JSON-serializable lists."""
    return {
        key: value.tolist() if hasattr(value, "tolist") else list(value)
        for key, value in data.items()
    }


class OmniLSSCoreClient:
    """gRPC client for the OmniLSS Core server."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 50051,
        timeout: float = 60.0,
    ) -> None:
        self._channel = grpc.insecure_channel(f"{host}:{port}")
        self._fit_stub = fit_pb2_grpc.FitServiceStub(self._channel)
        self._predict_stub = predict_pb2_grpc.PredictServiceStub(self._channel)
        self._sample_stub = sample_pb2_grpc.SampleServiceStub(self._channel)
        self._timeout = timeout

    def fit(
        self,
        formula: str,
        family: str,
        data: dict[str, Any],
        sigma_formula: str = "~ 1",
        nu_formula: str = "",
        tau_formula: str = "",
        method: str = "RS",
        max_iter: int = 20,
        verbose: bool = False,
    ) -> dict[str, Any]:
        """Fit a GAMLSS model through the Core gRPC server."""
        request = fit_pb2.FitRequest(
            formula=formula,
            family=family,
            data_json=json.dumps(_jsonable_mapping(data)),
            sigma_formula=sigma_formula,
            nu_formula=nu_formula,
            tau_formula=tau_formula,
            method=method,
            max_iter=max_iter,
            verbose=verbose,
        )
        response = self._fit_stub.Fit(request, timeout=self._timeout)
        if not response.success:
            raise RuntimeError(f"Core server fit failed: {response.error}")
        return {
            "model_id": response.model_id,
            "deviance": response.deviance,
            "iterations": response.iterations,
            "converged": response.converged,
        }

    def batch_fit(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Fit multiple GAMLSS models through Core's BatchFit RPC."""
        grpc_requests = []
        for item in requests:
            grpc_requests.append(
                fit_pb2.FitRequest(
                    formula=str(item["formula"]),
                    family=str(item["family"]),
                    data_json=json.dumps(_jsonable_mapping(item["data"])),
                    sigma_formula=str(item.get("sigma_formula", "~ 1")),
                    nu_formula=str(item.get("nu_formula", "")),
                    tau_formula=str(item.get("tau_formula", "")),
                    method=str(item.get("method", "RS")),
                    max_iter=int(item.get("max_iter", 20)),
                    verbose=bool(item.get("verbose", False)),
                )
            )
        response = self._fit_stub.BatchFit(
            fit_pb2.BatchFitRequest(requests=grpc_requests), timeout=self._timeout
        )
        if response.error and not response.responses:
            raise RuntimeError(f"Core server batch fit failed: {response.error}")
        return [
            {
                "model_id": result.model_id,
                "success": result.success,
                "error": result.error,
                "deviance": result.deviance,
                "iterations": result.iterations,
                "converged": result.converged,
            }
            for result in response.responses
        ]

    def list_models(self) -> list[str]:
        """List model ids currently registered by the Core server."""
        response = self._fit_stub.ListModels(
            fit_pb2.ListModelsRequest(), timeout=self._timeout
        )
        if not response.success:
            raise RuntimeError(f"Model listing failed: {response.error}")
        return list(response.model_ids)

    def delete_model(self, model_id: str) -> bool:
        """Delete a fitted model artifact from the Core server registry."""
        request = fit_pb2.DeleteModelRequest(model_id=model_id)
        response = self._fit_stub.DeleteModel(request, timeout=self._timeout)
        if not response.success:
            raise RuntimeError(f"Model deletion failed: {response.error}")
        return bool(response.deleted)

    def predict(self, model_id: str, newdata: dict[str, Any]) -> dict[str, list[float]]:
        """Predict distribution parameters for new data."""
        jsonable = _jsonable_mapping(newdata)
        columns = [
            predict_pb2.ColumnVector(name=name, values=list(map(float, values)))
            for name, values in jsonable.items()
        ]
        request = predict_pb2.PredictRequest(
            model_id=model_id,
            newdata_json=json.dumps(jsonable),
            newdata_columns=columns,
        )
        response = self._predict_stub.Predict(request, timeout=self._timeout)
        if not response.success:
            raise RuntimeError(f"Prediction failed: {response.error}")
        return json.loads(response.params_json)

    def sample(self, model_id: str, n: int) -> list[float]:
        """Sample from a fitted Core model."""
        request = sample_pb2.SampleRequest(model_id=model_id, n=n)
        response = self._sample_stub.Sample(request, timeout=self._timeout)
        if not response.success:
            raise RuntimeError(f"Sampling failed: {response.error}")
        return json.loads(response.samples_json)["samples"]

    def close(self) -> None:
        """Close the underlying gRPC channel."""
        self._channel.close()

    def __enter__(self) -> "OmniLSSCoreClient":
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
