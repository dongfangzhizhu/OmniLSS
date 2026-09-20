"""Service boundaries expose the same structured CG smooth rejection."""
import json
import numpy as np
import pytest

from omnilss.api.grpc import server as grpc_server
from omnilss.api.grpc.generated import fit_pb2


def test_grpc_cg_smooth_error_is_structured():
    service, *_ = grpc_server.create_service()
    response = service.Fit(fit_pb2.FitRequest(
        formula="y ~ s(x)", family="NO",
        data_json=json.dumps({"y":[1.,2.],"x":[0.,1.]}), method="CG"), None)
    assert response.success is False
    error = json.loads(response.error)
    assert error["type"] == "method_capability_error"
    assert error["code"] == "smooth_terms_unsupported_for_method"


def test_grpc_batch_cg_smooth_error_is_structured():
    service, *_ = grpc_server.create_service()
    response = service.BatchFit(fit_pb2.BatchFitRequest(requests=[fit_pb2.FitRequest(
        formula="y ~ x", sigma_formula="~ pb(x)", family="NO",
        data_json=json.dumps({"y":[1.,2.],"x":[0.,1.]}), method="CG")]), None)
    assert response.success is False
    error = json.loads(response.responses[0].error)
    assert error["code"] == "smooth_terms_unsupported_for_method"


def test_http_helper_rejects_cg_smooth_structured():
    pytest.importorskip("fastapi")
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[2] / "omnilss-server" / "main.py"
    spec = importlib.util.spec_from_file_location("omnilss_server_main", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    FitRequest, _reject_unsupported_smooth_method = module.FitRequest, module._reject_unsupported_smooth_method
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as caught:
        _reject_unsupported_smooth_method(FitRequest(
            formula="y ~ x", family="NO", data={"y":[1.],"x":[0.]},
            nu_formula="~ s(x)", method="CG"))
    assert caught.value.status_code == 400
    assert caught.value.detail["code"] == "smooth_terms_unsupported_for_method"
