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


def _load_server_module():
    """Import ``omnilss-server/main.py`` for the HTTP helper assertions.

    ``main.py`` uses ``from __future__ import annotations``, so pydantic sees
    its field annotations as strings.  Under ``spec_from_file_location`` the
    freshly created module namespace is not the one pydantic resolves names
    against, and instantiating ``FitRequest`` fails with
    "not fully defined; you should define ``Any``".  A normal import (how the
    service actually runs) has no such problem, so this is a test-harness
    artefact rather than a service defect -- hence the explicit
    ``model_rebuild()`` instead of editing the service source.
    """
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "omnilss-server" / "main.py"
    server_dir = str(path.parent)
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)
    spec = importlib.util.spec_from_file_location("omnilss_server_main", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    # Resolve the stringified annotations against the module's own globals.
    # Under ``spec_from_file_location`` pydantic cannot infer that namespace,
    # so ``Any`` (imported at main.py top level) must be handed over explicitly.
    module.FitRequest.model_rebuild(
        _types_namespace=vars(module),
    )
    return module


def test_http_helper_rejects_cg_smooth_structured():
    pytest.importorskip("fastapi")
    module = _load_server_module()
    FitRequest, _reject_unsupported_smooth_method = module.FitRequest, module._reject_unsupported_smooth_method
    from fastapi import HTTPException
    with pytest.raises(HTTPException) as caught:
        _reject_unsupported_smooth_method(FitRequest(
            formula="y ~ x", family="NO", data={"y":[1.],"x":[0.]},
            nu_formula="~ s(x)", method="CG"))
    assert caught.value.status_code == 400
    assert caught.value.detail["code"] == "smooth_terms_unsupported_for_method"


def test_server_fit_request_instantiates_under_normal_import():
    """The same model must work through a plain import (the deployed path).

    Guards the diagnosis above: if this ever fails, the service itself broke
    and the ``model_rebuild()`` in ``_load_server_module`` would be masking it.
    """
    pytest.importorskip("fastapi")
    import importlib
    import sys
    from pathlib import Path

    server_dir = str(Path(__file__).resolve().parents[2] / "omnilss-server")
    if server_dir not in sys.path:
        sys.path.insert(0, server_dir)
    main_module = importlib.import_module("main")
    instance = main_module.FitRequest(
        formula="y ~ x", family="NO", data={"y": [1.0], "x": [0.0]}, method="RS"
    )
    assert instance.formula == "y ~ x"
    assert instance.family == "NO"
