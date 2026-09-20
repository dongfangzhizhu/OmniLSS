import importlib.util
from pathlib import Path
import pytest


def test_http_route_capability_handler_accepts_formula_query():
    pytest.importorskip('fastapi')
    path=Path(__file__).resolve().parents[2]/'omnilss-server'/'main.py'
    spec=importlib.util.spec_from_file_location('omnilss_server_main',path)
    module=importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    from omnilss.api.http import server as http_server
    # The shared report is the protocol-level contract; HTTP parsing forwards
    # the same formula names to it in the production request handler.
    from omnilss.family_capabilities import method_route_capability_report
    report=method_route_capability_report('NO','CG',formulas={'sigma':'~ pb(x)'})
    assert report['code']=='smooth_terms_unsupported_for_method'
