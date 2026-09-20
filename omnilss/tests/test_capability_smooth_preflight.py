import json
import numpy as np
from omnilss.family_capabilities import method_route_capability_report
from omnilss.api.grpc import server as grpc_server
from omnilss.api.grpc.generated import capability_pb2


def test_python_preflight_reports_smooth_cg():
    report=method_route_capability_report('NO','CG',formulas={'sigma':'~ pb(x)'})
    assert report['ok'] is False
    assert report['code']=='smooth_terms_unsupported_for_method'
    assert report['smooth_parameters']==['sigma']


def test_python_preflight_allows_linear_cg():
    report=method_route_capability_report('NO','CG',formulas={'sigma':'~ x'})
    assert report['ok'] is True


def test_grpc_route_preflight_legacy_still_returns_capability_report():
    service,*_=grpc_server.create_service()
    response=service.RouteCapability(capability_pb2.RouteCapabilityRequest(
        family='NO',method='CG',strict=False),None)
    report=json.loads(response.report_json)
    assert response.success is True
    assert report['method']=='CG'
