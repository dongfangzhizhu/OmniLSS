from __future__ import annotations

import numpy as np

from omnilss.runtime.errors import (
    ConvergenceError,
    DistributionError,
    NumericalError,
    OmniLSSError,
    RuntimeExecutionError,
)
from omnilss.runtime.observability import StructuredRuntimeLogger, build_runtime_event


def test_error_hierarchy_subclasses_base():
    assert issubclass(NumericalError, OmniLSSError)
    assert issubclass(ConvergenceError, OmniLSSError)
    assert issubclass(DistributionError, OmniLSSError)
    assert issubclass(RuntimeExecutionError, OmniLSSError)


def test_structured_runtime_logger_emits_json_lines(tmp_path):
    logger = StructuredRuntimeLogger()
    event = build_runtime_event(
        iteration=3,
        family="NO",
        condition_number=123.0,
        values={"mu": np.array([0.0, np.nan])},
        level="WARNING",
        message="nan_detected",
    )
    logger.log(event)
    text = logger.as_json_lines()
    assert '"family": "NO"' in text
    assert '"has_nan": true' in text

    out = tmp_path / "runtime.jsonl"
    logger.export_json_lines(out)
    assert out.read_text(encoding="utf-8") == text
