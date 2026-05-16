import numpy as np
import pytest

from omnilss.tensor_protocol import validate_design_matrix, validate_vector


def test_validate_vector_ok():
    out = validate_vector("y", np.array([1.0, 2.0]), n=2)
    assert out.shape == (2,)


def test_validate_design_matrix_ok():
    out = validate_design_matrix("X", np.ones((3, 2)), n=3)
    assert out.shape == (3, 2)


def test_validate_design_matrix_raises_on_bad_rank():
    with pytest.raises(ValueError):
        validate_design_matrix("X", np.ones(3), n=3)
