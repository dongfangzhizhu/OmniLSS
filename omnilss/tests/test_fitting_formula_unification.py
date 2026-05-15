import numpy as np

from omnilss.fitting import _build_design_matrix


def test_formula_log_transform():
    data={"y":np.array([1.0,2.0,3.0]),"x":np.array([1.0,2.0,4.0])}
    _,X,labels=_build_design_matrix("y ~ log(x)",data)
    assert labels==["log(x)"]
    np.testing.assert_allclose(X[:,1], np.log(data["x"]))


def test_formula_no_intercept():
    data={"y":np.array([1.0,2.0,3.0]),"x1":np.array([1,2,3]),"x2":np.array([4,5,6])}
    _,X,labels=_build_design_matrix("y ~ x1 + x2 - 1",data)
    assert X.shape==(3,2)
    assert labels==["x1","x2"]


def test_formula_interaction_term():
    data={"y":np.array([1.0,2.0,3.0]),"x1":np.array([1,2,3]),"x2":np.array([4,5,6])}
    _,X,labels=_build_design_matrix("y ~ x1:x2",data)
    assert labels==["x1:x2"]
    np.testing.assert_allclose(X[:,1], data["x1"]*data["x2"])


def test_formula_identity_expression():
    data={"y":np.array([1.0,2.0,3.0]),"x":np.array([1.0,2.0,3.0])}
    _,X,labels=_build_design_matrix("y ~ I(x**2)",data)
    assert labels==["I(x**2)"]
    np.testing.assert_allclose(X[:,1], data["x"]**2)
