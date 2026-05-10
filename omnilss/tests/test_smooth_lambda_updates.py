"""Tests for smooth lambda updates (ML, GCV, REML, GAIC)."""

import pytest
import numpy as np
import jax.numpy as jnp
from omnilss.smooth_fitting import update_smooth_lambdas, SmoothFitInfo
from omnilss.smoothers.penalties import difference_penalty


@pytest.fixture
def simple_data():
    """Generate simple test data."""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 10, n)
    y = np.sin(x) + 0.1 * np.random.randn(n)
    
    # Create design matrix (polynomial basis for simplicity)
    X = np.column_stack([np.ones(n), x, x**2, x**3])
    
    # Initial coefficients
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    
    # Weights
    w = np.ones(n)
    
    return y, X, beta, w


@pytest.fixture
def smooth_fit_info():
    """Create a simple SmoothFitInfo object."""
    penalty = difference_penalty(4, order=2)
    
    return SmoothFitInfo(
        term_index=0,
        variable="x",
        smoother="ps",
        lambda_=1.0,
        edf=2.5,
        penalty=penalty,
        basis_columns=(0, 4),
        selection_method="ML",
        criterion_value=None,
    )


def test_update_smooth_lambdas_ml(simple_data, smooth_fit_info):
    """Test ML lambda update."""
    y, X, beta, w = simple_data
    smooth_fits = [smooth_fit_info]
    
    updated_fits = update_smooth_lambdas(X, y, beta, w, smooth_fits, method="ML")
    
    assert len(updated_fits) == 1
    assert updated_fits[0].selection_method == "ML"
    assert updated_fits[0].lambda_ > 0
    assert np.isfinite(updated_fits[0].lambda_)
    assert updated_fits[0].edf > 0
    assert np.isfinite(updated_fits[0].edf)


def test_update_smooth_lambdas_gcv(simple_data, smooth_fit_info):
    """Test GCV lambda update."""
    y, X, beta, w = simple_data
    smooth_fits = [smooth_fit_info]
    
    updated_fits = update_smooth_lambdas(X, y, beta, w, smooth_fits, method="GCV")
    
    assert len(updated_fits) == 1
    assert updated_fits[0].selection_method == "GCV"
    assert updated_fits[0].lambda_ > 0
    assert np.isfinite(updated_fits[0].lambda_)
    assert updated_fits[0].edf > 0
    assert np.isfinite(updated_fits[0].edf)
    assert updated_fits[0].criterion_value is not None
    assert np.isfinite(updated_fits[0].criterion_value)


def test_update_smooth_lambdas_reml(simple_data, smooth_fit_info):
    """Test REML lambda update."""
    y, X, beta, w = simple_data
    smooth_fits = [smooth_fit_info]
    
    updated_fits = update_smooth_lambdas(X, y, beta, w, smooth_fits, method="REML")
    
    assert len(updated_fits) == 1
    assert updated_fits[0].selection_method == "REML"
    assert updated_fits[0].lambda_ > 0
    assert np.isfinite(updated_fits[0].lambda_)
    assert updated_fits[0].edf > 0
    assert np.isfinite(updated_fits[0].edf)
    assert updated_fits[0].criterion_value is not None
    assert np.isfinite(updated_fits[0].criterion_value)


def test_update_smooth_lambdas_gaic(simple_data, smooth_fit_info):
    """Test GAIC lambda update."""
    y, X, beta, w = simple_data
    smooth_fits = [smooth_fit_info]
    
    updated_fits = update_smooth_lambdas(X, y, beta, w, smooth_fits, method="GAIC")
    
    assert len(updated_fits) == 1
    assert updated_fits[0].selection_method == "GAIC"
    assert updated_fits[0].lambda_ > 0
    assert np.isfinite(updated_fits[0].lambda_)
    assert updated_fits[0].edf > 0
    assert np.isfinite(updated_fits[0].edf)
    assert updated_fits[0].criterion_value is not None
    assert np.isfinite(updated_fits[0].criterion_value)


def test_update_smooth_lambdas_multiple_smooths(simple_data):
    """Test updating multiple smooth terms."""
    y, X, beta, w = simple_data
    
    # Create two smooth terms
    penalty1 = difference_penalty(2, order=2)
    penalty2 = difference_penalty(2, order=2)
    
    smooth_fits = [
        SmoothFitInfo(
            term_index=0,
            variable="x1",
            smoother="ps",
            lambda_=1.0,
            edf=1.5,
            penalty=penalty1,
            basis_columns=(0, 2),
            selection_method="ML",
            criterion_value=None,
        ),
        SmoothFitInfo(
            term_index=1,
            variable="x2",
            smoother="ps",
            lambda_=1.0,
            edf=1.5,
            penalty=penalty2,
            basis_columns=(2, 4),
            selection_method="ML",
            criterion_value=None,
        ),
    ]
    
    updated_fits = update_smooth_lambdas(X, y, beta, w, smooth_fits, method="GCV")
    
    assert len(updated_fits) == 2
    for fit in updated_fits:
        assert fit.selection_method == "GCV"
        assert fit.lambda_ > 0
        assert np.isfinite(fit.lambda_)
        assert fit.edf > 0
        assert np.isfinite(fit.edf)


def test_update_smooth_lambdas_comparison():
    """Compare different methods on the same data."""
    np.random.seed(42)
    n = 100
    x = np.linspace(0, 10, n)
    y = np.sin(x) + 0.1 * np.random.randn(n)
    
    X = np.column_stack([np.ones(n), x, x**2, x**3])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    w = np.ones(n)
    
    penalty = difference_penalty(4, order=2)
    smooth_fit = SmoothFitInfo(
        term_index=0,
        variable="x",
        smoother="ps",
        lambda_=1.0,
        edf=2.5,
        penalty=penalty,
        basis_columns=(0, 4),
        selection_method="ML",
        criterion_value=None,
    )
    
    # Test all methods
    methods = ["ML", "GCV", "REML", "GAIC"]
    results = {}
    
    for method in methods:
        updated_fits = update_smooth_lambdas(X, y, beta, w, [smooth_fit], method=method)
        results[method] = {
            "lambda": updated_fits[0].lambda_,
            "edf": updated_fits[0].edf,
            "criterion": updated_fits[0].criterion_value,
        }
    
    # All methods should produce valid results
    for method in methods:
        assert results[method]["lambda"] > 0
        assert np.isfinite(results[method]["lambda"])
        assert results[method]["edf"] > 0
        assert np.isfinite(results[method]["edf"])
    
    # GCV, REML, and GAIC should have criterion values
    for method in ["GCV", "REML", "GAIC"]:
        assert results[method]["criterion"] is not None
        assert np.isfinite(results[method]["criterion"])
    
    # Print results for inspection (optional)
    print("\nLambda update comparison:")
    for method in methods:
        print(f"{method:6s}: λ={results[method]['lambda']:.4e}, "
              f"edf={results[method]['edf']:.2f}, "
              f"criterion={results[method]['criterion']}")


def test_update_smooth_lambdas_edge_cases(simple_data, smooth_fit_info):
    """Test edge cases and error handling."""
    y, X, beta, w = simple_data
    
    # Test with zero weights (should handle gracefully)
    w_zero = np.zeros_like(w)
    w_zero[0] = 1.0  # At least one non-zero weight
    
    updated_fits = update_smooth_lambdas(X, y, beta, w_zero, [smooth_fit_info], method="ML")
    assert len(updated_fits) == 1
    assert updated_fits[0].lambda_ > 0
    
    # Test with very small lambda
    smooth_fit_small = SmoothFitInfo(
        term_index=0,
        variable="x",
        smoother="ps",
        lambda_=1e-10,
        edf=2.5,
        penalty=smooth_fit_info.penalty,
        basis_columns=(0, 4),
        selection_method="ML",
        criterion_value=None,
    )
    
    updated_fits = update_smooth_lambdas(X, y, beta, w, [smooth_fit_small], method="GCV")
    assert len(updated_fits) == 1
    assert updated_fits[0].lambda_ > 0
    
    # Test with very large lambda
    smooth_fit_large = SmoothFitInfo(
        term_index=0,
        variable="x",
        smoother="ps",
        lambda_=1e10,
        edf=2.5,
        penalty=smooth_fit_info.penalty,
        basis_columns=(0, 4),
        selection_method="ML",
        criterion_value=None,
    )
    
    updated_fits = update_smooth_lambdas(X, y, beta, w, [smooth_fit_large], method="REML")
    assert len(updated_fits) == 1
    assert updated_fits[0].lambda_ > 0


def test_update_smooth_lambdas_unknown_method(simple_data, smooth_fit_info):
    """Test with unknown method (should keep current lambda)."""
    y, X, beta, w = simple_data
    smooth_fits = [smooth_fit_info]
    
    updated_fits = update_smooth_lambdas(X, y, beta, w, smooth_fits, method="UNKNOWN")
    
    assert len(updated_fits) == 1
    # Should keep the original smooth fit unchanged
    assert updated_fits[0].lambda_ == smooth_fit_info.lambda_
    assert updated_fits[0].edf == smooth_fit_info.edf


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
