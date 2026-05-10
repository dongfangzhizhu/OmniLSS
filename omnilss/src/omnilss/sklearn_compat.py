"""scikit-learn compatible wrapper for OmniLSS.

Enables use in sklearn Pipelines while keeping the JAX backend.
This is an optional adapter layer — the primary OmniLSS API remains
formula-based (R-style). Use this when you need sklearn interoperability.

Examples
--------
>>> from omnilss.sklearn_compat import GAMLSSRegressor
>>> from sklearn.pipeline import Pipeline
>>> from sklearn.preprocessing import StandardScaler
>>> import numpy as np

>>> X = np.random.randn(200, 3)
>>> y = X[:, 0] + 0.5 * X[:, 1] + np.random.randn(200)

>>> pipe = Pipeline([
...     ("scaler", StandardScaler()),
...     ("gamlss", GAMLSSRegressor(family="NO")),
... ])
>>> pipe.fit(X, y)
>>> predictions = pipe.predict(X)
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from sklearn.base import BaseEstimator, RegressorMixin
    from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False
    # Provide stub base classes so the module can be imported without sklearn
    class BaseEstimator:  # type: ignore[no-redef]
        pass
    class RegressorMixin:  # type: ignore[no-redef]
        pass


def _require_sklearn() -> None:
    if not _SKLEARN_AVAILABLE:
        raise ImportError(
            "scikit-learn is required for GAMLSSRegressor. "
            "Install it with: pip install scikit-learn"
        )


class GAMLSSRegressor(BaseEstimator, RegressorMixin):
    """scikit-learn compatible GAMLSS regressor.

    Wraps OmniLSS's formula-based API in a scikit-learn estimator interface,
    enabling use in Pipelines, GridSearchCV, and other sklearn utilities.

    The JAX backend is preserved — GPU acceleration is available when JAX
    is configured with a GPU device.

    Parameters
    ----------
    family : str or FamilyDefinition, default="NO"
        Distribution family. Accepts string names (e.g., "NO", "GA", "NBI")
        or family instances (e.g., NO(), GA()).
    sigma_formula : str, optional
        Formula for the sigma (scale) parameter. If None, sigma is modeled
        as a constant intercept. Example: "~ x1 + x2".
    nu_formula : str, optional
        Formula for the nu (shape) parameter.
    tau_formula : str, optional
        Formula for the tau parameter.
    algorithm : str, default="RS"
        Fitting algorithm. One of:
        - "RS": Rigby-Stasinopoulos (default, most stable)
        - "CG": Cole-Green (alternative)
        - "Mixed": Automatic RS/CG selection
        - "Adam": Gradient descent via Optax
        - "L-BFGS": Quasi-Newton method
    feature_names : list of str, optional
        Names for the input features. If None, uses "x0", "x1", etc.
        Set automatically when fitting a pandas DataFrame.
    response_name : str, default="y"
        Internal name for the response variable.
    mu_smoother : str, optional
        Smoother for mu formula. E.g., "pb" wraps all terms in pb().
    fit_intercept : bool, default=True
        Whether to include an intercept in the mu formula.
    **gamlss_kwargs
        Additional keyword arguments passed to gamlss().

    Attributes
    ----------
    model_ : GAMLSSModel
        Fitted model after calling fit().
    feature_names_in_ : list of str
        Feature names seen during fit.
    n_features_in_ : int
        Number of features seen during fit.
    family_ : FamilyDefinition
        Resolved distribution family.

    Examples
    --------
    Basic usage:

    >>> from omnilss.sklearn_compat import GAMLSSRegressor
    >>> import numpy as np
    >>> X = np.random.randn(100, 2)
    >>> y = X[:, 0] + np.random.randn(100)
    >>> reg = GAMLSSRegressor(family="NO")
    >>> reg.fit(X, y)
    >>> reg.predict(X[:5])

    With sigma modeled as a function of features:

    >>> reg = GAMLSSRegressor(
    ...     family="NO",
    ...     sigma_formula="~ x0 + x1",
    ... )
    >>> reg.fit(X, y)

    In a Pipeline:

    >>> from sklearn.pipeline import Pipeline
    >>> from sklearn.preprocessing import StandardScaler
    >>> pipe = Pipeline([
    ...     ("scaler", StandardScaler()),
    ...     ("gamlss", GAMLSSRegressor(family="GA")),
    ... ])
    >>> pipe.fit(X, y)

    With cross-validation:

    >>> from sklearn.model_selection import cross_val_score
    >>> scores = cross_val_score(
    ...     GAMLSSRegressor(family="NO"),
    ...     X, y, cv=5, scoring="neg_mean_squared_error"
    ... )
    """

    def __init__(
        self,
        family: str | Any = "NO",
        sigma_formula: str | None = None,
        nu_formula: str | None = None,
        tau_formula: str | None = None,
        algorithm: str = "RS",
        feature_names: list[str] | None = None,
        response_name: str = "y",
        mu_smoother: str | None = None,
        fit_intercept: bool = True,
        **gamlss_kwargs: Any,
    ) -> None:
        _require_sklearn()
        self.family = family
        self.sigma_formula = sigma_formula
        self.nu_formula = nu_formula
        self.tau_formula = tau_formula
        self.algorithm = algorithm
        self.feature_names = feature_names
        self.response_name = response_name
        self.mu_smoother = mu_smoother
        self.fit_intercept = fit_intercept
        self.gamlss_kwargs = gamlss_kwargs

    def _get_feature_names(self, X: np.ndarray) -> list[str]:
        """Determine feature names from input."""
        n_features = X.shape[1]
        if self.feature_names is not None:
            if len(self.feature_names) != n_features:
                raise ValueError(
                    f"feature_names has {len(self.feature_names)} elements "
                    f"but X has {n_features} columns."
                )
            return list(self.feature_names)
        # Check if X is a DataFrame
        try:
            return list(X.columns)
        except AttributeError:
            return [f"x{i}" for i in range(n_features)]

    def _build_mu_formula(self, feature_names: list[str]) -> str:
        """Build the mu formula string."""
        response = self.response_name
        if not feature_names:
            return f"{response} ~ 1"

        if self.mu_smoother:
            terms = " + ".join(
                f"{self.mu_smoother}({name})" for name in feature_names
            )
        else:
            terms = " + ".join(feature_names)

        if not self.fit_intercept:
            terms = f"-1 + {terms}"

        return f"{response} ~ {terms}"

    def _build_data_dict(
        self, X: np.ndarray, y: np.ndarray | None, feature_names: list[str]
    ) -> dict[str, np.ndarray]:
        """Build the data dictionary for gamlss()."""
        data: dict[str, np.ndarray] = {}
        for i, name in enumerate(feature_names):
            data[name] = np.asarray(X[:, i], dtype=np.float64)
        if y is not None:
            data[self.response_name] = np.asarray(y, dtype=np.float64)
        return data

    def fit(self, X: Any, y: Any) -> "GAMLSSRegressor":
        """Fit the GAMLSS model.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target values.

        Returns
        -------
        self : GAMLSSRegressor
            Fitted estimator.
        """
        _require_sklearn()
        from .distributions import resolve_family
        from .fitting import gamlss

        X, y = check_X_y(X, y, dtype=np.float64, ensure_2d=True)

        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = self._get_feature_names(X)
        self.family_ = resolve_family(self.family)

        mu_formula = self._build_mu_formula(self.feature_names_in_)
        data = self._build_data_dict(X, y, self.feature_names_in_)

        # Build extra formula kwargs
        formula_kwargs: dict[str, Any] = {}
        if self.sigma_formula is not None:
            formula_kwargs["sigma_formula"] = self.sigma_formula
        if self.nu_formula is not None:
            formula_kwargs["nu_formula"] = self.nu_formula
        if self.tau_formula is not None:
            formula_kwargs["tau_formula"] = self.tau_formula

        self.model_ = gamlss(
            formula=mu_formula,
            family=self.family_,
            data=data,
            algorithm=self.algorithm,
            **formula_kwargs,
            **self.gamlss_kwargs,
        )

        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict the mean (mu parameter) for new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted mean values.
        """
        _require_sklearn()
        check_is_fitted(self, "model_")
        X = check_array(X, dtype=np.float64)

        newdata = self._build_data_dict(X, None, self.feature_names_in_)
        params = self.model_.predict_params(newdata)
        return np.asarray(params["mu"])

    def predict_distribution(self, X: Any) -> dict[str, np.ndarray]:
        """Predict all distribution parameters for new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.

        Returns
        -------
        params : dict of {str: ndarray}
            Predicted distribution parameters, e.g.,
            {"mu": array, "sigma": array}.
        """
        _require_sklearn()
        check_is_fitted(self, "model_")
        X = check_array(X, dtype=np.float64)

        newdata = self._build_data_dict(X, None, self.feature_names_in_)
        params = self.model_.predict_params(newdata)
        return {k: np.asarray(v) for k, v in params.items()}

    def predict_quantiles(
        self,
        X: Any,
        quantiles: list[float] = (0.05, 0.25, 0.5, 0.75, 0.95),
    ) -> dict[float, np.ndarray]:
        """Predict conditional quantiles for new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input features.
        quantiles : list of float
            Quantile levels in (0, 1).

        Returns
        -------
        results : dict of {float: ndarray}
            {quantile_level: predicted_values}.
        """
        _require_sklearn()
        check_is_fitted(self, "model_")
        X = check_array(X, dtype=np.float64)

        newdata = self._build_data_dict(X, None, self.feature_names_in_)
        results = self.model_.predict_quantiles(newdata, list(quantiles))
        return {k: np.asarray(v) for k, v in results.items()}

    def score(self, X: Any, y: Any) -> float:
        """Return the mean log-likelihood on the given data.

        Higher is better (less negative = better fit).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)

        Returns
        -------
        score : float
            Mean log-likelihood.
        """
        _require_sklearn()
        check_is_fitted(self, "model_")
        import jax.numpy as jnp

        X = check_array(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        newdata = self._build_data_dict(X, None, self.feature_names_in_)
        params = self.model_.predict_params(newdata)

        # Compute log-likelihood
        jax_params = {k: jnp.array(v) for k, v in params.items()}
        log_lik = self.family_.d(jnp.array(y), log=True, **jax_params)
        return float(jnp.mean(log_lik))

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get parameters for this estimator (sklearn API)."""
        params = {
            "family": self.family,
            "sigma_formula": self.sigma_formula,
            "nu_formula": self.nu_formula,
            "tau_formula": self.tau_formula,
            "algorithm": self.algorithm,
            "feature_names": self.feature_names,
            "response_name": self.response_name,
            "mu_smoother": self.mu_smoother,
            "fit_intercept": self.fit_intercept,
        }
        params.update(self.gamlss_kwargs)
        return params

    def set_params(self, **params: Any) -> "GAMLSSRegressor":
        """Set parameters for this estimator (sklearn API)."""
        known = {
            "family", "sigma_formula", "nu_formula", "tau_formula",
            "algorithm", "feature_names", "response_name",
            "mu_smoother", "fit_intercept",
        }
        for key, value in params.items():
            if key in known:
                setattr(self, key, value)
            else:
                self.gamlss_kwargs[key] = value
        return self

    def __repr__(self) -> str:
        family_name = (
            self.family if isinstance(self.family, str)
            else getattr(self.family, "name", str(self.family))
        )
        return (
            f"GAMLSSRegressor(family={family_name!r}, "
            f"algorithm={self.algorithm!r})"
        )
