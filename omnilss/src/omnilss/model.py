"""Core model container used during staged migration from R `gamlss`.

R source references:
- file: `gamlss/R/extra.R`
- functions: `is.gamlss`, `fitted.gamlss`, `coef.gamlss`, `coefAll`,
  `deviance.gamlss`, `lp`, `fv`, `IC`, `GAIC.gamlss`, `GAIC_table`,
  `GAIC_scaled`, `.hat.WX`, `numeric.deriv`
- file: `gamlss/R/DevianceIncr.R`
- function: `devianceIncr`
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any


@dataclass
class GAMLSSModel:
    """Minimal Python-side model state mirroring key R `gamlss` slots."""

    par: tuple[str, ...]
    family: Any
    df_fit: float
    g_dev: float
    n: int
    y: Any
    fitted_values: Mapping[str, Any] = field(default_factory=dict)
    coefficients: Mapping[str, Any] = field(default_factory=dict)
    linear_predictors: Mapping[str, Any] = field(default_factory=dict)
    working_vectors: Mapping[str, Any] = field(default_factory=dict)
    iterative_weights: Mapping[str, Any] = field(default_factory=dict)
    offsets: Mapping[str, Any] = field(default_factory=dict)
    formulas: Mapping[str, Any] = field(default_factory=dict)
    terms: Mapping[str, Any] = field(default_factory=dict)
    design_matrices: Mapping[str, Any] = field(default_factory=dict)
    xlevels: Mapping[str, Any] = field(default_factory=dict)
    additional_slots: Mapping[str, Any] = field(default_factory=dict)
    call: Mapping[str, Any] | None = None
    control: Mapping[str, Any] = field(default_factory=dict)
    iter: int = 0
    weights: Any = None
    residuals: Any = None
    type: str = "Continuous"
    parameters: tuple[str, ...] = field(default_factory=tuple)
    rqres: Callable[..., Any] | None = None
    class_name: str = "gamlss"
    lambda_selection_info: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Attach a design-matrix schema when enough fit metadata is available."""

        if not self.formulas or not self.design_matrices:
            return
        if isinstance(self.additional_slots, Mapping) and self.additional_slots.get("design_matrix_schema"):
            return
        try:
            from .design_schema import build_design_matrix_schema

            self.additional_slots = {
                **dict(self.additional_slots or {}),
                "design_matrix_schema": build_design_matrix_schema(self),
            }
        except Exception:
            # Model construction should not fail for legacy or partial objects;
            # prediction/serialization will raise structured schema errors later.
            return

    def has_parameter(self, what: str) -> bool:
        """Mirror the R object check `what %in% object$par`."""

        return what in self.par

    def get_slot(self, slot_name: str) -> Any:
        """Retrieve a migrated slot or raise a close R-style error."""

        if slot_name in self.additional_slots:
            return self.additional_slots[slot_name]
        raise KeyError(f"no slot named {slot_name!r}")
    
    @property
    def deviance(self) -> float:
        """Return the global deviance (g_dev).
        
        This mirrors R's deviance.gamlss() method.
        
        Returns
        -------
        deviance : float
            Global deviance
        """
        return self.g_dev
    
    def get_smooth_info(self, parameter: str = "mu") -> list[dict[str, Any]]:
        """Get information about smooth terms for a parameter.
        
        Parameters
        ----------
        parameter : str, default="mu"
            Parameter name (mu, sigma, nu, tau)
        
        Returns
        -------
        smooth_info : list of dict
            List of dictionaries containing smooth term information:
            - variable: variable name
            - smoother: smoother type (pb, ps, cs)
            - lambda_: smoothing parameter
            - edf: effective degrees of freedom
            - selection_method: method used for lambda selection
            - criterion_value: value of the optimization criterion
        
        Examples
        --------
        >>> model = gamlss("y ~ s(x1) + s(x2)", family="NO", data=data)
        >>> smooth_info = model.get_smooth_info("mu")
        >>> for info in smooth_info:
        ...     print(f"{info['variable']}: lambda={info['lambda_']:.4f}, "
        ...           f"edf={info['edf']:.2f}, method={info['selection_method']}")
        """
        if parameter not in self.lambda_selection_info:
            return []
        
        return self.lambda_selection_info[parameter]
    
    def predict_params(
        self,
        newdata: dict[str, Any],
        which: list[str] | None = None
    ) -> dict[str, Any]:
        """Predict distribution parameters for new data.
        
        Parameters
        ----------
        newdata : dict
            New data as {variable_name: values}
        which : list of str, optional
            Parameters to predict, defaults to all estimable parameters
        
        Returns
        -------
        params : dict
            Predicted parameter values {param_name: values}
        
        Examples
        --------
        >>> model = gamlss("y ~ x1 + x2", family="NO", data=train_data)
        >>> newdata = {"x1": np.array([1, 2, 3]), "x2": np.array([4, 5, 6])}
        >>> params = model.predict_params(newdata)
        >>> print(params["mu"])
        >>> print(params["sigma"])
        """
        from .prediction import predict_params
        return predict_params(self, newdata, which)
    
    def predict_quantiles(
        self,
        newdata: dict[str, Any],
        quantiles: list[float] = [0.05, 0.25, 0.5, 0.75, 0.95]
    ) -> dict[float, Any]:
        """Predict conditional quantiles for new data.
        
        Parameters
        ----------
        newdata : dict
            New data as {variable_name: values}
        quantiles : list of float
            Quantiles to predict (0, 1)
        
        Returns
        -------
        results : dict
            {quantile: predicted_values}
        
        Examples
        --------
        >>> model = gamlss("y ~ x", family="NO", data=data)
        >>> newdata = {"x": np.array([1, 2, 3])}
        >>> quantiles = model.predict_quantiles(newdata, quantiles=[0.05, 0.5, 0.95])
        >>> print("5% quantile:", quantiles[0.05])
        >>> print("Median:", quantiles[0.5])
        >>> print("95% quantile:", quantiles[0.95])
        """
        from .prediction import predict_quantiles
        return predict_quantiles(self, newdata, quantiles)
    
    def centiles(
        self,
        xvar: str,
        xvalues: Any | None = None,
        cent: list[float] = [5, 25, 50, 75, 95],
        n_points: int = 100,
        **fixed_vars
    ) -> Any:
        """Generate centile curves.
        
        Parameters
        ----------
        xvar : str
            X-axis variable name
        xvalues : array-like, optional
            X-axis values, defaults to training data range
        cent : list of float
            Percentiles to compute (0-100)
        n_points : int
            Number of points to generate
        **fixed_vars
            Fixed values for other variables
        
        Returns
        -------
        df : pd.DataFrame
            Centile curves data with columns: xvar, C5, C25, C50, C75, C95, ...
        
        Examples
        --------
        >>> model = gamlss("height ~ s(age)", family="NO", data=growth_data)
        >>> curves = model.centiles(xvar="age", cent=[5, 50, 95])
        >>> 
        >>> import matplotlib.pyplot as plt
        >>> for c in [5, 50, 95]:
        >>>     plt.plot(curves["age"], curves[f"C{c}"], label=f"{c}%")
        >>> plt.legend()
        >>> plt.show()
        """
        from .prediction import centiles
        return centiles(self, xvar, xvalues, cent, n_points, **fixed_vars)
    
    def predict(
        self,
        newdata: dict[str, Any],
        type: str = "response"
    ) -> Any:
        """Predict response variable.
        
        Parameters
        ----------
        newdata : dict
            New data as {variable_name: values}
        type : str
            Prediction type: "response", "link", or "terms"
        
        Returns
        -------
        predictions : array
            Predicted values
        
        Examples
        --------
        >>> model = gamlss("y ~ x", family="NO", data=data)
        >>> newdata = {"x": np.array([1, 2, 3])}
        >>> y_pred = model.predict(newdata, type="response")
        """
        from .prediction import predict_response
        return predict_response(self, newdata, type)
