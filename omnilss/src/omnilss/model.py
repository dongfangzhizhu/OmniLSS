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
    # Convergence reporting.  R's `gamlss()` object exposes `$converged`; the
    # Python container used to hide the same information inside
    # ``additional_slots["rs_converged"] / ["cg_converged"]``, so a diverged
    # fit was indistinguishable from a converged one at the model level.
    # ``None`` means "not reported by this construction site" (legacy
    # artifacts, ``gamlss_ml`` staging path, partial objects).
    converged: bool | None = None
    termination_reason: str | None = None
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

    def convergence_status(self) -> bool | None:
        """Return the reported convergence flag, or ``None`` when unknown.

        The first-class :attr:`converged` field wins.  Older construction
        sites (and third-party models that only populate
        ``additional_slots``) are handled by falling back to the per-route
        slot keys, so callers get one answer instead of silently defaulting.
        """

        if self.converged is not None:
            return bool(self.converged)
        slots = self.additional_slots if isinstance(self.additional_slots, Mapping) else {}
        for key in ("converged", "rs_converged", "cg_converged", "mixed_converged"):
            if key in slots:
                return bool(slots[key])
        return None

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

    def logLik(self):
        """R logLik.gamlss: log-likelihood (censored-aware via g_dev)."""

        from .logLik import log_likelihood

        return log_likelihood(self)

    def AIC(self, k: float = 2.0) -> float:
        """R AIC.gamlss: GAIC with penalty k (AIC at k=2)."""

        from .operations import gaic

        return float(gaic(self, k=k))

    def GAIC(self, k: float = 2.0) -> float:
        """R GAIC.gamlss."""

        from .operations import gaic

        return float(gaic(self, k=k))

    def IC(self, k: float = 2.0) -> float:
        """R IC.gamlss."""

        from .operations import ic

        return float(ic(self, k=k))

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
        if parameter in self.lambda_selection_info:
            return list(self.lambda_selection_info[parameter])

        # The fitting paths record the fitted smoothers in additional_slots;
        # derive the same view from them so the accessor works for models that
        # were fitted without an explicit lambda_selection_info mapping.
        slots = self.additional_slots or {}
        fitted = slots.get("smooth_fits") or {}
        entries = fitted.get(parameter)
        if entries is None:
            # The RS path records SmoothDesignInfo objects under smooth_infos.
            design = (slots.get("smooth_infos") or {}).get(parameter)
            entries = getattr(design, "smooth_fits", None) or []
        entries = entries or []
        # The edf recorded on the smoother is the seed value from design build;
        # the RS cycles update lambda in place, so recompute from the final
        # penalty/weights to report the edf of the fitted model.
        from .edf import smooth_term_edf

        final_edf = smooth_term_edf(self, parameter)
        return [
            {
                "variable": getattr(entry, "variable", None),
                "smoother": getattr(entry, "smoother", None),
                "lambda_": getattr(entry, "lambda_", None),
                "edf": final_edf.get(
                    str(getattr(entry, "variable", position)),
                    getattr(entry, "edf", None),
                ),
                "selection_method": getattr(entry, "selection_method", None),
                "criterion_value": getattr(entry, "criterion_value", None),
                "lambda_origin": getattr(entry, "lambda_origin", None),
                "basis_columns": getattr(entry, "basis_columns", None),
            }
            for position, entry in enumerate(entries)
        ]
    

    def get_method_routing(self, *, validate: bool = True) -> dict[str, Any] | None:
        """Return structured method-routing metadata from ``additional_slots``.

        Parameters
        ----------
        validate : bool, default=True
            When True, validate the payload schema using
            :func:`omnilss.config.validate_method_routing_decision`.

        Returns
        -------
        dict or None
            Routing payload dictionary when present, else ``None``.
        """
        payload = self.additional_slots.get("method_routing")
        if payload is None:
            return None
        payload_dict = dict(payload)
        if validate:
            from .config import validate_method_routing_decision

            validate_method_routing_decision(payload_dict)
        return payload_dict

    def has_method_routing(self) -> bool:
        """Return True when structured method-routing metadata is available."""
        return self.get_method_routing(validate=False) is not None


    def get_method_routing_reason(self, *, validate: bool = True) -> str | None:
        """Return the routing reason code when routing metadata is available."""
        routing = self.get_method_routing(validate=validate)
        if routing is None:
            return None
        return str(routing.get("reason"))

    def get_method_routing_reason_detail(self, *, validate: bool = True) -> str | None:
        """Return the user-facing routing reason detail string when available."""
        routing = self.get_method_routing(validate=validate)
        if routing is None:
            return None
        detail = routing.get("reason_detail")
        return None if detail is None else str(detail)


    def get_method_routing_summary(self, *, validate: bool = True) -> str | None:
        """Return a compact human-readable method-routing summary string."""
        routing = self.get_method_routing(validate=validate)
        if routing is None:
            return None
        req = routing.get("requested_method", "?")
        sel = routing.get("selected_method", "?")
        backend = routing.get("backend", "?")
        reason = routing.get("reason", "?")
        n_obs = routing.get("n_obs", None)
        threshold = routing.get("threshold", None)
        confidence = self.get_method_routing_confidence(validate=False)
        suffix = ""
        if n_obs is not None:
            suffix += f", n_obs={n_obs}"
        if threshold is not None:
            suffix += f", threshold={threshold}"
        if confidence is not None:
            suffix += f", confidence={confidence:.3f}"
        return f"requested={req} -> selected={sel} on {backend} ({reason}{suffix})"


    def explain_method_routing(self, *, validate: bool = True) -> str:
        """Return a user-facing explanation for the model routing decision."""
        detail = self.get_method_routing_reason_detail(validate=validate)
        if detail is not None:
            return detail
        summary = self.get_method_routing_summary(validate=validate)
        if summary is not None:
            return summary
        return "No method-routing metadata is available on this model."


    def get_method_routing_confidence(self, *, validate: bool = True) -> float | None:
        """Return routing confidence when present in routing metadata."""
        routing = self.get_method_routing(validate=validate)
        if routing is None:
            return None
        value = routing.get("confidence")
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

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
        newdata: dict[str, Any] | None = None,
        type: str = "response",
        what: str = "mu",
        se_fit: bool = False,
        terms: tuple[str, ...] | None = None,
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
        if type not in {"response", "link", "terms"}:
            raise ValueError(f"Unknown type: {type}")
        # Keep the historical mu-response/link surface for the common call, and
        # route what=/se_fit=/terms=/no-newdata to the richer R-aligned predict
        # (which also serves censored models via stored parameter covariance).
        if (
            what == "mu"
            and not se_fit
            and terms is None
            and newdata is not None
            and type in {"response", "link"}
        ):
            from .prediction import predict_response

            return predict_response(self, newdata, type)
        from .predict_gamlss_23_12_21 import predict as _rich_predict

        return _rich_predict(
            self, what=what, type=type, terms=terms, se_fit=se_fit, newdata=newdata
        )
