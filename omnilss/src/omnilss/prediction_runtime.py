"""Phase 1 unified prediction runtime wrappers."""

from __future__ import annotations

from .prediction import predictAll


def predict_mean(model, newdata):
    out = predictAll(model, newdata)
    return out.get("mu", out)


def predict_quantile(model, newdata, q=0.5):
    return model.family.q(q, **{p: predict_mean(model, newdata) for p in model.parameters if p == "mu"})


def predict_interval(model, newdata, alpha=0.1):
    lo = predict_quantile(model, newdata, alpha / 2.0)
    hi = predict_quantile(model, newdata, 1.0 - alpha / 2.0)
    return lo, hi


def predict_distribution(model, newdata):
    return {"family": model.family.name, "params": {"mu": predict_mean(model, newdata)}}
