"""R-aligned maximum likelihood fitting interfaces."""

from __future__ import annotations

import warnings

from .fitting import gamlss_ml


def gamlssML(*args, **kwargs):
    """Deprecated alias for :func:`gamlss_ml`."""
    warnings.warn(
        "gamlssML is deprecated; use gamlss_ml instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return gamlss_ml(*args, **kwargs)


__all__ = ["gamlssML", "gamlss_ml"]
