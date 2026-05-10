"""Deep GAMLSS module

使用神经网络建模分布参数，这是 OmniLSS 超越 R gamlss 的创新功能。

Deep GAMLSS 的核心思想是使用神经网络自动学习复杂的非线性关系，
而不需要手动指定平滑项或交互项。

Examples
--------
>>> from omnilss.deep import DeepGAMLSS, fit_deep_gamlss
>>> from omnilss import NO
>>> 
>>> # 拟合 Deep GAMLSS
>>> model, params = fit_deep_gamlss(X, y, family=NO())
>>> 
>>> # 预测
>>> pred_params = model.apply(params, X_new)
"""

from .deep_gamlss import (
    ParameterNetwork,
    DeepGAMLSS,
    fit_deep_gamlss,
    predict_deep_gamlss
)

__all__ = [
    "ParameterNetwork",
    "DeepGAMLSS",
    "fit_deep_gamlss",
    "predict_deep_gamlss"
]
