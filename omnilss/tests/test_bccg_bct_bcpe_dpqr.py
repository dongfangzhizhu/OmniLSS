"""Regression tests for BCCG/BCT/BCPE dpqr truncation normalization.

Validates that the Box-Cox family densities/CDFs/quantiles match R's
gamlss.dist implementations, which include truncation normalization:
- BCCG:  loglik - log(pnorm(1/(sigma*abs(nu))));  CDF = (Phi(z)-Phi(-1/(sigma|nu|)))/Phi(1/(sigma|nu|))
- BCT:   loglik - log(pt(1/(sigma*abs(nu)), df=tau))
- BCPE:  variance-standardized PE kernel + log(F.T(1/(sigma*abs(nu)))) truncation

Reference values are generated from R 4.5.0 gamlss.dist at machine precision
(x = [0.3, 1, 2, 4, 8]; p = [0.05, 0.5, 0.95]).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Machine-precision assertions require float64 (project precision policy).
import jax

jax.config.update("jax_enable_x64", True)

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from omnilss.dpqr_functions import (
    dBCCG, pBCCG, qBCCG,
    dBCT, pBCT, qBCT,
    dBCPE, pBCPE, qBCPE,
)

# R gamlss.dist reference values (digits=17)
BCCG_CASES = [
    (1.0, 0.1, 1.0),
    (1.0, 0.1, -1.0),
    (2.5, 0.2, 0.5),
]
BCCG_R = [
    {
        "d": [9.1347204083646573e-11, 3.9894228040143260, 7.6945986267063913e-22, 1.4736461348785630e-195, 0.0],
        "p": [1.2798125438782233e-12, 0.5, 1.0, 1.0, 1.0],
        "q": [0.83551463730485287, 1.0, 1.1644853626951472],
    },
    {
        "d": [2.6427652879718292e-117, 3.9894228040143260, 3.7167987868357430e-06, 1.5214503331431305e-13, 1.4770474815473197e-18],
        "p": [1.0174902686766813e-120, 0.5, 0.9999997133484281, 0.9999999999999681, 1.0],
        "q": [0.85874844977488296, 1.0, 1.1968671227900123],
    },
    {
        "d": [1.2197213167761104e-09, 1.4705882571030744e-03, 0.5109413465872300, 1.8879345965315640e-02, 1.3692494518490250e-14],
        "p": [3.1614417861708940e-11, 1.1871774404592030e-04, 0.1455462299257166, 0.9959648046995820, 0.9999999999999985],
        "q": [1.7452117728766494, 2.5, 3.3900653998281216],
    },
]

BCT_CASES = [
    (1.0, 0.1, 1.0, 5.0),
    (2.5, 0.2, 0.5, 3.0),
]
BCT_R = [
    {
        "d": [3.0136978850541290e-03, 3.7963913901773045, 4.0993320269704157e-04, 6.4022887174621902e-07, 4.0212767014279476e-09],
        "p": [0.00037293184522514976, 0.49995725945286778602, 0.99991451890573557204, 0.99999961403457904296, 0.99999999436528919095],
        "q": [0.79862233397195959, 1.00001125820360648, 1.20151153654155940],
    },
    {
        "d": [0.00914723774393472015, 0.03842275401924788447, 0.43738335324409216565, 0.05217360291096725944, 0.00087015445220396735],
        "p": [0.0025784036026087983, 0.0163870368233874943, 0.1834283643766213001, 0.9614327402067149864, 0.9978750820129447252],
        "q": [1.4702039757458427, 2.5007238943928995, 3.8158645278474115],
    },
]

BCPE_CASES = [
    (1.0, 0.1, 1.0, 2.0),
    (1.0, 0.1, -1.0, 1.5),
]
BCPE_R = [
    {
        "d": [9.1347204083646573e-11, 3.9894228040143260, 7.6945986267063913e-22, 1.4736461348785630e-195, 0.0],
        "p": [1.2798095916366492e-12, 0.5, 1.0, 1.0, 1.0],
        "q": [0.83551463730485254, 1.0, 1.16448536269514746],
    },
    {
        "d": [5.3517524156763881e-38, 4.7596665240782725, 1.6122157592115871e-04, 2.3297818795659866e-08, 8.2534415339868539e-11],
        "p": [0.0, 0.50000000000071299, 0.99997665923907375, 0.99999998882598740, 0.99999999985426258],
        "q": [0.85816732953941588, 0.99999999999985012, 1.19799777752105352],
    },
]

XS = np.array([0.3, 1.0, 2.0, 4.0, 8.0])
PS = np.array([0.05, 0.5, 0.95])


def _check(fn, r_vals, args, rtol=1e-7, atol=1e-10):
    got = np.asarray(fn(XS if len(r_vals) == 5 else PS, *args))
    ref = np.array(r_vals)
    np.testing.assert_allclose(got, ref, rtol=rtol, atol=atol)


def test_bccg_matches_r() -> None:
    for case, args in zip(BCCG_R, BCCG_CASES):
        _check(dBCCG, case["d"], args)
        _check(pBCCG, case["p"], args)
        _check(qBCCG, case["q"], args)


def test_bct_matches_r() -> None:
    for case, args in zip(BCT_R, BCT_CASES):
        _check(dBCT, case["d"], args)
        _check(pBCT, case["p"], args)
        _check(qBCT, case["q"], args)


def test_bcpe_matches_r() -> None:
    for case, args in zip(BCPE_R, BCPE_CASES):
        _check(dBCPE, case["d"], args)
        _check(pBCPE, case["p"], args)
        _check(qBCPE, case["q"], args)


def test_bccg_pq_roundtrip() -> None:
    for args in BCCG_CASES:
        q = np.asarray(qBCCG(PS, *args))
        p_back = np.asarray(pBCCG(q, *args))
        np.testing.assert_allclose(p_back, PS, rtol=1e-6, atol=1e-8)


def test_bcpe_tau2_reduces_to_bccg_style_normal() -> None:
    """BCPE with tau=2 uses the variance-standardized normal kernel, which
    equals the truncated normal BCCG density up to standardization."""
    d_bcpe = np.asarray(dBCPE(XS, 1.0, 0.1, 1.0, 2.0))
    d_bccg = np.asarray(dBCCG(XS, 1.0, 0.1, 1.0))
    # c = exp(0.5*(-log2 + lgamma(1/2) - lgamma(3/2))) = 1 for tau=2
    np.testing.assert_allclose(d_bcpe, d_bccg, rtol=1e-7, atol=1e-12)
