# DELAPORT compatibility correction

`DELAPORT` is the Python compatibility spelling of GAMLSS `DEL` (Delaporte).
The R source file `DELAPORT.R` exports `DEL`, `dDEL`, `pDEL`, `qDEL`, and
`rDEL`, not functions named `dDELAPORT` etc.

Both registry resolution and direct import from `omnilss.distributions_b13`
now retain the name `DELAPORT` and share DEL's d/p/q/r kernels. The former
single-term Poisson/Gamma approximation was not a normalized discrete PMF
and is no longer used.

## Parameter compatibility

- `mu > 0`, `sigma > 0`, `0 < nu < 1`.
- The default nu link changes from log to **logit**.
- Old results from the placeholder are not numerically compatible; refit
  models with valid DEL parameters. There is no supported conversion from
  the old placeholder parameterization.
- Mean is `mu`; variance is `mu + mu**2 * sigma * (1 - nu)**2`, except
  the inherited R-style Poisson approximation when `sigma <= 1e-4`.

## Numerical scope and verification

The alias inherits DEL's d/p integer evaluation limit of 10000 and q's
default search limit of 10000. Unresolved finite queries return NaN;
random sampling uses the untruncated Poisson-Gamma mixture.
Upper-tail probabilities are computed by subtraction and can lose precision
in extreme tails. This change does not promote any capability status.

Tests cover direct and registry entry points and compare central and tail
d/p/q profiles with R DEL. Sampling is checked against the shared DEL kernel.