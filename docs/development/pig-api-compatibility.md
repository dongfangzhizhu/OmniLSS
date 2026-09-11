# PIG / PIG2 numerical API repair

- PIG uses mean `mu > 0`, dispersion `sigma > 0`, and variance
  `mu + sigma * mu**2`. PIG2 maps its positive sigma to PIG dispersion
  `(hypot(mu, sigma) + mu) / sigma**2`, avoiding subtractive cancellation.
- Direct functions and family/registry APIs share JAX kernels; no parameter
  averaging, host NumPy conversion, or clipping counts to 512 remains in PIG.
  The log-PMF supports parameter gradients and Hessians.
- d/p numerically evaluate counts 0..10000; q searches up to `max_value`
  (default 10000). Unsupported finite counts or unresolved quantiles return
  NaN, not a fabricated endpoint. CDF floors fractional inputs.
- Noninteger PMF inputs have zero mass. R's legacy PIG C implementation can
  produce nonzero values for such inputs; parity tests use integer support.
- Invalid parameters/probabilities return NaN. Quantile endpoints are 0/Inf
  with tail and log transformations applied once. This intentionally does not
  reproduce R PIG2's double transformation or tail boundary overwrite bugs.
- Sampling is a direct Poisson/inverse-Gaussian mixture, not truncated inverse
  CDF sampling. Same JAX key is reproducible; no R sample-by-sample parity is
  claimed. `n` is a nonnegative integer, parameters broadcast to `(n,)`.
- Large extreme parameters may overflow; very small upper tails remain limited
  by floating-point CDF summation accuracy. Fixed 10000-step numerical loops
  favor explicit bounds and differentiability over fitting throughput.
- Regression evidence: independent SciPy mixture integration, JAX derivatives,
  broadcasting, support/tail boundaries, sampling moments, and live R integer
  d/p/q comparisons. No global capability/fitting validation status is upgraded.

Previous PIG2 likelihoods used a different, incorrect kernel; models fitted
with that implementation should be refitted.