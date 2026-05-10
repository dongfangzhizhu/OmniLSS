# RS Algorithm Structure Comparison

## R's glim.fit Structure

```r
glim.fit <- function(f, X, y, w, fv, os, step = 1, ...) {
  # INITIALIZATION (before loop)
  eta <- f$linkfun(fv)           # line 188
  dr <- 1/f$dr(eta)              # line 189-190
  dldp <- f$dldp(fv)             # line 191 - derivatives at INITIAL fv
  d2ldp2 <- f$d2ldp2(fv)         # line 192
  wt <- -(d2ldp2/(dr*dr))        # line 195
  wv <- (eta-os)+dldp/(dr*wt)    # line 199
  
  # ITERATION LOOP
  while (abs(olddv-dv) > cc && itn < cyc) {
    itn <- itn+1
    
    # 1. Fit with CURRENT wv and wt
    fit <- lm.wfit(X, wv, wt*w)
    lp <- fit$fitted.values
    
    # 2. Update eta and fv
    eta <- lp + os
    fv <- f$linkinv(eta)
    
    # 3. Compute deviance
    di <- f$G.di(fv)
    dv <- sum(w*di)
    
    # 4. Recompute derivatives at NEW fv for NEXT iteration
    dr <- 1/f$dr(eta)              # line 262-263
    dldp <- f$dldp(fv)             # line 265
    d2ldp2 <- f$d2ldp2(fv)         # line 266-267
    wt <- -(d2ldp2/(dr*dr))        # line 268
    wv <- (eta-os)+dldp/(dr*wt)    # line 271
  }
}
```

**Key: Derivatives computed at iteration i are used in iteration i+1**

## Python's rs_step Structure

```python
def rs_step(...):
    # INITIALIZATION
    eta = link_fun(fitted_values)
    
    # ITERATION LOOP
    for iteration in range(max_iter):
        # 1. Compute derivatives at CURRENT fitted_values
        first_deriv = dldp(**param_dict)
        second_deriv = d2ldp2(**param_dict)
        link_deriv = 1.0 / (dmu_deta + 1e-15)
        
        # 2. Compute working weights and response
        working_weights = -(second_deriv / (link_deriv ** 2))
        working_response = (eta - offset) + first_deriv / (link_deriv * working_weights)
        
        # 3. Fit with CURRENT wv and wt
        coef = lstsq(WX, Wy)
        eta_new = X @ coef + offset
        
        # 4. Update eta
        eta = step_size * eta_new + (1 - step_size) * eta_old
        
        # 5. Update fitted values
        fitted_values = link_inv(eta)
        
        # 6. Compute deviance
        deviance = sum(weights * dev_incr)
```

**Key: Derivatives computed at iteration i are used in iteration i (same iteration)**

## The Critical Difference

### R's Approach (Correct)
- Iteration 0: Use derivatives from initial values
- Iteration 1: Fit with derivatives from iteration 0, then compute new derivatives
- Iteration 2: Fit with derivatives from iteration 1, then compute new derivatives
- ...

This is the **standard Fisher Scoring / IRLS algorithm**:
- Use the information matrix (Hessian) from the PREVIOUS iteration
- This ensures stability and proper convergence

### Python's Approach (Bug)
- Iteration 0: Compute derivatives, fit immediately
- Iteration 1: Compute derivatives, fit immediately
- ...

This is **mixing the current and next iteration**, which can cause:
- Instability in the iteration
- Convergence to wrong local optima
- The algorithm is no longer pure Fisher Scoring

## The Fix

Python should follow R's structure:

```python
def rs_step(...):
    # INITIALIZATION (before loop)
    eta = link_fun(fitted_values)
    
    # Compute INITIAL derivatives
    param_dict = {"y": y, parameter: fitted_values, **other_parameters}
    first_deriv = dldp(**param_dict)
    second_deriv = d2ldp2(**param_dict)
    link_deriv_func = family.link_derivatives[parameter]
    dmu_deta = link_deriv_func(eta)
    link_deriv = 1.0 / (dmu_deta + 1e-15)
    
    # Compute INITIAL working weights and response
    working_weights = -(second_deriv / (link_deriv ** 2))
    working_response = (eta - offset) + first_deriv / (link_deriv * working_weights)
    
    # ITERATION LOOP
    for iteration in range(max_iter):
        # 1. Fit with CURRENT wv and wt (from previous iteration)
        W = working_weights * weights
        sqrt_W = np.sqrt(W)
        WX = X * sqrt_W[:, None]
        Wy = working_response * sqrt_W
        coef = lstsq(WX, Wy)
        
        # 2. Update eta
        eta_new = X @ coef + offset
        eta = step_size * eta_new + (1 - step_size) * eta_old
        
        # 3. Update fitted values
        fitted_values = link_inv(eta)
        
        # 4. Compute deviance
        deviance = sum(weights * dev_incr)
        
        # 5. Check convergence
        if abs(old_deviance - deviance) < tol:
            break
        
        # 6. Recompute derivatives at NEW fitted_values for NEXT iteration
        param_dict = {"y": y, parameter: fitted_values, **other_parameters}
        first_deriv = dldp(**param_dict)
        second_deriv = d2ldp2(**param_dict)
        dmu_deta = link_deriv_func(eta)
        link_deriv = 1.0 / (dmu_deta + 1e-15)
        
        # 7. Recompute working weights and response for NEXT iteration
        working_weights = -(second_deriv / (link_deriv ** 2))
        working_response = (eta - offset) + first_deriv / (link_deriv * working_weights)
```

## Why This Matters

The Fisher Scoring algorithm uses:
```
β^(t+1) = β^(t) + [I(β^(t))]^(-1) * U(β^(t))
```

Where:
- `I(β^(t))` is the information matrix at iteration t
- `U(β^(t))` is the score vector at iteration t

The key is that **both** the information matrix and score are evaluated at the **same** point β^(t), and then used to compute β^(t+1).

Python's current implementation is evaluating them at different points, which breaks the algorithm's convergence properties.
