# Month 1 Formula Safety Progress (2026-05-18)

> Chinese version: [month1-formula-safety-progress-2026-05-18_cn.md](month1-formula-safety-progress-2026-05-18_cn.md)
>
> Parent plan: [six-month-execution-plan-2026-05-17.md](six-month-execution-plan-2026-05-17.md)

## Scope

This note records progress against Month 1 / Workstream D2: **Formula Safety and Parser Hardening**.

## Implemented Progress

- Numeric formula expressions are evaluated through a strict AST whitelist.
- Attribute access is rejected, including `np.<function>` calls; use direct allowlisted calls such as `sqrt(x)` instead of `np.sqrt(x)`.
- Subscripts, lambdas, comprehensions, boolean/comparison expressions, collection literals, and overly deep expressions are rejected before evaluation.
- Malicious or unsupported expressions raise deterministic `FormulaEvaluationError` failures without executing arbitrary Python code.
- `FormulaEvaluationError` carries structured `term` and `reason` fields for diagnostics and service error mapping.
- Regression tests now cover direct allowlisted functions, explicit rejection paths for attribute calls, subscripts, lambdas, comprehensions, and non-allowlisted imports, plus structured error attributes.

## Remaining D2 Work

- Replace remaining ad hoc parser splitting in prediction and smooth/tensor argument handling with one shared constrained parser utility.
- Extend structured formula errors from expression-level `term`/`reason` to full public API errors that also include the distribution parameter name.
- Expand safety coverage to formula paths used by all public fitting aliases and service inputs.
