# CG Full Validation Report (Week 2 Progress)

> Date: 2026-05-19  
> Chinese version: [CG_FULL_VERIFICATION_2026_05_19_cn.md](./CG_FULL_VERIFICATION_2026_05_19_cn.md)

## Scope

This report documents current Week 2 progress on full CG validation:

- joint scoring matrix assembly (`build_joint_scoring_matrix`)
- joint linear solve (`solve_joint_system`)
- line-searched outer step (`cg_outer_step`)
- iterative convergence scaffold (`run_cg_outer_loop`)

## Completed Validation

1. Numeric block solve agreement against NumPy reference.
2. Line-search monotonicity on convex deviance toy systems.
3. Iterative convergence behavior via relative global deviance criterion (`c_crit`).
4. Week 2 validation harness test over acceptance family labels: NO/GA/WEI/NBI.

## Current Limitations

- Direct R `gamlss` final deviance alignment (`< 0.01`) for NO/GA/WEI/NBI is not yet wired through this new module path.
- Existing R consistency coverage remains available in repository test suites, but dedicated Week 2 full-CG vs R bridge execution is pending integration.

## Commands Executed

```bash
python -m pytest omnilss/tests/test_cg_algorithm_full.py omnilss/tests/test_cg_algorithm_full_validation.py -q
```
