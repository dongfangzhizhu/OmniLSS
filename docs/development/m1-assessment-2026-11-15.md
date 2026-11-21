# M1 assessment (target 0.5.0)

- promoted families: 22
- symbols in promoted families: 110 (golden 84, r_consistency 26)
- d/p/q/r: 84/88 golden directly, 4 suppressed by r_consistency precedence, 0 other
- golden:validated total: 84 (baseline 34, 2.4706x)
- kernel layer: {'implemented': 110}
- placeholder in scope: 0

| criterion | met | detail |
|---|---|---|
| highfreq_surface_parity | yes | no promoted family has a missing-surface symbol |
| highfreq_dpqr_parity_direct | no | 84/88 dpqr symbols carry golden parity directly; 4 are labelled r_consistency_test because that rule takes precedence over golden in generate_symbol_coverage.py:381 |
| highfreq_dpqr_parity_by_evidence | yes | every promoted dpqr symbol has either a golden fixture or a live-R r_consistency test; none is bare capability/tested/unverified |
| golden_validated_order_of_magnitude | no | 34 -> 84 (2.47x); the literal 10x target would need ~340 symbols, which the 40-family validated pool cannot supply (18 excluded for documented R-side/flat-ridge reasons) |
| kernel_implemented | yes | 22/22 promoted families have an implemented kernel. Kernel promotion was M3 admission work rather than M1 parity; D2.241 closed it by admitting the evidence-eligible batch, so this criterion tracks the registry rather than being an M1 target. |
| no_placeholder_in_scope | yes | in-scope placeholder symbols: 0 |

## Gate definition (revised 2026-11-15, D2.235)

- criteria: highfreq_surface_parity, highfreq_dpqr_parity_by_evidence, no_placeholder_in_scope
- revised from: high-frequency families all Surface+Kernel+DPQR parity; golden:validated raised by an order of magnitude; no placeholder in the target scope
- moved out: **golden_validated_order_of_magnitude** -- replaced by d/p/q/r evidence coverage: the 40-family validated pool caps golden:validated at ~110, so the 10x (340) target is unreachable by construction (18 families excluded with documented R-side/flat-ridge reasons)
- moved out: **kernel_implemented** -- M3 admission work: at the time of the M1 gate 21/22 promoted kernels were experimental because admission was blocked (production_safe: 1 family / 5 symbols). D2.241 admitted the evidence-eligible batch, so this is now closed; it was never a parity gap

**m1_gate_met: True**  
**core_m1_parity_met: True**  
**literal_all_criteria_met (original wording): False**

Under the D2.235 gate definition M1 is met: every promoted high-frequency family has surface parity, d/p/q/r evidence parity (golden, or a live-R r_consistency test), and no placeholder. The two literal original clauses are recorded as moved_out, not silently dropped. The r_consistency>golden precedence suppresses 24 golden labels; that is a reporting-rule finding, not a parity gap, and the coverage rule was not changed.
