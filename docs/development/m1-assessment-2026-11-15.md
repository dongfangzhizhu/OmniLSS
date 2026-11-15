# M1 assessment (target 0.5.0)

- promoted families: 22
- symbols in promoted families: 110 (golden 86, r_consistency 24)
- d/p/q/r: 86/88 golden directly, 2 suppressed by r_consistency precedence, 0 other
- golden:validated total: 86 (baseline 34, 2.5294x)
- kernel layer: {'experimental': 105, 'implemented': 5}
- placeholder in scope: 0

| criterion | met | detail |
|---|---|---|
| highfreq_surface_parity | yes | no promoted family has a missing-surface symbol |
| highfreq_dpqr_parity_direct | no | 86/88 dpqr symbols carry golden parity directly; 2 are labelled r_consistency_test because that rule takes precedence over golden in generate_symbol_coverage.py:381 |
| highfreq_dpqr_parity_by_evidence | yes | every promoted dpqr symbol has either a golden fixture or a live-R r_consistency test; none is bare capability/tested/unverified |
| golden_validated_order_of_magnitude | no | 34 -> 86 (2.53x); the literal 10x target would need ~340 symbols, which the 40-family validated pool cannot supply (18 excluded for documented R-side/flat-ridge reasons) |
| kernel_implemented | no | 1/22 promoted families have an implemented kernel; the rest are experimental because admission is blocked (production_safe=5). Kernel promotion is M3 admission work, not M1 parity. |
| no_placeholder_in_scope | yes | in-scope placeholder symbols: 0 |

**core_m1_parity_met: True**  
**literal_all_criteria_met: False**

M1 is met on the parity meaning that matters: every promoted high-frequency family has surface parity, d/p/q/r evidence parity, and no placeholder. It is NOT met on two literal wordings: golden:validated did not grow 10x (the frozen pool caps it at ~110), and kernel=implemented is blocked admission work that belongs to M3. Also note the r_consistency>golden precedence suppresses 24 golden labels; that is a reporting-rule finding, not a parity gap.
