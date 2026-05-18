# Phase 1 Progress: Device-Aware RS Routing

中文版本: [phase1-device-aware-rs-routing-progress-2026-05-18_cn.md](phase1-device-aware-rs-routing-progress-2026-05-18_cn.md)

## Completed in this step

- Routed the default `method="RS"` entry point through the existing device-aware backend selector.
- Preserved CPU behavior and infinite-threshold accelerator behavior as NumPy RS, while allowing configured GPU/TPU thresholds to select `RS_JAX` for supported families.
- Kept `method="auto"` as a compatibility alias for the same routing behavior.
- Marked explicit `method="RS_JAX"` use with a `DeprecationWarning`; users should prefer `method="RS"` plus crossover configuration.
- Changed method-routing diagnostic output to English so project runtime messages follow the default project language.
- Added integration-test coverage proving that default `method="RS"` can route to JAX when an accelerator threshold is configured.

## Notes

This completes the Phase 1 routing task without forcing JAX on current default GPU/TPU thresholds. The project still requires benchmark evidence before replacing the placeholder `math.inf` accelerator crossover thresholds with finite defaults.
