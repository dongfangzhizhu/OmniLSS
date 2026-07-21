# Release 1 Capability Evidence Matrix

This report is derived exclusively from `r-export-coverage.json`.

## Scope

Included fixed R sources: gamlss, gamlss.dist.

## Metrics

| Metric | Value |
| --- | ---: |
| exported_symbol_coverage | 141 |
| callable_python_entry_coverage | 141 |
| dedicated_kernel_coverage | 0 |
| e3_family_count | 0 |
| e4_feature_count | 0 |

## Excluded packages

| Package | Reason |
| --- | --- |
| gamlss.add | Release 1 excludes add-on terms from statistical migration metrics. |
| gamlss.cens | Release 1 excludes censored-response extensions. |
| gamlss.tr | Release 1 excludes truncated-distribution extensions. |
| gamlss2 | Release 1 excludes the next-generation interface package. |
| gamlss.mx | Release 1 excludes mixture-model extensions. |
| gamlss.spatial | Release 1 excludes spatial-model extensions. |
| gamlss.foreach | Release 1 excludes parallel foreach integration. |
| gamlss.prepdata | Release 1 excludes data-preparation helpers. |
| gamlss.ggplots | Release 1 excludes plotting integrations. |
| gamlss.data | Release 1 excludes bundled data from migration completion metrics. |
| WeatherGermany | Release 1 excludes the external example data package. |

## Fixed R source evidence

| Package | Version | Namespace SHA-256 |
| --- | --- | --- |
| gamlss | 5.5-3 | `709cd6a70b13e91d14bce45b2d59f9d8efe66aa50cca2c943e3711221116dfb8` |
| gamlss.dist | 6.1-8 | `a1b0cd62d169a22ba20c01e1a6283ceff49c90b3809205016261f182d57454e2` |
