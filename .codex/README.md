# Codex Cloud Setup

`setup.sh` is the post-checkout/post-create bootstrap for Codex Cloud and the
devcontainer. It is idempotent and should be safe to rerun after source changes.

Primary behavior:

- create or reuse `/opt/omnilss-venv`;
- install Python dependencies from `/opt/omnilss-wheelhouse` when
  `OMNILSS_OFFLINE=1`;
- install editable `omnilss` and optional `omnilss-pro`;
- verify the R bridge for `gamlss`, `gamlss.dist`, and `jsonlite`;
- regenerate gRPC stubs, falling back to committed stubs if generation is not
  possible;
- print the R-backed validation-gate command.

Recommended Codex Cloud path:

```bash
bash .codex/setup.sh
python benchmarks/run_local_validation.py --quick
```

The CPU devcontainer sets `OMNILSS_OFFLINE=1` by default. Run
`OMNILSS_OFFLINE=0 bash .codex/setup.sh` only during a connected maintenance
session when dependency caches should be refreshed.
