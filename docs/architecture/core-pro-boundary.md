# Core / Pro Boundary

[中文版本](core-pro-boundary_cn.md)

## Goal

Define the Open-Core boundary: what lives in the open-source **core** (OmniLSS)
and what lives in the commercial **Pro** layer (omnilss-pro), so the GPL core
internals are never copied into proprietary modules and the Pro layer talks to
the core only through public, process-level boundaries.

## Core — open source (GPL-3.0, `omnilss/`)

**L1 statistical engine** (Release 1 scope):
- distribution families (d/p/q/r) and the family registry
- fitting algorithms (RS / CG / MIXED) and numerical foundations
- smoothers (pb / ps / cs) and prediction (saved-state newdata rebuild)
- model selection / diagnostics / serialization (JSON + pickle)

**L4 validation & delivery**:
- R-consistency test suites and fixed-R golden artifacts
- capability evidence matrix and family capability gate
- tolerance audit, `--require-r` fail-closed phase gates
- CI workflows and release tooling

## Pro — commercial layer (`omnilss-pro/`)

**L3 hosted services**:
- gRPC / HTTP service layer and the client SDK (`omnilss_pro.client`)
- AutoML (candidate-family ranking, bootstrap intervals) — resource governed
- hosted orchestration / deployment tooling / enterprise adapters

## Boundary rules

1. **No copying**: Pro modules never import GPL core *internals*; they interact
   with the core only through public APIs (e.g. the gRPC batch-fit boundary)
   or the documented Python API of the core package.
2. **Process boundary**: hosted orchestration calls the core over gRPC/HTTP;
   the Pro package must not `import omnilss.<private module>`.
3. **Independent release channel**: the core publishes under GPL-3.0; the Pro
   layer ships through a separate commercial channel and version line.
4. **Verification belongs to the core**: R-consistency evidence, golden
   artifacts and capability matrices live in the open core so claims are
   auditable; Pro adds resource governance on top, not its own correctness
   claims.
