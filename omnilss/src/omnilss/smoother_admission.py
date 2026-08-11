"""Phase 3.7: smoother admission for Release 1 fitting routes.

design.md Property 12: a smoother is admitted to Release 1 fitting routes if
and only if parsing, basis/penalty construction, lambda/EDF lifecycle, fitting,
newdata prediction and R Golden Artifact validation are complete; otherwise
``UnsupportedSmootherError`` is raised containing the smoother name and the
missing capabilities.

This module is imported by both the formula parser and the smooth-fitting
layer so every smoother request — regardless of which layer first encounters
it — fails closed with the same structured error.
"""

from __future__ import annotations

# Only smoothers with complete parsing / basis / penalty / lambda-EDF /
# fitting / newdata prediction / R golden artifact validation are admitted.
ADMITTED_SMOOTHERS: frozenset[str] = frozenset({"pb", "ps", "cs"})

# Capabilities a smoother must complete to be admitted to Release 1 routes.
RELEASE1_CAPABILITIES: tuple[str, ...] = (
    "parsing",
    "basis",
    "penalty",
    "lambda_edf",
    "fitting",
    "prediction",
    "r_golden",
)


class UnsupportedSmootherError(ValueError):
    """Fail-closed rejection of a smoother that is not Release-1 admitted."""

    def __init__(
        self,
        smoother: str,
        missing_capabilities: list[str] | tuple[str, ...] | None = None,
        *,
        message: str | None = None,
    ) -> None:
        self.smoother = smoother
        self.missing_capabilities = list(missing_capabilities or RELEASE1_CAPABILITIES)
        super().__init__(
            message
            or (
                f"Smoother {smoother!r} is not admitted to Release 1 fitting "
                f"routes; missing capabilities: {', '.join(self.missing_capabilities)}"
            )
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "code": "unsupported_smoother",
            "smoother": self.smoother,
            "missing_capabilities": self.missing_capabilities,
            "message": str(self),
        }


def check_smoother_admission(smoother: str) -> None:
    """Admit a smoother to Release 1 fitting routes or raise fail-closed.

    Only ``pb``, ``ps`` and ``cs`` have completed the full capability chain.
    All other smoothers — including staged implementations (lo/random/re/ri/
    pcat) and unimplemented ones (fp/bfp/cy/pvc/pbm/pbz/pbo/pbc/pbp/scs/te/ti)
    — are rejected until validated.
    """
    if smoother in ADMITTED_SMOOTHERS:
        return
    raise UnsupportedSmootherError(smoother, missing_capabilities=RELEASE1_CAPABILITIES)
