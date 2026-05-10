"""Core optimization and JIT compilation modules for OmniLSS.

This package contains the core infrastructure for:
- Joint optimization (Optax, L-BFGS)
- GAMLSS integration (fit_with_joint_optimizer, fit_with_lbfgs)
- Device management (CPU/GPU/TPU)
- JIT compilation
"""

from .device import (
    DeviceInfo,
    DeviceManager,
    DeviceType,
    get_device,
    get_device_info,
    get_device_manager,
    list_devices,
    print_device_info,
    set_device,
    to_device,
    to_host,
)
from .gamlss_integration import (
    fit_with_joint_optimizer,
    fit_with_lbfgs,
)
from .lbfgs_optimizer import (
    LBFGSOptimizer,
    lbfgs_optimize,
)
from .optimizer import (
    JointOptimizer,
    OptimizationResult,
    Optimizer,
    create_optimizer,
    joint_optimize,
)

__all__ = [
    # Optimizers
    "Optimizer",
    "JointOptimizer",
    "LBFGSOptimizer",
    "OptimizationResult",
    "create_optimizer",
    "joint_optimize",
    "lbfgs_optimize",
    # GAMLSS Integration
    "fit_with_joint_optimizer",
    "fit_with_lbfgs",
    # Device Management
    "DeviceType",
    "DeviceInfo",
    "DeviceManager",
    "set_device",
    "get_device_manager",
    "get_device",
    "get_device_info",
    "to_device",
    "to_host",
    "list_devices",
    "print_device_info",
]

# ── Phase 2 新增：ParameterSpec 系统 ──
# ── Phase 2 新增：Laplace 近似 ──
from .laplace import (
    laplace_log_marginal,
    laplace_posterior_variance,
    log_det_cholesky,
    log_det_positive_semidefinite,
    reml_wood2011,
    stable_cholesky,
    trace_hat_matrix,
)
from .parameter_spec import (
    AUTO_LINK_MAP,
    CLASSIC_GAMLSS_PARAMS,
    MU_POS_SPEC,
    MU_SPEC,
    NU_SPEC,
    PI_SPEC,
    SIGMA_SPEC,
    TAU_SPEC,
    ConstraintType,
    ParameterSpec,
    infer_param_spec,
    specs_from_names,
)
