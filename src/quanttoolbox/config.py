"""Configuration dataclasses.

Replaces the MATLAB `global` variable blocks used throughout the original
QuantToolbox (e.g. RB_ADMM_*, RB_CCD_*, MVO_*, Proximal_Algorithm,
SVM_macheps, GMM_*, ML_*, WHITTLE_*). Each solver/estimator takes an
explicit config object (with sensible defaults) instead of relying on
module-level state, so calls are thread-safe and results are reproducible.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ADMMConfig:
    """Replaces RB_ADMM_* / ADMM_* globals."""

    varphi: float = 1.0
    varphi_method: int = 1
    max_iters: int = 500
    tol: float = 1e-10
    tau_primal: float = 1e-6
    tau_dual: float = 1e-6


@dataclass(frozen=True)
class CCDConfig:
    """Replaces RB_CCD_* globals (cyclical coordinate descent)."""

    tol: float = 1e-10
    max_iters: int = 500
    proximal: bool = True
    random_order: bool = False
    correction: bool = False
    x_max: float = 1e10


@dataclass(frozen=True)
class NewtonConfig:
    """Replaces RB_Newton_* globals."""

    tol: float = 1e-10
    max_iters: int = 100
    correction: bool = False
    x_max: float = 1e10
    print_iters: bool = False


@dataclass(frozen=True)
class ProximalConfig:
    """Replaces Proximal_Algorithm / Proximal_MaxIters globals."""

    algorithm: int = 1
    max_iters: int = 500


@dataclass(frozen=True)
class EstimationConfig:
    """Replaces Print_Results / Parameter_Labels / Header_Model /
    GMM_* / ML_* / WHITTLE_* globals used by econometrics estimators."""

    print_results: bool = False
    max_iters: int = 1000
    tol: float = 1e-8
    algorithm: str = "bfgs"
    parameter_labels: list[str] | None = None
    header_model: str | None = None


@dataclass(frozen=True)
class BisectionConfig:
    """Replaces the BISECTION_Tol global."""

    tol: float = 1e-8
    max_iters: int = 1000


@dataclass(frozen=True)
class SVMConfig:
    """Replaces SVM_macheps global."""

    macheps: float = 2.220446049250313e-16
