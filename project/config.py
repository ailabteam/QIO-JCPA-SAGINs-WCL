# =============================================================================
# File: config.py
# Description:
# Central configuration file for the QIO-JLSPA refactored project.
# =============================================================================

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class SystemConfig:
    """Physical/system-level parameters."""
    n_transmitters: int = 4
    n_channels: int = 1

    frequency_ghz: float = 28.0

    # Nominal propagation distances (for synthetic snapshot generation)
    d_leo_km: float = 600.0
    d_haps_km: float = 20.0

    # Rician fading parameter
    rician_k: float = 10.0

    # Power / noise settings
    p_max_per_tx: float = 1.0
    noise_power: float = 1e-4

    # Optional transmitter-side load cap in binary selection stage
    max_links_per_tx: Optional[int] = None

    # Force every user to be assigned in the binary stage
    allow_unserved_users: bool = False


@dataclass
class QUBOConfig:
    """QUBO surrogate parameters."""
    w_direct_reward: float = 6.0
    w_interference_penalty: float = 0.15

    # Stronger user assignment penalty
    lambda_user_constraint: float = 200.0
    lambda_tx_constraint: float = 20.0

    use_equal_power_template: bool = True
    eps: float = 1e-12

    # Keep all candidates for debugging first
    top_k_tx_per_user: Optional[int] = None


@dataclass
class PowerRefinementConfig:
    """Settings for classical power refinement."""
    method: str = "SLSQP"

    max_iter: int = 50
    convergence_tol: float = 1e-4

    # Initialization rule
    init_strategy: str = "uniform_per_tx"

    # Clamp tiny negatives from numerical optimizer
    projection_eps: float = 1e-10

    # NEW:
    # Minimum power allocated to any active/selected link:
    #   p[t,u] >= p_min_active_link * X[t,u]
    p_min_active_link: float = 0.05


@dataclass
class SolverConfig:
    """Solver-level settings for binary optimization."""
    neal_num_reads_default: int = 100
    neal_num_reads_high_quality: int = 1000

    local_search_max_iters: int = 200
    local_search_num_restarts: int = 10

    exact_max_users: int = 6


@dataclass
class ExperimentConfig:
    """Experiment design settings."""
    n_users_scales: List[int] = field(default_factory=lambda: [2, 3, 4, 5])

    n_random_seeds: int = 10
    seed_start: int = 100
    seed_stride: int = 10

    run_exact_small: bool = True
    run_local_search: bool = True
    run_greedy: bool = True
    run_qio_neal: bool = True

    save_csv: bool = True


@dataclass
class PlotConfig:
    """Plot/export settings."""
    figure_width: float = 7.0
    figure_height: float = 4.0

    dpi: int = 200

    quality_scale_filename: str = "figure_quality_vs_scale.pdf"
    runtime_tradeoff_filename: str = "figure_quality_runtime_tradeoff.pdf"
    scaling_runtime_filename: str = "figure_runtime_scaling.pdf"


@dataclass
class ProjectConfig:
    """Top-level project config bundle."""
    system: SystemConfig = field(default_factory=SystemConfig)
    qubo: QUBOConfig = field(default_factory=QUBOConfig)
    power: PowerRefinementConfig = field(default_factory=PowerRefinementConfig)
    solver: SolverConfig = field(default_factory=SolverConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)


def get_default_config() -> ProjectConfig:
    """Return a default project configuration."""
    return ProjectConfig()
