from .arbitrage import best_bounded_trader_profit
from .approachability import (
    AppErrCurve,
    app_err_curve_single_coordinate,
    app_err_curve_upper_orthant_dense,
    distance_to_box_linf,
    distance_to_upper_orthant_linf,
    summarize_top_box_violations,
)
from .calibration import expected_calibration_error
from .multiclass import multiclass_brier_loss, multiclass_log_loss, multiclass_sce, top_label_ece
from .proper import brier_loss, log_loss, squared_calibration_error
from .swap_regret import compute_swap_regret, compute_external_regret, swap_external_decomposition
from .multiscale_approachability import (
    multiscale_constraint_evaluation,
    approachability_dynamics,
    ApproachabilityDynamicsSpec,
)
from .hallucination import (
    HallucinationMetrics,
    compute_hallucination_rate,
    compute_abstention_aware_metrics,
    validate_kalai_bound,
    HallucinationTracker,
    compute_calibration_by_distance,
)

__all__ = [
    "AppErrCurve",
    "brier_loss",
    "log_loss",
    "squared_calibration_error",
    "expected_calibration_error",
    "best_bounded_trader_profit",
    "distance_to_box_linf",
    "distance_to_upper_orthant_linf",
    "app_err_curve_single_coordinate",
    "app_err_curve_upper_orthant_dense",
    "summarize_top_box_violations",
    "multiclass_brier_loss",
    "multiclass_log_loss",
    "multiclass_sce",
    "top_label_ece",
    "compute_swap_regret",
    "compute_external_regret",
    "swap_external_decomposition",
    "multiscale_constraint_evaluation",
    "approachability_dynamics",
    "ApproachabilityDynamicsSpec",
    "HallucinationMetrics",
    "compute_hallucination_rate",
    "compute_abstention_aware_metrics",
    "validate_kalai_bound",
    "HallucinationTracker",
    "compute_calibration_by_distance",
]


