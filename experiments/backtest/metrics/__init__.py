"""Metrics for backtest evaluation."""

# Lazy imports
def __getattr__(name: str):
    if name == "PnLTracker":
        from backtest.metrics.pnl import PnLTracker
        return PnLTracker
    if name == "compute_risk_metrics":
        from backtest.metrics.pnl import compute_risk_metrics
        return compute_risk_metrics
    if name == "RegretTracker":
        from backtest.metrics.regret import RegretTracker
        return RegretTracker
    if name == "compute_regret_exponent":
        from backtest.metrics.regret import compute_regret_exponent
        return compute_regret_exponent
    if name == "H4Validator":
        from backtest.metrics.h4_validation import H4Validator
        return H4Validator
    if name == "CtValidator":
        from backtest.metrics.ct_validation import CtValidator
        return CtValidator
    if name == "CtValidationConfig":
        from backtest.metrics.ct_validation import CtValidationConfig
        return CtValidationConfig
    if name == "quick_validate_ct":
        from backtest.metrics.ct_validation import quick_validate_ct
        return quick_validate_ct
    if name == "BlackwellApproachabilityTracker":
        from backtest.metrics.approachability import BlackwellApproachabilityTracker
        return BlackwellApproachabilityTracker
    if name == "TestFunctionFamily":
        from backtest.metrics.approachability import TestFunctionFamily
        return TestFunctionFamily
    if name == "run_approachability_test":
        from backtest.metrics.approachability import run_approachability_test
        return run_approachability_test
    if name == "CtApproximationAnalyzer":
        from backtest.metrics.ct_approximation_error import CtApproximationAnalyzer
        return CtApproximationAnalyzer
    if name == "CtApproximationMetrics":
        from backtest.metrics.ct_approximation_error import CtApproximationMetrics
        return CtApproximationMetrics
    if name == "compute_ct_error_from_samples":
        from backtest.metrics.ct_approximation_error import compute_ct_error_from_samples
        return compute_ct_error_from_samples
    raise AttributeError(f"module 'backtest.metrics' has no attribute {name!r}")

__all__ = [
    "PnLTracker",
    "compute_risk_metrics",
    "RegretTracker",
    "compute_regret_exponent",
    "H4Validator",
    "CtValidator",
    "CtValidationConfig",
    "quick_validate_ct",
    "BlackwellApproachabilityTracker",
    "TestFunctionFamily",
    "run_approachability_test",
    "CtApproximationAnalyzer",
    "CtApproximationMetrics",
    "compute_ct_error_from_samples",
]

