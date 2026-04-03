from pomdp_core import AQ_LABELS


def fuse_signals(neural_signals: dict, symbolic_signals: dict) -> dict:
    """
    Combine neural + symbolic AQ signals in [0,1], then map to 1-6.
    """
    fused_unit = {
        "LC": 0.65 * symbolic_signals["LC"] + 0.35 * neural_signals["LC_shape"],
        "RE": 0.55 * neural_signals["RE"] + 0.45 * symbolic_signals["RE"],
        "DR": 0.55 * symbolic_signals["DR"] + 0.45 * neural_signals["DR_neural"],
        "DN": 0.70 * symbolic_signals["DN"] + 0.30 * neural_signals["DN_neural"],
        "OQ": 0.50 * symbolic_signals["OQ"] + 0.50 * neural_signals["OQ_neural"],
    }

    # map [0,1] -> [1,6]
    aq_targets = {aq: 1.0 + 5.0 * fused_unit[aq] for aq in AQ_LABELS}

    # AQ strengths: give more weight to symbolically grounded categories
    aq_strengths = {
        "LC": 1.3,
        "RE": 1.0,
        "DR": 1.1,
        "DN": 1.0,
        "OQ": 0.8,
    }

    return aq_targets, aq_strengths, fused_unit


def print_belief_summary(summary: dict):
    print("Current belief summary:")
    for aq, stats in summary.items():
        print(f"  {aq}: mean={stats['mean']:.2f}, var={stats['var']:.3f}")