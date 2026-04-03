from pomdp_core import AQ_LABELS


def fuse_signals(neural_signals: dict, symbolic_aq: dict):
    """
    AQ-specific fusion.
    Symbolic is grounding-rich, neural captures softer discourse cues.
    """

    fused_unit = {
        "LC": 0.65 * symbolic_aq["LC"] + 0.35 * neural_signals["LC_shape"],
        "RE": 0.55 * symbolic_aq["RE"] + 0.45 * neural_signals["RE"],
        "DR": 0.65 * symbolic_aq["DR"] + 0.35 * neural_signals["DR_neural"],
        "DN": 0.65 * symbolic_aq["DN"] + 0.35 * neural_signals["DN_neural"],
        "OQ": 0.55 * symbolic_aq["OQ"] + 0.45 * neural_signals["OQ_neural"],
    }

    aq_targets = {aq: 1.0 + 5.0 * fused_unit[aq] for aq in AQ_LABELS}

    aq_strengths = {
        "LC": 1.25,
        "RE": 1.00,
        "DR": 1.15,
        "DN": 1.00,
        "OQ": 0.85,
    }

    return aq_targets, aq_strengths, fused_unit