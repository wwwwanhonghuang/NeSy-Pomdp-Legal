from pomdp_core import AQ_LABELS


class SimplePolicy:
    """
    Minimal policy layer.
    Chooses:
    - read next section
    - check criterion
    - grade now

    This is not a full POMDP solver.
    It is a heuristic approximation of value-of-information logic.
    """

    def __init__(self, grade_variance_threshold: float = 0.18, min_steps_before_grade: int = 2):
        self.grade_variance_threshold = grade_variance_threshold
        self.min_steps_before_grade = min_steps_before_grade

    def choose_action(self, belief_summary: dict, step_idx: int, remaining_files: int):
        variances = {aq: belief_summary[aq]["var"] for aq in AQ_LABELS}
        means = {aq: belief_summary[aq]["mean"] for aq in AQ_LABELS}

        most_uncertain_aq = max(variances, key=variances.get)
        avg_var = sum(variances.values()) / len(variances)

        if step_idx >= self.min_steps_before_grade and avg_var < self.grade_variance_threshold:
            return {
                "action": "grade_now",
                "focus_aq": None,
                "reason": f"Average uncertainty is low ({avg_var:.3f}); grading is now justified."
            }

        if remaining_files > 0:
            return {
                "action": "read_next_section",
                "focus_aq": most_uncertain_aq,
                "reason": f"Highest uncertainty is in {most_uncertain_aq}; reading more evidence is preferred."
            }

        return {
            "action": "check_criterion",
            "focus_aq": most_uncertain_aq,
            "reason": f"No sections remain; inspect the most uncertain criterion: {most_uncertain_aq}."
        }