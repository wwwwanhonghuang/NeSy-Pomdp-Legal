from typing import Dict, List, Optional

AQ_LABELS = ["LC", "RE", "DR", "DN", "OQ"]


class ClaimDrivenPolicy:
    """Policy that acts on unsupported claims and missing evidence."""

    def __init__(self, grade_variance_threshold: float = 0.18, min_steps_before_grade: int = 2):
        self.grade_variance_threshold = grade_variance_threshold
        self.min_steps_before_grade = min_steps_before_grade
        self.section_type_hints = {
            "statute": ["source", "analysis", "method", "intro"],
            "case_law": ["bge", "case", "method", "analysis", "conclusion"],
            "doctrine": ["literature", "review", "intro", "conclusion"],
            "comparative_source": ["comparative", "international", "conclusion", "literature"],
            "counterargument": ["discussion", "conclusion", "literature"],
            "method_statement": ["method", "intro", "analysis"],
        }

    def _avg_var(self, belief_summary: Dict) -> float:
        return sum(belief_summary[aq]["var"] for aq in AQ_LABELS) / len(AQ_LABELS)

    def _best_open_question(self, brief) -> Optional[Dict]:
        questions = brief.top_open_questions(10)
        if not questions:
            return None
        questions = sorted(
            questions,
            key=lambda q: (
                len(q.get("missing_evidence_types", [])),
                q.get("support_status") in {"unsupported", "contested"},
                1.0 - q.get("support_score", 0.0),
            ),
            reverse=True,
        )
        return questions[0]

    def _suggest_section(self, missing_types: List[str], remaining_sections: List[str]) -> Optional[str]:
        if not remaining_sections:
            return None
        lower_sections = [(s, s.lower()) for s in remaining_sections]
        for ev_type in missing_types:
            hints = self.section_type_hints.get(ev_type, [])
            for original, lower in lower_sections:
                if any(h in lower for h in hints):
                    return original
        return remaining_sections[0]

    def choose_action(self, belief_summary: Dict, brief, step_idx: int, remaining_sections: List[str]):
        avg_var = self._avg_var(belief_summary)
        open_q = self._best_open_question(brief)

        if step_idx >= self.min_steps_before_grade and avg_var < self.grade_variance_threshold and open_q is None:
            return {
                "action": "grade_now",
                "focus_aq": None,
                "focus_claim_id": None,
                "reason": f"Average uncertainty is low ({avg_var:.3f}) and no important open claims remain.",
                "suggested_section": None,
                "missing_evidence_types": [],
            }

        if open_q is not None:
            suggested_section = self._suggest_section(open_q.get("missing_evidence_types", []), remaining_sections)
            if remaining_sections:
                return {
                    "action": "read_next_section",
                    "focus_aq": open_q.get("aq_type"),
                    "focus_claim_id": open_q.get("claim_id"),
                    "reason": (
                        f"Claim is still {open_q.get('support_status')} with missing evidence "
                        f"{open_q.get('missing_evidence_types', [])}."
                    ),
                    "suggested_section": suggested_section,
                    "missing_evidence_types": open_q.get("missing_evidence_types", []),
                    "claim_text": open_q.get("claim_text", ""),
                }
            return {
                "action": "check_criterion",
                "focus_aq": open_q.get("aq_type"),
                "focus_claim_id": open_q.get("claim_id"),
                "reason": "No sections remain; inspect the unsupported claim directly.",
                "suggested_section": None,
                "missing_evidence_types": open_q.get("missing_evidence_types", []),
                "claim_text": open_q.get("claim_text", ""),
            }

        if remaining_sections:
            most_uncertain = max(AQ_LABELS, key=lambda aq: belief_summary[aq]["var"])
            return {
                "action": "read_next_section",
                "focus_aq": most_uncertain,
                "focus_claim_id": None,
                "reason": f"No severe claim gaps found; reduce uncertainty in {most_uncertain}.",
                "suggested_section": remaining_sections[0],
                "missing_evidence_types": [],
            }

        return {
            "action": "grade_now",
            "focus_aq": None,
            "focus_claim_id": None,
            "reason": "No sections remain and no unresolved high-priority claim gaps remain.",
            "suggested_section": None,
            "missing_evidence_types": [],
        }
