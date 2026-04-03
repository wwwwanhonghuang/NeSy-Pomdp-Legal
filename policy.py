from typing import Dict, List, Optional


class ClaimDrivenPolicy:
    """
    Policy acts on unresolved claims, not only AQ uncertainty.

    It chooses:
    - read_next_section
    - check_claim
    - grade_now

    based on:
    - top open claim in the brief
    - missing evidence types
    - remaining section types / names
    - belief uncertainty
    """

    def __init__(
        self,
        grade_variance_threshold: float = 0.18,
        min_steps_before_grade: int = 2
    ):
        self.grade_variance_threshold = grade_variance_threshold
        self.min_steps_before_grade = min_steps_before_grade

        self.section_hints = {
            "statute": ["source", "analysis", "method", "literature"],
            "case_law": ["bge", "case", "analysis", "method", "literature"],
            "doctrine": ["literature", "review", "analysis"],
            "comparative_source": ["comparative", "international", "discussion", "conclusion"],
            "counterargument": ["discussion", "conclusion", "literature"],
            "method_statement": ["method", "methodology", "approach"],
        }

    def choose_action(
        self,
        brief,
        belief_summary: Dict,
        step_idx: int,
        remaining_sections: List[str]
    ) -> Dict:
        avg_var = sum(v["var"] for v in belief_summary.values()) / len(belief_summary)

        top_open_claims = brief.get_top_open_claims(k=3)

        # If little uncertainty and no important open claims, grade
        if (
            step_idx >= self.min_steps_before_grade and
            avg_var < self.grade_variance_threshold and
            len(top_open_claims) == 0
        ):
            return {
                "action": "grade_now",
                "reason": f"Low uncertainty ({avg_var:.3f}) and no major open claims remain.",
                "focus_claim_id": None,
                "focus_aq": None,
                "missing_evidence": [],
                "suggested_section": None,
            }

        if len(top_open_claims) == 0:
            if remaining_sections:
                return {
                    "action": "read_next_section",
                    "reason": "No specific unresolved claim dominates; continue reading.",
                    "focus_claim_id": None,
                    "focus_aq": None,
                    "missing_evidence": [],
                    "suggested_section": remaining_sections[0],
                }
            return {
                "action": "grade_now",
                "reason": "No major open claims and no sections remain.",
                "focus_claim_id": None,
                "focus_aq": None,
                "missing_evidence": [],
                "suggested_section": None,
            }

        top_claim = top_open_claims[0]
        missing_evidence = top_claim.get("missing_evidence_types", [])
        suggested_section = self._suggest_section(missing_evidence, remaining_sections)

        if remaining_sections:
            return {
                "action": "read_next_section",
                "reason": (
                    f"Top unresolved claim in {top_claim['aq_type']} remains open; "
                    f"missing evidence types: {missing_evidence}."
                ),
                "focus_claim_id": top_claim["claim_id"],
                "focus_aq": top_claim["aq_type"],
                "missing_evidence": missing_evidence,
                "suggested_section": suggested_section,
            }

        return {
            "action": "check_claim",
            "reason": (
                f"No sections remain. Inspect unresolved claim {top_claim['claim_id']} "
                f"with missing evidence {missing_evidence}."
            ),
            "focus_claim_id": top_claim["claim_id"],
            "focus_aq": top_claim["aq_type"],
            "missing_evidence": missing_evidence,
            "suggested_section": None,
        }

    def _suggest_section(self, missing_evidence: List[str], remaining_sections: List[str]) -> Optional[str]:
        if not remaining_sections:
            return None
        if not missing_evidence:
            return remaining_sections[0]

        lower_sections = {s: s.lower() for s in remaining_sections}

        for ev in missing_evidence:
            hints = self.section_hints.get(ev, [])
            for sec, lower_name in lower_sections.items():
                for hint in hints:
                    if hint in lower_name:
                        return sec

        return remaining_sections[0]