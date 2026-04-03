from typing import Dict, List, Any
from collections import defaultdict


class FormalValidator:
    """
    Formal validation over observed objects and inferred claim hypotheses.
    """

    def __init__(self):
        self.schemas = {
            "doctrinal": {
                "required_any": ["statute", "case_law", "doctrine"],
                "preferred": ["method_statement"],
            },
            "methodological": {
                "required_any": ["method_statement"],
                "preferred": ["statute", "case_law"],
            },
            "comparative": {
                "required_any": ["comparative_source"],
                "preferred": ["doctrine"],
            },
            "rebuttal": {
                "required_any": ["counterargument"],
                "preferred": ["case_law", "doctrine"],
            },
            "conclusion": {
                "required_any": ["statute", "case_law", "doctrine"],
                "preferred": ["method_statement"],
            },
            "general": {
                "required_any": ["doctrine"],
                "preferred": [],
            },
        }

    def validate_claim_hypothesis(
        self,
        claim: Dict[str, Any],
        found_evidence_types: List[str],
        support_score: float,
        attack_score: float,
        method_fit: float,
        overclaim_risk: float,
    ) -> Dict[str, Any]:
        schema = self.schemas.get(claim["claim_type"], self.schemas["general"])

        has_required = any(t in found_evidence_types for t in schema["required_any"]) if schema["required_any"] else True
        missing_required = [] if has_required else schema["required_any"][:]
        missing_preferred = [t for t in schema["preferred"] if t not in found_evidence_types]

        reasoning_complete = True
        reasoning_notes = []

        if support_score < 0.25:
            reasoning_complete = False
            reasoning_notes.append("support too weak")

        if claim["claim_type"] in {"doctrinal", "conclusion"}:
            if not any(t in found_evidence_types for t in ["statute", "case_law", "doctrine"]):
                reasoning_complete = False
                reasoning_notes.append("no authority chain found")

        if claim["claim_type"] == "methodological":
            if "method_statement" not in found_evidence_types:
                reasoning_complete = False
                reasoning_notes.append("no explicit method support found")

        if method_fit < 0.40:
            reasoning_complete = False
            reasoning_notes.append("method fit too low")

        contradiction_flag = False
        contradiction_notes = []

        if attack_score > support_score and attack_score > 0.35:
            contradiction_flag = True
            contradiction_notes.append("attacks outweigh supports")

        if overclaim_risk > 0.65:
            contradiction_flag = True
            contradiction_notes.append("overclaim risk high")

        score = 1.0
        if not has_required:
            score -= 0.45
        score -= min(0.20, 0.08 * len(missing_preferred))
        if not reasoning_complete:
            score -= 0.25
        if contradiction_flag:
            score -= 0.20
        score = max(0.0, min(1.0, score))

        return {
            "schema_ok": has_required,
            "missing_required_any": missing_required,
            "missing_preferred": missing_preferred,
            "reasoning_complete": reasoning_complete,
            "reasoning_notes": reasoning_notes,
            "contradiction_flag": contradiction_flag,
            "contradiction_notes": contradiction_notes,
            "formal_validity_score": score,
        }

    def summarize(self, validations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        if not validations:
            return {
                "mean_formal_validity": 0.8,
                "schema_ok_rate": 0.0,
                "reasoning_complete_rate": 0.0,
                "contradiction_rate": 0.0,
            }

        vals = list(validations.values())
        return {
            "mean_formal_validity": float(sum(v["formal_validity_score"] for v in vals) / len(vals)),
            "schema_ok_rate": float(sum(1 for v in vals if v["schema_ok"]) / len(vals)),
            "reasoning_complete_rate": float(sum(1 for v in vals if v["reasoning_complete"]) / len(vals)),
            "contradiction_rate": float(sum(1 for v in vals if v["contradiction_flag"]) / len(vals)),
        }