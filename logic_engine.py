from collections import defaultdict
from typing import Dict, List
import numpy as np


class SymbolicGroundingEngine:
    """Convert claim support states into AQ-level symbolic signals."""

    def support_status_value(self, status: str) -> float:
        mapping = {
            "unsupported": 0.10,
            "weakly_supported": 0.35,
            "supported": 0.65,
            "strongly_supported": 0.90,
            "contested": 0.30,
        }
        return mapping.get(status, 0.20)

    def claims_to_aq_scores(self, claims: List[Dict], claim_states: Dict[str, Dict]) -> Dict[str, float]:
        aq_scores = defaultdict(list)

        for claim in claims:
            state = claim_states.get(claim["id"])
            if state is None:
                continue

            base = self.support_status_value(state["support_status"])
            missing = len(state.get("missing_evidence_types", []))
            if missing:
                base *= max(0.45, 1.0 - 0.15 * missing)

            if claim["claim_type"] == "methodological":
                base = min(1.0, base + 0.05)

            if claim["claim_type"] == "rebuttal" and state["attack_score"] > 0.35 and state["support_score"] < 0.4:
                base *= 0.8

            aq_scores[claim["aq_type"]].append(base)

        out = {}
        for aq in ["LC", "RE", "DR", "DN", "OQ"]:
            if aq_scores[aq]:
                out[aq] = float(np.mean(aq_scores[aq]))
            else:
                out[aq] = 0.40

        out["OQ"] = float(np.mean([out["LC"], out["RE"], out["DR"], out["DN"]]))
        return out
