from collections import defaultdict
from typing import Dict, List
import numpy as np


class SymbolicGroundingEngine:
    """
    Convert richer claim states into AQ-level symbolic scores.

    Important:
    AQ is not dominated only by evidence integrity.
    Integrity acts as grounding / gating, but quality also depends on:
    - reasoning quality
    - method fit
    - counterargument handling
    - clarity
    - overclaim risk
    """

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

            integrity = self.support_status_value(state["support_status"])
            reasoning_quality = state.get("reasoning_quality", 0.5)
            method_fit = state.get("method_fit", 0.5)
            clarity = state.get("clarity", 0.5)
            counterargument_handling = state.get("counterargument_handling", 0.5)
            overclaim_risk = state.get("overclaim_risk", 0.0)

            aq = claim["aq_type"]

            if aq == "LC":
                score = (
                    0.35 * integrity +
                    0.35 * reasoning_quality +
                    0.30 * method_fit
                )
                if integrity < 0.30:
                    score = min(score, 0.45)  # ceiling
                score -= 0.15 * overclaim_risk

            elif aq == "DR":
                score = (
                    0.30 * integrity +
                    0.45 * counterargument_handling +
                    0.25 * reasoning_quality
                )
                if integrity < 0.25:
                    score = min(score, 0.40)
                score -= 0.10 * overclaim_risk

            elif aq == "DN":
                score = (
                    0.30 * integrity +
                    0.35 * counterargument_handling +
                    0.20 * reasoning_quality +
                    0.15 * clarity
                )
                score -= 0.08 * overclaim_risk

            elif aq == "RE":
                score = (
                    0.20 * integrity +
                    0.50 * clarity +
                    0.30 * reasoning_quality
                )
                score -= 0.05 * overclaim_risk

            else:  # OQ or fallback
                score = (
                    0.25 * integrity +
                    0.25 * reasoning_quality +
                    0.20 * method_fit +
                    0.15 * clarity +
                    0.15 * counterargument_handling
                )
                score -= 0.10 * overclaim_risk

            aq_scores[aq].append(float(np.clip(score, 0.0, 1.0)))

        out = {}
        for aq in ["LC", "RE", "DR", "DN", "OQ"]:
            if aq_scores[aq]:
                out[aq] = float(np.mean(aq_scores[aq]))
            else:
                out[aq] = 0.40

        # OQ as constrained holistic aggregate, not plain average only
        oq_base = np.mean([out["LC"], out["RE"], out["DR"], out["DN"]])

        # If LC is very low, OQ ceiling should be limited
        if out["LC"] < 0.30:
            oq_base = min(oq_base, 0.45)

        # If DR is very low, holistic quality also cannot be too high
        if out["DR"] < 0.25:
            oq_base = min(oq_base, 0.50)

        out["OQ"] = float(np.clip(oq_base, 0.0, 1.0))
        return out