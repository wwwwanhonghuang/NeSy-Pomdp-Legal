from typing import Dict, List, Any
from belief_state import ClaimHypothesis, LatentBriefState


def build_latent_state_hypothesis(
    claims: List[Dict[str, Any]],
    claim_states: Dict[str, Dict[str, Any]],
    validations: Dict[str, Dict[str, Any]],
    symbolic_aq: Dict[str, float],
) -> LatentBriefState:
    claim_hypotheses = {}
    unresolved = []

    for claim in claims:
        cid = claim["id"]
        state = claim_states[cid]
        val = validations[cid]

        hyp = ClaimHypothesis(
            claim_id=cid,
            accepted_supports=[],
            accepted_attacks=[],
            support_status=state["support_status"],
            support_score=state["support_score"],
            attack_score=state["attack_score"],
            missing_evidence_types=state["missing_evidence_types"],
            reasoning_complete=val["reasoning_complete"],
            method_consistent=state["method_fit"] >= 0.5,
            contradiction_flag=val["contradiction_flag"],
            formal_validity_score=val["formal_validity_score"],
            reasoning_quality=state["reasoning_quality"],
            method_fit=state["method_fit"],
            clarity=state["clarity"],
            counterargument_handling=state["counterargument_handling"],
            overclaim_risk=state["overclaim_risk"],
        )
        claim_hypotheses[cid] = hyp

        if (
            hyp.support_status in {"unsupported", "weakly_supported", "contested"}
            or len(hyp.missing_evidence_types) > 0
            or not hyp.reasoning_complete
            or hyp.contradiction_flag
        ):
            unresolved.append(cid)

    mean_validity = 0.5
    if len(validations) > 0:
        mean_validity = sum(v["formal_validity_score"] for v in validations.values()) / len(validations)

    return LatentBriefState(
        claim_hypotheses=claim_hypotheses,
        aq_scores=symbolic_aq,
        global_formal_validity=float(mean_validity),
        unresolved_claim_ids=unresolved,
    )