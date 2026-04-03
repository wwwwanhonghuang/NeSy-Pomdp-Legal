from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import copy

from belief_state import LatentBriefState, ObservableClaim, ObservableEvidence, ObservableRelation


@dataclass
class EvaluationBrief:
    """
    First-class structured assessment state.

    This object is the evolving 'brief' of the evaluator.
    It stores claims, evidence, links, AQ summaries, open questions,
    and the current policy focus.
    """

    claims: List[Dict[str, Any]] = field(default_factory=list)
    evidences: List[Dict[str, Any]] = field(default_factory=list)
    links: List[Dict[str, Any]] = field(default_factory=list)

    claim_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    aq_scores_symbolic: Dict[str, float] = field(default_factory=dict)
    aq_scores_neural: Dict[str, float] = field(default_factory=dict)
    aq_scores_fused: Dict[str, float] = field(default_factory=dict)

    open_questions: List[Dict[str, Any]] = field(default_factory=list)

    policy_focus: Optional[Dict[str, Any]] = None
    action_history: List[Dict[str, Any]] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)
    
    formal_validations: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    formal_validation_summary: Dict[str, Any] = field(default_factory=dict)

    observed_claims: List[ObservableClaim] = field(default_factory=list)
    observed_evidences: List[ObservableEvidence] = field(default_factory=list)
    observed_relations: List[ObservableRelation] = field(default_factory=list)

    latent_state_mean: Optional[LatentBriefState] = None

    
    def snapshot(self) -> Dict[str, Any]:
        return copy.deepcopy({
            "claims": self.claims,
            "evidences": self.evidences,
            "links": self.links,
            "claim_states": self.claim_states,
            "aq_scores_symbolic": self.aq_scores_symbolic,
            "aq_scores_neural": self.aq_scores_neural,
            "aq_scores_fused": self.aq_scores_fused,
            "open_questions": self.open_questions,
            "policy_focus": self.policy_focus,
            "action_history": self.action_history,
            "metadata": self.metadata,
            "observed_claims": self.observed_claims,
            "observed_evidences": self.observed_evidences,
            "observed_relations": self.observed_relations,
            "latent_state_mean": self.latent_state_mean,
        })

    def set_claims(self, claims: List[Dict[str, Any]]):
        self.claims = claims

    def set_latent_state_mean(self, state: LatentBriefState):
        self.latent_state_mean = state
        
    def set_observed_claims(self, claims: List[ObservableClaim]):
        self.observed_claims = claims
    def set_observed_evidences(self, evidences: List[ObservableEvidence]):
        self.observed_evidences = evidences

    def set_observed_relations(self, relations: List[ObservableRelation]):
        self.observed_relations = relations
        
    def set_formal_validations(self, formal_validations: Dict[str, Dict[str, Any]]):
        self.formal_validations = formal_validations

    def set_formal_validation_summary(self, summary: Dict[str, Any]):
        self.formal_validation_summary = summary
        
    def set_evidences(self, evidences: List[Dict[str, Any]]):
        self.evidences = evidences

    def set_links(self, links: List[Dict[str, Any]]):
        self.links = links

    def set_claim_states(self, claim_states: Dict[str, Dict[str, Any]]):
        self.claim_states = claim_states
        self._refresh_open_questions()

    def set_symbolic_aq(self, aq_scores_symbolic: Dict[str, float]):
        self.aq_scores_symbolic = aq_scores_symbolic

    def set_neural_aq(self, aq_scores_neural: Dict[str, float]):
        self.aq_scores_neural = aq_scores_neural

    def set_fused_aq(self, aq_scores_fused: Dict[str, float]):
        self.aq_scores_fused = aq_scores_fused

    def set_policy_focus(self, policy_focus: Dict[str, Any]):
        self.policy_focus = policy_focus

    def add_action_record(self, record: Dict[str, Any]):
        self.action_history.append(record)

    def _refresh_open_questions(self):
        """
        Open questions are unresolved / weak claims that still lack grounding
        or have quality issues beyond simple support completeness.
        """
        claim_map = {c["id"]: c for c in self.claims}
        open_questions = []

        for cid, state in self.claim_states.items():
            claim = claim_map.get(cid, {})
            status = state.get("support_status", "unsupported")
            missing = state.get("missing_evidence_types", [])
            reasoning_quality = state.get("reasoning_quality", 0.5)
            overclaim_risk = state.get("overclaim_risk", 0.0)

            is_open = (
                status in {"unsupported", "weakly_supported", "contested"}
                or len(missing) > 0
                or reasoning_quality < 0.5
                or overclaim_risk > 0.5
            )

            if is_open:
                open_questions.append({
                    "claim_id": cid,
                    "claim_text": claim.get("text", ""),
                    "aq_type": claim.get("aq_type", ""),
                    "claim_type": claim.get("claim_type", ""),
                    "support_status": status,
                    "missing_evidence_types": missing,
                    "reasoning_quality": reasoning_quality,
                    "overclaim_risk": overclaim_risk,
                    "priority_score": self._priority_score(state, claim),
                })

        open_questions.sort(key=lambda x: x["priority_score"], reverse=True)
        self.open_questions = open_questions

    def _priority_score(self, state: Dict[str, Any], claim: Dict[str, Any]):
        """
        Higher means more important to inspect next.
        """
        support_status = state.get("support_status", "unsupported")
        support_penalty = {
            "unsupported": 1.00,
            "weakly_supported": 0.80,
            "contested": 0.85,
            "supported": 0.35,
            "strongly_supported": 0.15,
        }.get(support_status, 0.5)

        missing_count = len(state.get("missing_evidence_types", []))
        reasoning_quality = state.get("reasoning_quality", 0.5)
        overclaim_risk = state.get("overclaim_risk", 0.0)

        aq_weight = {
            "LC": 1.00,
            "DR": 0.95,
            "DN": 0.80,
            "RE": 0.65,
            "OQ": 0.75,
        }.get(claim.get("aq_type", "LC"), 0.7)

        return (
            0.40 * support_penalty +
            0.20 * min(missing_count / 3.0, 1.0) +
            0.20 * (1.0 - reasoning_quality) +
            0.20 * overclaim_risk
        ) * aq_weight

    def get_top_open_claims(self, k: int = 5):
        return self.open_questions[:k]

    def get_brief_diff(self, old_snapshot: Dict[str, Any]):
        """
        Produce a simple diff between two brief states for UI display.
        """
        old_claim_states = old_snapshot.get("claim_states", {})
        new_claim_states = self.claim_states

        changed_claims = []
        for cid, new_state in new_claim_states.items():
            old_state = old_claim_states.get(cid)
            if old_state is None:
                changed_claims.append({
                    "claim_id": cid,
                    "change_type": "new_claim_state",
                    "old_status": None,
                    "new_status": new_state.get("support_status"),
                })
                continue

            if (
                old_state.get("support_status") != new_state.get("support_status")
                or old_state.get("support_score") != new_state.get("support_score")
                or old_state.get("attack_score") != new_state.get("attack_score")
                or old_state.get("missing_evidence_types") != new_state.get("missing_evidence_types")
            ):
                changed_claims.append({
                    "claim_id": cid,
                    "change_type": "updated_claim_state",
                    "old_status": old_state.get("support_status"),
                    "new_status": new_state.get("support_status"),
                    "old_missing": old_state.get("missing_evidence_types", []),
                    "new_missing": new_state.get("missing_evidence_types", []),
                })

        old_fused = old_snapshot.get("aq_scores_fused", {})
        new_fused = self.aq_scores_fused
        aq_changes = {}
        for aq, new_val in new_fused.items():
            old_val = old_fused.get(aq)
            if old_val is None or old_val != new_val:
                aq_changes[aq] = {
                    "old": old_val,
                    "new": new_val,
                }

        return {
            "changed_claims": changed_claims,
            "aq_changes": aq_changes,
        }