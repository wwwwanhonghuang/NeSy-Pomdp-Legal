from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional

AQ_LABELS = ["LC", "RE", "DR", "DN", "OQ"]


@dataclass
class EvaluationBrief:
    claims: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    evidence: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    links: List[Dict[str, Any]] = field(default_factory=list)
    scored_links: List[Dict[str, Any]] = field(default_factory=list)
    claim_states: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    aq_summary: Dict[str, float] = field(default_factory=lambda: {aq: 0.4 for aq in AQ_LABELS})
    open_questions: List[Dict[str, Any]] = field(default_factory=list)
    policy_focus: Optional[Dict[str, Any]] = None

    def sync(self, claims: List[Dict], evidence: List[Dict], links: List[Dict], scored_links: List[Dict], claim_states: Dict[str, Dict], aq_summary: Dict[str, float]):
        self.claims = {c["id"]: c for c in claims}
        self.evidence = {e["id"]: e for e in evidence}
        self.links = links
        self.scored_links = scored_links
        self.claim_states = claim_states
        self.aq_summary = aq_summary
        self.open_questions = self._build_open_questions()

    def _build_open_questions(self) -> List[Dict[str, Any]]:
        questions = []
        for claim_id, state in self.claim_states.items():
            claim = self.claims.get(claim_id)
            if not claim:
                continue
            missing = state.get("missing_evidence_types", [])
            if state.get("support_status") in {"unsupported", "weakly_supported", "contested"} or missing:
                questions.append({
                    "claim_id": claim_id,
                    "claim_text": claim.get("text", ""),
                    "aq_type": claim.get("aq_type", "LC"),
                    "claim_type": claim.get("claim_type", "general"),
                    "support_status": state.get("support_status", "unsupported"),
                    "support_score": state.get("support_score", 0.0),
                    "attack_score": state.get("attack_score", 0.0),
                    "missing_evidence_types": missing,
                })
        questions.sort(key=lambda x: (len(x["missing_evidence_types"]), 1.0 - x["support_score"]), reverse=True)
        return questions

    def top_open_questions(self, n: int = 5) -> List[Dict[str, Any]]:
        return self.open_questions[:n]

    def set_policy_focus(self, action_info: Dict[str, Any]):
        self.policy_focus = action_info
