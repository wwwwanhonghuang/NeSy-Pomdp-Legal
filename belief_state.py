from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
import copy


@dataclass
class ObservableClaim:
    id: str
    text: str
    section: str
    sentence_idx: int
    claim_type: str
    aq_type: str
    required_evidence: List[str] = field(default_factory=list)


@dataclass
class ObservableEvidence:
    id: str
    text: str
    section: str
    sentence_idx: int
    evidence_types: List[str] = field(default_factory=list)


@dataclass
class ObservableRelation:
    claim_id: str
    evidence_id: str
    relation_type: str   # supports / attacks / rebuts / etc.
    candidate_score: float
    semantic_similarity: float
    type_score: float
    section_score: float


@dataclass
class ClaimHypothesis:
    """
    Latent interpretation of a claim given current observations.
    """
    claim_id: str
    accepted_supports: List[str] = field(default_factory=list)
    accepted_attacks: List[str] = field(default_factory=list)

    support_status: str = "unsupported"   # unsupported / weakly_supported / supported / strongly_supported / contested
    support_score: float = 0.0
    attack_score: float = 0.0

    missing_evidence_types: List[str] = field(default_factory=list)
    reasoning_complete: bool = False
    method_consistent: bool = False
    contradiction_flag: bool = False

    formal_validity_score: float = 0.0

    reasoning_quality: float = 0.5
    method_fit: float = 0.5
    clarity: float = 0.5
    counterargument_handling: float = 0.5
    overclaim_risk: float = 0.0


@dataclass
class LatentBriefState:
    """
    True POMDP-style latent state hypothesis.
    This is not just the observed objects.
    It is one interpretation of them.
    """
    claim_hypotheses: Dict[str, ClaimHypothesis] = field(default_factory=dict)

    aq_scores: Dict[str, float] = field(default_factory=lambda: {
        "LC": 0.4,
        "RE": 0.4,
        "DR": 0.4,
        "DN": 0.4,
        "OQ": 0.4,
    })

    global_formal_validity: float = 0.5
    unresolved_claim_ids: List[str] = field(default_factory=list)

    def snapshot(self):
        return copy.deepcopy(self)

    def top_unresolved_claims(self, k: int = 5) -> List[str]:
        return self.unresolved_claim_ids[:k]


@dataclass
class EvaluationBrief:
    """
    Full system brief:
    - observed graph
    - latent state summary
    - policy focus
    - history
    """
    observed_claims: List[ObservableClaim] = field(default_factory=list)
    observed_evidences: List[ObservableEvidence] = field(default_factory=list)
    observed_relations: List[ObservableRelation] = field(default_factory=list)

    latent_state_mean: Optional[LatentBriefState] = None

    policy_focus: Optional[Dict[str, Any]] = None
    action_history: List[Dict[str, Any]] = field(default_factory=list)

    def snapshot(self):
        return copy.deepcopy({
            "observed_claims": self.observed_claims,
            "observed_evidences": self.observed_evidences,
            "observed_relations": self.observed_relations,
            "latent_state_mean": self.latent_state_mean,
            "policy_focus": self.policy_focus,
            "action_history": self.action_history,
        })

    

    def set_observed_evidences(self, evidences: List[ObservableEvidence]):
        self.observed_evidences = evidences

    def set_observed_relations(self, relations: List[ObservableRelation]):
        self.observed_relations = relations

    def set_latent_state_mean(self, state: LatentBriefState):
        self.latent_state_mean = state

    def set_policy_focus(self, focus: Dict[str, Any]):
        self.policy_focus = focus

    def add_action_record(self, record: Dict[str, Any]):
        self.action_history.append(record)