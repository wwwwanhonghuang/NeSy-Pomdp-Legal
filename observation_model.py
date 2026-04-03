from dataclasses import dataclass, field
from typing import List, Dict, Any
from belief_state import ObservableClaim, ObservableEvidence, ObservableRelation


@dataclass
class Observation:
    timestep: int
    claims: List[ObservableClaim] = field(default_factory=list)
    evidences: List[ObservableEvidence] = field(default_factory=list)
    candidate_relations: List[ObservableRelation] = field(default_factory=list)

    metadata: Dict[str, Any] = field(default_factory=dict)


class ObservationBuilder:
    def build(
        self,
        timestep: int,
        raw_claims: List[Dict[str, Any]],
        raw_evidences: List[Dict[str, Any]],
        raw_links: List[Dict[str, Any]],
    ) -> Observation:
        claims = [
            ObservableClaim(
                id=c["id"],
                text=c["text"],
                section=c["section"],
                sentence_idx=c["sentence_idx"],
                claim_type=c["claim_type"],
                aq_type=c["aq_type"],
                required_evidence=c.get("required_evidence", []),
            )
            for c in raw_claims
        ]

        evidences = [
            ObservableEvidence(
                id=e["id"],
                text=e["text"],
                section=e["section"],
                sentence_idx=e["sentence_idx"],
                evidence_types=e.get("evidence_types", []),
            )
            for e in raw_evidences
        ]

        relations = [
            ObservableRelation(
                claim_id=l["claim_id"],
                evidence_id=l["evidence_id"],
                relation_type=l["relation"],
                candidate_score=l["score"],
                semantic_similarity=l["semantic_similarity"],
                type_score=l["type_score"],
                section_score=l["section_score"],
            )
            for l in raw_links
        ]

        return Observation(
            timestep=timestep,
            claims=claims,
            evidences=evidences,
            candidate_relations=relations,
            metadata={
                "num_claims": len(claims),
                "num_evidences": len(evidences),
                "num_relations": len(relations),
            }
        )