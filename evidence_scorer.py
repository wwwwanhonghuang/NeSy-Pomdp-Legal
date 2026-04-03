from typing import Dict


class EvidenceScorer:
    """
    Score evidence strength and reliability.
    """

    def __init__(self):
        self.authority_prior = {
            "statute": 0.95,
            "case_law": 0.90,
            "doctrine": 0.75,
            "comparative_source": 0.65,
            "counterargument": 0.60,
            "method_statement": 0.70,
            "empirical_result": 0.80,
        }

    def score_evidence(self, claim: Dict, evidence: Dict, link: Dict) -> Dict:
        ev_types = evidence["evidence_types"]

        authority = max(self.authority_prior.get(t, 0.50) for t in ev_types)
        relevance = min(1.0, 0.55 + 0.45 * link["semantic_similarity"])

        overlap = len(set(claim["required_evidence"]).intersection(set(ev_types)))
        specificity = min(1.0, 0.40 + 0.20 * overlap)

        consistency = 0.85
        if link["relation"] == "attacks":
            consistency = 0.55

        overall_strength = (
            0.35 * authority +
            0.30 * relevance +
            0.20 * specificity +
            0.15 * consistency
        )

        return {
            "claim_id": claim["id"],
            "evidence_id": evidence["id"],
            "relation": link["relation"],
            "authority": float(authority),
            "relevance": float(relevance),
            "specificity": float(specificity),
            "consistency": float(consistency),
            "overall_strength": float(overall_strength),
            "evidence_types": ev_types,
        }