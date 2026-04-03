from typing import List, Dict
from encoder import NeuralEncoder


class ClaimEvidenceLinker:
    """Link claims to evidence candidates using simple heuristics."""

    def __init__(self, encoder: NeuralEncoder):
        self.encoder = encoder

    def type_compatibility(self, claim: Dict, evidence: Dict) -> float:
        required = set(claim["required_evidence"])
        available = set(evidence["evidence_types"])

        overlap = len(required.intersection(available))
        if overlap == 0:
            return 0.15
        return min(1.0, 0.35 + 0.25 * overlap)

    def section_prior(self, claim: Dict, evidence: Dict) -> float:
        if claim["section"] == evidence["section"]:
            return 0.75
        return 0.90

    def link(self, claims: List[Dict], evidences: List[Dict]) -> List[Dict]:
        if not claims or not evidences:
            return []

        claim_texts = [c["text"] for c in claims]
        evidence_texts = [e["text"] for e in evidences]

        claim_vecs = self.encoder.encode_many(claim_texts)
        evidence_vecs = self.encoder.encode_many(evidence_texts)

        links = []

        for i, claim in enumerate(claims):
            for j, evidence in enumerate(evidences):
                sim = self.encoder.cosine_similarity(claim_vecs[i], evidence_vecs[j])
                type_score = self.type_compatibility(claim, evidence)
                section_score = self.section_prior(claim, evidence)
                link_score = 0.55 * sim + 0.30 * type_score + 0.15 * section_score

                relation = "supports"
                if "counterargument" in evidence["evidence_types"] and claim["aq_type"] != "DR":
                    relation = "attacks"

                if link_score >= 0.45:
                    links.append({
                        "claim_id": claim["id"],
                        "evidence_id": evidence["id"],
                        "relation": relation,
                        "score": float(link_score),
                        "semantic_similarity": float(sim),
                        "type_score": float(type_score),
                        "section_score": float(section_score),
                    })

        return links
