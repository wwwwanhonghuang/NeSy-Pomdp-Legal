import re
from typing import List, Dict


class ClaimExtractor:
    """Rule-based claim extractor for demo purposes."""

    def __init__(self):
        self.claim_markers = {
            "doctrinal": [
                r"\b(argues?|contends?|maintains?|holds?|supports?)\b",
                r"\b(OR|ZGB|StGB|BV)\s+(Art\.|Artikel)\s*\d+",
                r"\b(sufficient basis|legal basis|doctrinal basis|liability)\b",
            ],
            "methodological": [
                r"\b(method|methodology|approach|framework|interpretation|Auslegung)\b",
                r"\b(grammatical|historical|systematic|teleological)\b",
            ],
            "comparative": [
                r"\b(comparative|international|foreign law|EU law|European law)\b",
                r"\b(vergleichend|internationales Recht)\b",
            ],
            "rebuttal": [
                r"\b(however|nevertheless|although|but|yet)\b",
                r"\b(jedoch|allerdings|hingegen|gegenansicht)\b",
                r"\b(opposing view|counterargument|narrower reading)\b",
            ],
            "conclusion": [
                r"\b(therefore|thus|hence|in conclusion|thereby)\b",
                r"\b(daher|somit|folglich)\b",
            ]
        }

        self.inferential_markers = [
            r"\b(because|therefore|thus|hence|since|accordingly)\b",
            r"\b(daher|somit|folglich|weil|da)\b"
        ]

    def split_sentences(self, text: str) -> List[str]:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def detect_claim_type(self, sentence: str) -> str:
        hits = {}
        for claim_type, patterns in self.claim_markers.items():
            score = 0
            for p in patterns:
                if re.search(p, sentence, flags=re.IGNORECASE):
                    score += 1
            hits[claim_type] = score

        best_type = max(hits, key=hits.get)
        if hits[best_type] == 0:
            return "general"
        return best_type

    def assign_aq_type(self, claim_type: str, sentence: str) -> str:
        if claim_type in {"doctrinal", "methodological", "general"}:
            return "LC"
        if claim_type == "rebuttal":
            return "DR"
        if claim_type == "comparative":
            return "DN"
        if claim_type == "conclusion":
            return "RE"
        if re.search(r"\b(clear|coherent|well-structured|organized)\b", sentence, flags=re.IGNORECASE):
            return "RE"
        return "LC"

    def infer_required_evidence(self, claim_type: str, aq_type: str) -> List[str]:
        if claim_type == "doctrinal":
            return ["statute", "case_law", "doctrine"]
        if claim_type == "methodological":
            return ["method_statement", "case_law", "statute"]
        if claim_type == "comparative":
            return ["comparative_source", "doctrine"]
        if claim_type == "rebuttal":
            return ["counterargument", "case_law", "doctrine"]
        if claim_type == "conclusion":
            return ["statute", "case_law", "doctrine"]
        return ["doctrine"]

    def is_claim_sentence(self, sentence: str) -> bool:
        if len(sentence.split()) < 6:
            return False

        has_marker = False
        for patterns in self.claim_markers.values():
            for p in patterns:
                if re.search(p, sentence, flags=re.IGNORECASE):
                    has_marker = True
                    break

        has_inference = any(
            re.search(p, sentence, flags=re.IGNORECASE)
            for p in self.inferential_markers
        )

        has_assertive_shape = bool(re.search(r"\b(is|are|supports|provides|suggests|demonstrates|shows)\b", sentence, flags=re.IGNORECASE))

        return has_marker or has_inference or has_assertive_shape

    def extract_claims(self, text: str, section_name: str) -> List[Dict]:
        claims = []
        sentences = self.split_sentences(text)

        for idx, sent in enumerate(sentences):
            if not self.is_claim_sentence(sent):
                continue

            claim_type = self.detect_claim_type(sent)
            aq_type = self.assign_aq_type(claim_type, sent)
            required = self.infer_required_evidence(claim_type, aq_type)

            claims.append({
                "id": f"{section_name}::claim::{idx}",
                "text": sent,
                "section": section_name,
                "sentence_idx": idx,
                "claim_type": claim_type,
                "aq_type": aq_type,
                "required_evidence": required,
            })

        return claims
