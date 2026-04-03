import re
from typing import List, Dict


class EvidenceExtractor:
    """Extract evidence candidates from revealed text memory."""

    def __init__(self):
        self.patterns = {
            "statute": r"\b(OR|ZGB|StGB|BV)\s+(Art\.|Artikel)\s*\d+[a-zA-Z]*",
            "case_law": r"\bBGE\s+\d+\s+[IVX]+\s+\d+",
            "doctrine": r"\b([A-ZÄÖÜ][a-zäöüß]+,\s*\d{4}|[A-ZÄÖÜ][a-zäöüß]+\s+et al\.)\b",
            "comparative_source": r"\b(comparative|international|foreign law|EU law|European law|vergleichend|internationales Recht)\b",
            "counterargument": r"\b(however|on the other hand|nevertheless|although|but|yet|jedoch|allerdings|hingegen|gegenansicht|opposing view|counterargument)\b",
            "method_statement": r"\b(method|methodology|approach|framework|Auslegung|grammatical|historical|systematic|teleological)\b",
            "empirical_result": r"\b(experiment|result|data|finding|accuracy|performance|statistically significant)\b",
        }

    def split_sentences(self, text: str):
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s.strip() for s in sentences if s.strip()]

    def detect_evidence_types(self, sentence: str):
        ev_types = []
        for ev_type, pattern in self.patterns.items():
            if re.search(pattern, sentence, flags=re.IGNORECASE):
                ev_types.append(ev_type)
        return ev_types

    def extract_evidence(self, text: str, section_name: str) -> List[Dict]:
        evidences = []
        sentences = self.split_sentences(text)

        for idx, sent in enumerate(sentences):
            ev_types = self.detect_evidence_types(sent)
            if not ev_types:
                continue

            evidences.append({
                "id": f"{section_name}::evidence::{idx}",
                "text": sent,
                "section": section_name,
                "sentence_idx": idx,
                "evidence_types": ev_types,
            })

        return evidences
