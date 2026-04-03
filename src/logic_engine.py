import re
import numpy as np


class SymbolicGrounding:
    """
    Minimal symbolic grounding layer G.
    Produces AQ-specific symbolic evidence scores.
    """

    def __init__(self):
        self.patterns = {
            "statute": r"\b(OR|ZGB|StGB|BV)\s+(Art\.|Artikel)\s*\d+[a-zA-Z]*",
            "bge": r"\bBGE\s+\d+\s+[IVX]+\s+\d+",
            "doctrine": r"\b([A-ZÄÖÜ][a-zäöüß]+,\s*\d{4}|[A-ZÄÖÜ][a-zäöüß]+\s+et al\.)\b",
            "counterarg": r"\b(however|however,|on the other hand|nevertheless|yet|although|but|jedoch|allerdings|hingegen|gegenansicht)\b",
            "comparative": r"\b(comparative|international|foreign law|EU law|European law|vergleichend|internationales Recht)\b",
            "conclusion": r"\b(therefore|thus|hence|in conclusion|daher|somit|folglich)\b",
            "method": r"\b(method|methodology|approach|framework|Auslegung|teleological|systematic|historical|grammatical)\b",
            "citation_marker": r"(\(\w+,\s*\d{4}\)|\[\d+\]|footnote|\bFn\.\s*\d+)",
        }

    def _count(self, key: str, text: str) -> int:
        return len(re.findall(self.patterns[key], text, flags=re.IGNORECASE))

    def _sentences(self, text: str):
        sents = re.split(r"(?<=[.!?])\s+", text.strip())
        return [s for s in sents if len(s.strip()) > 0]

    def _statement_count(self, text: str) -> int:
        return max(1, len([s for s in self._sentences(text) if len(s.split()) >= 6]))

    def _grounding_ratio(self, text: str) -> float:
        statements = self._statement_count(text)
        anchors = (
            self._count("statute", text)
            + self._count("bge", text)
            + self._count("doctrine", text)
            + self._count("citation_marker", text)
        )
        ratio = anchors / max(1.0, statements / 2.0)
        return float(np.clip(ratio, 0.0, 1.0))

    def ground_component(self, text: str) -> dict:
        """
        AQ-specific symbolic scores in [0,1].
        """
        statute_n = self._count("statute", text)
        bge_n = self._count("bge", text)
        doctrine_n = self._count("doctrine", text)
        counterarg_n = self._count("counterarg", text)
        comparative_n = self._count("comparative", text)
        conclusion_n = self._count("conclusion", text)
        method_n = self._count("method", text)

        grounding_ratio = self._grounding_ratio(text)

        # LC: groundedness + legal support + explicit method
        lc = np.clip(
            0.40 * grounding_ratio
            + 0.25 * min(statute_n / 3.0, 1.0)
            + 0.20 * min(bge_n / 2.0, 1.0)
            + 0.15 * min(method_n / 3.0, 1.0),
            0.0, 1.0
        )

        # RE: symbolic structure markers only lightly
        re_score = np.clip(
            0.50 * min(conclusion_n / 2.0, 1.0)
            + 0.50 * min(doctrine_n / 3.0, 1.0),
            0.0, 1.0
        )

        # DR: counterarguments + case law engagement
        dr = np.clip(
            0.65 * min(counterarg_n / 3.0, 1.0)
            + 0.35 * min(bge_n / 2.0, 1.0),
            0.0, 1.0
        )

        # DN: comparative / alternative positions / broader deliberative engagement
        dn = np.clip(
            0.70 * min(comparative_n / 2.0, 1.0)
            + 0.30 * min(counterarg_n / 3.0, 1.0),
            0.0, 1.0
        )

        # OQ: do not score directly too aggressively
        oq = np.clip((lc + re_score + dr + dn) / 4.0, 0.0, 1.0)

        return {
            "LC": float(lc),
            "RE": float(re_score),
            "DR": float(dr),
            "DN": float(dn),
            "OQ": float(oq),
            "meta": {
                "statute_n": statute_n,
                "bge_n": bge_n,
                "doctrine_n": doctrine_n,
                "counterarg_n": counterarg_n,
                "comparative_n": comparative_n,
                "method_n": method_n,
                "grounding_ratio": grounding_ratio,
            }
        }