from sentence_transformers import SentenceTransformer
import numpy as np
import re


class NeuralEncoder:
    """
    Minimal neural encoder M1.
    Used for:
    - sentence embeddings
    - simple neural AQ proxies
    - claim/evidence semantic similarity
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode_text(self, text: str) -> np.ndarray:
        return self.model.encode(text)

    def encode_many(self, texts):
        if not texts:
            return []
        return self.model.encode(texts)

    def _sentence_count(self, text: str) -> int:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s for s in sentences if s.strip()]
        return max(1, len(sentences))

    def _avg_sentence_length(self, text: str) -> float:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s for s in sentences if s.strip()]
        if not sentences:
            return 0.0
        return float(np.mean([len(s.split()) for s in sentences]))

    def get_section_neural_signals(self, text: str) -> dict:
        """Lightweight neural AQ signals in [0,1]."""
        phi = self.encode_text(text)
        norm = np.linalg.norm(phi)
        sent_count = self._sentence_count(text)
        avg_len = self._avg_sentence_length(text)

        re_signal = np.clip(norm / 18.0, 0.0, 1.0)
        lc_shape = 1.0 - min(abs(avg_len - 22.0) / 22.0, 1.0)
        dr_signal = np.clip(sent_count / 12.0, 0.0, 1.0)
        dn_signal = 0.5
        oq_signal = np.clip((re_signal + lc_shape + dr_signal) / 3.0, 0.0, 1.0)

        return {
            "phi": phi,
            "RE": float(re_signal),
            "LC_shape": float(lc_shape),
            "DR_neural": float(dr_signal),
            "DN_neural": float(dn_signal),
            "OQ_neural": float(oq_signal),
        }

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        denom = (np.linalg.norm(vec_a) * np.linalg.norm(vec_b)) + 1e-12
        return float(np.dot(vec_a, vec_b) / denom)
