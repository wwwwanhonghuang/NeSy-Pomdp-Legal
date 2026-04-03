from sentence_transformers import SentenceTransformer
import numpy as np
import re


class NeuralEncoder:
    """
    Minimal neural layer M1.
    CPU-friendly sentence embedding encoder + simple AQ-related neural proxies.
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def encode_component(self, text: str) -> np.ndarray:
        """Map text -> phi."""
        return self.model.encode(text)

    def _sentence_count(self, text: str) -> int:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s for s in sentences if len(s.strip()) > 0]
        return max(1, len(sentences))

    def _avg_sentence_length(self, text: str) -> float:
        sentences = re.split(r"(?<=[.!?])\s+", text.strip())
        sentences = [s for s in sentences if len(s.strip()) > 0]
        if not sentences:
            return 0.0
        lengths = [len(s.split()) for s in sentences]
        return float(np.mean(lengths))

    def get_aq_signals(self, text: str) -> dict:
        """
        Produce lightweight neural AQ signals in [0,1].
        These are only demo proxies, not true learned AQ heads.
        """
        phi = self.encode_component(text)

        norm = np.linalg.norm(phi)
        sent_count = self._sentence_count(text)
        avg_sent_len = self._avg_sentence_length(text)

        # Proxy: denser / more structured text tends to look more "academic"
        re_signal = np.clip(norm / 18.0, 0.0, 1.0)

        # Proxy: moderate sentence length is often preferable to very short/fragmented
        lc_shape_signal = 1.0 - min(abs(avg_sent_len - 22.0) / 22.0, 1.0)

        # Proxy: longer sections may provide more room for dialectical development
        dr_signal = np.clip(sent_count / 12.0, 0.0, 1.0)

        # Proxy: deliberative norms not strongly neural here, keep weak neutral prior
        dn_signal = 0.5

        # OQ is not directly observed; keep as weak aggregate proxy
        oq_signal = np.clip((re_signal + lc_shape_signal + dr_signal) / 3.0, 0.0, 1.0)

        return {
            "phi": phi,
            "RE": float(re_signal),
            "LC_shape": float(lc_shape_signal),
            "DR_neural": float(dr_signal),
            "DN_neural": float(dn_signal),
            "OQ_neural": float(oq_signal),
        }