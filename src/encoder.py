from sentence_transformers import SentenceTransformer
import numpy as np

class NeuralEncoder:
    def __init__(self):
        # High-performance CPU model
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

    def get_phi(self, text):
        # The Manifold Mapping M1: Text -> Phi
        return self.model.encode(text)

    def get_rhetoric_likelihood(self, phi):
        # We use the vector norm and distribution as a proxy for 
        # "Academic Fluency" (RE). 
        norm = np.linalg.norm(phi)
        # Heuristic: Professional academic writing typically has 
        # higher density in this manifold.
        return np.clip(norm / 10, 0.1, 1.0)