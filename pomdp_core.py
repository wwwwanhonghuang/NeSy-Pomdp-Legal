import numpy as np
from typing import Dict, Any, Optional, List

AQ_LABELS = ["LC", "RE", "DR", "DN", "OQ"]


class ParticleFilter:
    """
    Belief over latent brief hypotheses.
    For now each particle keeps:
    - AQ latent vector
    - a lightweight formal-validity modifier
    - unresolved-count proxy

    This is still approximate, but now the interpretation is correct:
    particles represent latent evaluative states, not raw observations.
    """

    def __init__(self, num_particles: int = 300, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.num_particles = num_particles

        self.particles = self.rng.uniform(2.0, 5.0, size=(num_particles, 5))
        self.validity_particles = self.rng.uniform(0.4, 0.9, size=(num_particles,))
        self.unresolved_particles = self.rng.integers(1, 6, size=(num_particles,))

        self.weights = np.ones(num_particles) / num_particles

        self.last_observation = None
        self.last_action = None

    def build_observation(
        self,
        observation_graph: Dict[str, Any],
        latent_state_mean: Dict[str, Any],
        fused_aq: Dict[str, float],
        formal_validation_summary: Dict[str, Any],
    ) -> Dict[str, Any]:
        obs = {
            "observation_graph": observation_graph,
            "latent_state_mean": latent_state_mean,
            "fused_aq": fused_aq,
            "formal_validation_summary": formal_validation_summary,
        }
        self.last_observation = obs
        return obs

    def update_from_observation(
        self,
        observation: Dict[str, Any],
        action: Optional[str] = None,
        aq_strengths: Optional[Dict[str, float]] = None,
    ):
        self.last_action = action

        fused_aq = observation["fused_aq"]
        formal_summary = observation["formal_validation_summary"]
        latent_state_mean = observation["latent_state_mean"]

        if aq_strengths is None:
            aq_strengths = {k: 1.0 for k in AQ_LABELS}

        target_validity = formal_summary.get("mean_formal_validity", 0.8)
        target_unresolved = len(latent_state_mean.get("unresolved_claim_ids", []))

        joint = np.ones(self.num_particles)

        for idx, aq in enumerate(AQ_LABELS):
            target = 1.0 + 5.0 * fused_aq[aq] if fused_aq[aq] <= 1.0 else fused_aq[aq]
            strength = aq_strengths.get(aq, 1.0)
            lik = self._gaussian_likelihood(self.particles[:, idx], target, sigma=0.9)
            joint *= np.power(lik + 1e-12, strength)

        validity_lik = self._gaussian_likelihood(self.validity_particles, target_validity, sigma=0.15)
        unresolved_lik = self._gaussian_likelihood(self.unresolved_particles, target_unresolved, sigma=1.0)

        joint *= validity_lik
        joint *= unresolved_lik

        self.weights *= joint
        self.weights += 1e-300
        self.weights /= self.weights.sum()

        if self.effective_sample_size() < self.num_particles / 2:
            self.resample()

    def _gaussian_likelihood(self, values, target, sigma=0.85):
        return np.exp(-0.5 * ((values - target) ** 2) / (sigma ** 2))

    def effective_sample_size(self):
        return 1.0 / np.sum(self.weights ** 2)

    def resample(self):
        indices = self.rng.choice(self.num_particles, size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.validity_particles = self.validity_particles[indices]
        self.unresolved_particles = self.unresolved_particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def summary(self):
        mean = np.average(self.particles, axis=0, weights=self.weights)
        var = np.average((self.particles - mean) ** 2, axis=0, weights=self.weights)
        std = np.sqrt(var)

        return {
            aq: {
                "mean": float(mean[i]),
                "var": float(var[i]),
                "std": float(std[i]),
                "ci_low": float(mean[i] - 1.96 * std[i]),
                "ci_high": float(mean[i] + 1.96 * std[i]),
            }
            for i, aq in enumerate(AQ_LABELS)
        }

    def latent_meta_summary(self):
        validity_mean = float(np.average(self.validity_particles, weights=self.weights))
        unresolved_mean = float(np.average(self.unresolved_particles, weights=self.weights))
        return {
            "mean_formal_validity_latent": validity_mean,
            "mean_unresolved_claims_latent": unresolved_mean,
        }