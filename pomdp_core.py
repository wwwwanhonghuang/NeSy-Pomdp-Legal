import numpy as np

AQ_LABELS = ["LC", "RE", "DR", "DN", "OQ"]


class ParticleFilter:
    """Particle belief state over AQ scores."""

    def __init__(self, num_particles: int = 300, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.num_particles = num_particles
        self.particles = self.rng.uniform(2.0, 5.0, size=(num_particles, 5))
        self.weights = np.ones(num_particles) / num_particles
        self.claim_memory = {}

    def _gaussian_likelihood(self, values, target, sigma=0.85):
        return np.exp(-0.5 * ((values - target) ** 2) / (sigma ** 2))

    def update_claim_memory(self, claim_states):
        self.claim_memory.update(claim_states)

    def update(self, aq_targets: dict, aq_strengths: dict = None):
        if aq_strengths is None:
            aq_strengths = {k: 1.0 for k in AQ_LABELS}

        joint = np.ones(self.num_particles)
        for idx, aq in enumerate(AQ_LABELS):
            target = aq_targets[aq]
            strength = aq_strengths.get(aq, 1.0)
            lik = self._gaussian_likelihood(self.particles[:, idx], target, sigma=0.9)
            joint *= np.power(lik + 1e-12, strength)

        self.weights *= joint
        self.weights += 1e-300
        self.weights /= self.weights.sum()

        if self.effective_sample_size() < self.num_particles / 2:
            self.resample()

    def effective_sample_size(self):
        return 1.0 / np.sum(self.weights ** 2)

    def resample(self):
        indices = self.rng.choice(self.num_particles, size=self.num_particles, p=self.weights)
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def mean(self):
        return np.average(self.particles, axis=0, weights=self.weights)

    def variance(self):
        m = self.mean()
        diff = self.particles - m
        return np.average(diff ** 2, axis=0, weights=self.weights)

    def summary(self):
        mean = self.mean()
        var = self.variance()
        return {
            aq: {"mean": float(mean[i]), "var": float(var[i])}
            for i, aq in enumerate(AQ_LABELS)
        }
