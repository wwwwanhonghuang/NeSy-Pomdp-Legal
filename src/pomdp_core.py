import numpy as np

class ParticleBrief:
    def __init__(self, num_particles=500):
        # State: [LC, RE, DR, DN, OQ]
        # We initialize with a wide distribution (Presumption of Merit)
        self.particles = np.random.uniform(2.0, 5.0, (num_particles, 5))
        self.weights = np.ones(num_particles) / num_particles

    def recursive_filter(self, neural_lik, symbolic_lik):
        # Map observations to AQ categories
        # RE is driven by Neural (phi); LC is driven by Symbolic (G)
        
        # Likelihood for Logical Cogency (Index 0)
        target_lc = 1.0 + (symbolic_lik * 5.0)
        lc_lik = np.exp(-0.5 * ((self.particles[:, 0] - target_lc)**2) / 0.5)

        # Likelihood for Rhetorical Effectiveness (Index 1)
        target_re = 1.0 + (neural_lik * 5.0)
        re_lik = np.exp(-0.5 * ((self.particles[:, 1] - target_re)**2) / 0.5)

        # Update weights (Joint Likelihood)
        self.weights *= (lc_lik * re_lik)
        self.weights += 1e-300
        self.weights /= self.weights.sum()

        # Resample if effective sample size is low (prevent collapse)
        if 1.0 / np.sum(np.square(self.weights)) < len(self.particles) / 2:
            indices = np.random.choice(len(self.particles), len(self.particles), p=self.weights)
            self.particles = self.particles[indices]
            self.weights = np.ones(len(self.particles)) / len(self.particles)

    def get_uncertainty(self):
        return np.var(self.particles, axis=0)