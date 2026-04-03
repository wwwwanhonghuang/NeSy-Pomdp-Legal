import os
import numpy as np
from encoder import NeuralEncoder
from logic_engine import SymbolicGrounding
from pomdp_core import ParticleFilter

def run_evaluation_demo():
    # 1. Initialize Layers
    encoder = NeuralEncoder()
    grounder = SymbolicGrounding()
    pf = ParticleFilter(num_particles=200)
    
    data_dir = "data/"
    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".txt")])
    
    print(f"--- Starting NeSy-POMDP Evaluation ---")
    print(f"Initial Belief Mean (AQ Scores): {np.mean(pf.particles, axis=0)}")
    
    for t, filename in enumerate(files):
        with open(os.path.join(data_dir, filename), 'r') as f:
            text = f.read()
        
        print(f"\n[Step {t+1}] Reading: {filename}")
        
        # --- NEURAL LAYER (M1) ---
        # In a real model, this would be a classifier head. 
        # For the demo, we use the vector length/variance as a proxy for 'richness'.
        phi = encoder.encode_component(text)
        neural_quality_proxy = np.clip(np.linalg.norm(phi) / 10, 1, 6)
        
        # --- SYMBOLIC LAYER (G) ---
        symbolic_score = grounder.ground_component(text)
        
        # --- LIKELIHOOD CALCULATION ---
        # Particles close to the 'Neural Sense' AND the 'Symbolic Fact' get higher weights.
        # We target 'Logical Cogency' (Index 0) and 'Rhetorical Effectiveness' (Index 1).
        
        # Likelihood = How well does the particle's 'hypothesis' fit the evidence?
        # If symbolic_score is high, particles with high LC (index 0) are more likely.
        target_lc = 1.0 + (symbolic_score * 5.0) 
        
        # Calculate Gaussian Likelihood for each particle
        # particles[:, 0] is the LC score hypothesis for every particle
        likelihoods = np.exp(-0.5 * ((pf.particles[:, 0] - target_lc)**2) / 1.0)
        
        # --- POMDP UPDATE ---
        pf.update(likelihoods)
        
        current_mean = np.mean(pf.particles, axis=0)
        print(f"Update complete. Symbolic Evidence: {symbolic_score:.2f}")
        print(f"Current Estimated LC Score: {current_mean[0]:.2f}")
        print(f"Current Estimated RE Score: {current_mean[1]:.2f}")

    print("\n--- Final Assessment ---")
    final_scores = np.mean(pf.particles, axis=0)
    labels = ["LC", "RE", "DR", "DN", "OQ"]
    for label, score in zip(labels, final_scores):
        print(f"{label}: {score:.2f}")

if __name__ == "__main__":
    run_evaluation_demo()