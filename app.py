import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.encoder import NeuralEncoder
from src.logic_engine import SymbolicGrounding
from src.pomdp_core import ParticleBrief

st.set_page_config(page_title="NeSy-POMDP Thesis Eval", layout="wide")

# Persistent State
if 'agent' not in st.session_state:
    st.session_state.encoder = NeuralEncoder()
    st.session_state.grounder = SymbolicGrounding()
    st.session_state.brief = ParticleBrief(num_particles=1000)
    st.session_state.step = 0

st.title("⚖️ Neuro-Symbolic Thesis Evaluation")
st.markdown("### Sequential Paragraph-by-Paragraph Assessment")

# Input: The Thesis
thesis_input = st.text_area("Paste Thesis Text (separate paragraphs with double newlines)", 
                            height=200, placeholder="Enter legal text here...")

paragraphs = [p.strip() for p in thesis_input.split('\n\n') if p.strip()]

if st.button("Process Next Paragraph") and st.session_state.step < len(paragraphs):
    p_text = paragraphs[st.session_state.step]
    
    # 1. Neural Manifold (M1)
    phi = st.session_state.encoder.get_phi(p_text)
    n_lik = st.session_state.encoder.get_rhetoric_likelihood(phi)
    
    # 2. Symbolic Grounding (G)
    s_lik = st.session_state.grounder.get_logic_likelihood(p_text)
    
    # 3. POMDP Update
    st.session_state.brief.recursive_filter(n_lik, s_lik)
    st.session_state.step += 1

# --- Layout ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Current Evidence")
    if st.session_state.step > 0:
        st.info(f"**Last Read:** {paragraphs[st.session_state.step-1]}")
    else:
        st.write("Awaiting first paragraph...")

with col2:
    st.subheader("AQ State (POMDP Belief)")
    means = np.mean(st.session_state.brief.particles, axis=0)
    vars = st.session_state.brief.get_uncertainty()
    
    labels = ["Logic (LC)", "Rhetoric (RE)", "Dialectic (DR)", "Norms (DN)", "Overall (OQ)"]
    df = pd.DataFrame({'Category': labels, 'Mean Score': means, 'Uncertainty': vars})
    st.dataframe(df.style.highlight_max(axis=0))

    # Visualization of the Bimodal Fix
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(st.session_state.brief.particles[:, 0], bins=30, alpha=0.5, label="Logical Cogency")
    ax.hist(st.session_state.brief.particles[:, 1], bins=30, alpha=0.5, label="Rhetorical Effectiveness")
    ax.set_title("Multi-Hypothesis Particle Distribution")
    ax.legend()
    st.pyplot(fig)

# Active Inquiry Action
if st.session_state.step < len(paragraphs):
    uncertainty = np.max(vars)
    if uncertainty > 0.5:
        st.warning(f"⚠️ High Uncertainty Detected ({uncertainty:.2f}). Strategic Action: Request Detailed Citation Check.")
    else:
        st.success("Belief Converging. Action: Continue Sequential Reading.")