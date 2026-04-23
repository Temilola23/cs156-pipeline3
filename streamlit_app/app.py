"""
Pipeline 3 Streamlit App — Home Page
=====================================
Thesis: We model Temilola's taste as a probabilistic state over time,
using 14 complementary methods (GP, Kalman, HMM, GCN, VAE, causal, bandits)
to synthesize a 60-second generative trailer that embodies his ideal movie.
"""

import streamlit as st
import os
from pathlib import Path

st.set_page_config(page_title="CS156 Pipeline 3 — Taste Theory", layout="wide")

# Title and thesis
st.title("🎬 CS156 Pipeline 3: From Compression to Generation")

st.markdown("""
### Thesis
We model **Temilola's taste** as a probabilistic state that evolves over time.
Using 324 Streamlit-collected ratings across 100 movies, we apply 14 complementary methods
(Gaussian Processes, Kalman filtering, HMM regime-switching, relational networks, VAEs, causal inference, bandit optimization)
to synthesize an audience-of-one generative model. The output: a 60-second AI-crafted trailer
that captures the *ideal movie* embodying his taste.

**Key insight:** Taste is not a static vector; it's a trajectory. The pipeline artifacts show this journey
through uncertainty quantification, causal peer effects, and synthetic taste expansion.
""")

st.divider()

# Pipeline diagram
st.subheader("Pipeline Architecture")

diagram_path = Path("artifacts/plots/pipeline_diagram.png")
if diagram_path.exists():
    st.image(str(diagram_path), use_column_width=True, caption="ACT II → ACT III pipeline (14 methods)")
else:
    st.info("📊 Pipeline diagram (ACT II outputs → ACT III stage inputs) will be available after processing.")

st.divider()

# Page descriptions
st.subheader("📖 Explore the Analysis")

cols = st.columns(2)

with cols[0]:
    st.markdown("""
    #### 1️⃣ **Trailer Gallery**
    Placeholder gallery of ideal-movie posters and frames.

    #### 3️⃣ **HMM Regimes**
    Temilola's taste exhibits regime-switching behavior.
    Watch his rating patterns shift across latent states.
    """)

with cols[1]:
    st.markdown("""
    #### 2️⃣ **UKF Slider**
    Unscented Kalman Filter traces the latent taste state over time.
    Slide through the chronological rating sequence.

    #### 4️⃣ **GenMatch Explorer**
    Matched pairs of movies (treatment/control)
    that balance covariate similarity. Causal neighbor discovery.
    """)

st.divider()

# Artifact inventory
st.subheader("📦 Methods & Artifacts")

methods_info = {
    "1. Gaussian Processes": "Taste surface over 50K TMDB movies → posterior predictions + uncertainty",
    "2. Kalman Ladder": "Chronological rating dynamics → RTS smoother, EKF, UKF refinements",
    "3. Hidden Markov": "Viterbi path over 3 latent taste regimes",
    "4. Particle Filter": "Bootstrap resampling + FFBSi backward smoothing",
    "5. LightGCN": "User-movie bipartite graph embeddings → neighborhood precision",
    "6. HAN": "Heterogeneous network (movies ↔ genres ↔ directors)",
    "7. BNN (MC Dropout)": "Aleatoric + epistemic uncertainty quantification",
    "8. CVAE": "Taste centroid synthesis from conditional latent code",
    "9. TVAE": "Synthetic rare-taste samples → augmented training",
    "10. Bayesian ANOVA": "Genre/director/year variance decomposition",
    "11. Causal (IPW/AIPW)": "Peer effect estimation (conformal propensity overlap)",
    "12. Thompson Sampling": "Bandit exploration over GP posterior",
    "13. GenMatch": "Genetic algorithm + Mahalanobis metric for covariate balance",
    "14. Conformal Intervals": "Coverage-guaranteed prediction bands (90% nominal)",
}

col_size = 2
cols = st.columns(col_size)
for idx, (method, desc) in enumerate(methods_info.items()):
    cols[idx % col_size].write(f"**{method}**  \n{desc}")

st.divider()

st.markdown("""
---
**Data:** 324 ratings (primary hold-out) × 4 modalities (plot, poster, runtime, genre).
**Training:** MovieLens twin + GenMatch neighbors + TVAE synthetic (augmentation only).
**Evaluation integrity:** Metrics computed on 324-rating hold-out only.

Built with PyMC, scikit-learn, PyTorch, HuggingFace Diffusers, Streamlit.
""")
