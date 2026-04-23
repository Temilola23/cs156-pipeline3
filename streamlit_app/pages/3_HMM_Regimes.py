"""
Page 3: HMM Regimes
===================
Hidden Markov Model over rating sequence.
Viterbi path reveals 3 latent taste regimes (Exploratory, Stable, Discerning).
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path

st.set_page_config(page_title="HMM Regimes", layout="wide")

st.title("🔄 Taste Regimes: Hidden Markov Model")

st.markdown("""
**Hypothesis:** Temilola's taste doesn't flow linearly; it exhibits **regime-switching**.

The HMM discovers 3 latent regimes (e.g., Exploratory → Stable → Discerning)
and assigns each rating to its most likely regime via the **Viterbi path**.

This page shows the regime sequence and lets you highlight a specific regime.
""")

st.divider()

# Load HMM data
@st.cache_data
def load_hmm_data():
    """Load HMM regimes (Viterbi path)."""
    try:
        hmm_data = np.load("artifacts/hmm_regimes.npz", allow_pickle=True)
        return hmm_data
    except FileNotFoundError:
        return None

hmm_data = load_hmm_data()

# Show HMM timeline plot
plot_path = Path("artifacts/plots/08_hmm_timeline.png")
hmm_plot_path = Path("artifacts/plots/hmm_regimes.png")

if hmm_plot_path.exists() or plot_path.exists():
    st.subheader("📊 HMM Timeline")

    # Try to display the timeline plot first
    if plot_path.exists():
        st.image(str(plot_path), use_column_width=True, caption="HMM regimes over rating sequence (timeline)")
    elif hmm_plot_path.exists():
        st.image(str(hmm_plot_path), use_column_width=True, caption="HMM regimes visualization")

    st.success("✓ HMM Viterbi path computed.")
else:
    st.info("📊 HMM regime plots not yet rendered.")

# Interactive regime selector
st.subheader("🎯 Regime Spotlight")

regimes = {
    0: {"name": "Exploratory", "desc": "High variance, sampling broadly across genres"},
    1: {"name": "Stable", "desc": "Consistent preferences, narrower range"},
    2: {"name": "Discerning", "desc": "Low variance, refined taste, critical ratings"},
}

regime_choice = st.selectbox(
    "Select a regime to highlight:",
    [0, 1, 2],
    format_func=lambda x: f"Regime {x}: {regimes[x]['name']}",
)

st.write(f"**{regimes[regime_choice]['name']}**  \n{regimes[regime_choice]['desc']}")

if hmm_data is not None:
    # Check for Viterbi path in data
    if 'viterbi' in hmm_data:
        viterbi_path = hmm_data['viterbi']
        regime_counts = np.bincount(viterbi_path.astype(int), minlength=3)
        st.metric(
            f"Ratings in {regimes[regime_choice]['name']} regime",
            int(regime_counts[regime_choice]),
        )

        # Show transition matrix if available
        if 'transition_matrix' in hmm_data:
            trans_mat = hmm_data['transition_matrix']
            st.write("**Regime transition probabilities:**")
            trans_df = pd.DataFrame(
                trans_mat,
                columns=[f"To {r}" for r in range(3)],
                index=[f"From {r}" for r in range(3)],
            )
            st.dataframe(trans_df, use_container_width=True)

st.divider()

st.markdown("""
### 📚 HMM Methodology

- **States:** 3 latent regimes (discovered from data)
- **Observations:** Rating values (1–5) at each timestamp
- **Inference:** Forward-backward algorithm + Viterbi path
- **Output:** Regime assignment for each rating + transition probabilities

**Interpretation:**
The switching pattern reveals when Temilola's taste *changes character*.
This informs the generative model: scenes should reflect the dominant regime
or show regime transitions (plot turning points).
""")

st.info("""
**Why 3 regimes?**
Optimal trade-off between expressiveness and identifiability (BIC penalty).
See `scripts/13_fit_hmm.py` for grid search over 2–5 states.
""")
