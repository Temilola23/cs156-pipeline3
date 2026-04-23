"""
Page 2: UKF Slider
==================
Kalman ladder: chronological ratings → UKF latent taste state.
Slider to animate through time steps.
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import json

st.set_page_config(page_title="UKF Slider", layout="wide")

st.title("⏱️ Taste Evolution: Unscented Kalman Filter")

st.markdown("""
The **Kalman ladder** models Temilola's rating process as a hidden state (latent taste)
that evolves over time and is observed through noisy ratings.

**Steps:** Chronological ratings → Kalman filter → RTS backward smoother → UKF refinement.

This page animates the **UKF latent state** as you slide through time.
""")

st.divider()

# Load UKF data
@st.cache_data
def load_ukf_data():
    """Load UKF latent states."""
    try:
        ukf_data = np.load("artifacts/ukf_latent.npz", allow_pickle=True)
        return ukf_data
    except FileNotFoundError:
        return None

@st.cache_data
def load_ratings():
    """Load ratings data for timestamps."""
    try:
        ratings_df = pd.read_parquet("data/ml-latest-small/ratings.csv")
        return ratings_df
    except:
        try:
            # Fallback: check if ratings exist in artifacts
            ratings_df = pd.read_parquet("artifacts/movielens_twin_ratings.parquet")
            return ratings_df
        except:
            return None

ukf_data = load_ukf_data()
ratings_df = load_ratings()

if ukf_data is not None:
    st.subheader("📈 Latent State Trace")

    # Extract arrays from npz
    keys = list(ukf_data.keys())
    st.write(f"Available arrays: {', '.join(keys)}")

    # Typically: 'mean', 'var', 'timestamps', etc.
    if 'mean' in ukf_data:
        means = ukf_data['mean']
        n_steps = len(means) if hasattr(means, '__len__') else 1

        st.write(f"**UKF trace:** {n_steps} time steps")

        # Time slider
        time_idx = st.slider(
            "Time step:",
            min_value=0,
            max_value=max(0, n_steps - 1),
            value=0,
            step=1,
        )

        # Display current state
        col1, col2 = st.columns(2)

        with col1:
            st.metric(
                "Latent State (Mean)",
                f"{means[time_idx]:.4f}" if hasattr(means[time_idx], '__float__') else str(means[time_idx]),
            )

        if 'var' in ukf_data:
            vars_arr = ukf_data['var']
            with col2:
                st.metric(
                    "Uncertainty (Variance)",
                    f"{vars_arr[time_idx]:.4f}" if hasattr(vars_arr[time_idx], '__float__') else str(vars_arr[time_idx]),
                )

        # Show full UKF plot if available
        st.write("")
        plot_path = Path("artifacts/plots/kalman_ladder.png")
        if plot_path.exists():
            st.image(str(plot_path), use_column_width=True, caption="Full Kalman ladder (Kalman → RTS → UKF)")
        else:
            st.warning("⚠️ Full Kalman ladder plot not yet rendered.")
    else:
        st.warning("⚠️ UKF arrays not in expected format. Available keys: " + ", ".join(keys))

else:
    st.info("📊 UKF latent trace not yet computed. Running: `scripts/11_fit_kalman_ladder.py`")

    # Fallback: show static plot
    plot_path = Path("artifacts/plots/kalman_ladder.png")
    if plot_path.exists():
        st.image(str(plot_path), use_column_width=True, caption="Kalman Filter Ladder (static)")
    else:
        st.warning("No Kalman plot available yet.")

st.divider()

st.markdown("""
### 🔬 Kalman Ladder Explanation

1. **Forward Kalman:** Predicts $\\hat{x}_t$ given observations $y_1, \\ldots, y_t$.
2. **RTS Smoother:** Backward pass refines using future observations $y_{t+1}, \\ldots, y_T$.
3. **EKF/UKF:** Extended and Unscented variants handle nonlinear observation models.

**Output:** Latent taste state at each rating, with epistemic uncertainty bands.
This state conditions the downstream HMM, causal inference, and generative models.
""")
