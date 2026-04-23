"""
Page 4: GenMatch Explorer
==========================
Causal matching via genetic algorithm + Mahalanobis metric.
Displays matched pairs (treatment/control) with covariate balance.
"""

import streamlit as st
import numpy as np
import pandas as pd
from pathlib import Path
import json

st.set_page_config(page_title="GenMatch Explorer", layout="wide")

st.title("🔗 Causal Matching: GenMatch Explorer")

st.markdown("""
**Goal:** Expand the 100-movie corpus to ~500 candidates via **genetic algorithm matching**.

GenMatch uses Mahalanobis distance + genetic optimization to find movies
that are *similar* (balances covariates like genre, year, runtime)
but not in the original 100. This expansion feeds:
- **Trailer corpus** (more scenes to choose from)
- **Augmentation data** (training, but not eval)
""")

st.divider()

# Load GenMatch neighbors
@st.cache_data
def load_genmatch_neighbors():
    """Load matched neighbors from GenMatch."""
    try:
        with open("artifacts/genmatch_neighbors.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return None

neighbors_data = load_genmatch_neighbors()

if neighbors_data is not None:
    st.subheader("📋 Matched Pairs Table")

    # Convert to dataframe
    pairs_list = []
    if isinstance(neighbors_data, dict):
        # If it's a dict, extract pairs
        for movie_id, matches in neighbors_data.items():
            if isinstance(matches, list):
                for match in matches[:3]:  # Show top 3 matches per movie
                    pairs_list.append({
                        "Original Movie": str(movie_id),
                        "Matched Movie": str(match) if isinstance(match, (int, str)) else str(match.get("id", "?")),
                        "Distance": match.get("distance", "N/A") if isinstance(match, dict) else "N/A",
                    })
    elif isinstance(neighbors_data, list):
        # If it's already a list of pairs
        for pair in neighbors_data[:100]:
            pairs_list.append({
                "Original Movie": pair.get("original", pair.get("treatment", "?")),
                "Matched Movie": pair.get("neighbor", pair.get("control", "?")),
                "Distance": pair.get("distance", pair.get("mahalanobis", "N/A")),
            })

    if pairs_list:
        pairs_df = pd.DataFrame(pairs_list)
        st.dataframe(pairs_df, use_container_width=True, height=300)
        st.write(f"✓ Showing top {len(pairs_df)} matched pairs (of ~500 total).")
    else:
        st.warning("⚠️ GenMatch neighbors loaded but format unclear. Raw data keys: " + str(list(neighbors_data.keys())[:5] if isinstance(neighbors_data, dict) else "list"))

else:
    st.info("📊 GenMatch neighbors not yet computed. Running: `scripts/02_genmatch_expand.py`")

st.divider()

# Covariate balance plots
st.subheader("⚖️ Covariate Balance Before/After")

balance_before_path = Path("artifacts/plots/propensity_overlap.png")
balance_after_path = Path("artifacts/plots/genmatch_fitness.png")

col1, col2 = st.columns(2)

with col1:
    if balance_before_path.exists():
        st.image(str(balance_before_path), use_column_width=True, caption="Propensity overlap (unmatched)")
    else:
        st.warning("⚠️ Propensity overlap plot not found.")

with col2:
    if balance_after_path.exists():
        st.image(str(balance_after_path), use_column_width=True, caption="GenMatch fitness improvement")
    else:
        st.warning("⚠️ GenMatch fitness plot not found.")

st.divider()

st.markdown("""
### 🧬 GenMatch Methodology

1. **Genetic Algorithm:** Evolve distance metric weights to maximize covariate balance
2. **Mahalanobis Distance:** Flexible metric (inverse covariance-weighted) vs. fixed Euclidean
3. **Covariates:** Genre, release year, runtime, director popularity, vote count
4. **Output:** ~500 matched neighbors to original 100 movies

**Causal interpretation:**
If we treat the original 100 as "treatment" and matches as "control",
we can estimate *peer effects* or *taste contamination*
(e.g., "watching movies similar to my favorite changes my ratings").

See `scripts/02_genmatch_expand.py` for the GA fitness function
and `scripts/19_causal_ipw_aipw.py` for downstream IPW/AIPW estimation.
""")

st.info("""
**Why GenMatch?**
Standard matching (nearest neighbor in Euclidean space) can fail if covariates
are correlated or high-dimensional. GenMatch learns the *right* metric,
making balance maximization interpretable.
""")
