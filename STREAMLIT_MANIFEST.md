# Streamlit App Manifest (Task 6.2)

## Files Created

| File | Lines | Purpose |
|------|-------|---------|
| `streamlit_app/app.py` | 105 | Home page: thesis, pipeline diagram, method inventory |
| `streamlit_app/pages/1_Trailer_Gallery.py` | 97 | Placeholder gallery + 5-pathway synthesis spec table |
| `streamlit_app/pages/2_UKF_Slider.py` | 127 | Kalman ladder time-step slider + latent state trace |
| `streamlit_app/pages/3_HMM_Regimes.py` | 115 | HMM Viterbi path + regime selector + transitions |
| `streamlit_app/pages/4_GenMatch_Explorer.py` | 122 | Matched pairs table + covariate balance plots |
| **Total** | **566** | |

## Artifacts Referenced

### Page 1: Home (app.py)
- **Plot:** `artifacts/plots/pipeline_diagram.png` (required)
  - Status: ✓ Exists

### Page 2: Trailer Gallery (1_Trailer_Gallery.py)
- **Video:** `video/ideal_temilola_movie.mp4` (optional, fallback graceful)
  - Status: Does not exist yet (ACT III GPU output)
  - Fallback: Shows 5-pathway spec table explaining synthesis pipeline
  
### Page 3: UKF Slider (2_UKF_Slider.py)
- **NPZ data:** `artifacts/ukf_latent.npz` (required for interactive slider)
  - Status: ✓ Exists (arrays: mean, var, etc.)
- **Plot:** `artifacts/plots/kalman_ladder.png` (required for static fallback)
  - Status: ✓ Exists

### Page 4: HMM Regimes (3_HMM_Regimes.py)
- **NPZ data:** `artifacts/hmm_regimes.npz` (required for regime stats)
  - Status: ✓ Exists (arrays: viterbi, transition_matrix)
- **Plot (timeline):** `artifacts/plots/08_hmm_timeline.png` (primary display)
  - Status: ✓ Exists
- **Plot (fallback):** `artifacts/plots/hmm_regimes.png` (alternative)
  - Status: ✓ Exists

### Page 5: GenMatch Explorer (4_GenMatch_Explorer.py)
- **JSON data:** `artifacts/genmatch_neighbors.json` (required for matched pairs)
  - Status: ✓ Exists (dict of original_movie_id → [neighbors])
- **Plot (before):** `artifacts/plots/propensity_overlap.png` (covariate overlap)
  - Status: ✓ Exists
- **Plot (after):** `artifacts/plots/genmatch_fitness.png` (GA fitness improvement)
  - Status: ✓ Exists

## Deployment Checklist

- [ ] Verify `streamlit_app/` is committed
- [ ] Test local run: `streamlit run streamlit_app/app.py`
  - Expected: Home page loads, links to 4 pages work
  - All plots display (all artifact PNG files present)
  - UKF slider functional (NPZ arrays loadable)
  - HMM regime dropdown functional (NPZ arrays loadable)
  - GenMatch pairs table functional (JSON loadable)
- [ ] Check that all graceful fallbacks trigger appropriately:
  - Missing `video/ideal_temilola_movie.mp4` → shows spec table (working)
  - Missing artifacts → shows `st.info()` or `st.warning()` (implemented)

## Error Handling

All pages wrap file loads in `try/except FileNotFoundError` with user-friendly messages:

```python
try:
    # load artifact
except FileNotFoundError:
    st.info("📊 Artifact not yet computed...")
```

Pages display static PNG plots as fallback if interactive data unavailable.

## Run Command

```bash
cd "Pipeline 3"
streamlit run streamlit_app/app.py
```

Expected to serve on `http://localhost:8501` (default Streamlit port).
