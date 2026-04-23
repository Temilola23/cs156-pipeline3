# CS156 Pipeline 3 — "From Compression to Generation"

A probabilistic, causal, relational, and generative theory of Temilola's taste — ending in a 60-second AI-generated trailer for the ideal movie.

## Deliverables
- `notebooks/main_submission.ipynb` → PDF submitted to Minerva
- `notebooks/full_deep_dive.ipynb` → 200+pp exhaustive deep-dive (PDF on this repo)
- `video/ideal_temilola_movie.mp4` → the 60-second trailer
- `streamlit_app/` → interactive explainer (UKF slider, HMM regimes, GenMatch, motion LoRA timeline)

## Reproducibility
1. `pip install -r requirements.txt`
2. `python scripts/01_movielens_twin.py` (~2h laptop)
3. ... (full sequence in `RUN_ORDER.md`)
4. `jupyter nbconvert --to pdf notebooks/main_submission.ipynb`

## Data
- 324 Streamlit-collected ratings (100 movies × 4 modalities) — primary
- MovieLens 25M twin expansion (~50K pseudo-ratings) — training only
- GenMatch 500 neighbors — trailer corpus expansion
- TVAE synthetic rare-taste samples — training only

Eval uses the 324 only. See `PLAN.md` §Data provenance.
