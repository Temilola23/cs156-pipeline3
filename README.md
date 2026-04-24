# CS156 Pipeline 3 — "From Compression to Generation"

## Thesis

This project builds a probabilistic, causal, relational, and generative theory of Temilola's taste through 324 hand-elicited ratings across 100 films in 4 modalities, synthesized via 14–15 machine learning methods spanning dimensionality reduction, causal inference, heterogeneous networks, and generative models (ELICIT → MODEL → GENERATE), culminating in an original AI-generated movie trailer for the ideal film.

## How to Run

### Interactive Streamlit App
```bash
pip install -r requirements.txt
streamlit run streamlit_app/app.py
```

The app provides:
- **Trailer Gallery**: Explore trailers and recommendations
- **UKF Slider**: Interactive unscented Kalman filter timeline
- **HMM Regimes**: Hidden Markov model mood phase segmentation
- **GenMatch Explorer**: Causal neighbor matching visualization

### Full Pipeline (Laptop)
```bash
# Data augmentation pillars
python scripts/02_genmatch_expand.py      # GenMatch causal-balanced neighbours
python scripts/14_movielens_twin.py       # MovieLens 25M twin (~2 hours)
python scripts/15_tvae_synth.py           # TVAE rare-taste synthesis

# Model family fits (see scripts/ for full numbered sequence 10-22)
python scripts/10_fit_gp.py
python scripts/13_fit_hmm.py
python scripts/16_fit_lightgcn.py
python scripts/19_fit_hierarchical_anova.py
python scripts/20_causal_ipw_aipw.py
python scripts/20_conformal_wrap.py

# Export notebook to PDF
jupyter nbconvert --to pdf notebooks/main_submission.ipynb
```

## Repository Layout

```
Pipeline 3/
├── src/
│   ├── tvae.py                 # Variational autoencoders
│   ├── lightgcn.py             # Graph collaborative filtering
│   ├── han.py                  # Heterogeneous attention networks
│   ├── bnn_mcd.py              # Bayesian neural networks
│   ├── cvae.py                 # Conditional VAE generation
│   ├── thompson_gp.py          # Thompson sampling
│   ├── causal.py               # Causal discovery & DAG
│   ├── hierarchical_anova.py    # ANOVA decomposition
│   ├── hmm.py                  # Hidden Markov models
│   ├── kalman.py               # Kalman filters (UKF)
│   ├── genmatch.py             # Diamond & Sekhon GenMatch
│   └── ...
├── scripts/
│   ├── 02_genmatch_expand.py
│   ├── 10_fit_gp.py ... 22_fit_cvae.py
│   ├── ... (full pipeline scripts)
├── notebooks/
│   ├── main_submission.pdf     (<50 pp main narrative)
│   └── *.ipynb                 (source notebooks)
├── streamlit_app/
│   ├── app.py                  (main landing page)
│   └── pages/
│       ├── 1_Trailer_Gallery.py
│       ├── 2_UKF_Slider.py
│       ├── 3_HMM_Regimes.py
│       └── 4_GenMatch_Explorer.py
├── artifacts/
│   ├── plots/                  (static visualizations)
│   └── ...
├── derivations/                (intermediate outputs)
└── data/                        (raw & processed)
```

## Submission Artifacts

| File | Purpose |
|------|---------|
| [`notebooks/main_submission.pdf`](notebooks/main_submission.pdf) | <50 pp condensed narrative (Methods, Results, Conclusion) |

## Status

- **Phase 1A**: Complete (16 laptop methods: HMM, Kalman, BNN, GP, hierarchical ANOVA, causal discovery, LightGCN, HAN, CVAE, etc.)
- **Phase 1B**: Pending GPU compute (SDXL fine-tuning, Llama 2 caption generation, AnimateDiff video synthesis)
- **Phase 2**: ACT III trailer assembly (storyboard → motion LoRA keyframing)
- **Phase 3**: Final `ideal_temilola_movie.mp4` render

## Data Provenance

| Source | Quantity | Usage |
|--------|----------|-------|
| Streamlit-collected ratings | 324 (100 movies × 4 modalities) | Primary evaluation |
| MovieLens 25M twin | ~50K pseudo-ratings | Training only |
| GenMatch neighbors | 500 causal matches | Trailer corpus expansion |
| TVAE synthetic samples | Rare-taste coverage | Training only |

Evaluation metrics use only the 324 ground-truth ratings.
