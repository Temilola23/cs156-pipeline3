# CS156 Pipeline 3 — Final Report

**Author:** Temilola Olowolayemo · **Course:** CS156 (Prof. Watson) · **Date:** 2026-04-24

**One-sentence pitch:** A probabilistic, causal, relational, and generative theory of my own taste, elicited from 324 hand-annotated ratings across 100 films in four modalities, fused with augmented corpora, and carried all the way through to an AI-generated movie teaser for the ideal film.

**Goal of this document:** a guided tour of what was built, what the numbers say, what went wrong, and where each artifact lives. The exhaustive narrative is in `notebooks/main_submission.ipynb`; this file is the map.

---

## 1. How to read this repository

| If you want… | Go to |
|---|---|
| The main submission narrative (condensed, graded) | [`notebooks/main_submission.ipynb`](notebooks/main_submission.ipynb) · [`notebooks/main_submission.pdf`](notebooks/main_submission.pdf) |
| The exhaustive method-by-method deep dive | [`notebooks/full_deep_dive.ipynb`](notebooks/full_deep_dive.ipynb) · [`notebooks/full_deep_dive.pdf`](notebooks/full_deep_dive.pdf) |
| The pipeline diagram | [`artifacts/plots/pipeline_diagram.png`](artifacts/plots/pipeline_diagram.png) |
| The three generated teasers | [`video/FINAL/v4/`](video/FINAL/v4/) (MP4) · [`video/FINAL/v4/previews/`](video/FINAL/v4/previews/) (GIF) |
| The interactive app | [`streamlit_app/app.py`](streamlit_app/app.py) · manifest [`STREAMLIT_MANIFEST.md`](STREAMLIT_MANIFEST.md) |
| Method implementations | [`src/`](src/) (one module per method) |
| End-to-end fit scripts | [`scripts/`](scripts/) (ordered `01_` … `23_`) |
| The authoritative build plan | [`EXECUTION_PLAN.md`](EXECUTION_PLAN.md) · pitch [`MEGA_PITCH.md`](MEGA_PITCH.md) |

---

## 2. Architecture

Three acts, one pipeline:

- **ACT I — ELICIT.** Streamlit collects 324 ratings (4 modalities × ~81 films) from me. These are the ground-truth labels.
- **ACT II — MODEL.** 14+ ML methods compress, explain, relate, and quantify uncertainty over that taste.
- **ACT III — GENERATE.** A fine-tuned generative stack turns the taste model into an original movie teaser.

The architecture diagram lives at `artifacts/plots/pipeline_diagram.png` and is embedded inline in `notebooks/main_submission.ipynb` §5.18.

```
ELICIT         MODEL (ACT II)                                 GENERATE (ACT III)
─────────      ──────────────────────────────────────          ─────────────────────
324 ratings ─► preprocess ─► compress (TVAE/CVAE)   ─► taste ─► Llama 3.1 8B + QLoRA  ─► narrative JSON
(4 modal.)  ─► augment  ─► explain   (GP/Kalman/HMM)   DNA  ─► SDXL + LoRA            ─► keyframes
              (ML twin,             relate    (HAN/LightGCN)    ─► SVD-XT + LoRA       ─► motion clips
               GenMatch,            quantify  (BNN/ANOVA/IPW/    ─► StyleGAN3-ADA      ─► abstract scenes
               TVAE)                           Conformal/         ─► MusicGen medium   ─► audio
                                               Thompson)           ─► ffmpeg assembly  ─► MP4 teasers
```

---

## 3. Data

| Source | N | Role |
|---|---:|---|
| Streamlit elicitation (4 modalities × ~81 films) | **324 ratings** | Ground truth — the only data used for evaluation |
| MovieLens 25M twin (similarity-mapped) | ~50K pseudo-ratings | Training augmentation only |
| GenMatch neighbors (GA + Mahalanobis) | 465 matched films | Trailer corpus expansion; min KS p = 0.34 |
| TVAE synthetic samples (rare-taste coverage) | variable | Training augmentation only |
| Trailer frames for ACT III | ~20K frames from 1,392 trailers | SDXL/SVD/StyleGAN3 fine-tune corpora |

Evaluation metrics use only the 324 ground-truth ratings. Augmented data is never mixed into held-out eval. See `notebooks/main_submission.ipynb` §2–§3 for EDA and preprocessing choices.

Rating-distribution and modality-coverage EDA: `artifacts/plots/03_eda_overview.png`.

---

## 4. Methods inventory

Each method ships as a standalone module in `src/` and is fit by a numbered script in `scripts/`. Plots are in `artifacts/plots/`.

| # | Method | Module | Script | Plot | Purpose |
|--:|---|---|---|---|---|
| 1 | Preprocessing + EDA | `src/data_io.py` | `scripts/…` | `03_eda_overview.png` | Modality-stratified ratings, 1–10 scale |
| 2 | Gaussian Process regression | `src/gp.py` | `10_fit_gp.py` | `gp_rating_surface.png` | Smooth rating surface over year × runtime |
| 3 | Kalman filter (UKF) | `src/kalman.py` | `11_fit_kalman_ladder.py` | `kalman_ladder.png` | Temporal taste drift |
| 4 | Particle filter (N=500) | — | `12_fit_pf.py` | `pf_cloud.png` | Non-Gaussian posterior over rating state |
| 5 | Hidden Markov model | `src/hmm.py` | `13_fit_hmm.py` | `hmm_regimes.png`, `08_hmm_timeline.png` | Mood-phase segmentation |
| 6 | MovieLens twin | — | `14_movielens_twin.py` | — | Pseudo-rating augmentation |
| 7 | TVAE | `src/tvae.py` | `15_tvae_synth.py` | `tvae_overlap.png` | Tabular VAE; recon MSE = 0.42 |
| 8 | LightGCN | `src/lightgcn.py` | `16_fit_lightgcn.py` | `lightgcn_training.png` | Graph collaborative filtering |
| 9 | HAN | `src/han.py` | `17_fit_han.py` | `han_training.png` | Heterogeneous attention over film-graph |
| 10 | Bayesian NN (MC-Dropout) | `src/bnn_mcd.py` | `18_fit_bnn_mcd.py` | `bnn_mcd_uncertainty.png` | Epistemic uncertainty |
| 11 | Hierarchical Bayesian ANOVA | `src/hierarchical_anova.py` | `19_fit_hierarchical_anova.py` | `anova_forest.png`, `anova_variance_decomp.png` | Variance decomposition across modalities |
| 12 | IPW / AIPW (doubly-robust) | `src/causal.py` | `20_causal_ipw_aipw.py` | `propensity_overlap.png` | ATE of synopsis-vs-metadata modality |
| 13 | Conformal prediction | `src/conformal.py` | `20_conformal_wrap.py` | `conformal_bands.png` | α=0.10; empirical coverage **91.14%** |
| 14 | Thompson sampling over GPs | `src/thompson_gp.py` | `21_thompson_gp.py` | `thompson_regret.png` | Bandit over 39 film arms vs random baseline |
| 15 | Conditional VAE (from scratch) | `src/cvae.py` | `22_fit_cvae.py` | `cvae_conditional.png` | Rating-bin-conditioned feature synthesis |
| 16 | GenMatch (Diamond & Sekhon) | `src/genmatch.py` | `02_genmatch_expand.py` | `genmatch_fitness.png` | GA + Mahalanobis; min KS p = 0.34 |

ACT III generative stack (in `data/gpu_outputs/`):

| Stage | Model | Config | Output |
|---|---|---|---|
| A | Llama 3.1 8B + QLoRA | r=32, α=64, 4-bit NF4 | `llama_qlora_v2/generated_narrative_sanitized.json` |
| B | SDXL + LoRA | CFG=8.5 | `generated_keyframes/` |
| B′ | SVD-XT + LoRA | r=32, 2000 steps, motion_bucket=180 | `svd_xt/` |
| C | StyleGAN3-ADA | bf16, bs=256, 200 epochs, on 20K trailer frames | `stylegan3/slerp_morph.mp4` |
| D | MusicGen medium | — | `musicgen_v2/` |
| E | ffmpeg assembly | 50 s @ 1920×1080 @ 24fps, 8000k | `video/FINAL/v4/*.mp4` |

---

## 5. Headline results

- **Conformal coverage** (α = 0.10): empirical **91.14%** on held-out ratings → calibration passes.
- **TVAE reconstruction MSE:** 0.42 on normalized rating features.
- **GenMatch balance:** minimum KS p-value across covariates = **0.34** (no covariate rejects balance at α=0.05).
- **LightGCN / HAN:** training curves in `artifacts/plots/{lightgcn,han}_training.png` (see notebook for held-out metrics).
- **Thompson bandit:** beats uniform-random baseline on cumulative regret over 39 arms.
- **StyleGAN3-ADA:** 100 epochs of G/D loss captured in `data/gpu_outputs/stylegan3/training_losses.json` — G oscillates in the 1.6–2.3 band, D in 0.2–1.2, consistent with healthy adversarial equilibrium at this scale.

Full metrics and derivations are in `notebooks/main_submission.ipynb` §6–§8.

---

## 6. ACT III deliverable

Three distinct 50-second teasers, each rendered from a **disjoint mix of sources** so the variants are genuinely different artefacts — not the same cut with different thumbnails:

| Variant | Narrative keyframes | SVD-XT clips | StyleGAN3 scenes | Size | Preview |
|---|---:|---:|---:|---:|---|
| **Poster** | 13 | 0 | 0 | 32.5 MB | [`…v4_poster_6s.gif`](video/FINAL/v4/previews/ideal_temilola_movie_v4_poster_6s.gif) |
| **Trailer** | 0 | 11 | 6 | 26.4 MB | [`…v4_trailer_6s.gif`](video/FINAL/v4/previews/ideal_temilola_movie_v4_trailer_6s.gif) |
| **Both** | 3 | 7 | 7 | 27.6 MB | [`…v4_both_6s.gif`](video/FINAL/v4/previews/ideal_temilola_movie_v4_both_6s.gif) |

Source breakdowns are recorded verbatim in `video/FINAL/v4/assembly_meta_{poster,trailer,both}.json`. Each variant runs at 1920×1080 @ 24 fps, 8000 kbps, with MusicGen-generated score.

**The narrative (story):** `data/gpu_outputs/llama_qlora_v2/generated_narrative_sanitized.json` contains the film concept *"Ideal"* — a spy-thriller about Mara Okafor, an off-book intelligence asset forced to run a cold mission twice. Taste-DNA weights (Action 0.66, Adventure 0.58, Sci-Fi 0.44, Drama 0.28, avg rating 8.25) are deterministic aggregates over my real 78 ratings + 465 GenMatch neighbors + 26 discovered films.

---

## 7. Iteration log: v2 → v3 → v4

Each version was a distinct render with a specific fix-forward goal.

- **v2 — first full render.** Validated end-to-end pipeline. Audio was cut off on several transitions; teaser quality suffered from short SDXL→SVD handoff. Kept as baseline.
- **v3 — audio + quality pass.** Fixed the audio-cutoff bug in the ffmpeg assembly (Task #21); upgraded teaser duration and motion-bucket settings (Task #24). Single combined cut; the three "variants" at this stage were near-identical.
- **v4 — distinctness pass.** Rebuilt three teasers with **disjoint source mixes** (Task #30, #35): Poster = 13 SDXL keyframes only; Trailer = 11 SVD-XT clips + 6 StyleGAN3 scenes, no keyframes; Both = a 3/7/7 split. Verified by `assembly_meta_*.json` and by comparing MP4 SHA-12s: `ebdc8ca51ad7` / `79803ce9dbd5` / `9d65de1a9949`. A v5 retrain was scoped to re-generate narrative from a sanitized Llama LoRA but was not shipped because RunPod capacity was blocked on 2026-04-23/24 — see §8.

---

## 8. Honest limitations

1. **Llama-QLoRA memorization.** The raw `generated_narrative.json` from the first Llama 3.1 8B + QLoRA (r=32, α=64, 4-bit) run reproduced copyrighted Marvel Studios / *Black Widow* character names and plot beats (Natasha Romanoff, Dreykov, the Red Room, the Budapest flashback). This is a textbook memorization failure: the fine-tune corpus included IMDB-style plot summaries of films in my own highly MCU-weighted top-rated list, so the LoRA overfit to those tokens and regurgitated them. The remediation was to ship a **sanitized sibling file** (`generated_narrative_sanitized.json`) with the same `taste_dna` aggregates but an **original** story (*"Ideal" / Mara Okafor*) that preserves genre distribution, themes, runtime, and era without copying IP. The raw file is kept in the repository as evidence of the failure, and the notebook flags this as limitation #2 in §9.2. A proper retrain with stronger deduplication + template-scrub should be the first post-submission fix.
2. **Expected-Calibration-Error (ECE)** was not computed for the BNN head; we report MC-Dropout variance bands instead.
3. **Love-score plot** (requested as a taste-fit visualization of the generated teaser against the real ratings) was not produced — CVAE taste-fit scorer is stubbed but not wired into the notebook.
4. **SHAP** attributions were not run on HAN/LightGCN outputs; we lean on attention weights only.
5. **HAN scale-up** was run at a small film-graph size; we did not retrain at the full augmented corpus.
6. **Streamlit runtime** was smoke-tested locally but no cross-browser runtime verification pass was performed against every page.

None of these invalidate the core methodology. They are the concrete items on the post-submission improvement plan in §9.3 of the notebook.

---

## 9. CS156 LO / rubric coverage

Mapping to the 10-section rubric (see `notebooks/main_submission.ipynb` §1 for the full walkthrough; section numbers below refer to that notebook):

| Rubric section | Where it's covered |
|---|---|
| 1. Data Sample Selection | Notebook §2 — 324 ratings over 4 modalities; sampling rationale |
| 2. Data Explanation | Notebook §2.2 + `03_eda_overview.png` |
| 3. Data Loading / Conversion | `src/data_io.py`; notebook §3 |
| 4. Preprocessing + EDA | Notebook §3; augmentation via MovieLens twin + GenMatch + TVAE |
| 5. Analysis Definition | Notebook §4 (task framing); §5 (formal math, incl. §5.16 LoRA, §5.17 CFG) |
| 6. Model Selection & Math | Notebook §5 + this README table in §4 |
| 7. Training | Notebook §6; training curves in `artifacts/plots/` |
| 8. Predictions & Metrics | Notebook §7 — conformal 91.14%, TVAE MSE 0.42, GenMatch KS p≥0.34; §7.5 teaser-variant table |
| 9. Visualizations & Conclusions | All 19 PNGs in `artifacts/plots/`; teaser GIFs embedded in §7.5; §8 assembly table |
| 10. Executive Summary, References | Notebook §9 (rewritten with artifacts table) |

LOs hit explicitly: **#professionalism** (honest limitations, no hidden failure), **#organization** (atomic modules in `src/`, numbered scripts, versioned teasers), **#audience** (condensed main + deep-dive sibling), **#dataviz** (19 method-specific plots + pipeline diagram), **#computation** (from-scratch implementations of CVAE, IPW/AIPW, conformal, Thompson sampling, GenMatch GA).

---

## 10. Artifacts

| Kind | Path |
|---|---|
| Notebook (main, graded) | [`notebooks/main_submission.ipynb`](notebooks/main_submission.ipynb) |
| Notebook (deep dive) | [`notebooks/full_deep_dive.ipynb`](notebooks/full_deep_dive.ipynb) |
| Pipeline diagram | [`artifacts/plots/pipeline_diagram.png`](artifacts/plots/pipeline_diagram.png) · [`.svg`](artifacts/plots/pipeline_diagram.svg) |
| EDA | [`artifacts/plots/03_eda_overview.png`](artifacts/plots/03_eda_overview.png) |
| Method plots (19) | [`artifacts/plots/`](artifacts/plots/) |
| Narrative (sanitized) | [`data/gpu_outputs/llama_qlora_v2/generated_narrative_sanitized.json`](data/gpu_outputs/llama_qlora_v2/generated_narrative_sanitized.json) |
| Narrative (raw, retained as evidence) | [`data/gpu_outputs/llama_qlora_v2/generated_narrative.json`](data/gpu_outputs/llama_qlora_v2/generated_narrative.json) |
| Teaser — Poster variant | [`video/FINAL/v4/ideal_temilola_movie_v4_poster.mp4`](video/FINAL/v4/ideal_temilola_movie_v4_poster.mp4) |
| Teaser — Trailer variant | [`video/FINAL/v4/ideal_temilola_movie_v4_trailer.mp4`](video/FINAL/v4/ideal_temilola_movie_v4_trailer.mp4) |
| Teaser — Both variant | [`video/FINAL/v4/ideal_temilola_movie_v4_both.mp4`](video/FINAL/v4/ideal_temilola_movie_v4_both.mp4) |
| Teaser GIF previews | [`video/FINAL/v4/previews/`](video/FINAL/v4/previews/) |
| Assembly metadata | [`video/FINAL/v4/assembly_meta_*.json`](video/FINAL/v4/) |
| StyleGAN3 loss log | [`data/gpu_outputs/stylegan3/training_losses.json`](data/gpu_outputs/stylegan3/training_losses.json) |
| StyleGAN3 slerp morph | [`data/gpu_outputs/stylegan3/slerp_morph.mp4`](data/gpu_outputs/stylegan3/slerp_morph.mp4) |
| Streamlit app | [`streamlit_app/app.py`](streamlit_app/app.py) · manifest [`STREAMLIT_MANIFEST.md`](STREAMLIT_MANIFEST.md) |
| Build plan | [`EXECUTION_PLAN.md`](EXECUTION_PLAN.md) · pitch [`MEGA_PITCH.md`](MEGA_PITCH.md) |

---

## 11. Reproducibility

Everything in ACT I + II runs on a laptop (18 GB RAM budget, checkpointed). ACT III was executed on RunPod A100 and H100 pods with resume-safe orchestration (Task #33). The ordered pipeline is `scripts/01_…` through `scripts/23_…`; a RUN_ORDER.md inside the repo captures the intended sequence. Model weights for the generative stack are checkpointed to `/runpod-volume` and not committed to Git.

---

## 12. References

See `notebooks/main_submission.ipynb` §10 for the full bibliography (Diamond & Sekhon 2013 for GenMatch; Rombach et al. 2022 for SDXL; Blattmann et al. 2023 for SVD; Karras et al. 2021 for StyleGAN3-ADA; Hu et al. 2021 for LoRA; Ho & Salimans 2022 for CFG; Vovk et al. for conformal prediction; etc.).
