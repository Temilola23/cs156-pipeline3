# Pipeline 3 — Running Design Doc

**Last updated:** 2026-04-23 (Thursday)
**Soft deadline (passed / passing today):** Thursday 2026-04-23 at 4:00pm
**Hard deadline (late-accepted, our target):** Friday 2026-04-24 at 4:00pm
**Time remaining:** ~24+ hours
**Assignment weight:** 25%

> This is the single source of truth for Pipeline 3. Updated live as brainstorming progresses.
> When compaction hits, future-Claude reads this first.

---

## Status

Brainstorming phase (superpowers:brainstorming skill). All prior notebooks deleted. 324 Streamlit ratings preserved. Starting over with MEGA_PITCH as north star.

**Design LOCKED on 2026-04-23.** User approved via "Go!" Next step: writing-plans skill produces hour-by-hour implementation plan.

---

## Thesis (from MEGA_PITCH)

*"From Compression to Generation: A Probabilistic, Causal, Relational, and Generative Theory of Temilola's Taste."*

## 3-Act Arc

- **ACT I — ELICIT.** Modality ablation experiment. 324 ratings already collected via Streamlit (100 movies × 4 conditions: poster / poster+synopsis / poster+trailer / all). Bandit-driven selection, Latin-square for order effects.
- **ACT II — MODEL.** ~12-15 from-scratch methods targeting 5/5 rubric on every LO: GP (RBF+Periodic+String kernel, Cholesky), Kalman→RTS→EKF→UKF ladder, Bootstrap PF+FFBSi, HMM baseline, Conformal prediction wrapper, GenMatch GA+Mahalanobis (~300 LOC), IPW/AIPW, Thompson on GP, MC Dropout BNN, LightGCN, HAN, Bayesian hierarchical ANOVA (PyMC), TVAE, CVAE.
- **ACT III — GENERATE.** 60-sec trailer `ideal_temilola_movie.mp4`. 2 pathways × 3 ablations + CogVideoX third pathway = 7 clips. Llama 3.1 8B QLoRA voiceover. Process visualized in notebook + Streamlit explainer app.

---

## Decisions locked in (as of 2026-04-23)

### Strategy
- **Approach: B + C hybrid** — headline-first (ACT III built first, ACT II backfilled by priority), ACT II trimmed to ~12-15 methods instead of 20.

### Final deliverable packaging
- **Option R + assignment-required submission, narrative-first architecture:**
  - **Primary submission: narrative-first "curator" notebook → PDF.** Tells the 3-act story. Loads pre-computed artifacts. Shows key results, equations, plots, conclusions. Links out to everything deeper. Follows 10-section rubric below. Fast to re-render, predictable PDF size.
  - **Supplementary via public GitHub repo, linked from notebook:**
    - `notebooks/main_submission.ipynb` — the curator notebook (exported to PDF for Minerva)
    - `notebooks/full_deep_dive.ipynb` — single combined "everything" notebook, method-by-method, full equations + full training outputs + full plots. Exported as a second PDF on GitHub (~200-300 pp). Linked prominently from main_submission Section 9.
    - `scripts/` — training scripts (`train_*.py`) that produce artifacts
    - `artifacts/` — serialized outputs (model weights, embeddings, predictions, plots)
    - `data/` — expanded datasets
    - `streamlit_app/` — explainer app with interactive sliders (Tier III target)
    - `video/` — final MP4 + Pathway A/B/C/ablation variants + GIF previews
    - `derivations/` — 16 first-principles derivations (Markdown/LaTeX)
    - `README.md` — map of the repo, how to reproduce

### Architecture: "curator notebook + deep-dive repo"
- The PDF notebook is **narrative density**, not code density. Graders read the story, see the plots, watch the trailer (linked).
- GitHub has **implementation density**. Anyone who wants to dig can reproduce any piece.
- Clean separation of concerns: narrative ↔ implementation.

### ACT III shape
- **Option Z** — 60-second trailer, **2 pathways × 3 ablations + CogVideoX = 7 clips**, Llama-written voiceover, process viz.
- Pathway A: generic AnimateDiff motion LoRA
- Pathway B: AnimateDiff motion LoRA **fine-tuned on expanded trailer corpus**, 3 ablations at p ∈ {0, 1, 2} fusion strengths
- Pathway C (stretch): CogVideoX-2B as a third comparison

### Data augmentation — 3 pillars kept, all in parallel
1. **MovieLens twin** — find MovieLens 25M users whose rating patterns match yours, borrow their rating vectors. Laptop, ~2 hr.
2. **GenMatch GA + Mahalanobis** — genetic algorithm + Mahalanobis nearest-neighbor on (genre, year, director, tags, cast embeddings) from 50K-movie TMDB catalog. ~300 LOC from scratch. Laptop, ~3 hr. **Also feeds ACT III trailer corpus.**
3. **TVAE** — Tabular VAE to synthesize rare rating patterns. Laptop (MPS), ~1 hr. Runs in parallel with pillar 2.

### Compute split — both machines active
| Where | Tasks |
|---|---|
| **Laptop (M5 Pro, 48GB unified, 2TB disk)** | MovieLens twin, GenMatch, all ACT II methods (GP, Kalman ladder, PF, HMM, Conformal, IPW/AIPW, Thompson, MC Dropout, PyMC, LightGCN, HAN, TVAE, CVAE), yt-dlp trailer downloads, PySceneDetect, StyleGAN3-ADA (tiny posters), MusicGen, ffmpeg assembly, notebook authoring |
| **RunPod A100 80GB** (~$1.19/hr, $29.99 balance, volume `rthk2teqhv`) | SDXL DreamBooth, AnimateDiff motion LoRA A+B+ablations, Llama 3.1 8B QLoRA, SVD-XT, CogVideoX-2B, Real-ESRGAN upscale. Spin up only for heavy training, ~15-18 A100 hours total. |

### ACT II → ACT III coupling (THIS is the narrative thread)

ACT II methods train in parallel, but their outputs **condition** ACT III generation. Every method has a job:

| ACT II output | → | ACT III use |
|---|---|---|
| GP posterior (rating surface over 50K TMDB movies) | → | weights trailer/scene selection |
| UKF latent taste state (final filtered estimate) | → | seeds CVAE taste centroid |
| Particle Filter posterior samples | → | N alternative CVAE centroids (taste variants) |
| HMM taste regimes (3 hidden states) | → | structures trailer into 3 acts |
| LightGCN / HAN movie embeddings | → | CVAE input space |
| Bayesian ANOVA genre/decade effect sizes | → | SDXL prompt weighting |
| IPW / AIPW causal effect of modality | → | visual weighting (motion vs stills) |
| Thompson sampling policy | → | clip selection in final assembly |
| MC Dropout BNN uncertainty | → | gates low-confidence scenes |
| Conformal prediction intervals | → | overlay in final deliverable |
| GenMatch neighbors | → | trailer corpus for motion LoRA B |
| MovieLens twin expansion | → | LightGCN/HAN training graph |
| Llama 3.1 8B QLoRA (trained on synopses) | → | writes the voiceover |

---

## Assignment rubric constraints (from CS156 Pipeline Final Draft PDF)

### Submission format
- **ONE Jupyter notebook exported as PDF.** Not zip, not raw .ipynb, not multiple docs.
- Supplementary code/data allowed via **public GitHub link**.

### 10 required notebook sections
1. **Data explanation** — what's included, how obtained, sampling details
2. **Python conversion** — well-commented loading into np/pd/glob/etc.
3. **Preprocessing + feature engineering + EDA** — cleaning, viz of samples, descriptive stats
4. **Analysis framing** — classification/regression/clustering framing + train/test splits
5. **Model selection + math** — markdown discussion of mathematical underpinnings, typeset equations, pseudocode
6. **Training** — code + explanations for CV and hyperparameter tuning
7. **Predictions + metrics** — out-of-sample predictions, performance metrics
8. **Visualization + conclusions**
9. **Executive summary** — steps, pipeline diagram, key viz, insights, shortcomings, improvement discussion
10. **References**

### Assignment 3 specific
- MUST showcase a model/method **not covered in class**. Extensions of class-covered methods are fine (e.g., UKF extends Kalman extends HMM).
- Compare to similar class-covered method (apples-to-apples).
- Include diagrams, equations.
- Target practical / humorous goal.

### LO targets (aiming 5/5 each)
- #professionalism — polished PDF, clean formatting
- #dataviz — every ACT II method produces a plot
- #algorithms — from-scratch implementations
- #cs156-MLCode — working, readable, performant
- #cs156-MLExplaination — written + mathematical + visual
- #cs156-MLFlexibility — novel work, 13+ beyond-class methods
- #cs156-MLMath — 16 first-principles derivations (GP posterior, Kalman gain, UKF sigma weights, PF importance ratios, ELBO, LoRA low-rank, CFG, conformal coverage, GenMatch optimization, Thompson regret, cross-frame attention, etc.)

---

## Critical path (high-level)

### Phase 1 — Parallel sprint (laptop + A100 both hot)
- **Laptop:** MovieLens twin → GenMatch → GP → Kalman ladder → PF → HMM → Conformal → IPW/AIPW → Thompson → MC Dropout → PyMC ANOVA → LightGCN → HAN → TVAE → CVAE
- **A100 (from start):** SDXL DreamBooth, Llama 8B QLoRA, motion LoRA Pathway A (generic)

### Phase 2 — Sequential backbone
- MovieLens twin + GenMatch complete → yt-dlp downloads expanded trailer set → scene detection → motion LoRA Pathway B trains on A100

### Phase 3 — Conditioning + assembly
- Extract UKF latent → seed CVAE → CVAE taste centroid
- HMM regimes → 3-act trailer structure
- LightGCN embeddings → SDXL prompt conditioning
- Thompson policy → clip selection
- BNN/Conformal → quality gating
- Llama QLoRA → voiceover generation
- SDXL → AnimateDiff → SVD-XT → CogVideoX (stretch) → ffmpeg stitches → MusicGen audio → Real-ESRGAN upscale → RIFE interpolation → final MP4

### Phase 4 — Notebook + PDF
- Write all 10 sections, embed final MP4 preview (GIF or first-frame + GitHub link), export PDF.
- Deploy Streamlit explainer app, link from notebook.

**Total wall-clock estimate:** ~18-20 hours of critical-path work. Fits in the ~24-hour window to Friday 2026-04-24 4pm with one short sleep block. Hour-by-hour schedule will be produced by writing-plans skill.

---

## Files preserved (in /Pipeline 3/)

- `data/modality_ratings.jsonl` — 324 Streamlit ratings (CRITICAL, DO NOT TOUCH)
- `data/movies_meta.json` — 82 movie metadata entries
- `data/trailers/` — 80 raw trailer MP4s
- `data/posters_hq/` — 34 high-quality poster JPGs
- `MEGA_PITCH.md` — authoritative vision
- `research/` — 12 research docs (research_*.md consolidated here 2026-04-23)
- `LOs.txt` — learning outcome notes
- Streamlit app files
- `CS156 - Pipeline - Final Draft _ Forum _ Minerva.pdf` — assignment rubric

## Files deleted on 2026-04-22

- `pipeline3.ipynb`, `pipeline3_executed.ipynb`, `pipeline3_backup.ipynb`, `pipeline3_debug.ipynb`, `pipeline3_debug2.ipynb` — all old notebooks
- Stale `HANDOFF.md`, `CONTINUATION_PLAN.md`, `EXECUTION_PLAN.md`, `RUN_NOW.md`, `INFRA_README.md` will be consolidated/removed once design locks.

---

## Data provenance (honest framing, goes in notebook Section 1)

| Source | Count | Role |
|---|---|---|
| My ratings, Streamlit modality experiment (4 conditions, bandit + Latin-square) | **324** | **Seed — training + held-out evaluation** |
| MovieLens 25M twin users (top-K by rating correlation) | ~50K-200K pseudo-ratings | Training signal for graph methods (LightGCN, HAN) |
| GenMatch neighbors from TMDB 50K catalog | ~500 movies | Feature/trailer expansion for motion LoRA Pathway B |
| TVAE synthetic rare-taste samples | ~5K pseudo-ratings | Covers sparse genre×decade combinations |
| **Evaluation always uses:** | **324 only** | Strict hold-out; augmented data never appears in eval |

**Evaluation integrity principle:** augmented data is for training only. All final predictions (including the taste-fit score on the generated trailer) are computed on held-out samples from the 324.

## Brainstorm progress

- [x] Q1: Path — **B + C hybrid**
- [x] Q2: TVAE — **keep, in parallel (not stretch)**
- [x] Q3: ACT III shape — **Option Z** (60s, 2 pathways × 3 ablations + CogVideoX)
- [x] Q4: Deliverable packaging — **Option R + assignment PDF**
- [x] Q5: Assignment rubric absorbed (one PDF from one notebook; Streamlit = supplementary via GitHub)
- [x] Q6: Notebook architecture — **β curator-notebook + GitHub repo with per-method deep-dive notebooks**
- [x] Q7: Central narrative — **Hook (4) nested structure** — surface: "can AI generate a 60s trailer I'd love?"; mid-reveal: "which modality moves me most?"; backbone: "what are my hidden taste regimes?"
- [x] Q7b: Opener phrasing — **(C) punchy + humorous** ("Can I teach an AI to make a movie I'd love, starting from 324 ratings I gave it the hard way? This pipeline says: yes, within a calibrated 90% confidence interval.") — workshop as we go
- [x] Q7c: Evaluation integrity — augmented data trains only, 324 hold-out evaluates
- [x] Q8: A100 budget — **Option (i) full Z with buffer, user adds credit as needed** (no scope trimming for budget)
- [x] Q9: Streamlit app scope — **Target Tier (III) full**, guard-rail at (II). Trailer gallery + UKF slider + HMM regime switcher + GenMatch explorer + **motion LoRA training-process animation**. Last-priority, cut-first-if-needed. ⚠ Hard requirement: motion LoRA training scripts MUST save checkpoints every N steps to support the training-process animation.
- [x] Q10: MP4 embedding — **(d) with a twist** — GIF in Section 8, storyboard in Section 9, MP4 + Streamlit links prominent on exec-summary page
- [x] Q11: GitHub repo structure for supplementary materials — **covered by Q13 (2-notebook structure + repo layout under "Final deliverable packaging")**
- [x] Q12: Method list locked — 14 ACT II methods + MovieLens twin. Tier 1 (10 load-bearing, non-negotiable), Tier 2 (4 cuttable in order: TVAE → PyMC ANOVA → BNN → HAN). 5 apples-to-apples comparisons baked in.
- [x] Q13: GitHub repo structure + RunPod workflow — **2-notebook structure (main_submission + full_deep_dive), RunPod spins Thursday evening, tears down Friday noon**
- [x] Final: design lock → writing-plans skill for implementation plan (locked 2026-04-23)

---

## Open risks / things to watch

- **8B Llama QLoRA must finish training before voiceover writing step.** Schedule it earliest on A100.
- **Motion LoRA Pathway B depends on expanded trailer corpus.** MovieLens twin + GenMatch + yt-dlp must complete by end of Day 1.
- **PDF render of notebook with 15+ plots and a 60s MP4 = large file.** May need to embed MP4 as GIF preview + GitHub link, not raw video tag.
- **Streamlit "live slider" conditioning requires serialized ACT II artifacts.** Must export UKF state, LightGCN embeddings, HMM regimes, etc., as portable files.
- **RunPod cuDNN is broken on A100 pod class** — set `torch.backends.cudnn.enabled = False` at top of every training script.
- **RunPod balance is $29.99, user will top up if needed** — 15-18 A100 hours at $1.19/hr = ~$18-22 baseline. User has explicitly said "I can add more stuff if there's more stuff to add" — no scope trimming for budget reasons. Still: don't leave pods idle, tear down between phases.
