# Pipeline 3 — The Mega-Pitch (v2)

> Synthesized from **11 parallel research streams**: 5 original (generative, data augmentation, sequence/state-space, SOTA recommenders, validation+Bayesian+causal) + 6 redeploy (diffusion fine-tuning, GAN fine-tuning, RL bandits, GNNs, particle filter/UKF, Diamond & Sekhon GenMatch deep-dive).
>
> **Constraints update (Apr 16 2026):** Cloud GPU paid by user. Fine-tuning explicitly OK. Class coverage now includes Sessions 1–24 (HMM, AE, RNN, Attention, Transformers all covered — novelty claims re-anchored accordingly).

---

## 0. The single sentence

**"From Compression to Generation: A Probabilistic, Causal, Relational, and Generative Theory of Temilola's Taste."**

P1 clustered. P2 embedded. P3 *infers, predicts under uncertainty, corrects bias, decides what to watch next, and creates the movies that don't exist yet*.

---

## 1. The narrative arc (three acts, one notebook)

### ACT I — ELICIT (the experiment)

We don't know what *causes* Temilola to rate a movie 9.0 vs 6.0 (his ratings are on the 0–10 scale he's used since Pipeline 1). So we **run an experiment**.

Within-subjects modality ablation across **100 stratified-sampled movies × 4 conditions = 400 modality-conditional ratings** (poster-only, title+metadata only, synopsis-only, all-three), Latin-square counterbalanced, ≥48h washout. **Movie selection itself is bandit-driven**: instead of random sampling, an information-gain bandit picks the 100 movies whose posterior modality-effect is most uncertain — turning the experiment design into an active-learning problem. **Rating collection is via a deployed Streamlit website (mobile-friendly, persistent JSON store), not a Jupyter widget** — Temilola can rate from his phone in spare moments across the 7-day window.

This converts the "Madam Web problem" from P1 into a **measurable, statistical artifact**: how much does each modality (visual / lexical / semantic) actually drive the rating?

### ACT II — MODEL (the probabilistic + causal + relational stack)

1. **Augment** — break N=162 ceiling with MovieLens 25M + IMDB datasets + a "Synthetic Temilola Twin" (his k-nearest user in MovieLens) → ~10K richer rating signal
2. **De-bias with Diamond & Sekhon GenMatch (CORE)** — instead of (or alongside) IPW, run **genetic matching** to find the cohort of MovieLens users whose covariate distribution is balanced against Temilola's. The GA optimizes weights `W` on the Mahalanobis distance to maximize the *worst-case* covariate-balance p-value. This produces nonparametric, multivariate-aware counterfactuals — pairs IPW + AIPW (parametric, doubly-robust) with GenMatch (nonparametric, balance-optimal). All three from scratch.
3. **Predict with uncertainty** — three uncertainty engines:
   - **Gaussian Process** with composite kernel (RBF on embeddings + Periodic on Date Watched + String on titles) — **from scratch**: kernel matrix, Cholesky, posterior mean & variance derived from first principles
   - **MC-Dropout Bayesian NN** as cross-check (Gal & Ghahramani ELBO derivation)
   - **Conformal prediction wrapper** giving distribution-free 90% coverage guarantee on top of either
4. **Decide under uncertainty (NEW)** — **Thompson sampling on the GP posterior** picks which unwatched movie Temilola should rate next. Closes the loop: posterior → action → new label → updated posterior. Apples-to-apples vs ε-greedy and naïve argmax-over-mean.
5. **Sequence — nonlinear ladder (EXPANDED)** — `Date Watched` was sitting in the data unused. Frame his ratings as a state-space and climb the ladder of generality:
   - **HMM** (class baseline, S20) — discrete latent regimes
   - **Linear Kalman + RTS smoother** (from scratch) — continuous-Gaussian
   - **EKF** (Jacobian-linearized) — first-order nonlinearity
   - **UKF** (sigma-point unscented transform) — derivative-free nonlinearity, **headline method**
   - **Bootstrap particle filter + FFBSi smoother** — fully nonparametric, non-Gaussian
     This is one continuous mathematical ladder, each step a strictly more general extension of the previous, all on the *same* Date-Watched series, evaluated by smoothed RMSE.
6. **Relational embeddings (NEW)** — content-only embeddings (P2 sentence-transformer + ResNet) miss collaborative signal. **LightGCN** on the user-movie bipartite + **HAN heterogeneous attention** on the user-movie-actor-director-genre graph produce embeddings that capture *who else liked what*. These can substitute or fuse with content embeddings into downstream GP / Kalman / CVAE.

### ACT III — GENERATE (the wild card)

Now that we have a probabilistic, causally de-biased, sequentially-aware, relationally-grounded taste model, **invert it**. The deliverable is no longer just a poster — it's an actual **30–60 second AI-generated TEASER TRAILER** for "the ideal Temilola movie." A real, playable MP4. Per `research/research_video_generation.md`.

**Why not Sora / Veo 3 / Midjourney / DALL-E?** All closed-weights / inference-only / no fine-tuning. Zero rubric credit (nothing from-scratch, no math derivation, no personalization on internals). They're ruled out as the deliverable; the open-source stack below replaces them.

**Two parallel video pathways ship side-by-side as an apples-to-apples** (Pathway A = image-only training, Pathway B = image + trailer training). Pathway A is the safe headline; Pathway B is the ambition extension. If Pathway B's training run fails, A still ships and B is discussed as future work. Both consume the same shared `v_taste` vector from the CVAE encoder — one taste embedding conditions every generator (image, motion, text, audio).

The video pipeline (Stage A → Stage B → Stage C):

1. **CVAE on movie embeddings (taste centroid)** → sample from the high-rating posterior region → decode an "ideal-Temilola-movie" embedding vector that conditions everything downstream (encoder/decoder + ELBO from scratch)
2. **STAGE A — Keyframe generation (poster diffusion)** — fine-tune SDXL via DreamBooth + LoRA on his ⭐8.5+ posters (A100, ~1.5hr). Then **classifier-free guidance** with rating *and* modality as conditioning signal directly closes the loop to ACT I's ablation. Generate **5–10 keyframe posters** of the ideal movie aesthetic. **Wild-card extension**: diffusion *on the CVAE latent* (not pixels) — sample ideal embeddings cheaply, retrieve real-world k-NN movies as a sanity check.
3. **STAGE B — Motion generation (image → video, NEW)** — turn each keyframe into a 2–8 sec animated clip. Two paths, evaluated apples-to-apples:
   - **AnimateDiff (primary)** — bolts a temporal-attention motion module onto our already-fine-tuned SDXL+LoRA. Maximum synergy; only ~12–16 GB VRAM to LoRA the motion module. Frozen spatial layers preserve our learned poster aesthetic; new temporal cross-frame attention learns motion. Math hook: extension of Session 23 Attention into the time axis.
   - **Stable Video Diffusion (SVD-XT, comparator)** — pure image-to-video, 25 frames at ~8 fps. Inference-only baseline; if AnimateDiff fine-tune fails, this still ships clips.
3a. **PATHWAY B — Trailer-fine-tuned motion LoRA (parallel extension, NEW)** — instead of using AnimateDiff's generic motion prior, **fine-tune the motion module on Temilola's own ⭐ corpus**: pull all 162 movies' official trailers via TMDB API → YouTube → `yt-dlp`, segment each ~2-min trailer at shot boundaries via PySceneDetect (~15 clips × 162 movies ≈ 2,400 raw clips → stratified-sample 1,500 for tractable training). Train AnimateDiff motion LoRA with **rating-weighted loss** `w_i = (rating_i / 10)^p`. Run three weighting ablations (`p ∈ {0, 1, 2}`) → uniform / linear / quadratic emphasis on top-rated movies. **The math machinery is the same as ACT II's IPW** — importance weighting applied at the training-loss layer. The trailer-fine-tuned motion LoRA learns trailer-pacing + cinematography patterns *specific to movies he loves*. Pathway A vs Pathway B becomes a within-ACT-III apples-to-apples (FVD, CLIP-sim vs centroid, blind rating). Acquisition is **decoupled from training** — trailer download + scene segmentation can run as a Day 0 background task; training waits for Day 6 like the rest of ACT III.
4. **STAGE C — Stitch + score (NEW)** — `ffmpeg`/`moviepy` assembles 5–10 motion clips into a 30–60 sec teaser with crossfades. **MusicGen** (Meta, open-source, transformer over EnCodec discrete audio tokens — Session 24 LLM extension to audio) generates a soundtrack conditioned on the synopsis. **Llama 3.2 3B + QLoRA** generates title + synopsis (S24 is class-covered, but PEFT/QLoRA isn't); LLM-generated title overlays as a title card.
5. **GAN as fast comparator + SLERP morph wild-card** — fine-tune StyleGAN3-ADA on the same posters; GAN-invert his favorites to recover w-space coordinates; **SLERP through 5 cluster centroids over 60 frames → "the trailer that morphs through your taste space"** ("begins as your sci-fi taste, ends as your romance taste"). Speed (1–2 sec sampling) and smooth-manifold properties argue for GAN where diffusion is overkill. Apples-to-apples GAN vs Diffusion vs CVAE.
6. **Validation loop closes the arc** → take the generated poster + title + synopsis + motion clip + final teaser back to Temilola, blind-rate them, run a Turing test, measure CLIP-similarity to his taste centroid, check temporal consistency via optical flow. The Act-I experimental machinery validates the Act-III generations.

**Final deliverable:** `ideal_temilola_movie.mp4` — 30–60 sec trailer with personalized aesthetic, AI-generated title, synopsis, and soundtrack. Plus a fallback ladder (animated single keyframe → static keyframe grid) if any stage fails.

**This is the closed loop**: experiment → model → decide → generate → **animate** → re-experiment. Nobody in CS156 has ever submitted that.

---

## 2. Why this hits every rubric box

| Rubric criterion     | What gives us a 5                                                                                                                                                                                                                                                                                                                                                                                          |
| -------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **MLCode**           | From-scratch RBF kernel + GP posterior, Kalman + RTS, EKF, UKF (sigma points), bootstrap particle filter, CVAE encoder/decoder + ELBO, IPW propensity, AIPW, **GenMatch genetic algorithm + Mahalanobis matching (~300 LOC)**, conformal calibration, Thompson sampling (~30 LOC), LoRA adapter math — every one paired with a library cross-check                                                          |
| **MLExplanation**    | Every method motivated by a specific failure mode of P1/P2 (selection bias → GenMatch + IPW + AIPW; N=162 ceiling → augmentation; deterministic predictions → GP/conformal; static features → Kalman ladder; content-only embeddings → GNN; passive predictor → bandit)                                                                                                                                    |
| **MLFlexibility**    | 13+ methods strictly outside class coverage (S1–24): GP, Kalman/EKF/UKF/PF, CVAE (extension of S21 AE), GenMatch + IPW + AIPW + Rosenbaum bounds, conformal, Thompson sampling, GraphSAGE/LightGCN/HAN, diffusion (DDPM/CFG/SDS), GAN (StyleGAN3-ADA/inversion), LoRA/QLoRA fine-tuning, **AnimateDiff temporal-attention motion module fine-tuned on his own trailer corpus (Pathway B), Stable Video Diffusion (image-to-video diffusion), MusicGen (transformer over discrete audio tokens, extension of S24), unified taste-vector conditioning across image/motion/text/audio generators**. **The HMM, vanilla RNN, vanilla attention, vanilla transformer, vanilla AE all appear as class-covered baselines**. |
| **MLMath**           | First-principles derivations: GP posterior (Bayes + marginalization), Kalman gain (innovation + covariance update), UKF sigma-point weights, particle-filter SIS importance ratios + ESS, ELBO (KL + reconstruction), LoRA low-rank (ΔW = BA), classifier-free guidance (∇log p(x\|y) ≈ (1+w)·ε(x,y) − w·ε(x,∅)), conformal coverage proof, GenMatch worst-case-balance optimization, Thompson regret bound, **temporal cross-frame attention (4D Q/K/V over space×time), why freezing spatial layers preserves the SDXL prior in AnimateDiff, video-diffusion 3D-conv factorization, optical-flow consistency loss, EnCodec discrete-audio token entropy, rating-weighted loss as IPW at the training-loss layer (`L = Σ w_i · L_i` with `w_i = (rating_i / 10)^p`) — same importance-weighting machinery as ACT II's IPW, applied at a different layer** |
| **#datavis**         | DAG of rating-generation, GP uncertainty bands, posterior credible intervals, Latin-square heatmap, modality-info-gain bars, latent-space traversal of CVAE, SLERP interpolation grid, Kalman-vs-EKF-vs-UKF-vs-PF smoothed overlay, GenMatch covariate-balance Love plot, GNN attention heatmap (which neighbors matter), Thompson-vs-ε-greedy regret curves, generated-vs-real poster grid                 |
| **#algorithms**      | Causal inference (GenMatch + IPW + AIPW + Rosenbaum bounds), nonlinear state-space inference (EKF/UKF/PF + FFBSi), MCMC (PyMC NUTS), variational inference (CVAE + MC dropout), distribution-free prediction (conformal), low-rank adaptation (LoRA/QLoRA), graph message-passing, sequential decision-making (Thompson + LinUCB), score-based generation (SDS), genetic algorithms (GenMatch GA)           |
| **#professionalism** | One coherent arc, no Frankenstein. Section headers are full prose paragraphs, not bullets (Watson's exact feedback). Equation-numbered, citation-style references, executive summary stands alone.                                                                                                                                                                                                         |

---

## 3. MUST-HAVE vs NICE-TO-HAVE (7-day gate, revised)

### MUST-HAVE (these alone get all 5s)

- **Day 0 (infrastructure, parallel, ~3 hr work)** — (a) **Streamlit rating-collection website** built and deployed (mobile-friendly, persistent JSON, Latin-square + bandit-driven movie ordering baked in). (b) **TMDB trailer fetch + yt-dlp + PySceneDetect** pipeline runs in background, pulls all 162 trailers, segments to ~2,400 raw motion clips, stratified-samples to ~1,500, saves to `data/trailer_clips/`. Both jobs complete before Day 1.
- **Day 1** — Modality ablation: bandit-selected 100 movies (info-gain), Latin square. Temilola starts using the Streamlit site to rate (Session 1: ~40 movies × 4 cond = 160 ratings, ~2.5 hr at his pace, can be split across day on phone).
- **Day 2** — Augmented dataset assembly: MovieLens 25M, IMDB datasets, GenMatch'd Synthetic Twin. Temilola continues rating on Streamlit (Session 2: ~60 movies × 4 cond = 240 ratings, split across day).
- **Day 3** — **GenMatch core** (genetic algorithm + Mahalanobis matching from scratch) + IPW + AIPW. Gaussian Process (from-scratch RBF + Cholesky posterior). Conformal wrapper. Apples-to-apples vs P2 Ridge.
- **Day 4** — Sequence ladder: HMM (library) → Kalman+RTS (from scratch) → **UKF (from scratch, headline)**. Apples-to-apples on Date-Watched RMSE. EKF + PF as Day-7 stretch.
- **Day 5** — CVAE on movie embeddings (encoder + decoder + ELBO from scratch), train, sample ideal-movie embedding from high-rating posterior region. Latent traversal visualization. **Thompson sampling** on GP posterior (~30 LOC) recommends next-watch.
- **Day 6 — TWO PARALLEL VIDEO PATHWAYS** —
   - **STAGE A keyframes (shared)**: SDXL DreamBooth + LoRA fine-tune on his ⭐8.5+ posters (Colab Pro A100, ~1.5hr); generate 5–10 keyframe posters with classifier-free guidance.
   - **PATHWAY A motion**: AnimateDiff motion module (off-the-shelf generic motion prior) on top of SDXL+LoRA → each keyframe → 2–8 sec clip. SVD as comparator. Stitch → `teaser_pathway_A.mp4`.
   - **PATHWAY B motion (parallel, NEW)**: AnimateDiff motion LoRA fine-tuned on the Day-0-acquired trailer clips with rating-weighted loss (3 ablations: `p∈{0,1,2}`). ~6–10 hr A100 per ablation, training kicks off Day 5 evening or Day 6 morning. Stitch → `teaser_pathway_B.mp4`.
   - **STAGE C (shared)**: `ffmpeg` assembly + MusicGen soundtrack + LLM title card on top of both pathways' clips.
   - **Validation**: blind-rate generations, CLIP similarity vs centroid, optical-flow temporal consistency, FVD comparison Pathway A vs B.
- **Day 7** — Apples-to-apples summary table (P1+P2+P3, all rows), executive summary paragraph, all visuals polished, references appendix, PDF export, formatting pass. **Embed final teaser MP4 in notebook + screenshots in PDF.**

### NICE-TO-HAVE (drop in if ahead of schedule, in priority order)

1. **EKF + bootstrap PF** to complete the nonlinear-Kalman ladder (huge MLMath payoff, ~250 LOC total)
2. **AIPW** doubly-robust on top of IPW + GenMatch
3. **MC Dropout BNN** as second uncertainty engine alongside GP
4. **LightGCN** on user-movie bipartite (3–4 days from-scratch in PyG, ~150 LOC) — relational embedding track
5. **GAN fine-tune** (StyleGAN3-ADA on posters + GAN inversion + SLERP between cluster centroids) — apples-to-apples vs diffusion
6. **Llama 3.2 3B + QLoRA** plot/title generator (~3hr Colab)
7. **HAN heterogeneous GNN** on user-movie-actor-director-genre graph
8. **Diffusion-on-latent** wild-card: train a tiny DDPM on the CVAE latent space
9. **Score Distillation Sampling** to use ResNet preference signal as a refinement reward on diffusion
10. **Modality-conditional Bayesian NN** with KL info-gain table
11. **Bayesian hierarchical ANOVA** in PyMC (replaces frequentist ablation analysis)
12. **CogVideoX-2B fine-tune** as a third video-gen comparator alongside AnimateDiff and SVD (text-to-video alternative; ~3–4 days extra)
13. **SLERP-morph trailer wild-card** — extend the 30s teaser to a 60–90s "journey through Temilola's taste space" (5-centroid latent walk through StyleGAN3-ADA w-space, AnimateDiff-rendered)

### NO LONGER OUT-OF-SCOPE (rehabilitated by the 6-agent redeploy)

The following were previously marked out-of-scope. The redeploy research files demonstrate viable fine-tune-not-from-scratch paths for each, all under cloud-GPU budget:

- ~~Full diffusion from scratch~~ → **DreamBooth + LoRA fine-tune of SDXL** (`research/research_diffusion_redeploy.md`)
- ~~Full GAN training~~ → **StyleGAN3-ADA fine-tune + GAN inversion** (`research/research_gan_redeploy.md`)
- ~~RL bandit~~ → **Thompson sampling on GP posterior, ~30 LOC, on-narrative as decision layer** (`research/research_bandits_redeploy.md`)
- ~~GraphSAGE / PinSAGE / HetGNN~~ → **LightGCN + HAN, narrative fit as relational-embedding track** (`research/research_gnn_redeploy.md`)
- ~~Particle filter / UKF~~ → **Nonlinear ladder extending Kalman, headline = UKF** (`research/research_particle_ukf_redeploy.md`)
- ~~Generic "synthetic data"~~ → **Diamond & Sekhon GenMatch deep-dive, elevated to core de-bias step** (`research/research_genmatch_redeploy.md`)

The remaining genuinely-out-of-scope items (only because they don't add narrative value, not because they're hard):
- Score-based DDPM from scratch (use SDXL pretrained — fine-tuning is the path)
- Full multi-week reinforcement learning beyond bandits (off-narrative; bandits already cover the decision-under-uncertainty story)

---

## 4. The apples-to-apples comparison table (the rubric closer)

This is the table that anchors the executive summary. **Class-covered = Sessions 1–24**, including HMM (S20), AE (S21), RNN (S22), Attention (S23), Transformers/LLMs (S24).

| Method                       | Pipeline | From-scratch?         | Class-covered?            | Test R² / metric            | Captures...                                       |
| ---------------------------- | -------- | --------------------- | ------------------------- | --------------------------- | ------------------------------------------------- |
| K-Means clustering           | P1       | ✅                     | ✅                         | ARI 0.48                    | Genre groupings                                   |
| OLS / Ridge                  | P1       | partial               | ✅                         | R² 0.16                     | Linear feature → rating                           |
| Kernel SVM (CVXOPT)          | P2       | ✅                     | ✅                         | —                           | Non-linear classification                         |
| Lasso CD                     | P2       | ✅                     | ✅                         | —                           | Sparse feature selection                          |
| Vanilla Autoencoder          | P2       | ✅                     | ✅ (S21)                   | —                           | Embedding compression                             |
| **HMM (Viterbi/F-B)**        | **P3**   | partial               | ✅ (S20)                   | smoothed log-lik            | Discrete-regime baseline for state-space          |
| **Gaussian Process**         | **P3**   | ✅                     | ❌                         | **R² + 90% CI**             | **Uncertainty-aware regression**                  |
| **Conformal wrapper**        | **P3**   | ✅                     | ❌                         | **coverage 0.90 ± 0.02**    | **Distribution-free intervals**                   |
| **GenMatch (Diamond+Sekhon)** | **P3**   | ✅ (~300 LOC GA)       | ❌                         | **balance KS-p ↑ to >0.5**  | **Nonparametric causal counterfactuals**          |
| **IPW + AIPW**               | **P3**   | ✅                     | ❌                         | **ΔR² ≈ +0.10**             | **Parametric doubly-robust de-bias**              |
| **Kalman + RTS**             | **P3**   | ✅                     | ❌                         | **smoothed RMSE**           | **Linear-Gaussian temporal taste drift**          |
| **EKF**                      | **P3**   | ✅                     | ❌                         | RMSE vs Kalman              | First-order nonlinearity                          |
| **UKF (sigma points)**       | **P3**   | ✅ (~120 LOC)          | ❌                         | **RMSE vs Kalman/EKF**      | **Derivative-free nonlinearity (headline)**       |
| **Bootstrap Particle Filter** | **P3**   | ✅ (~150 LOC)          | ❌                         | RMSE + ESS                  | Fully nonparametric non-Gaussian state            |
| **Thompson sampling**        | **P3**   | ✅ (~30 LOC)           | ❌                         | **regret vs ε-greedy**      | **Decision under uncertainty (active learning)**  |
| **LightGCN**                 | **P3**   | ✅ (~150 LOC PyG)      | ❌                         | NDCG@10 vs MF               | Collaborative-relational embeddings               |
| **HAN heterogeneous GNN**    | **P3**   | partial (DGL)         | ❌                         | NDCG@10 vs LightGCN         | Multi-typed graph (actor/director/genre)          |
| **CVAE**                     | **P3**   | ✅                     | ❌ (extension of S21 AE)   | **ELBO ↓**                  | **Generative ideal-movie sampling**               |
| **SDXL + DreamBooth + LoRA** | **P3**   | adapter from scratch  | ❌                         | **CLIP-sim 0.78**           | **Personalized keyframe (poster) generation**     |
| **StyleGAN3-ADA fine-tune**  | **P3**   | adapter from scratch  | ❌                         | FID + CLIP-sim              | Fast-sampling alt + GAN inversion + SLERP morph   |
| **AnimateDiff (motion LoRA, generic prior)** | **P3** | LoRA adapter from scratch | ❌ (extends S23 Attention) | optical-flow consistency    | **PATHWAY A: image → 2–8 sec clip (off-the-shelf motion)** |
| **AnimateDiff motion LoRA, fine-tuned on his trailers** | **P3** | LoRA adapter from scratch + rating-weighted loss (3 ablations p∈{0,1,2}) | ❌ (extends S23 + IPW from ACT II) | FVD vs Pathway A; CLIP-sim vs taste centroid | **PATHWAY B: image → personalized-motion clip (his-trailer-trained, rating-weighted)** |
| **Stable Video Diffusion (SVD-XT)** | **P3** | inference comparator  | ❌                         | clip FVD vs AnimateDiff     | Image → 4 sec clip alternative (Pathway A fallback) |
| **Llama 3.2 3B + QLoRA**     | **P3**   | adapter from scratch  | ❌ (LLM is S24, but PEFT isn't) | BLEU + human-rate           | Personalized synopsis/title                       |
| **MusicGen (Meta)**          | **P3**   | inference + prompt    | ❌ (extends S24 LLM to audio) | CLAP score                  | AI-generated soundtrack for the teaser            |
| **Stitched MP4 teaser (×2: Pathway A + B)** | **P3** | full pipeline         | ❌ (final integration)     | **2× 30–60 sec MP4 + comparison** | **The actual ideal-Temilola-movie deliverables (parallel pathways for apples-to-apples)** |

**13+ rows of "✅/partial from-scratch + ❌ class-covered"** is the structural proof of MLFlexibility = 5.

---

## 5. Hardware / runtime plan (cloud GPU available)

- **Local (M-series, ≤18GB)**: data assembly, GenMatch GA, IPW/AIPW, GP, Kalman/EKF/UKF/PF, Thompson sampling, conformal, LightGCN-small, CVAE training, all analysis & viz
- **Colab Pro (A100 40GB) — paid, no longer the bottleneck**:
  - SDXL DreamBooth + LoRA (~1.5 hr)
  - StyleGAN3-ADA fine-tune (~2 hr)
  - Llama-3.2-3B QLoRA (~3 hr)
  - HAN GNN on heterogeneous graph (~1 hr)
  - MusicGen / AnimateDiff inference (minutes)
- **Two-phase checkpoint architecture from P2**: reuse pattern. Phase A (cells 0–N) saves embeddings + GenMatch'd cohort + augmented dataset to `.pkl`; Phase B (cells N+1–end) loads from `.pkl`, runs all heavy ML.

---

## 6. The exact next-step list (what to do in the next 4 hours)

1. **Confirm the v3 pitch** — read this doc, push back on anything that feels wrong / overscoped / underscoped
2. **Lock the must-have list** — anything to swap in/out? GenMatch is Day-3 core; Pathway A+B parallel video-gen is Day-6; OK?
3. **Streamlit rating-collection website** (Day 0, ~2–3 hr build): mobile-friendly, persistent JSON store, deployed to Streamlit Cloud free tier so Temilola can rate from his phone. Latin-square + bandit-driven movie ordering baked in. Replaces the earlier "Jupyter ipywidget" plan because 400 ratings spread over the week need phone access.
4. **Trailer acquisition pipeline** (Day 0, runs in background): TMDB API `/movie/{id}/videos` for all 162 movies → `yt-dlp` → ~13 GB MP4s → PySceneDetect segments to ~2,400 raw clips → stratified-sample to ~1,500 → save to `data/trailer_clips/` ready for Day 6 motion LoRA training.
5. **Pull the augmentation data NOW** (it takes the longest):
   - MovieLens 25M (`ml-25m.zip`, ~250MB)
   - IMDB datasets (title.basics + title.ratings, ~1GB)
   - TMDB unwatched-movie sample for propensity (~1000 movies, your existing TMDB key)
6. **Decide GenMatch implementation path** — (a) rpy2 + R `Matching::GenMatch()` (1–2 hr, canonical), (b) from-scratch GA in pymoo + Mahalanobis (4–5 days, ~300 LOC, MLCode payoff), (c) hybrid. Recommendation: **(c) hybrid** — pymoo for the GA, your own Mahalanobis distance + KS fitness, then validate against `Matching::GenMatch()` via rpy2.
7. **Generate Latin square + bandit-driven 100-movie sample** from your existing 162-movie dataset (info-gain bandit on P2 best model's predicted rating + posterior variance)
8. **Set up the empty Pipeline 3 notebook** with the 10-section skeleton, full-prose section intros (no bullets in Watson's view — the bullets here are for *us*, not the deliverable)

---

## 7. References to the supporting research

Per-stream deep dives are in this folder:

**Original 5 streams:**
- `research/research_generative.md` — full generative stack (CVAE, SDXL+LoRA, Llama, MusicGen, AnimateDiff, IP-Adapter)
- `research/research_data_augmentation.md` — datasets + GenMatch + propensity + SMOTE/CTGAN + Synthetic Twin
- `research/research_sequence.md` — Kalman/RTS, Mamba, SASRec, PELT, Hawkes, Neural ODE
- `research/research_recommenders.md` — Two-Tower, SASRec, LightGCN, fan-class generation
- `research/research_validation_bayesian_causal.md` — modality ablation, GP, MC dropout, conformal, IPW/AIPW, Rosenbaum

**6 redeploy streams (Apr 16 2026):**
- `research/research_diffusion_redeploy.md` — DreamBooth/LoRA fine-tune, classifier-free guidance, score distillation, latent diffusion (centerpiece of ACT III)
- `research/research_gan_redeploy.md` — StyleGAN3-ADA fine-tune, GAN inversion, SLERP, contrastive style mixing
- `research/research_bandits_redeploy.md` — Thompson on GP posterior, LinUCB, info-gain experiment design (decision-under-uncertainty layer)
- `research/research_gnn_redeploy.md` — GraphSAGE, LightGCN, HAN, PinSAGE-style (relational embedding track for ACT II)
- `research/research_particle_ukf_redeploy.md` — EKF, UKF, bootstrap PF, FFBSi smoother (nonlinear ladder extending Kalman)
- `research/research_genmatch_redeploy.md` — Diamond & Sekhon (2013) deep-dive, GA + Mahalanobis from scratch, three use cases (twin construction, de-bias VAE, validate ablation)

**1 video-gen stream (Apr 16 2026, replaces poster-only ACT III):**
- `research/research_video_generation.md` — closed/proprietary tool ruling (DALL-E/Midjourney/Sora/Veo 3 ruled out), open-source ranking (AnimateDiff, SVD, CogVideoX, Open-Sora, HunyuanVideo, Mochi, LTX-Video), realistic 30–60s teaser pipeline, math hooks, day-by-day plan, fallback ladder

All twelve are fully referenced (papers, libraries, code skeletons).
