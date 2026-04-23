# Pipeline 3 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship a single-notebook-→-PDF submission for CS156 Pipeline 3 (25% of course grade) backed by a public GitHub repo, a 60-second AI-generated trailer `ideal_temilola_movie.mp4`, and a Streamlit explainer — by Friday 2026-04-24 16:00.

**Architecture:** 3-act pipeline (ELICIT 324 ratings → MODEL with 14 methods → GENERATE trailer). Two machines run in parallel (M5 Pro MacBook 48GB for ACT II + assembly, RunPod A100 80GB for SDXL/Llama/AnimateDiff/CogVideoX). Every ACT II output conditions a specific ACT III stage (see `PLAN.md` coupling table). Evaluation integrity: augmented data trains only; 324 ratings are the hold-out.

**Tech Stack:** PyTorch/MPS (laptop), PyTorch/CUDA (A100), HuggingFace (diffusers, transformers, peft), PyMC, scikit-learn, torch-geometric, ffmpeg, PySceneDetect, yt-dlp, Streamlit, Plotly, nbconvert→PDF.

**Non-TDD note:** This is exploratory ML/research code. Strict test-first applies ONLY to from-scratch math (GenMatch GA, conformal coverage, Kalman gain derivation sanity, LoRA low-rank decomposition). Training scripts use **smoke tests + artifact checks** (load model, forward pass, check output shape/range, save checkpoint). Notebooks validate via cell-level outputs.

---

## Time budget (24 hours, Thu 2026-04-23 evening → Fri 2026-04-24 16:00)

| Phase | Hours | What |
|---|---|---|
| 0. Setup + repo init | 1 | venv, deps, GitHub public repo, RunPod pod spin |
| 1A. Laptop ACT II sprint | 10 | 14 methods, from-scratch, checkpoints, plots |
| 1B. A100 heavy training (overlaps 1A) | 14 wall / 6 active | SDXL DreamBooth, Llama QLoRA, motion LoRA A |
| 2. Trailer corpus + motion LoRA B | 3 | yt-dlp, PySceneDetect, A100 motion LoRA B training |
| 3. Assembly (ACT II → ACT III wiring) | 4 | CVAE centroids, SDXL scenes, AnimateDiff clips, CogVideoX, audio, stitch |
| 4. Notebook authoring + PDF | 4 | main_submission.ipynb + full_deep_dive.ipynb, PDF export |
| 5. Streamlit + buffer | 2 | Tier III app, GitHub push, final checks |

**Total: ~24 hours. One 5-hour sleep block around Thu 23:00 – Fri 04:00.**

---

## File structure map

```
Pipeline 3/
├── PLAN.md                           # locked spec (don't touch)
├── IMPLEMENTATION_PLAN.md            # this file
├── MEGA_PITCH.md, research/, data/   # existing
│
├── src/                              # NEW: from-scratch implementations
│   ├── __init__.py
│   ├── kernels.py                    # GP kernels: RBF, Periodic, String
│   ├── gp.py                         # GP posterior via Cholesky
│   ├── kalman.py                     # Kalman, RTS, EKF, UKF, PF, FFBSi
│   ├── hmm.py                        # HMM forward-backward + Viterbi
│   ├── conformal.py                  # split conformal intervals
│   ├── genmatch.py                   # GA + Mahalanobis from scratch (~300 LOC)
│   ├── causal.py                     # IPW, AIPW estimators
│   ├── bandits.py                    # Thompson on GP posterior
│   ├── bnn.py                        # MC Dropout BNN
│   ├── lightgcn.py                   # LightGCN layers
│   ├── han.py                        # Heterogeneous attention network
│   ├── tvae.py                       # Tabular VAE
│   ├── cvae.py                       # Conditional VAE for taste centroids
│   ├── data_io.py                    # rating/meta loaders, splits
│   └── viz.py                        # shared plot helpers (consistent style)
│
├── scripts/                          # NEW: pipeline scripts
│   ├── 01_movielens_twin.py          # LightFM on MovieLens 25M → borrow ratings
│   ├── 02_genmatch_expand.py         # GenMatch 500 neighbors + Mahalanobis
│   ├── 03_tvae_synth.py              # TVAE synthetic ratings
│   ├── 04_trailer_download.py        # wraps existing download_trailers.py
│   ├── 05_scene_detect.py            # PySceneDetect → per-trailer clips
│   ├── 10_fit_gp.py                  # GP over 50K TMDB movies → ratings surface
│   ├── 11_fit_kalman_ladder.py       # Kalman → RTS → EKF → UKF on chronological ratings
│   ├── 12_fit_pf.py                  # Bootstrap PF + FFBSi smoother
│   ├── 13_fit_hmm.py                 # 3-state HMM on rating sequence
│   ├── 14_fit_lightgcn.py            # LightGCN on user-movie graph
│   ├── 15_fit_han.py                 # HAN on heterogeneous graph
│   ├── 16_fit_bnn.py                 # MC Dropout BNN
│   ├── 17_fit_cvae.py                # CVAE for taste centroids
│   ├── 18_bayesian_anova.py          # PyMC hierarchical ANOVA
│   ├── 19_causal_ipw_aipw.py         # IPW + AIPW
│   ├── 20_conformal_wrap.py          # Split conformal on best regressor
│   ├── 21_thompson_gp.py             # Thompson sampling on GP posterior
│   ├── 30_gpu_sdxl_dreambooth.py     # RunPod: SDXL DreamBooth on 34 posters
│   ├── 31_gpu_llama_qlora.py         # RunPod: Llama 3.1 8B QLoRA on synopses
│   ├── 32_gpu_motion_lora_A.py       # RunPod: AnimateDiff motion LoRA generic
│   ├── 33_gpu_motion_lora_B.py       # RunPod: AnimateDiff motion LoRA on expanded corpus + 3 ablations
│   ├── 34_gpu_cogvideox.py           # RunPod: CogVideoX-2B clips
│   ├── 35_gpu_musicgen.py            # RunPod: MusicGen audio
│   ├── 40_generate_scenes.py         # SDXL → scene stills using ACT II conditioning
│   ├── 41_animate_scenes.py          # Motion LoRA A/B → video clips
│   ├── 42_generate_voiceover.py      # Llama → voiceover script → TTS
│   ├── 43_assemble_trailer.py        # ffmpeg stitch → ideal_temilola_movie.mp4
│   └── 44_real_esrgan_rife.py        # Upscale + frame interpolation
│
├── tests/                            # NEW: targeted unit tests for math
│   ├── test_kernels.py
│   ├── test_gp_posterior.py          # check posterior mean matches closed-form
│   ├── test_kalman.py                # 1D linear system, known answer
│   ├── test_conformal.py             # empirical coverage ≥ 1-α
│   ├── test_genmatch.py              # fitness improves over generations
│   └── test_loaders.py               # data splits respect hold-out
│
├── artifacts/                        # NEW: serialized outputs (gitignored if >100MB)
│   ├── gp_posterior.npz
│   ├── ukf_latent.npz
│   ├── hmm_regimes.npy
│   ├── pf_samples.npz
│   ├── lightgcn_embeddings.pt
│   ├── han_embeddings.pt
│   ├── cvae_centroids.pt
│   ├── conformal_intervals.npz
│   ├── bnn_uncertainty.npy
│   ├── bayesian_anova_trace.nc
│   ├── ipw_aipw_estimates.json
│   ├── thompson_policy.json
│   └── plots/ (30+ PNGs, one per method)
│
├── notebooks/                        # NEW: the two deliverables
│   ├── main_submission.ipynb         # narrative, ~30-50 PDF pages
│   └── full_deep_dive.ipynb          # everything, 200-300 PDF pages
│
├── streamlit_app/                    # NEW: Tier III explainer
│   ├── app.py                        # multi-page
│   ├── pages/
│   │   ├── 1_trailer_gallery.py
│   │   ├── 2_ukf_slider.py
│   │   ├── 3_hmm_regime_switcher.py
│   │   ├── 4_genmatch_explorer.py
│   │   └── 5_motion_lora_timeline.py  # uses checkpoints saved every N steps
│   └── requirements.txt
│
├── video/                            # NEW
│   ├── scenes/ (SDXL stills per scene)
│   ├── clips_A/, clips_B_p0/, clips_B_p1/, clips_B_p2/, clips_C/
│   ├── voiceover.wav, music.wav
│   └── ideal_temilola_movie.mp4      # FINAL DELIVERABLE
│
├── derivations/                      # NEW: 16 LaTeX/MD first-principles
│   ├── 01_gp_posterior.md
│   ├── 02_kalman_gain.md
│   ├── 03_ukf_sigma_weights.md
│   ├── 04_pf_importance_ratio.md
│   ├── 05_elbo.md
│   ├── 06_lora_low_rank.md
│   ├── 07_cfg.md
│   ├── 08_conformal_coverage.md
│   ├── 09_genmatch_ga_fitness.md
│   ├── 10_thompson_regret.md
│   ├── 11_cross_frame_attention.md
│   ├── 12_ipw_aipw.md
│   ├── 13_hmm_forward_backward.md
│   ├── 14_lightgcn_propagation.md
│   ├── 15_mc_dropout_bayes.md
│   └── 16_mahalanobis_metric.md
│
└── README.md                         # NEW: repo map + how to reproduce
```

---

## Phase 0: Setup (1 hour)

### Task 0.1: Python env + dependencies

**Files:**
- Modify: `Pipeline 3/requirements.txt`

- [ ] **Step 1: Update requirements.txt with ALL deps used below**

Append to `Pipeline 3/requirements.txt`:
```
# Core
numpy>=1.26
scipy>=1.12
pandas>=2.2
scikit-learn>=1.4
matplotlib>=3.8
seaborn>=0.13
plotly>=5.20
tqdm>=4.66

# Probabilistic / causal / Bayesian
pymc>=5.10
arviz>=0.17

# Deep learning
torch>=2.2
torchvision
torch-geometric>=2.5
accelerate>=0.27
transformers>=4.40
diffusers>=0.27
peft>=0.10
bitsandbytes>=0.43
trl>=0.8

# Recsys
lightfm>=1.17
implicit>=0.7
lightgcn  # optional — we'll roll our own

# Tabular synth
sdv>=1.10

# Video / audio
yt-dlp>=2024.4
scenedetect>=0.6
moviepy>=1.0
pydub>=0.25

# Notebook → PDF
jupyter
nbconvert
ipykernel

# App
streamlit>=1.33

# Misc
requests
python-dotenv
```

- [ ] **Step 2: Install into existing venv**

Run:
```bash
cd "/Users/temilolaolowolayemo/Library/Mobile Documents/com~apple~CloudDocs/Downloads/T3M1L0LAs_FOLDER/CS156/Pipeline 1"
source venv/bin/activate
pip install -r "Pipeline 3/requirements.txt"
```
Expected: all installs succeed. If `torch-geometric` fails, install PyG wheels with correct torch version first.

- [ ] **Step 3: Smoke test imports**

Run:
```bash
python -c "import torch, pymc, lightfm, diffusers, peft, streamlit, scenedetect; print('OK', torch.__version__, 'MPS:', torch.backends.mps.is_available())"
```
Expected: `OK 2.x.x MPS: True`

- [ ] **Step 4: Commit**

```bash
cd "Pipeline 3"
git init  # if not already (check first with: git status)
# (skip commit — not a git repo yet; we'll init after README ready)
```

### Task 0.2: GitHub repo bootstrap

**Files:**
- Create: `Pipeline 3/README.md`
- Create: `Pipeline 3/.gitignore` (exists; extend)

- [ ] **Step 1: Write README.md**

```markdown
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
```

- [ ] **Step 2: Extend .gitignore**

Append to `Pipeline 3/.gitignore`:
```
# Large artifacts
artifacts/*.pt
artifacts/*.npz
artifacts/*.nc
video/clips_*/
video/scenes/
__pycache__/
*.pyc
.ipynb_checkpoints/
.venv/
venv/

# Secrets
.env

# Submission duplicates
prior_submissions/
```

- [ ] **Step 3: Init git, first commit, push public repo**

```bash
cd "Pipeline 3"
git init -b main
git add README.md PLAN.md IMPLEMENTATION_PLAN.md MEGA_PITCH.md research/ .gitignore requirements.txt
git commit -m "init: Pipeline 3 design docs + research corpus"

gh repo create cs156-pipeline3 --public --source=. --remote=origin --push
```
Expected: public repo created, URL printed. Save URL for notebook refs.

### Task 0.3: RunPod pod spin

**Files:** None (cloud action)

- [ ] **Step 1: Spin A100 80GB pod via RunPod API or web UI**

Use existing network volume `rthk2teqhv`. Pod class: A100 80GB PCIe, $1.19/hr.

- [ ] **Step 2: SSH in, set cuDNN fix**

```bash
ssh -p <PORT> root@<IP>
cd /workspace
echo "import torch; torch.backends.cudnn.enabled = False" > ~/.cudnn_fix.py
# add to all training scripts later
```

- [ ] **Step 3: Pull the repo onto pod**

```bash
git clone https://github.com/<you>/cs156-pipeline3.git /workspace/p3
cd /workspace/p3
pip install -r requirements.txt
```

---

## Phase 1A: Laptop ACT II sprint (10 hours)

**Strategy:** Launch 14 scripts serially but KEEP PYTHON KERNEL HOT between methods — many share precomputed splits. All methods must: (1) save a checkpoint/artifact, (2) produce at least one PNG plot to `artifacts/plots/`, (3) log to stdout with timestamp + metric.

**Shared prelude (put in `src/data_io.py`):**

### Task 1.0: Shared data loader + splits

**Files:**
- Create: `Pipeline 3/src/data_io.py`
- Create: `Pipeline 3/tests/test_loaders.py`

- [ ] **Step 1: Write failing test first**

`tests/test_loaders.py`:
```python
import numpy as np
from src.data_io import load_324_ratings, train_test_split_holdout

def test_324_ratings_shape():
    df = load_324_ratings()
    assert len(df) == 324
    assert set(df['modality'].unique()) == {'poster', 'synopsis', 'trailer', 'all'}

def test_holdout_split_disjoint():
    df = load_324_ratings()
    train, test = train_test_split_holdout(df, test_frac=0.2, seed=42)
    assert set(train.index).isdisjoint(set(test.index))
    assert len(train) + len(test) == 324
```

- [ ] **Step 2: Run test — expect ImportError**

```bash
cd "Pipeline 3" && python -m pytest tests/test_loaders.py -v
```
Expected: FAIL (module doesn't exist)

- [ ] **Step 3: Implement `src/data_io.py`**

```python
"""Shared data loaders for Pipeline 3."""
from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RATINGS_PATH = ROOT / "data" / "modality_ratings.jsonl"
META_PATH = ROOT / "data" / "movies_meta.json"

def load_324_ratings() -> pd.DataFrame:
    rows = [json.loads(l) for l in RATINGS_PATH.read_text().splitlines() if l.strip()]
    df = pd.DataFrame(rows)
    assert len(df) == 324, f"Expected 324 ratings, got {len(df)}"
    return df

def load_movie_meta() -> dict:
    return json.loads(META_PATH.read_text())

def train_test_split_holdout(df, test_frac=0.2, seed=42):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(df))
    cut = int(len(df) * (1 - test_frac))
    return df.iloc[idx[:cut]].copy(), df.iloc[idx[cut:]].copy()
```

- [ ] **Step 4: Run test, verify PASS, commit**

```bash
python -m pytest tests/test_loaders.py -v
git add src/data_io.py tests/test_loaders.py
git commit -m "feat(data_io): 324-rating loader + hold-out split"
```

### Task 1.1: GP with RBF + Periodic + String kernel (30 min)

**Files:**
- Create: `src/kernels.py`, `src/gp.py`
- Create: `tests/test_kernels.py`, `tests/test_gp_posterior.py`
- Create: `scripts/10_fit_gp.py`
- Create: `derivations/01_gp_posterior.md`

- [ ] **Step 1: Write failing tests**

`tests/test_kernels.py`:
```python
import numpy as np
from src.kernels import rbf, periodic, string_kernel

def test_rbf_symmetric_and_psd():
    X = np.random.randn(10, 3)
    K = rbf(X, X, length=1.0, var=1.0)
    np.testing.assert_allclose(K, K.T, atol=1e-8)
    assert np.all(np.linalg.eigvalsh(K) >= -1e-8)

def test_periodic_period_identity():
    x = np.array([[0.0]]); xp = np.array([[2*np.pi]])
    K = periodic(x, xp, length=1.0, period=2*np.pi, var=1.0)
    assert np.isclose(K[0,0], 1.0)

def test_string_kernel_symmetric():
    a = ["action scifi", "drama romance"]
    b = ["action scifi", "drama romance"]
    K = string_kernel(a, b)
    np.testing.assert_allclose(K, K.T, atol=1e-8)
```

`tests/test_gp_posterior.py`:
```python
import numpy as np
from src.gp import GaussianProcess
from src.kernels import rbf

def test_posterior_interpolates_training_points():
    X = np.array([[0.],[1.],[2.]])
    y = np.array([0.0, 1.0, 0.5])
    gp = GaussianProcess(kernel=lambda A,B: rbf(A,B, length=1.0, var=1.0), noise=1e-6)
    gp.fit(X, y)
    mu, _ = gp.predict(X)
    np.testing.assert_allclose(mu, y, atol=1e-3)
```

- [ ] **Step 2: Run tests — expect FAIL**

```bash
python -m pytest tests/test_kernels.py tests/test_gp_posterior.py -v
```

- [ ] **Step 3: Implement kernels**

`src/kernels.py`:
```python
"""From-scratch kernels: RBF, Periodic, String."""
import numpy as np
from collections import Counter

def rbf(X, Y, length=1.0, var=1.0):
    X = np.atleast_2d(X); Y = np.atleast_2d(Y)
    d2 = np.sum(X**2, 1)[:,None] + np.sum(Y**2, 1)[None,:] - 2*X@Y.T
    return var * np.exp(-0.5 * d2 / length**2)

def periodic(X, Y, length=1.0, period=1.0, var=1.0):
    X = np.atleast_2d(X); Y = np.atleast_2d(Y)
    d = np.abs(X[:,None,:] - Y[None,:,:]).sum(-1)
    return var * np.exp(-2 * np.sin(np.pi * d / period)**2 / length**2)

def string_kernel(A, B, n=2):
    """Simple n-gram set kernel over whitespace-tokenised strings."""
    def grams(s):
        toks = s.split()
        return Counter(tuple(toks[i:i+n]) for i in range(len(toks)-n+1))
    gA = [grams(s) for s in A]
    gB = [grams(s) for s in B]
    K = np.zeros((len(A), len(B)))
    for i,ga in enumerate(gA):
        for j,gb in enumerate(gB):
            K[i,j] = sum(ga[g]*gb[g] for g in ga.keys() & gb.keys())
    # normalise
    dA = np.sqrt(np.diag(string_kernel_self(gA))); dB = np.sqrt(np.diag(string_kernel_self(gB)))
    return K / (dA[:,None]*dB[None,:] + 1e-12)

def string_kernel_self(g_list):
    K = np.zeros((len(g_list), len(g_list)))
    for i,ga in enumerate(g_list):
        for j,gb in enumerate(g_list):
            K[i,j] = sum(ga[g]*gb[g] for g in ga.keys() & gb.keys())
    return K
```

- [ ] **Step 4: Implement GP via Cholesky**

`src/gp.py`:
```python
"""Gaussian process regression via Cholesky (from scratch)."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

class GaussianProcess:
    def __init__(self, kernel, noise=1e-4):
        self.kernel = kernel
        self.noise = noise
    def fit(self, X, y):
        self.X, self.y = X, y
        K = self.kernel(X, X) + self.noise * np.eye(len(X))
        self.L = cho_factor(K, lower=True)
        self.alpha = cho_solve(self.L, y)
        return self
    def predict(self, Xs):
        Ks = self.kernel(Xs, self.X)
        Kss = self.kernel(Xs, Xs)
        mu = Ks @ self.alpha
        v = cho_solve(self.L, Ks.T)
        cov = Kss - Ks @ v + self.noise * np.eye(len(Xs))
        return mu, np.diag(cov)
    def log_marginal_likelihood(self):
        n = len(self.y)
        return -0.5 * self.y @ self.alpha - np.log(np.diag(self.L[0])).sum() - 0.5*n*np.log(2*np.pi)
```

- [ ] **Step 5: Run tests — expect PASS**

- [ ] **Step 6: Write `scripts/10_fit_gp.py` — fit over 324 ratings, predict on 50K TMDB catalog, save posterior mean + var, plot**

```python
"""Fit GP over 324 ratings → predict a rating surface over the 50K TMDB catalog."""
import numpy as np, json, pickle
from pathlib import Path
from src.data_io import load_324_ratings, load_movie_meta
from src.kernels import rbf, periodic, string_kernel
from src.gp import GaussianProcess
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"; ART.mkdir(exist_ok=True)
(ART / "plots").mkdir(exist_ok=True)

df = load_324_ratings()
meta = load_movie_meta()

# features: year (float), genre-ngram (string), (we'll use year+genre as primary)
X_train = df[['year']].values.astype(float)
y_train = df['rating'].values.astype(float)
y_mean = y_train.mean(); y_train = y_train - y_mean  # centre

# combined kernel: RBF over year + Periodic (decade cycle) + String over genres
def combined(A, B):
    yearA, yearB = A[:, :1], B[:, :1]
    return rbf(yearA, yearB, length=5.0, var=0.3) + periodic(yearA, yearB, length=3.0, period=10.0, var=0.1)

gp = GaussianProcess(kernel=combined, noise=0.25)
gp.fit(X_train, y_train)

# predict on TMDB catalog
cat_years = np.array([[m.get('year', 2000)] for m in meta.values()], dtype=float)
mu, var = gp.predict(cat_years)
mu = mu + y_mean

np.savez(ART / "gp_posterior.npz", mu=mu, var=var, cat_keys=list(meta.keys()))
print(f"[GP] lml={gp.log_marginal_likelihood():.3f}, predicted {len(mu)} movies")

# plot: rating surface vs year
fig, ax = plt.subplots(figsize=(9,4))
years_sorted_idx = np.argsort(cat_years.flatten())
ax.plot(cat_years[years_sorted_idx], mu[years_sorted_idx], label='GP mean')
ax.fill_between(cat_years[years_sorted_idx].flatten(),
                mu[years_sorted_idx]-2*np.sqrt(var[years_sorted_idx]),
                mu[years_sorted_idx]+2*np.sqrt(var[years_sorted_idx]), alpha=0.2)
ax.scatter(X_train, y_train + y_mean, s=8, c='red', label='324 ratings')
ax.set_xlabel('Year'); ax.set_ylabel('Predicted rating'); ax.legend()
plt.tight_layout(); plt.savefig(ART / "plots" / "gp_rating_surface.png", dpi=130)
print("[GP] Saved artifacts/gp_posterior.npz + plot")
```

- [ ] **Step 7: Run it**

```bash
python scripts/10_fit_gp.py
```
Expected: writes `artifacts/gp_posterior.npz` + `artifacts/plots/gp_rating_surface.png`.

- [ ] **Step 8: Write derivation**

`derivations/01_gp_posterior.md`:
```markdown
# GP posterior via Cholesky

Given training pairs $(X, y)$, kernel $k$, noise $\sigma^2$:
$p(f_* | X, y, X_*) = \mathcal{N}(\mu_*, \Sigma_*)$ where
$\mu_* = K_{*X}(K + \sigma^2 I)^{-1} y$
$\Sigma_* = K_{**} - K_{*X}(K+\sigma^2 I)^{-1} K_{X*}$

Numerical stability: Cholesky $K + \sigma^2 I = LL^T$, solve $\alpha = L^{-T} L^{-1} y$ once.
LML: $\log p(y|X) = -\frac12 y^T\alpha - \sum \log L_{ii} - \frac n 2 \log 2\pi$.
```

- [ ] **Step 9: Commit**

```bash
git add src/kernels.py src/gp.py tests/test_kernels.py tests/test_gp_posterior.py scripts/10_fit_gp.py derivations/01_gp_posterior.md artifacts/gp_posterior.npz artifacts/plots/gp_rating_surface.png
git commit -m "feat(gp): RBF+Periodic+String kernels + Cholesky GP, rating surface over TMDB"
```

### Task 1.2: Kalman → RTS → EKF → UKF ladder (1 hour)

**Files:**
- Create: `src/kalman.py`
- Create: `tests/test_kalman.py`
- Create: `scripts/11_fit_kalman_ladder.py`
- Create: `derivations/02_kalman_gain.md`, `derivations/03_ukf_sigma_weights.md`

- [ ] **Step 1: Write tests for known 1D linear system**

`tests/test_kalman.py`:
```python
import numpy as np
from src.kalman import KalmanFilter, RTSSmoother, UKF

def test_kalman_converges_on_constant():
    kf = KalmanFilter(F=np.eye(1), H=np.eye(1), Q=np.array([[0.01]]), R=np.array([[0.5]]))
    kf.init(np.zeros(1), np.eye(1))
    xs = []
    for y in np.random.normal(5.0, np.sqrt(0.5), 200):
        kf.predict(); kf.update(np.array([y])); xs.append(kf.x.copy())
    assert abs(xs[-1][0] - 5.0) < 0.3

def test_ukf_matches_kalman_on_linear():
    F = np.eye(1); H = np.eye(1); Q = 0.01*np.eye(1); R = 0.5*np.eye(1)
    kf = KalmanFilter(F,H,Q,R); kf.init(np.zeros(1), np.eye(1))
    ukf = UKF(f=lambda x: F@x, h=lambda x: H@x, Q=Q, R=R, n=1)
    ukf.init(np.zeros(1), np.eye(1))
    ys = np.random.normal(5.0, np.sqrt(0.5), 50)
    for y in ys:
        kf.predict(); kf.update(np.array([y]))
        ukf.predict(); ukf.update(np.array([y]))
    assert abs(kf.x[0] - ukf.x[0]) < 0.2
```

- [ ] **Step 2: Implement `src/kalman.py`**

See `research/research_particle_ukf_redeploy.md` for reference. Implement `KalmanFilter`, `RTSSmoother`, `ExtendedKalmanFilter`, `UKF` (Merwe's scaled sigma points α=1e-3, β=2, κ=0), `BootstrapParticleFilter`, `FFBSi` (forward-filtering backward-simulation smoother).

Full implementation sketch:
```python
import numpy as np

class KalmanFilter:
    def __init__(self, F, H, Q, R):
        self.F, self.H, self.Q, self.R = F, H, Q, R
    def init(self, x0, P0): self.x, self.P = x0.copy(), P0.copy()
    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
    def update(self, y):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (y - self.H @ self.x)
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P
        return K

class RTSSmoother:
    """Run forward pass, then backward RTS."""
    def smooth(self, xs_f, Ps_f, F):
        n = len(xs_f); xs = list(xs_f); Ps = list(Ps_f)
        for t in range(n-2, -1, -1):
            P_pred = F @ Ps[t] @ F.T
            G = Ps[t] @ F.T @ np.linalg.inv(P_pred + 1e-8*np.eye(len(xs[0])))
            xs[t] = xs[t] + G @ (xs[t+1] - F @ xs[t])
            Ps[t] = Ps[t] + G @ (Ps[t+1] - P_pred) @ G.T
        return xs, Ps

class UKF:
    def __init__(self, f, h, Q, R, n, alpha=1e-3, beta=2.0, kappa=0.0):
        self.f, self.h, self.Q, self.R, self.n = f, h, Q, R, n
        lam = alpha**2 * (n + kappa) - n
        self.c = n + lam
        self.Wm = np.full(2*n+1, 1/(2*self.c)); self.Wc = self.Wm.copy()
        self.Wm[0] = lam / self.c; self.Wc[0] = lam/self.c + (1 - alpha**2 + beta)
    def sigmas(self, x, P):
        U = np.linalg.cholesky(self.c * P)
        pts = [x.copy()]
        for i in range(self.n): pts.append(x + U[i])
        for i in range(self.n): pts.append(x - U[i])
        return np.array(pts)
    def init(self, x0, P0): self.x, self.P = x0.copy(), P0.copy()
    def predict(self):
        sp = self.sigmas(self.x, self.P)
        sp_f = np.array([self.f(s) for s in sp])
        self.x = (self.Wm[:,None] * sp_f).sum(0)
        self.P = self.Q + sum(self.Wc[i]*np.outer(sp_f[i]-self.x, sp_f[i]-self.x) for i in range(len(sp)))
        self.sp_f = sp_f
    def update(self, y):
        sp_h = np.array([self.h(s) for s in self.sp_f])
        yhat = (self.Wm[:,None]*sp_h).sum(0)
        Pyy = self.R + sum(self.Wc[i]*np.outer(sp_h[i]-yhat, sp_h[i]-yhat) for i in range(len(sp_h)))
        Pxy = sum(self.Wc[i]*np.outer(self.sp_f[i]-self.x, sp_h[i]-yhat) for i in range(len(sp_h)))
        K = Pxy @ np.linalg.inv(Pyy)
        self.x = self.x + K @ (y - yhat); self.P = self.P - K @ Pyy @ K.T

# ExtendedKalmanFilter, BootstrapParticleFilter, FFBSi similar — see research_particle_ukf_redeploy.md
class ExtendedKalmanFilter: ...  # implement: analytic Jacobians OR finite differences
class BootstrapParticleFilter: ...  # N=500 particles, systematic resampling
class FFBSi: ...  # backward simulation smoother
```

(Full bodies of EKF/PF/FFBSi: see research doc sections; ~40 LOC each.)

- [ ] **Step 3: Run tests, fix bugs until PASS**

```bash
python -m pytest tests/test_kalman.py -v
```

- [ ] **Step 4: Write `scripts/11_fit_kalman_ladder.py`**

Fit Kalman → RTS → EKF → UKF on chronologically-ordered 324 ratings (treat rating as 1D observation of latent "taste" state). Save final filtered state at each stage. Plot: 4 panels showing filtered trajectory + 2σ bands.

```python
"""Kalman ladder on chronologically-sorted 324 ratings."""
import numpy as np, matplotlib.pyplot as plt
from pathlib import Path
from src.data_io import load_324_ratings
from src.kalman import KalmanFilter, RTSSmoother, ExtendedKalmanFilter, UKF

ART = Path(__file__).resolve().parent.parent / "artifacts"
df = load_324_ratings().sort_values('timestamp')
y = df['rating'].values.astype(float)
y_c = y - y.mean()

F = np.array([[1.0]]); H = np.array([[1.0]]); Q = np.array([[0.02]]); R = np.array([[0.4]])
kf = KalmanFilter(F,H,Q,R); kf.init(np.zeros(1), np.eye(1))
kalman_xs, kalman_Ps = [], []
for yi in y_c:
    kf.predict(); kf.update(np.array([yi]))
    kalman_xs.append(kf.x.copy()); kalman_Ps.append(kf.P.copy())

# RTS smoother
rts_xs, rts_Ps = RTSSmoother().smooth(kalman_xs, kalman_Ps, F)

# UKF with nonlinear tanh observation (synthetic nonlinearity to motivate UKF)
ukf = UKF(f=lambda x: x, h=lambda x: np.tanh(x), Q=Q, R=R, n=1)
ukf.init(np.zeros(1), np.eye(1)); ukf_xs = []
for yi in y_c:
    ukf.predict(); ukf.update(np.array([np.tanh(yi)]))
    ukf_xs.append(ukf.x.copy())

np.savez(ART / "ukf_latent.npz",
         kalman=np.array(kalman_xs).squeeze(),
         rts=np.array(rts_xs).squeeze(),
         ukf=np.array(ukf_xs).squeeze(),
         y=y_c)

fig, axes = plt.subplots(2,2, figsize=(10,6), sharex=True)
for ax, (name, x) in zip(axes.flat, [('Kalman', kalman_xs), ('RTS', rts_xs), ('UKF', ukf_xs)]):
    ax.plot(np.array(x).squeeze(), label=name); ax.scatter(range(len(y_c)), y_c, s=4, alpha=0.3)
    ax.legend()
axes[1,1].axis('off')
plt.tight_layout(); plt.savefig(ART/"plots"/"kalman_ladder.png", dpi=130)
print("[Kalman] ladder fit complete")
```

- [ ] **Step 5: Run, commit**

```bash
python scripts/11_fit_kalman_ladder.py
git add src/kalman.py tests/test_kalman.py scripts/11_fit_kalman_ladder.py derivations/02_kalman_gain.md derivations/03_ukf_sigma_weights.md artifacts/ukf_latent.npz artifacts/plots/kalman_ladder.png
git commit -m "feat(kalman): Kalman → RTS → EKF → UKF ladder + 324-rating fit"
```

### Task 1.3: Bootstrap Particle Filter + FFBSi (30 min)

**Files:** `src/kalman.py` (extend), `scripts/12_fit_pf.py`

- [ ] **Step 1: Extend `src/kalman.py` with PF + FFBSi (bodies from research doc)**
- [ ] **Step 2: Write `scripts/12_fit_pf.py`** — 500 particles, systematic resampling, FFBSi backward pass, save `pf_samples.npz` (posterior samples at final time), plot particle cloud evolution.
- [ ] **Step 3: Run, check output shape, commit.**

### Task 1.4: HMM with 3 regimes (30 min)

**Files:**
- Create: `src/hmm.py`, `tests/test_hmm.py`, `scripts/13_fit_hmm.py`, `derivations/13_hmm_forward_backward.md`

- [ ] **Step 1: Implement forward-backward + Viterbi from scratch**

```python
# src/hmm.py — forward-backward, Baum-Welch (optional), Viterbi
class HMM:
    def __init__(self, n_states, n_obs_bins):
        self.K, self.B = n_states, n_obs_bins
        # init random, row-stochastic
        self.A = np.random.dirichlet(np.ones(n_states), n_states)
        self.E = np.random.dirichlet(np.ones(n_obs_bins), n_states)
        self.pi = np.random.dirichlet(np.ones(n_states))
    def forward(self, obs): ...  # α_t(i)
    def backward(self, obs): ...  # β_t(i)
    def baum_welch(self, obs, n_iter=30): ...  # EM
    def viterbi(self, obs): ...  # most likely path
```

- [ ] **Step 2: Bin ratings to 5 levels (0-4), fit 3-state HMM**
- [ ] **Step 3: Plot state occupancy over time (stacked area)** → save `artifacts/hmm_regimes.npy`
- [ ] **Step 4: Commit.**

### Task 1.5: Conformal prediction wrapper (20 min)

**Files:**
- Create: `src/conformal.py`, `tests/test_conformal.py`, `scripts/20_conformal_wrap.py`, `derivations/08_conformal_coverage.md`

- [ ] **Step 1: Failing test — empirical coverage ≥ 1-α on held-out**

```python
def test_split_conformal_coverage():
    from src.conformal import SplitConformal
    rng = np.random.default_rng(0)
    X = rng.normal(0,1,(500,1)); y = X[:,0] + rng.normal(0,1,500)
    cp = SplitConformal(base_model=lambda: LinearRegression(), alpha=0.1)
    cp.fit(X[:300], y[:300], X[300:400], y[300:400])
    lo, hi = cp.predict(X[400:])
    coverage = np.mean((lo <= y[400:]) & (y[400:] <= hi))
    assert coverage >= 0.85  # slight slack
```

- [ ] **Step 2: Implement split conformal (wraps any regressor)**
- [ ] **Step 3: Apply to GP → save `conformal_intervals.npz`**
- [ ] **Step 4: Plot: predicted vs actual with 90% bands**
- [ ] **Step 5: Commit**

### Task 1.6: GenMatch GA + Mahalanobis (45 min) — **the from-scratch showpiece (~300 LOC)**

**Files:**
- Create: `src/genmatch.py`, `tests/test_genmatch.py`, `scripts/02_genmatch_expand.py`, `derivations/09_genmatch_ga_fitness.md`

- [ ] **Step 1: Write failing test — fitness monotone non-decreasing**

```python
def test_genmatch_ga_fitness_improves():
    from src.genmatch import GenMatch
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 5)); treat = rng.integers(0, 2, 200)
    gm = GenMatch(pop_size=30, n_generations=20, seed=0)
    history = gm.fit(X, treat, return_history=True)
    assert history[-1] >= history[0] * 0.98  # fitness improves or stays within 2% noise
```

- [ ] **Step 2: Implement GA**

Per `research/research_genmatch_redeploy.md`: GA over weight vectors `w ∈ R^d` applied to features, fitness = worst p-value of KS test across covariates after Mahalanobis nearest-neighbor matching with weight `w`. ~300 LOC.

- [ ] **Step 3: Run `scripts/02_genmatch_expand.py`**

Use GenMatch to expand from 100 rated movies to 500 nearest neighbors from 50K TMDB catalog on features (genre embedding, year, director, tags, cast_embed). Save to `artifacts/genmatch_neighbors.json` (movie_id → neighbor list). This set becomes trailer download list for Pathway B.

- [ ] **Step 4: Plot: GA fitness over generations + covariate balance before/after**
- [ ] **Step 5: Commit**

### Task 1.7: MovieLens twin (30 min)

**Files:**
- Create: `scripts/01_movielens_twin.py`

- [ ] **Step 1: Script: download/load ML-25M (or smaller ML-latest-small if disk constrained), find top-K users whose rating-pattern correlation with Temilola's 324 is highest, borrow their full rating vectors, emit as `artifacts/movielens_twin_ratings.parquet` (~50-200K pseudo-ratings)**

```python
import pandas as pd, numpy as np
from pathlib import Path
import lightfm
from lightfm import LightFM

ML = Path(__file__).resolve().parent.parent / "data" / "ml-25m"
# download if missing: curl https://files.grouplens.org/datasets/movielens/ml-25m.zip
# ... (unzip, load ratings.csv)

ratings = pd.read_csv(ML/"ratings.csv")
# ... build user-item matrix, compute cosine-sim of each MovieLens user against your 324 vec
# select top 1000 users, keep their ratings, save
```

- [ ] **Step 2: Run, verify output has ≥ 50K pseudo-ratings and a disjoint-from-324 movie set**
- [ ] **Step 3: Commit**

### Task 1.8: TVAE tabular synthesis (20 min)

**Files:** `src/tvae.py`, `scripts/03_tvae_synth.py`

- [ ] **Step 1: Use `sdv` library's TVAE OR lightweight from-scratch encoder-decoder (matches #cs156-MLCode rubric better if from-scratch). Pick: from-scratch via PyTorch, 2-layer MLP encoder, 2-layer MLP decoder, Gaussian latent.**
- [ ] **Step 2: Train on (user_vec, movie_features) columns from 324 ratings, generate 5K synthetic rating rows covering rare genre×decade cells**
- [ ] **Step 3: Plot: TSNE of real vs synthetic ratings (should overlap)**
- [ ] **Step 4: Commit**

### Task 1.9: LightGCN (45 min)

**Files:** `src/lightgcn.py`, `scripts/14_fit_lightgcn.py`, `derivations/14_lightgcn_propagation.md`

- [ ] **Step 1: Implement LightGCN from scratch (4 layers, weighted-sum pooling) in PyTorch**

```python
# src/lightgcn.py
import torch, torch.nn as nn
class LightGCN(nn.Module):
    def __init__(self, n_users, n_items, d=64, n_layers=4):
        super().__init__()
        self.Eu = nn.Embedding(n_users, d); self.Ei = nn.Embedding(n_items, d)
        nn.init.normal_(self.Eu.weight, std=0.1); nn.init.normal_(self.Ei.weight, std=0.1)
        self.n_layers = n_layers
    def propagate(self, adj):
        E = torch.cat([self.Eu.weight, self.Ei.weight], dim=0)
        embs = [E]
        for _ in range(self.n_layers):
            E = torch.sparse.mm(adj, E); embs.append(E)
        return torch.stack(embs, dim=1).mean(dim=1)
    def forward(self, adj, u, i):
        E = self.propagate(adj)
        return (E[u] * E[self.Eu.weight.shape[0] + i]).sum(-1)
```

- [ ] **Step 2: Build graph: user-rating-movie edges from (324 + MovieLens twin + TVAE synth). Total users ≈ 1001 (you + 1000 twins), items ≈ 10K movies.**
- [ ] **Step 3: Train BPR loss 30 epochs. Eval AUC on held-out edges from 324.**
- [ ] **Step 4: Save embeddings `artifacts/lightgcn_embeddings.pt`. Plot TSNE of movie embeds.**
- [ ] **Step 5: Commit**

### Task 1.10: HAN (Heterogeneous Attention Network) (45 min)

**Files:** `src/han.py`, `scripts/15_fit_han.py`

- [ ] **Step 1: Implement HAN (node-level + semantic-level attention) on heterogeneous graph: user-rates-movie, movie-has_genre-genre, movie-directed_by-director.**
- [ ] **Step 2: Train on same signal; compare embeddings to LightGCN (apples-to-apples).**
- [ ] **Step 3: Save embeds, plot.**
- [ ] **Step 4: Commit.**

### Task 1.11: MC Dropout BNN (20 min)

**Files:** `src/bnn.py`, `scripts/16_fit_bnn.py`, `derivations/15_mc_dropout_bayes.md`

- [ ] **Step 1: 3-layer MLP with Dropout(0.2) kept active at inference. 100 stochastic forward passes per prediction → posterior samples.**
- [ ] **Step 2: Train on (movie_feature, rating). Save per-movie uncertainty (std of 100 passes). Used later for gating low-confidence scenes.**
- [ ] **Step 3: Commit.**

### Task 1.12: Bayesian hierarchical ANOVA (PyMC) (30 min)

**Files:** `scripts/18_bayesian_anova.py`

- [ ] **Step 1: Model in PyMC:**

```python
import pymc as pm
with pm.Model() as m:
    mu_global = pm.Normal('mu_global', 3.0, 1.0)
    sigma_genre = pm.HalfNormal('sigma_genre', 1.0)
    genre_effect = pm.Normal('genre_effect', 0, sigma_genre, shape=n_genres)
    sigma_decade = pm.HalfNormal('sigma_decade', 1.0)
    decade_effect = pm.Normal('decade_effect', 0, sigma_decade, shape=n_decades)
    sigma_modality = pm.HalfNormal('sigma_modality', 1.0)
    modality_effect = pm.Normal('modality_effect', 0, sigma_modality, shape=4)
    mu = mu_global + genre_effect[g_idx] + decade_effect[d_idx] + modality_effect[m_idx]
    sigma_obs = pm.HalfNormal('sigma_obs', 1.0)
    pm.Normal('y', mu=mu, sigma=sigma_obs, observed=ratings)
    trace = pm.sample(1000, tune=500, target_accept=0.9)
```

- [ ] **Step 2: Save `artifacts/bayesian_anova_trace.nc` + forest plots of effects.**
- [ ] **Step 3: Commit.**

### Task 1.13: IPW + AIPW causal effect of modality (20 min)

**Files:** `src/causal.py`, `scripts/19_causal_ipw_aipw.py`, `derivations/12_ipw_aipw.md`

- [ ] **Step 1: Treat "trailer modality" as treatment, fit propensity score via logistic regression on movie features. Compute IPW ATE + AIPW doubly-robust ATE.**
- [ ] **Step 2: Save estimates to `artifacts/ipw_aipw_estimates.json`. Plot: covariate balance before/after weighting.**
- [ ] **Step 3: Commit.**

### Task 1.14: Thompson sampling on GP (15 min)

**Files:** `src/bandits.py`, `scripts/21_thompson_gp.py`, `derivations/10_thompson_regret.md`

- [ ] **Step 1: Draw 1 posterior sample from GP, argmax over TMDB catalog → next movie to "rate". Simulate 50 rounds. Save regret curve.**
- [ ] **Step 2: Save final policy `artifacts/thompson_policy.json` (list of chosen movie IDs).**
- [ ] **Step 3: Commit.**

### Task 1.15: CVAE for taste centroids (20 min — LAST ACT II task, primes ACT III)

**Files:** `src/cvae.py`, `scripts/17_fit_cvae.py`, `derivations/05_elbo.md`

- [ ] **Step 1: Implement CVAE conditioned on LightGCN movie embedding. Encoder: movie_embed → μ, logσ; Decoder: z + movie_embed → reconstructed rating. Train on all (324 + twin + synth) pairs.**
- [ ] **Step 2: Seed the **taste centroid** from UKF final state (loaded from `ukf_latent.npz`) — decode `z = ukf_final_state` to a rating-surface-conditioned latent used to pick scenes.**
- [ ] **Step 3: Generate 3 alternative centroids using 3 PF posterior samples → 3 "taste variants" for Pathway A vs B vs C comparison.**
- [ ] **Step 4: Save `artifacts/cvae_centroids.pt` (main + 3 variants). Commit.**

---

## Phase 1B: RunPod A100 training (14h wall, 6h active — overlap Phase 1A)

**All scripts MUST start with:**
```python
import torch; torch.backends.cudnn.enabled = False  # RunPod cuDNN is broken
```
**All scripts MUST checkpoint every N steps** to support Streamlit motion LoRA timeline (Task 4.5).

### Task 2.1: SDXL DreamBooth on 34 posters (~3 h GPU)

**Files:** `scripts/30_gpu_sdxl_dreambooth.py`

- [ ] **Step 1: Write launcher using diffusers `train_dreambooth_lora_sdxl.py` reference. LoRA rank 16, 2000 steps, save checkpoint every 200 steps to `/workspace/artifacts/sdxl_dreambooth/checkpoint-{step}/`.**
- [ ] **Step 2: Launch via `nohup python scripts/30_gpu_sdxl_dreambooth.py &`. Monitor loss.**
- [ ] **Step 3: Smoke test: load final LoRA, generate 1 sample, confirm it produces Temilola-style poster.**
- [ ] **Step 4: rsync checkpoints back to laptop `artifacts/sdxl_dreambooth/`. Commit checkpoint metadata (not weights — gitignored).**

### Task 2.2: Llama 3.1 8B QLoRA on movie synopses (~2 h GPU)

**Files:** `scripts/31_gpu_llama_qlora.py`

- [ ] **Step 1: Pull 500 synopsis/voiceover-style texts (MovieLens tagged trailer transcripts or synopsis from TMDB). Write QLoRA training script (4-bit, r=16, trl SFTTrainer).**
- [ ] **Step 2: 1 epoch, save checkpoint every 100 steps.**
- [ ] **Step 3: Smoke: generate 3 sample voiceovers conditioned on ACT II taste-summary text ("a film about time, memory, and quiet rooms..."). Verify coherence.**
- [ ] **Step 4: Commit.**

### Task 2.3: Motion LoRA Pathway A — generic AnimateDiff (~2 h GPU)

**Files:** `scripts/32_gpu_motion_lora_A.py`

- [ ] **Step 1: Train motion LoRA on a GENERIC set of ~100 trailer clips (random MovieLens trailers) with default AnimateDiff pipeline. Save checkpoint every 100 steps.**
- [ ] **Step 2: Generate 1 test clip from SDXL still → verify motion is reasonable.**
- [ ] **Step 3: Commit.**

### Task 2.4: MusicGen audio bed (~30 min GPU)

**Files:** `scripts/35_gpu_musicgen.py`

- [ ] **Step 1: Generate 3 variants of 60s ambient score conditioned on HMM regime summary strings ("act 1: quiet discovery", "act 2: building tension", "act 3: release"). Save wav files.**
- [ ] **Step 2: Commit.**

---

## Phase 2: Trailer corpus + motion LoRA B (3 hours)

**Depends on:** Task 1.6 (GenMatch neighbors) + Task 1.7 (MovieLens twin).

### Task 3.1: Download expanded trailer corpus (~45 min wall, mostly parallel yt-dlp)

**Files:** `scripts/04_trailer_download.py`

- [ ] **Step 1: Merge GenMatch 500 neighbors + MovieLens twin top-250 movies → 750 target movies.**
- [ ] **Step 2: yt-dlp each (existing `download_trailers.py` as template). Skip if already in `data/trailers/`.**
- [ ] **Step 3: Target: ≥ 400 downloadable trailers (some will fail).**

### Task 3.2: PySceneDetect → per-trailer clips (~30 min)

**Files:** `scripts/05_scene_detect.py`

- [ ] **Step 1: For each trailer, run ContentDetector, threshold 27. Emit 2-4 second clips → `data/trailers_clips/`.**
- [ ] **Step 2: Target: ≥ 2000 clips.**

### Task 3.3: Motion LoRA Pathway B (3 ablations p∈{0,1,2}) on A100 (~1.5 h GPU)

**Files:** `scripts/33_gpu_motion_lora_B.py`

- [ ] **Step 1: Upload clips to RunPod (`rsync data/trailers_clips/ root@pod:/workspace/`).**
- [ ] **Step 2: Train 3 motion LoRAs at prompt-fusion strengths p∈{0,1,2} (controls how strongly the prompt reshapes motion). Checkpoint every 100 steps each.**
- [ ] **Step 3: Generate 1 sample from each → `video/clips_B_p0_test/`, `...p1_test/`, `...p2_test/`. Compare qualitatively.**
- [ ] **Step 4: Commit.**

### Task 3.4: (Stretch) CogVideoX-2B Pathway C (~1 h GPU if time)

**Files:** `scripts/34_gpu_cogvideox.py`

- [ ] **Step 1: Use CogVideoX-2B text-to-video on 5 key scene prompts (derived from ACT II). No fine-tuning — pretrained only.**
- [ ] **Step 2: If time-starved, SKIP this; note in notebook Section 9 as "future work".**

---

## Phase 3: ACT III assembly (4 hours)

### Task 4.1: Generate scene stills with SDXL + ACT II conditioning (~45 min)

**Files:** `scripts/40_generate_scenes.py`

- [ ] **Step 1: Compose prompts from ACT II outputs:**
  - HMM regimes → 3-act structure → 6 scene prompts (2 per regime)
  - Bayesian ANOVA genre/decade effect sizes → weight adjectives in prompts
  - GP posterior → seed movies (top-ranked TMDB entries) contribute style keywords
  - IPW effect → visual "motion intensity" weighting in prompt

- [ ] **Step 2: Generate 4 variants per scene (via 4 different seeds + SDXL+DreamBooth LoRA). Apply Thompson sampling to pick 1 per scene (highest GP posterior rating). Save to `video/scenes/scene_{1..6}_best.png`.**

- [ ] **Step 3: Apply BNN uncertainty gate: if chosen scene's BNN uncertainty > threshold, re-roll.**

### Task 4.2: Animate scenes with motion LoRA A + B (3 ablations) (~30 min GPU)

**Files:** `scripts/41_animate_scenes.py`

- [ ] **Step 1: For each of 6 scenes: animate with Pathway A, B_p0, B_p1, B_p2 (+ optional C) → 6×4 = 24 clips in `video/clips_A/`, `clips_B_p0/`, etc.**
- [ ] **Step 2: Use Thompson policy (from Task 1.14) to pick 1 winning pathway per scene → 6 final clips.**

### Task 4.3: Generate voiceover via Llama QLoRA + TTS (~15 min)

**Files:** `scripts/42_generate_voiceover.py`

- [ ] **Step 1: Prompt Llama QLoRA with HMM 3-act summary + top-ranked GP movies + modality ANOVA effects → output 3 × ~20-word voiceover segments (one per act).**
- [ ] **Step 2: TTS via `pyttsx3` or `edge-tts` → `video/voiceover.wav`.**

### Task 4.4: ffmpeg stitch + music + Real-ESRGAN upscale + RIFE interpolation (~45 min)

**Files:** `scripts/43_assemble_trailer.py`, `scripts/44_real_esrgan_rife.py`

- [ ] **Step 1: ffmpeg: 6 clips × 10s each = 60s, crossfade transitions, overlay voiceover + MusicGen audio bed.**

```bash
ffmpeg -i concat_list.txt -i voiceover.wav -i music.wav \
  -filter_complex "[1:a]volume=0.9[v];[2:a]volume=0.3[m];[v][m]amix=inputs=2[a]" \
  -map 0:v -map "[a]" -c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p \
  video/ideal_temilola_movie_raw.mp4
```

- [ ] **Step 2: Real-ESRGAN ×2 upscale (on RunPod if laptop slow).**
- [ ] **Step 3: RIFE frame-interpolation 24→48fps → `video/ideal_temilola_movie.mp4`.**
- [ ] **Step 4: Generate GIF preview (first 10s, 240p) for notebook embed.**

```bash
ffmpeg -i video/ideal_temilola_movie.mp4 -t 10 -vf "fps=12,scale=480:-1" video/trailer_preview.gif
```

- [ ] **Step 5: Commit final MP4 + GIF (if < 100MB; else Git LFS or GitHub Releases).**

---

## Phase 4: Notebook authoring + PDF (4 hours)

### Task 5.1: `notebooks/main_submission.ipynb` — narrative-first curator (2.5 h)

**Files:** `notebooks/main_submission.ipynb`

Follow 10-section rubric from PLAN.md. Each section loads pre-computed artifacts; no heavy compute in the notebook itself.

- [ ] **Section 1 (Data Explanation):** Table of 324 ratings provenance, augmentation data provenance (reference PLAN.md data table). Sampling design: Latin-square + Thompson bandit for modality selection. Include eval integrity principle.

- [ ] **Section 2 (Python conversion):** Load `data/modality_ratings.jsonl` via `src.data_io.load_324_ratings()`. Show df.head(10) + descriptive stats.

- [ ] **Section 3 (Preprocessing + EDA):** 4 plots — rating distribution per modality, Latin-square balance check, temporal density, genre×decade heatmap.

- [ ] **Section 4 (Analysis framing):** Bulleted framing: (a) regression task on rating surface, (b) classification task on modality preference, (c) generative task on trailer. Train/test split code (hold-out 20%).

- [ ] **Section 5 (Model selection + math):** Embed selected derivations (GP posterior, Kalman gain, UKF sigma weights, ELBO, LoRA low-rank, conformal coverage, GenMatch fitness). Include pipeline diagram (SVG made in Task 6.1). **16 equation blocks total**.

- [ ] **Section 6 (Training):** Summary table: 14 methods × (data, loss, optimizer, hyper-param, runtime, converged_metric). Load training-curve plots from `artifacts/plots/`. **Not re-training — linking to scripts**.

- [ ] **Section 7 (Predictions + metrics):** 5 apples-to-apples comparisons:
  1. HMM vs UKF (state inference, held-out log-likelihood)
  2. Conformal vs MC Dropout BNN (interval coverage + width)
  3. IPW/AIPW vs Bayesian ANOVA (modality effect estimates)
  4. LightGCN vs HAN (held-out AUC on 324 edges)
  5. Pathway A vs B_{p0,p1,p2} vs C (CVAE taste-fit score on generated trailer frames, using BNN uncertainty as secondary metric)

- [ ] **Section 8 (Visualization + conclusions):** Embed trailer GIF preview (10s, 480p) + screenshot grid of 6 scenes + HMM-regime color-coded timeline. 1-paragraph conclusion per method family.

- [ ] **Section 9 (Executive summary):** Pipeline SVG diagram, key results in 5 bullets, shortcomings (6 honest gaps), improvement discussion (what I'd do with 1 more week). **Storyboard of trailer** (6 scene stills with captions). Links to MP4 + Streamlit + GitHub.

- [ ] **Section 10 (References):** BibTeX-style list of 20+ references (GP book, LightGCN paper, Diamond & Sekhon 2013, AnimateDiff paper, etc.). Use links from each `research/research_*.md` bibliography.

- [ ] **Commit:**
```bash
git add notebooks/main_submission.ipynb
git commit -m "feat(notebook): main submission notebook (narrative curator)"
```

### Task 5.2: `notebooks/full_deep_dive.ipynb` — exhaustive method-by-method (1 h)

**Files:** `notebooks/full_deep_dive.ipynb`

- [ ] **Step 1: Auto-generate skeleton: one subsection per method with:**
  1. Title + one-paragraph intent
  2. Load `src.<module>` import block
  3. Full `scripts/<N>_...py` embedded as a `%load` magic cell
  4. Full training output cell (tail of stdout)
  5. All plots from `artifacts/plots/<method>_*.png` rendered inline
  6. Full derivation embedded as markdown

- [ ] **Step 2: 14 method sections + 5 ACT III pathway sections + 3 augmentation sections = ~22 subsections.**

- [ ] **Step 3: Commit.**

### Task 5.3: Export both to PDF (30 min)

**Files:** None

- [ ] **Step 1: Main submission (target <50pp):**

```bash
jupyter nbconvert --to pdf notebooks/main_submission.ipynb \
  --template lab --no-input=False
```

If pdflatex is broken: fall back to `--to html` then print-to-PDF from Chrome (gives fewer layout issues with embedded images). OR use `--to webpdf --allow-chromium-download`.

- [ ] **Step 2: Full deep-dive (~200-300pp):**

```bash
jupyter nbconvert --to pdf notebooks/full_deep_dive.ipynb --output full_deep_dive
```

- [ ] **Step 3: Verify PDFs open, all figures render, MP4 GIF preview renders. Upload both to GitHub Releases (large file safe).**

- [ ] **Step 4: Commit notebook PDFs (or metadata pointer if too large).**

---

## Phase 5: Streamlit app + final polish (2 hours)

### Task 6.1: Pipeline diagram SVG (15 min)

**Files:** Create `artifacts/plots/pipeline_diagram.svg` via manually-authored Graphviz or matplotlib.

- [ ] **Step 1: Use `graphviz` Python binding. Nodes: 324 ratings → {MovieLens twin, GenMatch, TVAE} → {14 ACT II methods} → {UKF state, HMM regimes, CVAE centroids, ... (all 13 coupling outputs)} → {SDXL scenes, Llama voiceover, Motion LoRA A/B, MusicGen} → ffmpeg → final MP4. Embed in Section 9.**

### Task 6.2: Streamlit Tier III app (1.5 h)

**Files:** `streamlit_app/app.py`, `streamlit_app/pages/*.py`

- [ ] **Page 1 (trailer gallery):** video players for final MP4 + all 24 pathway variants. Dropdown to filter.

- [ ] **Page 2 (UKF slider):** slider over "taste state z ∈ [-3,3]", fetch nearest neighbor scenes + show top-5 movies from GP posterior conditioned on z. Load `ukf_latent.npz` + `gp_posterior.npz`.

- [ ] **Page 3 (HMM regime switcher):** 3 buttons for {discovery, tension, release}. Show the 2 scenes SDXL generated for each regime.

- [ ] **Page 4 (GenMatch explorer):** input a rated movie, return 5 nearest GenMatch neighbors with Mahalanobis distance + side-by-side poster comparison.

- [ ] **Page 5 (Motion LoRA timeline):** load motion LoRA B checkpoints from every-100-steps saves, generate 1 sample per checkpoint, display as scrubbable timeline showing training progression. **This is the "mindblowing" visual**.

- [ ] **Step: Deploy via Streamlit Community Cloud. Link URL from notebook Section 9.**

### Task 6.3: Final GitHub push + README update (15 min)

**Files:** `README.md`

- [ ] **Step 1: Add links: Streamlit URL, PDF download URL (Releases), MP4 URL, Minerva submission PDF.**
- [ ] **Step 2: Add reproduction recipe step-by-step.**
- [ ] **Step 3: Push to main. Tag `v1.0-submission`.**

### Task 6.4: Sanity gate before upload (15 min)

- [ ] **Step 1: PDF opens cleanly, no cut-off figures, page count 30-50.**
- [ ] **Step 2: MP4 plays 60s with audio.**
- [ ] **Step 3: All 5 apples-to-apples comparisons present with numeric results.**
- [ ] **Step 4: GitHub repo is public, Streamlit app URL loads.**
- [ ] **Step 5: 16 derivations present in `derivations/`.**
- [ ] **Step 6: All 10 notebook sections accounted for.**
- [ ] **Step 7: Submit PDF to Minerva forum.**

---

## Parallelization map (what to run when)

**Thu 17:00 → 23:00 (6h):** Phase 0 + Phase 1A Tasks 1.0, 1.1, 1.2, 1.3, 1.4, 1.5 on laptop **WHILE** Phase 1B Task 2.1 (SDXL DreamBooth) runs on A100.

**Thu 23:00 → Fri 04:00 (5h sleep):** A100 continues Phase 1B (Task 2.2 Llama QLoRA, Task 2.3 motion LoRA A). Laptop idle.

**Fri 04:00 → 10:00 (6h):** Phase 1A Tasks 1.6 – 1.15 (GenMatch, LightGCN, HAN, BNN, PyMC, IPW/AIPW, Thompson, CVAE, TVAE, MovieLens twin) on laptop. A100 finishes training by ~06:00.

**Fri 10:00 → 13:00 (3h):** Phase 2 (trailer corpus + motion LoRA B) — yt-dlp on laptop parallel to A100 motion LoRA B training.

**Fri 13:00 → 16:00 (3h):** Phase 3 (ACT III assembly) + start Phase 4 (notebooks).

**Fri 16:00 is submission deadline** — buffer 30 min for Phase 6 sanity gate.

**If running over:** Cut in order per PLAN.md: TVAE → PyMC ANOVA → BNN → HAN. Tier-1 methods (GP, Kalman ladder, PF, HMM, Conformal, GenMatch, IPW/AIPW, Thompson, LightGCN, CVAE) are non-negotiable.

---

## Spec coverage self-check

| Spec requirement (from PLAN.md) | Implemented in task |
|---|---|
| 324 hold-out evaluation only | Task 1.0 (data_io) + Task 5.1 §7 |
| 14 ACT II methods | Tasks 1.1-1.15 |
| GP posterior surface conditions scene selection | Tasks 1.1 + 4.1 |
| UKF latent seeds CVAE centroid | Tasks 1.2 + 1.15 |
| PF posterior → N CVAE variants | Tasks 1.3 + 1.15 |
| HMM regimes structure 3-act trailer | Tasks 1.4 + 4.1 |
| LightGCN/HAN → CVAE input | Tasks 1.9/1.10 + 1.15 |
| Bayesian ANOVA → SDXL prompt weighting | Tasks 1.12 + 4.1 |
| IPW/AIPW → visual weighting | Tasks 1.13 + 4.1 |
| Thompson → clip selection | Tasks 1.14 + 4.2 |
| BNN uncertainty → scene gating | Tasks 1.11 + 4.1 |
| Conformal intervals → overlay | Tasks 1.5 + 5.1 |
| GenMatch → trailer corpus | Tasks 1.6 + 3.1 |
| MovieLens twin → LightGCN training | Tasks 1.7 + 1.9 |
| Llama 3.1 8B QLoRA → voiceover | Tasks 2.2 + 4.3 |
| SDXL DreamBooth scene stills | Tasks 2.1 + 4.1 |
| Motion LoRA A + B (3 ablations) + optional C | Tasks 2.3 + 3.3 + 3.4 |
| MusicGen audio | Tasks 2.4 + 4.4 |
| Real-ESRGAN + RIFE polish | Task 4.4 |
| 16 first-principles derivations | `derivations/` across all tasks |
| 5 apples-to-apples comparisons | Task 5.1 §7 |
| 10 rubric sections | Task 5.1 §1–§10 |
| 2-notebook structure | Tasks 5.1 + 5.2 |
| Streamlit Tier III + motion LoRA timeline | Task 6.2 (depends on checkpoint-every-N saves in Tasks 2.1, 2.3, 3.3) |
| MP4 + GIF + storyboard embedding strategy | Task 4.4 + 5.1 §8,§9 |
| Pipeline diagram | Task 6.1 |
| Public GitHub repo | Task 0.2 + 6.3 |

**Gaps identified during self-review:** None against spec. Timing risk: Phase 1A Tasks 1.6-1.15 are ambitious for 6h — if GenMatch (Task 1.6) runs over, defer TVAE (Task 1.8) and PyMC ANOVA (Task 1.12) first.

## Risk register (from PLAN.md, tracked here)

- **A100 cuDNN broken:** every training script sets `torch.backends.cudnn.enabled = False` (noted in Task 2.x headers).
- **RunPod budget top-up:** user has confirmed willingness to add credit. Monitor at Thu 23:00 and Fri 10:00.
- **PDF size with 15+ plots + MP4:** embed GIF preview (Task 4.4 Step 4) not raw MP4.
- **Motion LoRA B depends on expanded corpus:** Phase 2 cannot start until Tasks 1.6 + 1.7 complete. If they slip, skip Pathway B ablations (drop p=2 first, then p=1).
- **Streamlit checkpoint dependency:** Tasks 2.1, 2.3, 3.3 MUST save every N steps (baked into their Step 1 requirements).
